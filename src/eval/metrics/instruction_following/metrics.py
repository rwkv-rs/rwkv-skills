from __future__ import annotations

"""Compute instruction-following metrics from pipeline JSONL 输出。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import orjson

from . import instructions_registry


@dataclass(slots=True)
class InstructionFollowingSample:
    key: int
    prompt: str
    response: str
    instruction_ids: list[str]
    kwargs_list: list[dict]
    sample_id: int | None = None


@dataclass(slots=True)
class InstructionFollowingSampleResult:
    sample: InstructionFollowingSample
    follow_instruction_list: list[bool]

    @property
    def follow_all(self) -> bool:
        return all(self.follow_instruction_list)


@dataclass(slots=True)
class InstructionFollowingMetrics:
    prompt_accuracy: float
    instruction_accuracy: float
    tier0_accuracy: dict[str, float]
    tier1_accuracy: dict[str, float]
    samples: list[InstructionFollowingSampleResult]
    avg_at_k: dict[str, float] | None = None


def load_samples_from_jsonl(path: str | Path) -> list[InstructionFollowingSample]:
    records: list[InstructionFollowingSample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            instruction_ids = payload.get("instruction_ids") or []
            kwargs_list = payload.get("kwargs") or []
            if len(instruction_ids) != len(kwargs_list):
                raise ValueError(
                    f"instruction_ids 与 kwargs 长度不一致: {len(instruction_ids)} vs {len(kwargs_list)}"
                )
            response = payload.get("response_clean") or payload.get("output1") or ""
            prompt = payload.get("prompt") or payload.get("prompt1") or ""
            key = payload.get("key")
            if key is None:
                key = len(records)
            records.append(
                InstructionFollowingSample(
                    key=int(key),
                    prompt=prompt,
                    response=response,
                    instruction_ids=list(instruction_ids),
                    kwargs_list=list(kwargs_list),
                    sample_id=payload.get("sample_id"),
                )
            )
    return records


def evaluate_samples(
    samples: Iterable[InstructionFollowingSample],
    *,
    strict: bool = True,
) -> InstructionFollowingMetrics:
    registry = instructions_registry.INSTRUCTION_DICT
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    tier0_total: dict[str, int] = {}
    tier0_correct: dict[str, int] = {}
    tier1_total: dict[str, int] = {}
    tier1_correct: dict[str, int] = {}

    sample_results: list[InstructionFollowingSampleResult] = []

    for sample in samples:
        follow_list: list[bool] = []
        variants = None if strict else _build_loose_variants(sample.response)

        for idx, instruction_id in enumerate(sample.instruction_ids):
            instruction_cls = registry[instruction_id]
            instruction = instruction_cls(instruction_id)
            kwargs = sample.kwargs_list[idx]
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=sample.prompt)

            if strict:
                is_following = bool(sample.response.strip() and instruction.check_following(sample.response))
            else:
                is_following = False
                for variant in variants:
                    if variant.strip() and instruction.check_following(variant):
                        is_following = True
                        break
            follow_list.append(is_following)

            tier0_key = instruction_id.split(":")[0]
            tier0_total[tier0_key] = tier0_total.get(tier0_key, 0) + 1
            tier1_total[instruction_id] = tier1_total.get(instruction_id, 0) + 1
            if is_following:
                tier0_correct[tier0_key] = tier0_correct.get(tier0_key, 0) + 1
                tier1_correct[instruction_id] = tier1_correct.get(instruction_id, 0) + 1

        prompt_total += 1
        if all(follow_list):
            prompt_correct += 1
        instruction_total += len(follow_list)
        instruction_correct += sum(follow_list)
        sample_results.append(InstructionFollowingSampleResult(sample, follow_list))

    prompt_accuracy = prompt_correct / prompt_total if prompt_total else 0.0
    instruction_accuracy = instruction_correct / instruction_total if instruction_total else 0.0
    tier0_accuracy = {
        key: tier0_correct.get(key, 0) / total if total else 0.0
        for key, total in tier0_total.items()
    }
    tier1_accuracy = {
        key: tier1_correct.get(key, 0) / total if total else 0.0
        for key, total in tier1_total.items()
    }

    return InstructionFollowingMetrics(
        prompt_accuracy=prompt_accuracy,
        instruction_accuracy=instruction_accuracy,
        tier0_accuracy=tier0_accuracy,
        tier1_accuracy=tier1_accuracy,
        samples=sample_results,
    )


def compute_avg_at_k(
    samples: Iterable[InstructionFollowingSampleResult],
    ks: Iterable[int],
) -> dict[str, float]:
    """Average prompt-level accuracy across the first k samples per problem."""

    grouped: dict[int, list[tuple[int, bool]]] = {}
    for result in samples:
        sample = result.sample
        sample_id = sample.sample_id if sample.sample_id is not None else 0
        grouped.setdefault(sample.key, []).append((int(sample_id), result.follow_all))

    metrics: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        correct = 0
        total = 0
        for entries in grouped.values():
            ordered = sorted(entries, key=lambda pair: pair[0])
            if len(ordered) < k:
                continue
            selected = ordered[:k]
            correct += sum(1 for _, flag in selected if flag)
            total += k
        if total > 0:
            metrics[f"avg@{k}"] = correct / total
    return metrics


def _build_loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    variants = [response]
    if len(lines) > 1:
        variants.append("\n".join(lines[1:]).strip())
        variants.append("\n".join(lines[:-1]).strip())
        variants.append("\n".join(lines[1:-1]).strip())
    stripped = response.replace("*", "")
    variants.append(stripped)
    more = [text.replace("*", "") for text in variants if text]
    variants.extend(more)
    return [v for v in variants if v]


def write_sample_results(
    results: Iterable[InstructionFollowingSampleResult],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        for result in results:
            payload = {
                "sample_key": result.sample.key,
                "prompt": result.sample.prompt,
                "response": result.sample.response,
                "instruction_ids": result.sample.instruction_ids,
                "kwargs_list": result.sample.kwargs_list,
                "follow_instruction_list": result.follow_instruction_list,
                "follow_all": result.follow_all,
                "sample_id": result.sample.sample_id,
            }
            fh.write(orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE))
    return target


__all__ = [
    "InstructionFollowingSample",
    "InstructionFollowingSampleResult",
    "InstructionFollowingMetrics",
    "load_samples_from_jsonl",
    "evaluate_samples",
    "write_sample_results",
    "compute_avg_at_k",
]
