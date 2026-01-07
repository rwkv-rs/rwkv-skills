from __future__ import annotations

"""Schema helpers for `results/completions` and `results/eval` artifacts.

This project enforces a strict separation:
- `results/completions`: model generation traces (prompts/completions/stop reasons only)
- `results/eval`: evaluator judgments (context + pass/fail + fail reason + extracted answer + reference answer)

Both formats aim to stay stable and reusable. Eval artifacts are derived from
completions but may store additional evaluator-facing fields to enable
downstream analysis (e.g. wrong-answer checking).
"""

from dataclasses import asdict
from typing import Any, Iterable

from src.eval.scheduler.dataset_utils import canonical_slug, split_benchmark_and_split
from src.infer.sampling import SamplingConfig


def sampling_config_to_dict(config: SamplingConfig) -> dict[str, object]:
    raw = asdict(config)
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def dataset_slug_parts(dataset_slug: str) -> tuple[str, str]:
    """Return (benchmark_name, dataset_split) from a canonical dataset slug."""
    return split_benchmark_and_split(canonical_slug(dataset_slug))


def iter_stage_indices(payload: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    for key in payload:
        if not key.startswith("prompt") and not key.startswith("completion") and not key.startswith("stop_reason"):
            continue
        suffix = key.removeprefix("prompt").removeprefix("completion").removeprefix("stop_reason")
        if suffix.isdigit():
            indices.add(int(suffix))
    return sorted(indices)


def build_context_from_completions(payload: dict[str, Any]) -> str:
    """Concatenate prompt+completion segments into the final model context."""
    parts: list[str] = []
    for idx in iter_stage_indices(payload):
        prompt = payload.get(f"prompt{idx}")
        completion = payload.get(f"completion{idx}")
        if prompt is None or completion is None:
            continue
        parts.append(str(prompt))
        parts.append(str(completion))
    return "".join(parts)


def make_eval_payload(
    completions_payload: dict[str, Any],
    *,
    is_passed: bool,
    fail_reason: str | None = None,
    answer: str | None = None,
    ref_answer: str | None = None,
) -> dict[str, Any]:
    """Build a canonical `results/eval` line from a `results/completions` line."""
    passed = bool(is_passed)
    reason = "" if passed else (fail_reason or "incorrect")
    return {
        "benchmark_name": str(completions_payload.get("benchmark_name", "")),
        "dataset_split": str(completions_payload.get("dataset_split", "")),
        "sample_index": int(completions_payload.get("sample_index", 0)),
        "repeat_index": int(completions_payload.get("repeat_index", 0)),
        "context": build_context_from_completions(completions_payload),
        "answer": "" if answer is None else str(answer),
        "ref_answer": "" if ref_answer is None else str(ref_answer),
        "is_passed": passed,
        "fail_reason": reason,
    }


def prompt_delta(full_prompt: str, prior_context: str) -> str:
    """Return the suffix of `full_prompt` after `prior_context` (must be a strict prefix)."""
    if full_prompt.startswith(prior_context):
        return full_prompt[len(prior_context) :]
    raise ValueError("stage prompt is not prefixed by prior context; cannot compute delta")


def strip_artifact_suffix(dataset_stem: str) -> str:
    """Strip known artifact-only suffixes from a dataset stem (e.g. '__cot')."""
    stem = canonical_slug(dataset_stem)
    if stem.endswith("__cot"):
        return stem[: -len("__cot")]
    return stem


def normalize_sampling_config_by_stage(items: Iterable[tuple[int, SamplingConfig]]) -> dict[str, object]:
    """Build the `sampling_config` payload: only stages that actually sampled."""
    payload: dict[str, object] = {}
    for stage_idx, cfg in items:
        payload[f"stage{int(stage_idx)}"] = sampling_config_to_dict(cfg)
    return payload
