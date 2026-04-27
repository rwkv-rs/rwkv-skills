from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY

from ..data_utils import iter_hf_dataset, write_jsonl

_DATASET_ID = "AssistantBench/AssistantBench"
_SUPPORTED_SPLITS = ("validation",)


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"raw_metadata": value}


@FUNCTION_CALL_REGISTRY.register("assistantbench_offline")
def prepare_assistantbench_offline(output_root: Path, split: str = "validation") -> list[Path]:
    if split not in _SUPPORTED_SPLITS:
        raise ValueError(
            "assistantbench_offline 当前仅提供 validation split；"
            "官方 test split 没有公开答案。"
        )

    rows: list[dict[str, Any]] = []
    for payload in iter_hf_dataset(_DATASET_ID, split=split):
        task = payload.get("task")
        answer = payload.get("answer")
        if not isinstance(task, str) or not task.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue

        metadata = _normalize_metadata(payload.get("metadata"))
        metadata.update(
            {
                "gold_url": payload.get("gold_url"),
                "explanation": payload.get("explanation"),
                "difficulty": payload.get("difficulty"),
                "set": payload.get("set"),
                "source": _DATASET_ID,
                "benchmark_family": "assistantbench",
                "mode": "offline_static_baseline",
            }
        )

        rows.append(
            {
                "task_id": str(payload.get("id") or ""),
                "instruction": task.strip(),
                "expected_answer": answer.strip(),
                "env": {
                    "type": "single_turn_qa",
                },
                "scorer": {
                    "type": "normalized_text_exact",
                    "ignore_case": True,
                    "strip": True,
                },
                "tools": [],
                "attachments": [],
                "max_steps": 1,
                "time_limit_s": 60,
                "metadata": metadata,
            }
        )

    dataset_dir = output_root / "assistantbench_offline"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, rows)
    return [target]


__all__ = ["prepare_assistantbench_offline"]
