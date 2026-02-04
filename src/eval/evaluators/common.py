from __future__ import annotations

"""Shared helper structures for evaluator pipelines (调试工具)."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Sequence

import orjson


@dataclass(slots=True)
class ProbeConfig:
    """用于评估阶段的轻量化探测：限制样本数/生成长度。"""

    max_samples: int | None = None
    max_tokens: int | None = None

    def apply(self, records: Sequence) -> list:
        if not self.max_samples or self.max_samples <= 0:
            return list(records)
        return list(records)[: max(1, min(self.max_samples, len(records)))]


@dataclass(slots=True)
class StageRecord:
    prompt: str
    completion: str
    stop_reason: str


@dataclass(slots=True)
class SampleRecord:
    benchmark_name: str
    dataset_split: str
    sample_index: int
    repeat_index: int
    stages: list[StageRecord]
    sampling_config: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        payload = {
            "benchmark_name": self.benchmark_name,
            "dataset_split": self.dataset_split,
            "sample_index": int(self.sample_index),
            "repeat_index": int(self.repeat_index),
            "sampling_config": self.sampling_config or {},
        }
        for idx, stage in enumerate(self.stages, start=1):
            payload[f"prompt{idx}"] = stage.prompt
            payload[f"completion{idx}"] = stage.completion
            payload[f"stop_reason{idx}"] = stage.stop_reason
        return payload


@dataclass(slots=True)
class ResumeState:
    next_index: int
    completed: int
    append: bool

    @property
    def has_progress(self) -> bool:
        return self.append



@dataclass(slots=True)
class DebugCaptureConfig:
    path: Path
    limit: int
    exit_on_limit: bool = False

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and str(self.path)

    @classmethod
    def from_env(cls) -> DebugCaptureConfig | None:
        raw_path = os.environ.get("RUN_DEBUG_CAPTURE_PATH")
        limit = _env_int(os.environ.get("RUN_DEBUG_CAPTURE_LIMIT"), 0)
        exit_flag = _env_flag(os.environ.get("RUN_DEBUG_CAPTURE_EXIT"))
        if not raw_path or limit <= 0:
            return None
        return cls(path=Path(raw_path).expanduser(), limit=limit, exit_on_limit=exit_flag)


@dataclass(slots=True)
class DebugCaptureBuffer:
    config: DebugCaptureConfig
    records: list[dict] = field(default_factory=list)

    def add(self, record: dict) -> bool:
        if not self.config.enabled:
            return False
        if len(self.records) >= self.config.limit:
            return True
        self.records.append(record)
        return len(self.records) >= self.config.limit

    def flush(self, **metadata) -> None:
        if not self.config.enabled:
            return
        payload = {
            **metadata,
            "limit": self.config.limit,
            "collected": len(self.records),
            "records": self.records,
        }
        target = self.config.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=4)


def _env_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _env_flag(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}


def detect_resume_state(path: str | Path, *, repeats: int = 1) -> ResumeState:
    target = Path(path)
    if not target.exists():
        return ResumeState(next_index=0, completed=0, append=False)
    repeats = max(1, int(repeats))
    seen: set[int] = set()
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                sample_index = payload.get("sample_index")
                repeat_index = payload.get("repeat_index", 0)
                if not (isinstance(sample_index, int) and sample_index >= 0):
                    continue
                if not (isinstance(repeat_index, int) and repeat_index >= 0):
                    continue
                seen.add(sample_index * repeats + repeat_index)
    except OSError:
        return ResumeState(next_index=0, completed=0, append=False)

    next_index = 0
    while next_index in seen:
        next_index += 1
    return ResumeState(next_index=next_index, completed=len(seen), append=next_index > 0)


def _max_repeat_index(path: Path) -> int | None:
    if not path.exists():
        return None
    max_id: int | None = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                repeat_index = payload.get("repeat_index")
                if isinstance(repeat_index, int) and repeat_index >= 0:
                    if max_id is None or repeat_index > max_id:
                        max_id = repeat_index
    except OSError:
        return None
    return max_id


def ensure_resume_samples_compatible(path: Path, samples_per_task: int) -> None:
    max_id = _max_repeat_index(path)
    if max_id is None:
        return
    required = max_id + 1
    if samples_per_task >= required:
        return
    raise ValueError(
        f"已有日志 {path} 含 repeat_index={max_id}，需要每题至少 {required} 个样本才能继续（由 pass-k/avg-k 最大值决定）；"
        "请删除旧日志或使用不小于该值的生成次数重试。"
    )


__all__ = [
    "ProbeConfig",
    "StageRecord",
    "SampleRecord",
    "ResumeState",
    "DebugCaptureConfig",
    "DebugCaptureBuffer",
    "detect_resume_state",
    "ensure_resume_samples_compatible",
]
