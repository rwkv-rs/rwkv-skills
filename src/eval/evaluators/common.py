from __future__ import annotations

"""Shared helper structures for evaluator pipelines (JSONL 输出 & 调试工具)."""

from dataclasses import dataclass, field
import errno
import json
import os
from pathlib import Path
import threading
from queue import Queue
from typing import Sequence
import sys

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
    output: str | None = None
    finish_reason: str | None = None
    logits: dict[str, float] | None = None


@dataclass(slots=True)
class SampleRecord:
    index: int
    dataset: str
    stages: list[StageRecord]
    metadata: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        payload = {"sample_index": self.index, "dataset": self.dataset}
        for idx, stage in enumerate(self.stages, start=1):
            payload[f"prompt{idx}"] = stage.prompt
            if stage.output is not None:
                payload[f"output{idx}"] = stage.output
            if stage.finish_reason is not None:
                payload[f"finish_reason{idx}"] = stage.finish_reason
            if stage.logits is not None:
                payload[f"logits{idx}"] = stage.logits
        payload.update(self.metadata)
        return payload


@dataclass(slots=True)
class ResumeState:
    next_index: int
    completed: int
    append: bool

    @property
    def has_progress(self) -> bool:
        return self.append


class JsonlStageWriter:
    _SENTINEL = object()

    def __init__(self, path: str | Path, *, resume: bool = False):
        self.path = Path(path)
        self._mode = "ab" if resume else "wb"
        self._fh = None
        self._open_file()
        self._queue: Queue[SampleRecord | object] = Queue()
        self._closed = False
        self._worker_exc: BaseException | None = None
        self._worker = threading.Thread(
            target=self._writer_loop,
            name=f"JsonlStageWriter[{self.path.name}]",
            daemon=True,
        )
        self._worker.start()

    def write(self, record: SampleRecord) -> None:
        self._ensure_ready()
        self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._SENTINEL)
        if self._worker.is_alive():
            self._worker.join()
        if self._worker_exc:
            raise RuntimeError("JSONL writer thread failed") from self._worker_exc

    def __enter__(self) -> "JsonlStageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._SENTINEL:
                        break
                    payload = orjson.dumps(item.as_payload(), option=orjson.OPT_APPEND_NEWLINE)
                    self._write_payload(payload)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._worker_exc = exc
        finally:
            try:
                if self._fh:
                    self._fh.close()
            except OSError:
                pass

    def _ensure_ready(self) -> None:
        if self._worker_exc:
            raise RuntimeError("JSONL writer thread failed") from self._worker_exc
        if self._closed:
            raise RuntimeError("JSONL writer already closed")

    def _open_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(self._mode)
        self._mode = "ab"

    def _reopen_file(self) -> None:
        try:
            if self._fh:
                self._fh.close()
        except OSError:
            pass
        self._open_file()

    def _write_payload(self, payload: bytes) -> None:
        attempts = 0
        while True:
            try:
                if not self._fh:
                    self._open_file()
                self._fh.write(payload)
                self._fh.flush()
                return
            except OSError as exc:
                if exc.errno != errno.ENOENT or attempts >= 1:
                    raise FileNotFoundError(f"结果文件 {self.path} 无法写入：{exc}") from exc
                attempts += 1
                print(f"[writer] 结果目录缺失，正在重新创建：{self.path.parent}", file=sys.stderr)
                self._reopen_file()


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


def detect_resume_state(path: str | Path) -> ResumeState:
    target = Path(path)
    if not target.exists():
        return ResumeState(next_index=0, completed=0, append=False)
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
                idx = payload.get("sample_index")
                if isinstance(idx, int) and idx >= 0:
                    seen.add(idx)
    except OSError:
        return ResumeState(next_index=0, completed=0, append=False)

    next_index = 0
    while next_index in seen:
        next_index += 1
    return ResumeState(next_index=next_index, completed=len(seen), append=next_index > 0)


def _max_sample_id(path: Path) -> int | None:
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
                sample_id = payload.get("sample_id")
                if isinstance(sample_id, int) and sample_id >= 0:
                    if max_id is None or sample_id > max_id:
                        max_id = sample_id
    except OSError:
        return None
    return max_id


def ensure_resume_samples_compatible(path: Path, samples_per_task: int) -> None:
    max_id = _max_sample_id(path)
    if max_id is None:
        return
    required = max_id + 1
    if samples_per_task >= required:
        return
    raise ValueError(
        f"已有日志 {path} 含 sample_id={max_id}，需要每题至少 {required} 个样本才能继续（由 pass-k 最大值决定）；"
        "请删除旧日志或使用不小于该值的生成次数重试。"
    )


__all__ = [
    "ProbeConfig",
    "StageRecord",
    "SampleRecord",
    "JsonlStageWriter",
    "ResumeState",
    "DebugCaptureConfig",
    "DebugCaptureBuffer",
    "detect_resume_state",
    "ensure_resume_samples_compatible",
]
