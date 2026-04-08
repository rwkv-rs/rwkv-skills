from __future__ import annotations

"""Runtime control files and state enums for the scheduler admin service."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any


def current_unix_millis() -> int:
    return int(time.time() * 1000)


class DesiredState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ObservedStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        return self in {self.CANCELLED, self.COMPLETED, self.FAILED}


@dataclass(slots=True)
class SchedulerControlFile:
    desired_state: str
    updated_at_unix_ms: int


@dataclass(slots=True)
class SchedulerProgressSnapshot:
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    queue_head: tuple[str, ...] = ()
    active_jobs: tuple[str, ...] = ()
    available_gpus: tuple[str, ...] = ()


@dataclass(slots=True)
class SchedulerRuntimeFile:
    observed_status: str
    started_at_unix_ms: int
    updated_at_unix_ms: int
    finished_at_unix_ms: int | None = None
    error: str | None = None
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    queue_head: list[str] = field(default_factory=list)
    active_jobs: list[str] = field(default_factory=list)
    available_gpus: list[str] = field(default_factory=list)

    def status_enum(self) -> ObservedStatus:
        return ObservedStatus(self.observed_status)


class SchedulerRuntimeControl:
    def __init__(self, *, control_path: Path, runtime_path: Path) -> None:
        self.control_path = control_path
        self.runtime_path = runtime_path

    @classmethod
    def from_dir(cls, directory: Path) -> "SchedulerRuntimeControl":
        return cls(
            control_path=directory / "control.json",
            runtime_path=directory / "runtime.json",
        )

    def desired_state(self) -> DesiredState:
        payload = self.read_control_file()
        if payload is None:
            return DesiredState.RUNNING
        return DesiredState(payload.desired_state)

    def write_desired_state(self, desired_state: DesiredState) -> SchedulerControlFile:
        payload = SchedulerControlFile(
            desired_state=desired_state.value,
            updated_at_unix_ms=current_unix_millis(),
        )
        self._write_json(self.control_path, asdict(payload))
        return payload

    def read_control_file(self) -> SchedulerControlFile | None:
        payload = self._read_json(self.control_path)
        if payload is None:
            return None
        return SchedulerControlFile(
            desired_state=str(payload.get("desired_state", DesiredState.RUNNING.value)),
            updated_at_unix_ms=int(payload.get("updated_at_unix_ms", current_unix_millis())),
        )

    def read_runtime_file(self) -> SchedulerRuntimeFile | None:
        payload = self._read_json(self.runtime_path)
        if payload is None:
            return None
        return SchedulerRuntimeFile(
            observed_status=str(payload.get("observed_status", ObservedStatus.STARTING.value)),
            started_at_unix_ms=int(payload.get("started_at_unix_ms", current_unix_millis())),
            updated_at_unix_ms=int(payload.get("updated_at_unix_ms", current_unix_millis())),
            finished_at_unix_ms=_maybe_int(payload.get("finished_at_unix_ms")),
            error=_maybe_str(payload.get("error")),
            pending_jobs=int(payload.get("pending_jobs", 0)),
            running_jobs=int(payload.get("running_jobs", 0)),
            completed_jobs=int(payload.get("completed_jobs", 0)),
            failed_jobs=int(payload.get("failed_jobs", 0)),
            queue_head=_coerce_str_list(payload.get("queue_head")),
            active_jobs=_coerce_str_list(payload.get("active_jobs")),
            available_gpus=_coerce_str_list(payload.get("available_gpus")),
        )

    def write_status(
        self,
        observed_status: ObservedStatus,
        *,
        error: str | None = None,
        progress: SchedulerProgressSnapshot | None = None,
    ) -> SchedulerRuntimeFile:
        now = current_unix_millis()
        current = self.read_runtime_file()
        started_at = current.started_at_unix_ms if current is not None else now
        finished_at = now if observed_status.is_terminal() else None
        snapshot = progress or SchedulerProgressSnapshot()
        payload = SchedulerRuntimeFile(
            observed_status=observed_status.value,
            started_at_unix_ms=started_at,
            updated_at_unix_ms=now,
            finished_at_unix_ms=finished_at,
            error=error,
            pending_jobs=snapshot.pending_jobs,
            running_jobs=snapshot.running_jobs,
            completed_jobs=snapshot.completed_jobs,
            failed_jobs=snapshot.failed_jobs,
            queue_head=list(snapshot.queue_head),
            active_jobs=list(snapshot.active_jobs),
            available_gpus=list(snapshot.available_gpus),
        )
        self._write_json(self.runtime_path, asdict(payload))
        return payload

    def heartbeat(self, *, progress: SchedulerProgressSnapshot | None = None) -> SchedulerRuntimeFile | None:
        current = self.read_runtime_file()
        if current is None:
            return None
        snapshot = progress or SchedulerProgressSnapshot(
            pending_jobs=current.pending_jobs,
            running_jobs=current.running_jobs,
            completed_jobs=current.completed_jobs,
            failed_jobs=current.failed_jobs,
            queue_head=tuple(current.queue_head),
            active_jobs=tuple(current.active_jobs),
            available_gpus=tuple(current.available_gpus),
        )
        return self.write_status(current.status_enum(), error=current.error, progress=snapshot)

    def snapshot(self) -> tuple[DesiredState, SchedulerRuntimeFile | None]:
        return self.desired_state(), self.read_runtime_file()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.replace(path)


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if stripped else None
    raise TypeError(f"unsupported integer payload: {value!r}")


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


__all__ = [
    "DesiredState",
    "ObservedStatus",
    "SchedulerControlFile",
    "SchedulerProgressSnapshot",
    "SchedulerRuntimeControl",
    "SchedulerRuntimeFile",
    "current_unix_millis",
]
