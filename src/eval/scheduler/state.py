from __future__ import annotations

"""State tracking helpers for dispatcher."""

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .dataset_utils import safe_slug
from .jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService


@dataclass(frozen=True)
class CompletedKey:
    job: str
    model_slug: str
    dataset_slug: str
    is_cot: bool


@dataclass(frozen=True)
class CompletedRecord:
    job_id: str
    key: CompletedKey
    model_name: str
    samples: int | None = None
    problems: int | None = None
    metrics: dict | None = None
    task_details: dict | None = None
    version_id: str | None = None


@dataclass
class RunningEntry:
    pid: int
    gpu: str | None
    log_path: Path | None = None


_JOB_DETECTION_ALERTS: set[tuple[Path, str]] = set()


def scan_completed_jobs(log_dir: Path) -> tuple[set[CompletedKey], dict[str, CompletedRecord]]:
    completed: set[CompletedKey] = set()
    records: dict[str, CompletedRecord] = {}
    if not DEFAULT_DB_CONFIG.enabled:
        return completed, records
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    for raw in service.list_latest_scores():
        if not isinstance(raw, Mapping):
            continue
        model_name = raw.get("model")
        dataset_slug = raw.get("dataset")
        if not isinstance(model_name, str) or not isinstance(dataset_slug, str):
            continue
        detected_is_cot = bool(raw.get("cot"))
        job_name = detect_job_from_dataset(dataset_slug, detected_is_cot)
        key_is_cot = detected_is_cot
        if not job_name:
            job_name, key_is_cot = _infer_job_from_task_field(
                Path("<db>"),
                raw,
                dataset_slug,
                detected_is_cot,
            )
        if not job_name:
            _warn_job_detection(
                Path("<db>"),
                "unknown_job",
                f"⚠️  无法从结果解析 job，已忽略: dataset={dataset_slug} cot={detected_is_cot} task={raw.get('task')!r}",
            )
            continue
        raw_problems = raw.get("problems")
        raw_samples = raw.get("samples")
        if isinstance(raw_samples, dict):
            raw_samples = raw_samples.get("total")
        raw_counts = raw_problems if raw_problems is not None else raw_samples
        try:
            samples = int(raw_counts) if raw_counts is not None else None
        except (TypeError, ValueError):
            samples = None
        if samples is not None and samples <= 0:
            samples = None
        key = CompletedKey(
            job=job_name,
            model_slug=safe_slug(model_name),
            dataset_slug=dataset_slug,
            is_cot=key_is_cot,
        )
        job_id = _completed_job_id(key)
        completed.add(key)
        records[job_id] = CompletedRecord(
            job_id=job_id,
            key=key,
            model_name=model_name,
            samples=samples,
            problems=raw.get("problems"),
            metrics=raw.get("metrics") if isinstance(raw.get("metrics"), dict) else None,
            task_details=raw.get("task_details") if isinstance(raw.get("task_details"), dict) else None,
            version_id=str(raw.get("version_id")) if raw.get("version_id") is not None else None,
        )
    return completed, records


def _completed_job_id(key: CompletedKey) -> str:
    suffix = "cot" if key.is_cot else "nocot"
    run_slug = safe_slug(f"{key.dataset_slug}_{suffix}_{key.model_slug}")
    return f"{key.job}__{run_slug}"


def _warn_job_detection(json_path: Path, category: str, message: str) -> None:
    key = (json_path, category)
    if key in _JOB_DETECTION_ALERTS:
        return
    _JOB_DETECTION_ALERTS.add(key)
    print(message)


def _infer_job_from_task_field(
    json_path: Path,
    payload: Mapping[str, object],
    dataset_slug: str,
    detected_is_cot: bool,
) -> tuple[str | None, bool]:
    """Best-effort mapping for legacy/buggy score payloads.

    Some bin scripts historically wrote mismatched `cot` flags, which prevents the
    `dataset_slug + cot` heuristic from mapping back to a JobSpec. When the score
    payload contains a `task` field that matches a known job, prefer the JobSpec
    metadata so the scheduler can still converge.
    """

    task = payload.get("task")
    if not isinstance(task, str) or not task:
        return None, detected_is_cot
    spec = JOB_CATALOGUE.get(task)
    if spec is None:
        return None, detected_is_cot
    if dataset_slug not in spec.dataset_slugs:
        return None, detected_is_cot
    if spec.is_cot != detected_is_cot:
        _warn_job_detection(
            json_path,
            "cot_mismatch",
            (
                "⚠️  结果 cot 标记与任务定义不一致，已按 JobSpec 纠正: "
                f"task={task} detected_cot={detected_is_cot} expected_cot={spec.is_cot} ({json_path})"
            ),
        )
    return task, spec.is_cot


def load_running(pid_dir: Path) -> dict[str, RunningEntry]:
    running: dict[str, RunningEntry] = {}
    if not pid_dir.exists():
        return running
    for pid_file in pid_dir.glob("*.pid"):
        job_id = pid_file.stem
        lines = pid_file.read_text().splitlines()
        if not lines:
            continue
        try:
            pid = int(lines[0])
        except ValueError:
            continue
        if not pid_alive(pid):
            pid_file.unlink(missing_ok=True)
            continue
        gpu = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
        log_path: Path | None = None
        if len(lines) > 2 and lines[2].strip():
            log_path = Path(lines[2].strip())
        running[job_id] = RunningEntry(pid=pid, gpu=gpu, log_path=log_path)
    return running


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_pid_file(pid_dir: Path, job_id: str, pid: int, gpu: str | None, log_name: str) -> None:
    pid_dir.mkdir(parents=True, exist_ok=True)
    payload = [str(pid), gpu or "", log_name]
    (pid_dir / f"{job_id}.pid").write_text("\n".join(payload), encoding="utf-8")


def stop_job(job_id: str, pid_dir: Path) -> None:
    pid_file = pid_dir / f"{job_id}.pid"
    if not pid_file.exists():
        print(f"ℹ️  {job_id} 未找到 PID 文件")
        return
    lines = pid_file.read_text().splitlines()
    if not lines:
        pid_file.unlink(missing_ok=True)
        return
    try:
        pid = int(lines[0])
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return
    if pid_alive(pid):
        print(f"⏹  停止 {job_id} (pid={pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        print(f"ℹ️  PID {pid} 已退出")
    pid_file.unlink(missing_ok=True)


def stop_all_jobs(pid_dir: Path) -> None:
    running = load_running(pid_dir)
    if not running:
        return
    for job_id in sorted(running.keys()):
        stop_job(job_id, pid_dir)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def tail_file(path: Path, tail_lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()[-tail_lines:]
    except Exception:
        return []


def terminate_process(pid: int) -> None:
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


__all__ = [
    "CompletedKey",
    "CompletedRecord",
    "RunningEntry",
    "scan_completed_jobs",
    "load_running",
    "pid_alive",
    "write_pid_file",
    "stop_job",
    "stop_all_jobs",
    "ensure_dirs",
    "tail_file",
]
