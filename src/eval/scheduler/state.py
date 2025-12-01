from __future__ import annotations

"""State tracking helpers for dispatcher."""

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import (
    DEFAULT_COMPLETION_DIR,
    DEFAULT_EVAL_RESULT_DIR,
    DEFAULT_RUN_LOG_DIR,
    REPO_ROOT,
    RESULTS_ROOT,
)
from .dataset_utils import canonical_slug, infer_dataset_slug_from_path, safe_slug
from .jobs import JOB_CATALOGUE, JobSpec, detect_job_from_dataset


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
    score_path: Path
    completion_path: Path
    eval_result_path: Path | None
    dataset_path: str
    model_name: str
    missing_artifacts: tuple[str, ...] = ()


@dataclass
class RunningEntry:
    pid: int
    gpu: str | None
    log_path: Path | None = None


_MISSING_ARTIFACT_ALERTS: set[tuple[Path, str]] = set()


def scan_completed_jobs(log_dir: Path) -> tuple[set[CompletedKey], dict[str, CompletedRecord]]:
    completed: set[CompletedKey] = set()
    records: dict[str, CompletedRecord] = {}
    if not log_dir.exists():
        return completed, records

    for json_path in log_dir.glob("*.json"):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, Mapping):
            continue

        model_name = raw.get("model") or raw.get("raw_model")
        dataset_path = raw.get("dataset")
        log_path_str = raw.get("log_path")
        if not isinstance(model_name, str) or not isinstance(dataset_path, str) or not isinstance(log_path_str, str):
            continue

        completion_path = _resolve_run_path(log_path_str)
        missing: list[str] = []
        if not completion_path.exists():
            _warn_missing_artifact(json_path, "completion", completion_path)
            missing.append("completion")

        eval_details_path: Path | None = None
        task_details = raw.get("task_details")
        if isinstance(task_details, Mapping):
            details_path = task_details.get("eval_details_path")
            if isinstance(details_path, str):
                candidate = _resolve_run_path(details_path)
                if not candidate.exists():
                    _warn_missing_artifact(json_path, "eval_details", candidate)
                else:
                    eval_details_path = candidate

        dataset_slug = infer_dataset_slug_from_path(dataset_path)
        is_cot = _detect_is_cot(json_path, raw)
        job_name = detect_job_from_dataset(dataset_slug, is_cot)
        if not job_name:
            continue
        key = CompletedKey(
            job=job_name,
            model_slug=safe_slug(model_name),
            dataset_slug=dataset_slug,
            is_cot=is_cot,
        )
        if not missing:
            completed.add(key)
        records[json_path.stem] = CompletedRecord(
            job_id=json_path.stem,
            key=key,
            score_path=json_path,
            completion_path=completion_path,
            eval_result_path=eval_details_path,
            dataset_path=dataset_path,
            model_name=model_name,
            missing_artifacts=tuple(missing),
        )
    return completed, records


def _warn_missing_artifact(json_path: Path, category: str, path: Path) -> None:
    key = (json_path, category)
    if key in _MISSING_ARTIFACT_ALERTS:
        return
    _MISSING_ARTIFACT_ALERTS.add(key)
    print(f"⚠️  结果 {json_path} 缺少 {category} 文件：{path}")


def _resolve_run_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    candidates: list[Path] = []

    def _push(path: Path | None) -> None:
        if path is None:
            return
        normalized = path.expanduser()
        if normalized not in candidates:
            candidates.append(normalized)

    if raw.is_absolute():
        _push(raw)
        _push(_rebase_to_repo(raw))
    else:
        _push((REPO_ROOT / raw).expanduser())

    _push(_match_known_results_root(raw))

    name = raw.name
    if name:
        for base in (
            DEFAULT_COMPLETION_DIR,
            DEFAULT_EVAL_RESULT_DIR,
            DEFAULT_RUN_LOG_DIR,
            RESULTS_ROOT,
        ):
            _push(base / name)

    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else raw


def _rebase_to_repo(path: Path) -> Path | None:
    parts = path.parts
    repo_name = REPO_ROOT.name
    for idx, part in enumerate(parts):
        if part == repo_name:
            relative = Path(*parts[idx + 1 :])
            return (REPO_ROOT / relative) if relative.parts else REPO_ROOT
    for idx, part in enumerate(parts):
        if part == "results":
            relative = Path(*parts[idx:])
            return (REPO_ROOT / relative) if relative.parts else (REPO_ROOT / "results")
    return None


def _match_known_results_root(path: Path) -> Path | None:
    if not path.is_absolute():
        return None
    marker = RESULTS_ROOT.name
    for idx, part in enumerate(path.parts):
        if part == marker:
            relative = Path(*path.parts[idx + 1 :])
            return (RESULTS_ROOT / relative) if relative.parts else RESULTS_ROOT
    return None


def _detect_is_cot(log_path: Path, payload: Mapping[str, object]) -> bool:
    cot_flag = payload.get("cot")
    if isinstance(cot_flag, bool):
        return cot_flag
    cot_markers = (
        "cot_generation_template",
        "final_answer_generation_template",
        "cot_generation_input_example",
        "answer_generation_input_example",
    )
    if any(key in payload for key in cot_markers):
        return True
    name = log_path.name.lower()
    if "nocot" in name:
        return False
    return "cot" in name


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
