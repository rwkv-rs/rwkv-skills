from __future__ import annotations

"""File-backed evaluation persistence for inspectable local experiment runs."""

import json
import math
import os
import subprocess
import threading
from collections.abc import Iterable, Mapping as AbcMapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from src.db.eval_db_service import ResumeContext, TaskLookup
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.results.schema import iter_stage_indices, strict_nonneg_int
from src.eval.scheduler.config import REPO_ROOT, RESULTS_ROOT
from src.eval.scheduler.dataset_utils import split_benchmark_and_split
from src.eval.scheduler.models import normalize_model_name


_GIT_SHA_CACHE: str | None = None


def _get_git_sha() -> str:
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    env_sha = os.environ.get("RWKV_GIT_SHA", "").strip()
    if env_sha:
        _GIT_SHA_CACHE = env_sha
        return env_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    _GIT_SHA_CACHE = result.stdout.strip()
    return _GIT_SHA_CACHE


def _now_cn() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False).isoformat()


def _json_root() -> Path:
    raw = os.environ.get("RWKV_EVAL_JSON_ROOT") or os.environ.get("RUN_JSON_RESULT_DIR")
    return Path(raw).expanduser() if raw else RESULTS_ROOT / "json_runs"


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").replace("\x00", "")
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_json(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _sanitize_json(value.model_dump())
        except Exception:
            return str(value)
    if isinstance(value, AbcMapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized_key = _sanitize_json(key)
            if not isinstance(sanitized_key, str):
                sanitized_key = str(sanitized_key)
            sanitized[sanitized_key] = _sanitize_json(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_json(item) for item in sorted(value, key=str)]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_sanitize_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize_json(dict(payload)), fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_json_dumps(dict(row)))
            fh.write("\n")
    tmp.replace(path)


def _attempt_key(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        strict_nonneg_int(payload.get("sample_index"), "sample_index"),
        strict_nonneg_int(payload.get("repeat_index"), "repeat_index"),
        strict_nonneg_int(payload.get("pass_index", 0), "pass_index"),
    )


def _completion_stage(payload: Mapping[str, Any]) -> str:
    return str(payload.get("_stage", "answer") or "answer").strip().lower()


def _build_completion_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    payload_dict = dict(payload)
    for idx in iter_stage_indices(payload_dict):
        stages.append(
            {
                "prompt": payload.get(f"prompt{idx}"),
                "completion": payload.get(f"completion{idx}"),
                "stop_reason": payload.get(f"stop_reason{idx}"),
            }
        )
    context: dict[str, Any] = {
        "stages": stages,
        "sampling_config": payload.get("sampling_config", {}),
    }
    for key in ("agent_result", "agent_info", "agent_trace", "task_id", "domain", "instruction"):
        value = payload.get(key)
        if value is not None:
            context[key] = value
    sanitized = _sanitize_json(context)
    return sanitized if isinstance(sanitized, dict) else {}


def _resolve_task_config_path(benchmark_name: str, model: str) -> str | None:
    config_path = config_path_for_benchmark(benchmark_name, model)
    if config_path.exists():
        return str(config_path)
    fallback_path = config_path_for_benchmark(benchmark_name, None)
    if fallback_path.exists():
        return str(fallback_path)
    return None


class EvalJsonService:
    """Drop-in subset of EvalDbService backed by structured JSON artifacts."""

    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else _json_root()
        self.tasks_root = self.root / "tasks"
        self._lock = threading.RLock()
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    def _task_dir(self, task_id: str | int) -> Path:
        return self.tasks_root / str(task_id)

    def _task_path(self, task_id: str | int) -> Path:
        return self._task_dir(task_id) / "task.json"

    def _completions_path(self, task_id: str | int) -> Path:
        return self._task_dir(task_id) / "completions.jsonl"

    def _eval_path(self, task_id: str | int) -> Path:
        return self._task_dir(task_id) / "eval.jsonl"

    def _checker_path(self, task_id: str | int) -> Path:
        return self._task_dir(task_id) / "checker.jsonl"

    def _score_path(self, task_id: str | int) -> Path:
        return self._task_dir(task_id) / "score.json"

    def _counter_path(self) -> Path:
        return self.root / "task_counter.txt"

    def _next_task_id(self) -> int:
        with self._lock:
            counter_path = self._counter_path()
            current = 0
            if counter_path.exists():
                raw = counter_path.read_text(encoding="utf-8").strip()
                current = int(raw) if raw.isdigit() else 0
            if current <= 0:
                existing = [
                    int(path.name)
                    for path in self.tasks_root.iterdir()
                    if path.is_dir() and path.name.isdigit()
                ]
                current = max(existing, default=0)
            next_id = current + 1
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            counter_path.write_text(f"{next_id}\n", encoding="utf-8")
            return next_id

    def _iter_task_meta(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.tasks_root.glob("*/task.json"), key=lambda item: int(item.parent.name) if item.parent.name.isdigit() else 0):
            meta = _read_json(path)
            if isinstance(meta, dict):
                rows.append(meta)
        return rows

    def get_resume_context(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
        job_name: str | None = None,
        sampling_config: dict[str, Any] | None = None,
        force_new_task: bool = False,
    ) -> ResumeContext:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        normalized_model = normalize_model_name(model)
        sanitized_sampling = _sanitize_json(sampling_config) if sampling_config is not None else None
        ctx = ResumeContext()
        if force_new_task:
            return ctx

        git_sha = _get_git_sha()
        config_path = _resolve_task_config_path(benchmark_name, normalized_model)
        matches: list[TaskLookup] = []
        completed_ids: list[int] = []
        resumable_ids: list[int] = []
        for meta in self._iter_task_meta():
            if meta.get("benchmark_name") != benchmark_name:
                continue
            if meta.get("benchmark_split") != benchmark_split:
                continue
            if meta.get("model") != normalized_model:
                continue
            if bool(meta.get("is_param_search", False)) != bool(is_param_search):
                continue
            if str(meta.get("evaluator") or "") != str(job_name or ""):
                continue
            if str(meta.get("git_hash") or "") != git_sha:
                continue
            if (meta.get("config_path") or None) != config_path:
                continue
            if meta.get("sampling_config") != sanitized_sampling:
                continue
            task_id = int(meta["task_id"])
            status = str(meta.get("status") or "")
            if self._score_path(task_id).exists():
                status = "Completed"
            lookup = TaskLookup(task_id=task_id, status=status)
            matches.append(lookup)
            normalized = status.strip().lower()
            if normalized == "completed":
                completed_ids.append(task_id)
            elif normalized in {"running", "failed"}:
                resumable_ids.append(task_id)

        ctx.matching_tasks = tuple(matches)
        ctx.completed_task_ids = tuple(completed_ids)
        ctx.resumable_task_ids = tuple(resumable_ids)
        if not completed_ids and len(resumable_ids) == 1:
            ctx.task_id = resumable_ids[0]
            ctx.can_resume = True
            ctx.completed_keys = self.list_completion_keys(task_id=str(ctx.task_id), status="Completed")
        elif completed_ids:
            ctx.task_id = completed_ids[-1]
        elif resumable_ids:
            ctx.task_id = resumable_ids[-1]
        return ctx

    def create_task_from_context(
        self,
        *,
        ctx: ResumeContext,
        job_name: str | None,
        dataset: str,
        model: str,
        is_param_search: bool,
        sampling_config: dict[str, Any] | None = None,
    ) -> str:
        if ctx.can_resume and ctx.task_id is not None:
            self.update_task_status(task_id=str(ctx.task_id), status="running")
            score_path = self._score_path(ctx.task_id)
            if score_path.exists():
                score_path.unlink()
            return str(ctx.task_id)

        task_id = self._next_task_id()
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        normalized_model = normalize_model_name(model)
        meta = {
            "task_id": task_id,
            "persistence": "json",
            "created_at": _now_cn(),
            "status": "running",
            "git_hash": _get_git_sha(),
            "config_path": _resolve_task_config_path(benchmark_name, normalized_model),
            "evaluator": job_name or "",
            "dataset": str(dataset),
            "benchmark_name": benchmark_name,
            "benchmark_split": benchmark_split,
            "model": normalized_model,
            "is_param_search": bool(is_param_search),
            "desc": os.environ.get("RWKV_TASK_DESC"),
            "sampling_config": _sanitize_json(sampling_config) if sampling_config is not None else None,
            "artifacts": {
                "completions": "completions.jsonl",
                "eval": "eval.jsonl",
                "checker": "checker.jsonl",
                "score": "score.json",
            },
        }
        _write_json(self._task_path(task_id), meta)
        return str(task_id)

    def insert_completion_payload(self, *, payload: dict[str, Any], task_id: str) -> None:
        if _completion_stage(payload) != "answer":
            return
        record = dict(payload)
        record["status"] = "Completed"
        record["created_at"] = _now_cn()
        record["context"] = _build_completion_context(record)
        self._upsert_jsonl(self._completions_path(task_id), record)

    def insert_completion_payloads_batch(self, *, payloads: Sequence[dict[str, Any]], task_id: str) -> int:
        inserted = 0
        for payload in payloads:
            if _completion_stage(payload) != "answer":
                continue
            self.insert_completion_payload(payload=payload, task_id=task_id)
            inserted += 1
        return inserted

    def ingest_eval_payloads(self, *, payloads: Iterable[dict[str, Any]], task_id: str) -> int:
        existing_completion_keys = self.list_completion_keys(task_id=task_id, status="Completed")
        inserted = 0
        for payload in payloads:
            key = _attempt_key(payload)
            if key not in existing_completion_keys:
                continue
            record = dict(payload)
            record["created_at"] = _now_cn()
            self._upsert_jsonl(self._eval_path(task_id), record)
            inserted += 1
        return inserted

    def ingest_checker_payloads(self, *, payloads: Iterable[dict[str, Any]], task_id: str) -> int:
        existing_eval_keys = {_attempt_key(row) for row in _read_jsonl(self._eval_path(task_id))}
        inserted = 0
        for payload in payloads:
            key = _attempt_key(payload)
            if key not in existing_eval_keys:
                continue
            record = dict(payload)
            record["created_at"] = _now_cn()
            record["needs_human_review"] = bool(record.get("needs_human_review", False))
            self._upsert_jsonl(self._checker_path(task_id), record)
            inserted += 1
        return inserted

    def record_score_payload(self, *, payload: dict[str, Any], task_id: str) -> None:
        score_payload = dict(payload)
        score_payload.setdefault("task_id", str(task_id))
        score_payload.setdefault("created_at", _now_cn())
        _write_json(self._score_path(task_id), score_payload)
        self.update_task_status(task_id=task_id, status="completed")
        try:
            from src.space.score_index import append_score_index_entry

            append_score_index_entry(score_payload, task_id=task_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[space] failed to append score index for JSON task {task_id}: {exc}")

    def count_completions(self, *, task_id: str, status: str | None = None) -> int:
        return len(self.list_completion_payloads(task_id=task_id, status=status))

    def list_completion_payloads(self, *, task_id: str, status: str | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self._completions_path(task_id))
        if status is not None:
            rows = [row for row in rows if str(row.get("status") or "") == status]
        return rows

    def list_completion_keys(self, *, task_id: str, status: str | None = None) -> set[tuple[int, int, int]]:
        return {_attempt_key(row) for row in self.list_completion_payloads(task_id=task_id, status=status)}

    def get_score_payload(self, *, task_id: str) -> dict[str, Any] | None:
        return _read_json(self._score_path(task_id))

    def get_task_bundle(self, *, task_id: str) -> dict[str, Any] | None:
        task = _read_json(self._task_path(task_id))
        if not task:
            return None
        benchmark = {
            "benchmark_name": task.get("benchmark_name", ""),
            "benchmark_split": task.get("benchmark_split", ""),
        }
        model = {"model_name": task.get("model", "")}
        return {"task": task, "model": model, "benchmark": benchmark}

    def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        rows = _read_jsonl(self._eval_path(task_id))
        if only_wrong:
            rows = [row for row in rows if not bool(row.get("is_passed", False))]
        start = max(0, int(offset or 0))
        end = None if limit is None else start + max(0, int(limit))
        selected = rows[start:end]
        if include_context:
            return selected
        return [{key: value for key, value in row.items() if key != "context"} for row in selected]

    def get_eval_context_for_space(
        self,
        *,
        task_id: str,
        sample_index: int,
        repeat_index: int,
        pass_index: int = 0,
    ) -> Any | None:
        target = (int(sample_index), int(repeat_index), int(pass_index))
        for row in _read_jsonl(self._eval_path(task_id)):
            if _attempt_key(row) == target:
                return row.get("context")
        return None

    def list_checker_keys(self, *, task_id: str) -> set[tuple[int, int, int]]:
        return {_attempt_key(row) for row in _read_jsonl(self._checker_path(task_id))}

    def list_completions_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._completions_path(task_id))

    def list_eval_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._eval_path(task_id))

    def list_checker_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._checker_path(task_id))

    def list_scores_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        score = self.get_score_payload(task_id=task_id)
        return [] if score is None else [score]

    def update_task_status(self, *, task_id: str, status: str) -> None:
        with self._lock:
            path = self._task_path(task_id)
            meta = _read_json(path) or {"task_id": int(task_id) if str(task_id).isdigit() else str(task_id)}
            meta["status"] = str(status).lower() if status.islower() else str(status)
            meta["updated_at"] = _now_cn()
            _write_json(path, meta)

    def _upsert_jsonl(self, path: Path, record: Mapping[str, Any]) -> None:
        with self._lock:
            key = _attempt_key(record)
            rows = _read_jsonl(path)
            replaced = False
            normalized = _sanitize_json(dict(record))
            for idx, row in enumerate(rows):
                if _attempt_key(row) == key:
                    rows[idx] = normalized
                    replaced = True
                    break
            if not replaced:
                rows.append(normalized)
            rows.sort(key=_attempt_key)
            _write_jsonl(path, rows)


__all__ = ["EvalJsonService"]
