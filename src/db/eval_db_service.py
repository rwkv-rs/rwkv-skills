from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
from collections.abc import Mapping as AbcMapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from zoneinfo import ZoneInfo
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.results.schema import iter_stage_indices, strict_nonneg_int
from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import make_dataset_slug, split_benchmark_and_split
from src.eval.scheduler.datasets import DATASET_ROOTS, find_dataset_file
from src.eval.scheduler.models import _normalize_model_identifier, _parse_model_tags, normalize_model_name

from .pool import init_db_pool
from .sql_repo import SqlEvalDbRepository

# Git SHA cache - resolved once per process
_GIT_SHA_CACHE: str | None = None


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)

# Avoid generating oversized multi-row INSERT statements for eval payloads.
# LiveCodeBench `ref_answer` can be very large; flushing in small chunks keeps
# each SQL statement bounded and prevents psycopg buffer allocation failures.
_EVAL_INSERT_FLUSH_ROWS = _positive_int_env("RWKV_EVAL_INSERT_FLUSH_ROWS", 32)
_EVAL_INSERT_FLUSH_CHARS = _positive_int_env("RWKV_EVAL_INSERT_FLUSH_CHARS", 2_000_000)
_EVAL_ANSWER_MAX_CHARS = _positive_int_env("RWKV_EVAL_ANSWER_MAX_CHARS", 65_536)
_EVAL_REF_ANSWER_MAX_CHARS = _positive_int_env("RWKV_EVAL_REF_ANSWER_MAX_CHARS", 4_096)
_EVAL_FAIL_REASON_MAX_CHARS = _positive_int_env("RWKV_EVAL_FAIL_REASON_MAX_CHARS", 2_048)
_CHECKER_INSERT_FLUSH_ROWS = _positive_int_env("RWKV_CHECKER_INSERT_FLUSH_ROWS", 64)
_COMPLETION_EXTRA_TEXT_MAX_CHARS = _positive_int_env("RWKV_COMPLETION_EXTRA_TEXT_MAX_CHARS", 4_096)
_COMPLETION_EXTRA_LIST_MAX_ITEMS = _positive_int_env("RWKV_COMPLETION_EXTRA_LIST_MAX_ITEMS", 32)
_COMPLETION_EXTRA_DICT_MAX_ITEMS = _positive_int_env("RWKV_COMPLETION_EXTRA_DICT_MAX_ITEMS", 128)
_COMPLETION_EXTRA_MAX_DEPTH = _positive_int_env("RWKV_COMPLETION_EXTRA_MAX_DEPTH", 6)
_EVAL_REF_ANSWER_KEYS = (
    "ref_answer",
    "expected_answer",
    "reference_answer",
    "expected_judgement",
    "reference_solution",
    "canonical_solution",
    "solution",
    "output",
    "target",
    "final_answer",
)
_DATASET_REF_ANSWER_KEYS = (
    *_EVAL_REF_ANSWER_KEYS[1:],
    "answer",
    "answers",
    "gold",
    "test_cases",
)
_DATASET_REFERENCE_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}


def _get_cached_git_sha() -> str:
    """Get git SHA with caching - only resolves once per process."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    env_sha = os.environ.get("RWKV_GIT_SHA", "").strip()
    if env_sha:
        _GIT_SHA_CACHE = env_sha
        return _GIT_SHA_CACHE
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = "unknown"
    if not sha:
        sha = "unknown"
    _GIT_SHA_CACHE = sha
    return _GIT_SHA_CACHE


@dataclass(slots=True)
class TaskLookup:
    task_id: int
    status: str


@dataclass(slots=True)
class ResumeContext:
    """三层级联检索结果：一次查询获取所有续跑信息。

    Layer 1: benchmark_id, model_id (实体标识)
    Layer 2: matching_tasks / task_id / can_resume (任务状态)
    Layer 3: completed_keys (已完成 attempt)
    """
    benchmark_id: int | None = None
    model_id: int | None = None
    task_id: int | None = None
    can_resume: bool = False
    matching_tasks: tuple[TaskLookup, ...] = ()
    completed_task_ids: tuple[int, ...] = ()
    resumable_task_ids: tuple[int, ...] = ()
    completed_keys: set[tuple[int, int, int]] = field(default_factory=set)

    @property
    def is_new_task(self) -> bool:
        """是否需要创建新任务"""
        return self.task_id is None or not self.can_resume


class EvalDbService:
    """Database service for evaluation tasks.

    Usage:
        # Auto-initialize with default config
        service = EvalDbService()
    """

    def __init__(self) -> None:
        init_db_pool()
        self._repo = SqlEvalDbRepository()

    @staticmethod
    def _now_cn() -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)

    @staticmethod
    def _estimate_eval_payload_chars(payload: Mapping[str, Any]) -> int:
        """Estimate text size of a single eval row for flush chunking."""
        answer = payload.get("answer")
        ref_answer = payload.get("ref_answer")
        fail_reason = payload.get("fail_reason")
        return (
            len(str(answer or ""))
            + len(str(ref_answer or ""))
            + len(str(fail_reason or ""))
        )

    @staticmethod
    def _bounded_eval_text(value: Any, *, max_chars: int) -> str:
        text = str(value or "").replace("\x00", "")
        if len(text) <= max_chars:
            return text
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        marker = f"\n...[truncated chars={len(text)} sha256={digest}]"
        if max_chars <= len(marker):
            return marker[-max_chars:]
        return text[: max_chars - len(marker)].rstrip() + marker

    @classmethod
    def _compact_completion_extra(cls, value: Any, *, depth: int = 0) -> Any:
        if depth > _COMPLETION_EXTRA_MAX_DEPTH:
            return "[truncated depth]"
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return cls._bounded_eval_text(value, max_chars=_COMPLETION_EXTRA_TEXT_MAX_CHARS)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return cls._bounded_eval_text(
                value.decode("utf-8", errors="replace"),
                max_chars=_COMPLETION_EXTRA_TEXT_MAX_CHARS,
            )
        if is_dataclass(value) and not isinstance(value, type):
            return cls._compact_completion_extra(asdict(value), depth=depth + 1)
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return cls._compact_completion_extra(value.model_dump(), depth=depth + 1)
            except Exception:
                return cls._bounded_eval_text(value, max_chars=_COMPLETION_EXTRA_TEXT_MAX_CHARS)
        if isinstance(value, AbcMapping):
            compact: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:_COMPLETION_EXTRA_DICT_MAX_ITEMS]:
                compact_key = cls._sanitize_json_text(key)
                if not isinstance(compact_key, str):
                    compact_key = str(compact_key)
                compact[compact_key] = cls._compact_completion_extra(item, depth=depth + 1)
            if len(items) > _COMPLETION_EXTRA_DICT_MAX_ITEMS:
                compact["__truncated_items__"] = len(items) - _COMPLETION_EXTRA_DICT_MAX_ITEMS
            return compact
        if isinstance(value, (list, tuple)):
            compact_list = [
                cls._compact_completion_extra(item, depth=depth + 1)
                for item in value[:_COMPLETION_EXTRA_LIST_MAX_ITEMS]
            ]
            if len(value) > _COMPLETION_EXTRA_LIST_MAX_ITEMS:
                compact_list.append({"__truncated_items__": len(value) - _COMPLETION_EXTRA_LIST_MAX_ITEMS})
            return compact_list
        if isinstance(value, set):
            items = sorted(value, key=str)
            compact_list = [
                cls._compact_completion_extra(item, depth=depth + 1)
                for item in items[:_COMPLETION_EXTRA_LIST_MAX_ITEMS]
            ]
            if len(items) > _COMPLETION_EXTRA_LIST_MAX_ITEMS:
                compact_list.append({"__truncated_items__": len(items) - _COMPLETION_EXTRA_LIST_MAX_ITEMS})
            return compact_list
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return cls._bounded_eval_text(value, max_chars=_COMPLETION_EXTRA_TEXT_MAX_CHARS)

    @staticmethod
    def _normalize_reference_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            normalized = str(value).strip()
            return normalized or None
        try:
            normalized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            normalized = str(value)
        normalized = normalized.strip()
        return normalized or None

    @classmethod
    def _extract_reference_from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        keys: Sequence[str] = _EVAL_REF_ANSWER_KEYS,
    ) -> str | None:
        for key in keys:
            if key not in payload:
                continue
            normalized = cls._normalize_reference_value(payload.get(key))
            if normalized:
                return normalized
        raw_record = payload.get("raw_record")
        if isinstance(raw_record, Mapping):
            return cls._extract_reference_from_mapping(raw_record, keys=_DATASET_REF_ANSWER_KEYS)
        return None

    @classmethod
    def _dataset_references(cls, benchmark_name: str, dataset_split: str) -> tuple[str, ...]:
        key = (benchmark_name, dataset_split)
        cached = _DATASET_REFERENCE_CACHE.get(key)
        if cached is not None:
            return cached

        dataset_path: Path | None = None
        direct_candidates: list[Path] = []
        for root in DATASET_ROOTS:
            if dataset_split:
                direct_candidates.append((root / benchmark_name / f"{dataset_split}.jsonl").resolve())
            direct_candidates.append((root / f"{benchmark_name}.jsonl").resolve())
        for candidate in direct_candidates:
            if candidate.exists():
                dataset_path = candidate
                break
        if dataset_path is None:
            dataset_slug = make_dataset_slug(benchmark_name, dataset_split) if dataset_split else benchmark_name
            dataset_path = find_dataset_file(dataset_slug, DATASET_ROOTS)
        references: list[str] = []
        if dataset_path is not None:
            with dataset_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        references.append("")
                        continue
                    if isinstance(payload, Mapping):
                        references.append(
                            cls._extract_reference_from_mapping(
                                payload,
                                keys=_DATASET_REF_ANSWER_KEYS,
                            )
                            or ""
                        )
                    else:
                        references.append("")
        resolved = tuple(references)
        _DATASET_REFERENCE_CACHE[key] = resolved
        return resolved

    @classmethod
    def _resolve_eval_ref_answer(cls, payload: Mapping[str, Any]) -> str:
        explicit = cls._extract_reference_from_mapping(payload)
        if explicit:
            return explicit

        benchmark_name = str(payload.get("benchmark_name") or "").strip()
        dataset_split = str(payload.get("dataset_split") or "").strip()
        if not benchmark_name:
            return ""
        try:
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        except Exception:
            return ""

        references = cls._dataset_references(benchmark_name, dataset_split)
        if 0 <= sample_index < len(references):
            return references[sample_index]
        return ""

    @classmethod
    def _normalize_eval_payload_for_db(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Keep eval rows small; full prompts/completions live in completions.context."""
        return {
            "answer": cls._bounded_eval_text(
                payload.get("answer"),
                max_chars=_EVAL_ANSWER_MAX_CHARS,
            ),
            "ref_answer": cls._bounded_eval_text(
                cls._resolve_eval_ref_answer(payload),
                max_chars=_EVAL_REF_ANSWER_MAX_CHARS,
            ),
            "is_passed": bool(payload.get("is_passed", False)),
            "fail_reason": cls._bounded_eval_text(
                payload.get("fail_reason"),
                max_chars=_EVAL_FAIL_REASON_MAX_CHARS,
            ),
        }

    @classmethod
    def _sanitize_json_text(cls, value: Any) -> Any:
        """Return a JSONB-safe value, stripping NUL bytes recursively."""
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
            return cls._sanitize_json_text(asdict(value))
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return cls._sanitize_json_text(value.model_dump())
            except Exception:
                return str(value)
        if isinstance(value, AbcMapping):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                sanitized_key = cls._sanitize_json_text(key)
                if not isinstance(sanitized_key, str):
                    sanitized_key = str(sanitized_key)
                sanitized[sanitized_key] = cls._sanitize_json_text(item)
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize_json_text(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_json_text(item) for item in value]
        if isinstance(value, set):
            return [cls._sanitize_json_text(item) for item in sorted(value, key=str)]
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _normalize_task_status(value: object) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _is_resumable_task_status(value: object) -> bool:
        return EvalDbService._normalize_task_status(value) in {"running", "failed"}

    @staticmethod
    def _is_completed_task_status(value: object) -> bool:
        return EvalDbService._normalize_task_status(value) == "completed"

    @classmethod
    def _effective_resumable_task_ids(cls, matches: Sequence[TaskLookup]) -> tuple[int, ...]:
        resumable_ids = tuple(
            int(task.task_id)
            for task in matches
            if cls._is_resumable_task_status(task.status)
        )
        if len(resumable_ids) <= 1:
            return resumable_ids
        running_ids = tuple(
            int(task.task_id)
            for task in matches
            if cls._normalize_task_status(task.status) == "running"
        )
        if len(running_ids) == 1:
            return running_ids
        return resumable_ids

    @staticmethod
    def _resolve_task_config_path(benchmark_name: str, model: str) -> str | None:
        config_path = config_path_for_benchmark(benchmark_name, model)
        if config_path.exists():
            return str(config_path)
        fallback_path = config_path_for_benchmark(benchmark_name, None)
        if fallback_path.exists():
            return str(fallback_path)
        return None

    @staticmethod
    def _completion_stage(payload: Mapping[str, Any]) -> str:
        return str(payload.get("_stage", "answer") or "answer").strip().lower()

    @classmethod
    def _should_persist_completion_payload(cls, payload: Mapping[str, Any]) -> bool:
        return cls._completion_stage(payload) == "answer"

    @classmethod
    def _checker_needs_human_review(cls, payload: Mapping[str, Any]) -> bool:
        explicit = payload.get("needs_human_review")
        if isinstance(explicit, bool):
            return explicit
        return False

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
        """三层级联检索：一次查询获取所有续跑信息。

        Layer 1: 查找/创建 benchmark 和 model
        Layer 2: 查找最新的未完成 task
        Layer 3: 获取已完成的 completion keys（仅 answer 阶段计入 completed）
        """
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        model = normalize_model_name(model)
        normalized = _normalize_model_identifier(model)
        arch, data_version, num_params = _parse_model_tags(normalized)
        if not arch or not data_version or not num_params:
            fallback_arch, fallback_data, fallback_params = self._fallback_parse_model_tags(model)
            arch = arch or fallback_arch
            data_version = data_version or fallback_data
            num_params = num_params or fallback_params
        arch_version = arch or "unknown"
        data_version = data_version or "unknown"
        num_params = num_params or "unknown"

        ctx = ResumeContext()

        sanitized_sampling = self._sanitize_json_text(sampling_config) if sampling_config is not None else None

        ctx.benchmark_id = self._repo.get_benchmark_id(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
        )
        if ctx.benchmark_id is None:
            resolved_samples = self._resolve_dataset_sample_count(dataset)
            ctx.benchmark_id = self._repo.insert_benchmark(
                benchmark_name=benchmark_name,
                benchmark_split=benchmark_split,
                url=None,
                status="Todo",
                num_samples=resolved_samples if resolved_samples is not None else 0,
            )
        else:
            existing = self._parse_num_samples(
                self._repo.get_benchmark_num_samples(benchmark_id=ctx.benchmark_id)
            )
            if not existing:
                resolved_samples = self._resolve_dataset_sample_count(dataset)
                if resolved_samples is not None and resolved_samples > 0:
                    self._repo.update_benchmark_num_samples(
                        benchmark_id=ctx.benchmark_id,
                        num_samples=resolved_samples,
                    )

        ctx.model_id = self._repo.get_model_id(
            model_name=model,
            arch_version=arch_version,
            data_version=data_version,
            num_params=num_params,
        )
        if ctx.model_id is None:
            ctx.model_id = self._repo.insert_model(
                model_name=model,
                arch_version=arch_version,
                data_version=data_version,
                num_params=num_params,
            )

        if force_new_task:
            return ctx

        git_sha = _get_cached_git_sha()
        config_path_str = self._resolve_task_config_path(benchmark_name, model)
        raw_matches = self._repo.find_tasks_by_identity(
            config_path=config_path_str,
            evaluator=job_name or "",
            git_hash=git_sha,
            model_id=ctx.model_id,
            benchmark_id=ctx.benchmark_id,
            sampling_config=sanitized_sampling if isinstance(sanitized_sampling, dict) else None,
        )

        matches: list[TaskLookup] = []
        completed_ids: list[int] = []
        for row in raw_matches:
            task_id = int(row["task_id"])
            status = str(row.get("status") or "")
            if self._repo.task_has_score(task_id=task_id):
                status = "Completed"
            lookup = TaskLookup(task_id=task_id, status=status)
            matches.append(lookup)
            if self._is_completed_task_status(status):
                completed_ids.append(task_id)

        ctx.matching_tasks = tuple(matches)
        ctx.completed_task_ids = tuple(completed_ids)
        ctx.resumable_task_ids = self._effective_resumable_task_ids(ctx.matching_tasks)

        if not completed_ids and len(ctx.resumable_task_ids) == 1:
            ctx.task_id = ctx.resumable_task_ids[0]
            ctx.can_resume = True
            answer_rows = self._repo.fetch_completion_keys(
                task_id=ctx.task_id,
                status="Completed",
            )
            ctx.completed_keys = set(answer_rows)
        elif completed_ids:
            ctx.task_id = completed_ids[-1]
        elif ctx.resumable_task_ids:
            ctx.task_id = ctx.resumable_task_ids[-1]

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
        """基于 ResumeContext 创建或恢复任务。

        如果 ctx.can_resume 为 True，返回已有的 task_id；
        否则创建新任务。
        """
        if ctx.can_resume and ctx.task_id is not None:
            self._repo.update_task_status(task_id=ctx.task_id, status="running")
            self._repo.delete_scores_by_task_id(task_id=ctx.task_id)
            return str(ctx.task_id)

        benchmark_name, _ = split_benchmark_and_split(dataset)
        model = normalize_model_name(model)
        git_sha = _get_cached_git_sha()
        config_path_str = self._resolve_task_config_path(benchmark_name, model)
        desc = os.environ.get("RWKV_TASK_DESC")
        is_tmp = os.environ.get("RWKV_TASK_IS_TMP", "").strip().lower() in {"1", "true", "yes", "on"}

        task_id = self._repo.insert_task(
            config_path=config_path_str,
            evaluator=job_name or "",
            is_param_search=is_param_search,
            is_tmp=is_tmp,
            created_at=self._now_cn(),
            status="running",
            git_hash=git_sha,
            model_id=ctx.model_id,
            benchmark_id=ctx.benchmark_id,
            desc=desc,
            sampling_config=(
                self._sanitize_json_text(sampling_config) if sampling_config is not None else None
            ),
            log_path=os.environ.get("RWKV_SKILLS_LOG_PATH", ""),
        )
        return str(task_id)

    def get_or_create_task(
        self,
        *,
        job_name: str | None,
        job_id: str | None,
        dataset: str,
        model: str,
        is_param_search: bool,
        sampling_config: dict[str, Any] | None = None,
        allow_resume: bool = True,
    ) -> str:
        ctx = self.get_resume_context(
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            job_name=job_name,
            sampling_config=sampling_config,
            force_new_task=not allow_resume,
        )
        if allow_resume and ctx.can_resume:
            return self.create_task_from_context(
                ctx=ctx,
                job_name=job_name,
                dataset=dataset,
                model=model,
                is_param_search=is_param_search,
                sampling_config=sampling_config,
            )

        fresh_ctx = self.get_resume_context(
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            job_name=job_name,
            sampling_config=sampling_config,
            force_new_task=True,
        )
        return self.create_task_from_context(
            ctx=fresh_ctx,
            job_name=job_name,
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            sampling_config=sampling_config,
        )

    def get_benchmark_num_samples(self, *, dataset: str) -> int | None:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        benchmark_id = self._repo.get_benchmark_id(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
        )
        if benchmark_id is None:
            return None
        return self._parse_num_samples(
            self._repo.get_benchmark_num_samples(benchmark_id=benchmark_id)
        )

    def ensure_benchmark_num_samples(self, *, dataset: str, num_samples: int) -> None:
        if num_samples <= 0:
            return
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        benchmark_id = self._repo.get_benchmark_id(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
        )
        if benchmark_id is None:
            self._repo.insert_benchmark(
                benchmark_name=benchmark_name,
                benchmark_split=benchmark_split,
                url=None,
                status="Todo",
                num_samples=num_samples,
            )
            return
        existing = self._parse_num_samples(
            self._repo.get_benchmark_num_samples(benchmark_id=benchmark_id)
        )
        if existing == num_samples:
            return
        self._repo.update_benchmark_num_samples(
            benchmark_id=benchmark_id,
            num_samples=num_samples,
        )

    def insert_completion_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        if not self._should_persist_completion_payload(payload):
            return
        context = self._build_completion_context(payload)
        self._repo.insert_completion(
            task_id=int(task_id),
            payload=payload,
            context=context,
            created_at=self._now_cn(),
            status="Completed",
        )

    def insert_completion_payloads_batch(
        self,
        *,
        payloads: Sequence[dict[str, Any]],
        task_id: str,
    ) -> int:
        """Batch insert completion payloads.

        Returns the number of payloads inserted.
        """
        if not payloads:
            return 0
        task_id_int = int(task_id)
        now = self._now_cn()
        rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for payload in payloads:
            if not self._should_persist_completion_payload(payload):
                continue
            context = self._build_completion_context(payload)
            rows.append((payload, context))
        if not rows:
            return 0
        return self._repo.insert_completions_batch(
            task_id=task_id_int,
            rows=rows,
            created_at=now,
            status="Completed",
        )

    def ingest_eval_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        task_id: str,
    ) -> int:
        inserted = 0
        pending_rows = 0
        pending_chars = 0
        pending_payloads: list[tuple[int, dict[str, Any]]] = []
        created_at = self._now_cn()
        task_id_int = int(task_id)
        mapping = self._repo.fetch_completion_id_map(
            task_id=task_id_int,
            status="Completed",
        )
        existing_eval_ids = self._repo.fetch_existing_eval_completion_ids(
            task_id=task_id_int,
        )
        for payload in payloads:
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
            pass_index = strict_nonneg_int(payload.get("pass_index", 0), "pass_index")

            completions_id = mapping.get((sample_index, repeat_index, pass_index))
            if completions_id is None or completions_id in existing_eval_ids:
                continue

            db_payload = self._normalize_eval_payload_for_db(payload)
            pending_payloads.append((completions_id, db_payload))
            existing_eval_ids.add(completions_id)
            pending_rows += 1
            pending_chars += self._estimate_eval_payload_chars(db_payload)

            if pending_rows >= _EVAL_INSERT_FLUSH_ROWS or pending_chars >= _EVAL_INSERT_FLUSH_CHARS:
                inserted += self._insert_eval_payload_chunk(
                    task_id=task_id_int,
                    rows=pending_payloads,
                    created_at=created_at,
                )
                pending_payloads = []
                pending_rows = 0
                pending_chars = 0

        if pending_payloads:
            inserted += self._insert_eval_payload_chunk(
                task_id=task_id_int,
                rows=pending_payloads,
                created_at=created_at,
            )
        return inserted

    def _insert_eval_payload_chunk(
        self,
        *,
        task_id: int,
        rows: Sequence[tuple[int, dict[str, Any]]],
        created_at: datetime,
    ) -> int:
        if not rows:
            return 0
        return self._repo.insert_eval_batch(
            rows=rows,
            created_at=created_at,
        )

    def ingest_eval_payload_groups(
        self,
        *,
        task_id: str,
        completion_payloads: Sequence[dict[str, Any]],
        payloads_by_group: Mapping[str, Sequence[dict[str, Any]]],
        primary_group: str,
    ) -> dict[str, int]:
        """Persist strategy eval rows without adding a grouping column."""
        parent_task_id = int(task_id)
        task_ids: dict[str, int] = {}

        primary_payloads = list(payloads_by_group.get(primary_group, ()))
        self.ingest_eval_payloads(payloads=primary_payloads, task_id=str(parent_task_id))
        task_ids[primary_group] = parent_task_id

        for group, payloads in payloads_by_group.items():
            if group == primary_group:
                continue
            strategy_task_id = self.create_eval_strategy_task(
                parent_task_id=parent_task_id,
                strategy=group,
            )
            self.insert_completion_payloads_batch(
                payloads=completion_payloads,
                task_id=str(strategy_task_id),
            )
            self.ingest_eval_payloads(payloads=list(payloads), task_id=str(strategy_task_id))
            self.update_task_status(task_id=str(strategy_task_id), status="completed")
            task_ids[group] = strategy_task_id

        return task_ids

    def create_eval_strategy_task(self, *, parent_task_id: int, strategy: str) -> int:
        parent = self._repo.fetch_task(task_id=int(parent_task_id))
        if parent is None:
            raise RuntimeError(f"parent task not found: {parent_task_id}")

        parent_desc = str(parent.get("desc") or "")
        desc_parts = [
            part
            for part in (
                parent_desc,
                f"parent_task_id={parent_task_id}",
                f"eval_strategy={strategy}",
            )
            if part
        ]
        sampling_config = parent.get("sampling_config")
        return self._repo.insert_task(
            config_path=parent.get("config_path"),
            evaluator=f"{parent.get('evaluator') or 'eval'}:{strategy}",
            is_param_search=True,
            is_tmp=True,
            created_at=self._now_cn(),
            status="running",
            git_hash=str(parent.get("git_hash") or _get_cached_git_sha()),
            model_id=int(parent["model_id"]),
            benchmark_id=int(parent["benchmark_id"]),
            desc="; ".join(desc_parts),
            sampling_config=sampling_config if isinstance(sampling_config, dict) else None,
            log_path=str(parent.get("log_path") or ""),
        )

    def record_score_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        self._repo.insert_score(
            task_id=int(task_id),
            payload=payload,
        )
        self._repo.update_task_status(task_id=int(task_id), status="completed")

    def ingest_checker_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        task_id: str,
    ) -> int:
        task_id_int = int(task_id)
        created_at = self._now_cn()
        mapping = self._repo.fetch_completion_id_map(
            task_id=task_id_int,
            status="Completed",
        )
        existing_checker_ids = self._repo.fetch_existing_checker_completion_ids(
            task_id=task_id_int,
        )
        inserted = 0
        pending_rows: list[tuple[int, dict[str, Any]]] = []
        for payload in payloads:
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
            pass_index = strict_nonneg_int(payload.get("pass_index", 0), "pass_index")
            completions_id = mapping.get((sample_index, repeat_index, pass_index))
            if completions_id is None or completions_id in existing_checker_ids:
                continue
            checker_payload = dict(payload)
            checker_payload["needs_human_review"] = self._checker_needs_human_review(payload)
            pending_rows.append((completions_id, checker_payload))
            existing_checker_ids.add(completions_id)
            if len(pending_rows) >= _CHECKER_INSERT_FLUSH_ROWS:
                inserted += self._repo.insert_checker_batch(
                    rows=pending_rows,
                    created_at=created_at,
                )
                pending_rows = []
        if pending_rows:
            inserted += self._repo.insert_checker_batch(
                rows=pending_rows,
                created_at=created_at,
            )
        return inserted

    def list_latest_scores(self) -> list[dict[str, Any]]:
        return self._repo.fetch_latest_scores()

    def list_latest_scores_for_space(self, *, include_param_search: bool = False) -> list[dict[str, Any]]:
        return self._repo.fetch_latest_scores_for_space(
            include_param_search=include_param_search,
        )

    def list_scores_by_dataset(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        model = normalize_model_name(model)
        return self._repo.fetch_scores_by_benchmark(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            model_name=model,
            is_param_search=is_param_search,
        )

    def list_score_history(self, *, model: str, dataset: str) -> list[dict[str, Any]]:
        """Every official score for one model+benchmark (no dedup / latest-only)."""
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        return self._repo.fetch_score_history(
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            model_name=normalize_model_name(model),
        )

    def list_score_history_pairs(self) -> list[dict[str, Any]]:
        """Distinct (model, dataset) options for the score-history picker."""
        return self._repo.fetch_score_history_pairs()

    def get_score_history_detail(self, *, task_id: str) -> dict[str, Any] | None:
        """Score + task + representative completion context for one task."""
        tid = int(task_id)
        score = self._repo.fetch_score_by_task(task_id=tid)
        task = self._repo.fetch_task(task_id=tid)
        if score is None and task is None:
            return None
        context = self._repo.fetch_first_completion_context(task_id=tid)
        return {"score": score, "task": task, "context": context}

    def count_completions(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> int:
        return self._repo.count_completions(
            task_id=int(task_id),
            status=status,
        )

    def list_completion_payloads(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._repo.fetch_completions(
            task_id=int(task_id),
            status=status,
        )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, Mapping):
                row_dict: dict[str, Any] = dict(row)
            elif isinstance(row, dict):
                row_dict = row
            else:
                row_dict = {}
            context = row_dict.get("context")
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except json.JSONDecodeError:
                    context = None
            sampling_cfg = None
            if isinstance(context, dict):
                sampling_cfg = context.get("sampling_config")
            payload: dict[str, Any] = {
                "benchmark_name": row_dict.get("benchmark_name", ""),
                "dataset_split": row_dict.get("benchmark_split", "") or row_dict.get("dataset_split", ""),
                "sample_index": strict_nonneg_int(row_dict.get("sample_index"), "sample_index"),
                "repeat_index": strict_nonneg_int(row_dict.get("repeat_index"), "repeat_index"),
                "pass_index": strict_nonneg_int(row_dict.get("pass_index", 0), "pass_index"),
                "sampling_config": sampling_cfg if isinstance(sampling_cfg, dict) else {},
                "context": context if isinstance(context, dict) else None,
            }
            if isinstance(context, dict):
                stages = context.get("stages")
                if isinstance(stages, list):
                    for idx, stage in enumerate(stages, start=1):
                        if not isinstance(stage, dict):
                            continue
                        payload[f"prompt{idx}"] = stage.get("prompt")
                        payload[f"completion{idx}"] = stage.get("completion")
                        payload[f"stop_reason{idx}"] = stage.get("stop_reason")
                stats = context.get("stats")
                if isinstance(stats, dict):
                    payload["stats"] = stats
                agent_result = context.get("agent_result")
                if isinstance(agent_result, dict):
                    payload["agent_result"] = agent_result
                agent_info = context.get("agent_info")
                if isinstance(agent_info, dict):
                    payload["agent_info"] = agent_info
                agent_trace = context.get("agent_trace")
                if isinstance(agent_trace, list):
                    payload["agent_trace"] = agent_trace
                task_name = context.get("task_id")
                if isinstance(task_name, str):
                    payload["task_id"] = task_name
                domain = context.get("domain")
                if isinstance(domain, str):
                    payload["domain"] = domain
                instruction = context.get("instruction")
                if isinstance(instruction, str):
                    payload["instruction"] = instruction
            payloads.append(payload)
        return payloads

    def list_completion_keys(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> set[tuple[int, int, int]]:
        rows = self._repo.fetch_completion_keys(
            task_id=int(task_id),
            status=status,
        )
        return set(rows)

    def get_score_payload(
        self,
        *,
        task_id: str,
    ) -> dict[str, Any] | None:
        return self._repo.fetch_score_by_task(task_id=int(task_id))

    def get_task_bundle(self, *, task_id: str) -> dict[str, Any] | None:
        task = self._repo.fetch_task(task_id=int(task_id))
        if not task:
            return None
        model_id = task.get("model_id")
        benchmark_id = task.get("benchmark_id")
        model = self._repo.fetch_model(model_id=int(model_id)) if model_id else None
        benchmark = self._repo.fetch_benchmark(benchmark_id=int(benchmark_id)) if benchmark_id else None
        return {"task": task, "model": model, "benchmark": benchmark}

    def get_latest_task_generation_progress(
        self,
        *,
        evaluator: str,
        model_name: str,
        benchmark_name: str,
        benchmark_split: str,
    ) -> dict[str, Any] | None:
        return self._repo.fetch_latest_task_generation_progress(
            evaluator=evaluator,
            model_name=model_name,
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
        )

    def list_completions_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return self._repo.fetch_completions_rows(task_id=int(task_id))

    def list_eval_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return self._repo.fetch_eval_rows(task_id=int(task_id))

    def list_checker_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return self._repo.fetch_checker_rows(task_id=int(task_id))

    def list_checker_keys(self, *, task_id: str) -> set[tuple[int, int, int]]:
        return self._repo.fetch_checker_keys(task_id=int(task_id))

    def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        safe_limit = int(limit) if isinstance(limit, int) or (isinstance(limit, str) and limit.isdigit()) else None
        if safe_limit is not None and safe_limit <= 0:
            safe_limit = None
        try:
            safe_offset = int(offset)
        except (TypeError, ValueError):
            safe_offset = 0
        safe_offset = max(0, safe_offset)

        rows = self._repo.fetch_eval_with_completions_by_task(
            task_id=int(task_id),
            only_wrong=bool(only_wrong),
            limit=safe_limit,
            offset=safe_offset,
            include_context=bool(include_context),
            include_preview=bool(include_preview),
        )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            mapping = dict(row) if isinstance(row, Mapping) else row
            if not isinstance(mapping, dict):
                continue
            payload: dict[str, Any] = {
                "sample_index": int(mapping.get("sample_index", 0)),
                "repeat_index": int(mapping.get("repeat_index", 0)),
                "pass_index": int(mapping.get("pass_index", 0)),
                "is_passed": bool(mapping.get("is_passed", False)),
                "answer": str(mapping.get("answer") or ""),
                "ref_answer": str(mapping.get("ref_answer") or ""),
                "fail_reason": str(mapping.get("fail_reason") or ""),
                "context_preview": str(mapping.get("context_preview") or ""),
            }
            if include_context:
                payload["context"] = mapping.get("context")
            payloads.append(payload)
        return payloads

    def get_eval_context_for_space(
        self,
        *,
        task_id: str,
        sample_index: int,
        repeat_index: int,
        pass_index: int = 0,
    ) -> Any | None:
        return self._repo.fetch_eval_context_by_task_attempt(
            task_id=int(task_id),
            sample_index=int(sample_index),
            repeat_index=int(repeat_index),
            pass_index=int(pass_index),
        )

    def list_scores_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        return self._repo.fetch_scores_rows(task_id=int(task_id))

    def update_task_status(self, *, task_id: str, status: str) -> None:
        self._repo.update_task_status(task_id=int(task_id), status=status)

    @staticmethod
    def _build_completion_context(payload: dict[str, Any]) -> dict[str, Any]:
        stages: list[dict[str, Any]] = []
        for idx in iter_stage_indices(payload):
            stages.append(
                {
                    "prompt": payload.get(f"prompt{idx}"),
                    "completion": payload.get(f"completion{idx}"),
                    "stop_reason": payload.get(f"stop_reason{idx}"),
                }
            )
        context = {
            "stages": stages,
            "sampling_config": payload.get("sampling_config", {}),
        }
        stats = payload.get("stats")
        if isinstance(stats, Mapping):
            context["stats"] = dict(stats)
        for key in ("agent_result", "agent_info", "agent_trace", "task_id", "domain", "instruction"):
            value = payload.get(key)
            if value is not None:
                context[key] = EvalDbService._compact_completion_extra(value)
        sanitized = EvalDbService._sanitize_json_text(context)
        return sanitized if isinstance(sanitized, dict) else {}

    @staticmethod
    def _fallback_parse_model_tags(raw: str | None) -> tuple[str | None, str | None, str | None]:
        if not raw:
            return None, None, None
        lowered = raw.lower().replace("_", "-")
        parts = lowered.split("-")
        arch = parts[0] if parts and parts[0].startswith("rwkv") else None
        data_version = None
        num_params = None
        match = re.search(r"\bg\d[a-z0-9]*\b", lowered)
        if match:
            data_version = match.group(0)
        match = re.search(r"\b\d+(?:\.\d+)?b\b", lowered)
        if match:
            num_params = match.group(0)
        return arch, data_version, num_params

    @staticmethod
    def _parse_num_samples(value: object) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _resolve_dataset_sample_count(cls, dataset: str) -> int | None:
        path = find_dataset_file(dataset, DATASET_ROOTS)
        if path is None or not path.exists():
            return None
        try:
            count = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        count += 1
        except OSError:
            return None
        return count if count > 0 else None

    @staticmethod
    def _resolve_git_sha() -> str:
        """Deprecated: use _get_cached_git_sha() instead."""
        return _get_cached_git_sha()


__all__ = ["EvalDbService", "ResumeContext", "TaskLookup"]
