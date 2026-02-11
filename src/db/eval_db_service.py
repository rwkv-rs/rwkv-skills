from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from zoneinfo import ZoneInfo
from src.db.orm import get_session, init_orm, transaction
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.results.schema import iter_stage_indices, strict_nonneg_int
from src.eval.scheduler.config import DEFAULT_DB_CONFIG, REPO_ROOT
from src.eval.scheduler.dataset_utils import split_benchmark_and_split
from src.eval.scheduler.datasets import DATASET_ROOTS, find_dataset_file
from src.eval.scheduler.models import _normalize_model_identifier, _parse_model_tags, normalize_model_name

from .eval_db_repo import EvalDbRepository

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


def _get_cached_git_sha() -> str:
    """Get git SHA with caching - only resolves once per process."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    env_sha = os.environ.get("RWKV_GIT_SHA", "").strip()
    if env_sha:
        _GIT_SHA_CACHE = env_sha
        return _GIT_SHA_CACHE
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse HEAD returned empty output")
    _GIT_SHA_CACHE = sha
    return _GIT_SHA_CACHE


@dataclass(slots=True)
class ResumeContext:
    """三层级联检索结果：一次查询获取所有续跑信息。

    Layer 1: benchmark_id, model_id (实体标识)
    Layer 2: task_id, can_resume (任务状态)
    Layer 3: completed_keys / cot_only_keys (已完成样本 / 仅完成第一阶段样本)
    """
    benchmark_id: int | None = None
    model_id: int | None = None
    task_id: int | None = None
    can_resume: bool = False
    completed_keys: set[tuple[int, int]] = field(default_factory=set)
    cot_only_keys: set[tuple[int, int]] = field(default_factory=set)

    @property
    def is_new_task(self) -> bool:
        """是否需要创建新任务"""
        return self.task_id is None or not self.can_resume


class EvalDbService:
    """Database service for evaluation tasks.

    Usage:
        # Option 1: Auto-initialize with default config
        service = EvalDbService()

        # Option 2: Initialize with custom config
        init_orm(custom_config)
        service = EvalDbService()
    """

    def __init__(self) -> None:
        self._repo = EvalDbRepository()
        # Auto-initialize if not already done
        init_orm()

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

    @classmethod
    def _sanitize_json_text(cls, value: Any) -> Any:
        """PostgreSQL JSONB rejects NUL bytes in text values; strip them recursively."""
        if isinstance(value, str):
            return value.replace("\x00", "")
        if isinstance(value, dict):
            sanitized: dict[Any, Any] = {}
            for key, item in value.items():
                if isinstance(key, str):
                    sanitized_key = cls._sanitize_json_text(key)
                else:
                    sanitized_key = key
                sanitized[sanitized_key] = cls._sanitize_json_text(item)
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize_json_text(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_json_text(item) for item in value]
        return value

    def get_resume_context(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
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

        with get_session() as session:
            # Layer 1: benchmark & model
            ctx.benchmark_id = self._repo.get_benchmark_id(
                session, benchmark_name=benchmark_name, benchmark_split=benchmark_split
            )
            if ctx.benchmark_id is None:
                resolved_samples = self._resolve_dataset_sample_count(dataset)
                ctx.benchmark_id = self._repo.insert_benchmark(
                    session,
                    benchmark_name=benchmark_name,
                    benchmark_split=benchmark_split,
                    url=None,
                    status="Todo",
                    num_samples=resolved_samples if resolved_samples is not None else 0,
                )
            else:
                existing = self._parse_num_samples(
                    self._repo.get_benchmark_num_samples(session, benchmark_id=ctx.benchmark_id)
                )
                if not existing:
                    resolved_samples = self._resolve_dataset_sample_count(dataset)
                    if resolved_samples is not None and resolved_samples > 0:
                        self._repo.update_benchmark_num_samples(
                            session,
                            benchmark_id=ctx.benchmark_id,
                            num_samples=resolved_samples,
                        )

            ctx.model_id = self._repo.get_model_id(
                session,
                model_name=model,
                arch_version=arch_version,
                data_version=data_version,
                num_params=num_params,
            )
            if ctx.model_id is None:
                ctx.model_id = self._repo.insert_model(
                    session,
                    model_name=model,
                    arch_version=arch_version,
                    data_version=data_version,
                    num_params=num_params,
                )

            if force_new_task:
                return ctx

            # Layer 2: 查找可续跑的 task
            ctx.task_id = self._repo.get_latest_task_id(
                session,
                benchmark_id=ctx.benchmark_id,
                model_id=ctx.model_id,
                is_param_search=is_param_search,
            )
            if ctx.task_id is not None:
                has_score = self._repo.task_has_score(session, task_id=ctx.task_id)
                ctx.can_resume = not has_score
                # Layer 3: 获取已完成的 keys
                if ctx.can_resume:
                    answer_rows = self._repo.fetch_completion_keys(
                        session,
                        task_id=ctx.task_id,
                        status="answer",
                    )
                    cot_rows = self._repo.fetch_completion_keys(
                        session,
                        task_id=ctx.task_id,
                        status="cot",
                    )
                    ctx.completed_keys = set(answer_rows)
                    ctx.cot_only_keys = set(cot_rows) - ctx.completed_keys

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
            with get_session() as session:
                self._repo.update_task_status(session, task_id=ctx.task_id, status="running")
            return str(ctx.task_id)

        benchmark_name, _ = split_benchmark_and_split(dataset)
        model = normalize_model_name(model)
        git_sha = _get_cached_git_sha()
        config_path = config_path_for_benchmark(benchmark_name, model)
        if config_path.exists():
            config_path_str = str(config_path)
        else:
            fallback_path = config_path_for_benchmark(benchmark_name, None)
            config_path_str = str(fallback_path) if fallback_path.exists() else None
        desc = os.environ.get("RWKV_TASK_DESC")

        with get_session() as session:
            task_id = self._repo.insert_task(
                session,
                config_path=config_path_str,
                evaluator=job_name or "",
                is_param_search=is_param_search,
                created_at=self._now_cn(),
                status="running",
                git_hash=git_sha,
                model_id=ctx.model_id,
                benchmark_id=ctx.benchmark_id,
                desc=desc,
                sampling_config=sampling_config,
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
        with get_session() as session:
            benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
            benchmark_id = self._repo.get_benchmark_id(
                session, benchmark_name=benchmark_name, benchmark_split=benchmark_split
            )
            if benchmark_id is None:
                resolved_samples = self._resolve_dataset_sample_count(dataset)
                benchmark_id = self._repo.insert_benchmark(
                    session,
                    benchmark_name=benchmark_name,
                    benchmark_split=benchmark_split,
                    url=None,
                    status="Todo",
                    num_samples=resolved_samples if resolved_samples is not None else 0,
                )
            else:
                existing = self._parse_num_samples(
                    self._repo.get_benchmark_num_samples(session, benchmark_id=benchmark_id)
                )
                if not existing:
                    resolved_samples = self._resolve_dataset_sample_count(dataset)
                    if resolved_samples is not None and resolved_samples > 0:
                        self._repo.update_benchmark_num_samples(
                            session,
                            benchmark_id=benchmark_id,
                            num_samples=resolved_samples,
                        )
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
            model_id = self._repo.get_model_id(
                session,
                model_name=model,
                arch_version=arch_version,
                data_version=data_version,
                num_params=num_params,
            )
            if model_id is None:
                model_id = self._repo.insert_model(
                    session,
                    model_name=model,
                    arch_version=arch_version,
                    data_version=data_version,
                    num_params=num_params,
                )

            if allow_resume:
                latest = self._repo.get_latest_task_id(
                    session,
                    benchmark_id=benchmark_id,
                    model_id=model_id,
                    is_param_search=is_param_search,
                )
                if latest and not self._repo.task_has_score(session, task_id=latest):
                    self._repo.update_task_status(session, task_id=latest, status="running")
                    return str(latest)

            git_sha = _get_cached_git_sha()
            config_path = config_path_for_benchmark(benchmark_name, model)
            if config_path.exists():
                config_path_str = str(config_path)
            else:
                fallback_path = config_path_for_benchmark(benchmark_name, None)
                config_path_str = str(fallback_path) if fallback_path.exists() else None
            desc = os.environ.get("RWKV_TASK_DESC")
            task_id = self._repo.insert_task(
                session,
                config_path=config_path_str,
                evaluator=job_name or "",
                is_param_search=is_param_search,
                created_at=self._now_cn(),
                status="running",
                git_hash=git_sha,
                model_id=model_id,
                benchmark_id=benchmark_id,
                desc=desc,
                sampling_config=sampling_config,
                log_path=os.environ.get("RWKV_SKILLS_LOG_PATH", ""),
            )
            return str(task_id)

    def get_benchmark_num_samples(self, *, dataset: str) -> int | None:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        with get_session() as session:
            benchmark_id = self._repo.get_benchmark_id(
                session, benchmark_name=benchmark_name, benchmark_split=benchmark_split
            )
            if benchmark_id is None:
                return None
            return self._parse_num_samples(
                self._repo.get_benchmark_num_samples(session, benchmark_id=benchmark_id)
            )

    def ensure_benchmark_num_samples(self, *, dataset: str, num_samples: int) -> None:
        if num_samples <= 0:
            return
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        with get_session() as session:
            benchmark_id = self._repo.get_benchmark_id(
                session, benchmark_name=benchmark_name, benchmark_split=benchmark_split
            )
            if benchmark_id is None:
                self._repo.insert_benchmark(
                    session,
                    benchmark_name=benchmark_name,
                    benchmark_split=benchmark_split,
                    url=None,
                    status="Todo",
                    num_samples=num_samples,
                )
                return
            existing = self._parse_num_samples(
                self._repo.get_benchmark_num_samples(session, benchmark_id=benchmark_id)
            )
            if existing == num_samples:
                return
            self._repo.update_benchmark_num_samples(
                session,
                benchmark_id=benchmark_id,
                num_samples=num_samples,
            )

    def insert_completion_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        with get_session() as session:
            context = self._build_completion_context(payload)
            status = payload.get("_stage", "answer")
            self._repo.insert_completion(
                session,
                task_id=int(task_id),
                payload=payload,
                context=context,
                created_at=self._now_cn(),
                status=status,
            )

    def insert_completion_payloads_batch(
        self,
        *,
        payloads: Sequence[dict[str, Any]],
        task_id: str,
    ) -> int:
        """Batch insert multiple completion payloads in a single transaction.

        Returns the number of payloads inserted.
        """
        if not payloads:
            return 0
        task_id_int = int(task_id)
        now = self._now_cn()
        with transaction() as session:
            for payload in payloads:
                context = self._build_completion_context(payload)
                status = payload.get("_stage", "answer")
                self._repo.insert_completion(
                    session,
                    task_id=task_id_int,
                    payload=payload,
                    context=context,
                    created_at=now,
                    status=status,
                )
        return len(payloads)

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
        with get_session() as session:
            mapping = self._repo.fetch_completion_id_map(session, task_id=task_id_int)
            existing_eval_ids = self._repo.fetch_existing_eval_completion_ids(
                session,
                task_id=task_id_int,
            )
        for payload in payloads:
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")

            completions_id = mapping.get((sample_index, repeat_index))
            if completions_id is None or completions_id in existing_eval_ids:
                continue

            pending_payloads.append((completions_id, payload))
            existing_eval_ids.add(completions_id)
            pending_rows += 1
            pending_chars += self._estimate_eval_payload_chars(payload)

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
        with get_session() as session:
            known_ids = self._repo.fetch_existing_eval_completion_ids(
                session,
                task_id=task_id,
            )
            inserted = 0
            for completions_id, payload in rows:
                if completions_id in known_ids:
                    continue
                self._repo.insert_eval(
                    session,
                    completions_id=completions_id,
                    payload=payload,
                    created_at=created_at,
                )
                known_ids.add(completions_id)
                inserted += 1
        return inserted

    def record_score_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        with get_session() as session:
            self._repo.insert_score(
                session,
                task_id=int(task_id),
                payload=payload,
            )
            self._repo.update_task_status(session, task_id=int(task_id), status="completed")

    def list_latest_scores(self) -> list[dict[str, Any]]:
        with get_session() as session:
            return self._repo.fetch_latest_scores(session)

    def list_latest_scores_for_space(self, *, include_param_search: bool = False) -> list[dict[str, Any]]:
        with get_session() as session:
            return self._repo.fetch_latest_scores_for_space(
                session,
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
        with get_session() as session:
            return self._repo.fetch_scores_by_benchmark(
                session,
                benchmark_name=benchmark_name,
                benchmark_split=benchmark_split,
                model_name=model,
                is_param_search=is_param_search,
            )

    def should_allow_resume(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
        is_cot: bool,
    ) -> bool:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        model = normalize_model_name(model)
        with get_session() as session:
            latest = self._repo.fetch_latest_task_by_names(
                session,
                benchmark_name=benchmark_name,
                benchmark_split=benchmark_split,
                model_name=model,
                is_param_search=is_param_search,
            )
            if not latest:
                return True
            status = str(latest.get("status") or "").lower()
            task_id = latest.get("task_id")
            if status == "completed":
                return False
            if isinstance(task_id, int) and self._repo.task_has_score(session, task_id=task_id):
                return False
        return True

    def count_completions(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> int:
        with get_session() as session:
            return self._repo.count_completions(
                session,
                task_id=int(task_id),
                status=status,
            )

    def list_completion_payloads(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with get_session() as session:
            rows = self._repo.fetch_completions(
                session,
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
            payloads.append(payload)
        return payloads

    def list_completion_keys(
        self,
        *,
        task_id: str,
        status: str | None = None,
    ) -> set[tuple[int, int]]:
        with get_session() as session:
            rows = self._repo.fetch_completion_keys(
                session,
                task_id=int(task_id),
                status=status,
            )
        return set(rows)

    def get_score_payload(
        self,
        *,
        task_id: str,
    ) -> dict[str, Any] | None:
        with get_session() as session:
            return self._repo.fetch_score_by_task(session, task_id=int(task_id))

    def get_task_bundle(self, *, task_id: str) -> dict[str, Any] | None:
        with get_session() as session:
            task = self._repo.fetch_task(session, task_id=int(task_id))
            if not task:
                return None
            model_id = task.get("model_id")
            benchmark_id = task.get("benchmark_id")
            model = self._repo.fetch_model(session, model_id=int(model_id)) if model_id else None
            benchmark = (
                self._repo.fetch_benchmark(session, benchmark_id=int(benchmark_id)) if benchmark_id else None
            )
            return {"task": task, "model": model, "benchmark": benchmark}

    def list_completions_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with get_session() as session:
            return self._repo.fetch_completions_rows(session, task_id=int(task_id))

    def list_eval_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with get_session() as session:
            return self._repo.fetch_eval_rows(session, task_id=int(task_id))

    def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        safe_limit = int(limit) if isinstance(limit, int) or (isinstance(limit, str) and limit.isdigit()) else None
        if safe_limit is not None and safe_limit <= 0:
            safe_limit = None
        try:
            safe_offset = int(offset)
        except (TypeError, ValueError):
            safe_offset = 0
        safe_offset = max(0, safe_offset)

        with get_session() as session:
            rows = self._repo.fetch_eval_with_completions_by_task(
                session,
                task_id=int(task_id),
                only_wrong=bool(only_wrong),
                limit=safe_limit,
                offset=safe_offset,
                include_context=bool(include_context),
            )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            mapping = dict(row) if isinstance(row, Mapping) else row
            if not isinstance(mapping, dict):
                continue
            payload: dict[str, Any] = {
                "sample_index": int(mapping.get("sample_index", 0)),
                "repeat_index": int(mapping.get("repeat_index", 0)),
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
    ) -> Any | None:
        with get_session() as session:
            return self._repo.fetch_eval_context_by_task_sample_repeat(
                session,
                task_id=int(task_id),
                sample_index=int(sample_index),
                repeat_index=int(repeat_index),
            )

    def list_scores_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with get_session() as session:
            return self._repo.fetch_scores_rows(session, task_id=int(task_id))

    def update_task_status(self, *, task_id: str, status: str) -> None:
        with get_session() as session:
            self._repo.update_task_status(session, task_id=int(task_id), status=status)

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


__all__ = ["EvalDbService", "ResumeContext"]
