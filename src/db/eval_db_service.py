from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from typing import Any, Iterable, Mapping

from zoneinfo import ZoneInfo
from src.db.database import DatabaseManager
from src.db.orm import get_session, init_orm
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.results.schema import iter_stage_indices
from src.eval.scheduler.config import DEFAULT_DB_CONFIG, REPO_ROOT
from src.eval.scheduler.dataset_utils import split_benchmark_and_split
from src.eval.scheduler.datasets import DATASET_ROOTS, find_dataset_file
from src.eval.scheduler.models import _normalize_model_identifier, _parse_model_tags, normalize_model_name

from .eval_db_repo import EvalDbRepository


class EvalDbService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._repo = EvalDbRepository()
        init_orm(db.config or DEFAULT_DB_CONFIG)

    @staticmethod
    def _now_cn() -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)

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

            git_sha = self._resolve_git_sha()
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
            context = self._merge_completion_context(
                self._repo.fetch_completion_context(
                    session,
                    task_id=int(task_id),
                    sample_index=self._parse_index(payload.get("sample_index", 0)),
                    repeat_index=self._parse_index(payload.get("repeat_index", 0)),
                ),
                self._build_completion_context(payload),
            )
            status = payload.get("_stage", "answer")
            self._repo.insert_completion(
                session,
                task_id=int(task_id),
                payload=payload,
                context=context,
                created_at=self._now_cn(),
                status=status,
            )

    def ingest_eval_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        task_id: str,
    ) -> int:
        inserted = 0
        with get_session() as session:
            mapping = self._repo.fetch_completion_id_map(session, task_id=int(task_id))
            for payload in payloads:
                completions_id = mapping.get(
                    (int(payload.get("sample_index", 0)), int(payload.get("repeat_index", 0)))
                )
                if completions_id is None:
                    continue
                self._repo.insert_eval(
                    session,
                    completions_id=completions_id,
                    payload=payload,
                    created_at=self._now_cn(),
                )
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
    ) -> int:
        with get_session() as session:
            return self._repo.count_completions(
                session,
                task_id=int(task_id),
            )

    def list_completion_payloads(
        self,
        *,
        task_id: str,
    ) -> list[dict[str, Any]]:
        with get_session() as session:
            rows = self._repo.fetch_completions(
                session,
                task_id=int(task_id),
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
                "sample_index": int(row_dict.get("sample_index", 0)),
                "repeat_index": int(row_dict.get("repeat_index", 0)),
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
    ) -> set[tuple[int, int]]:
        with get_session() as session:
            rows = self._repo.fetch_completion_keys(
                session,
                task_id=int(task_id),
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
        return {
            "stages": stages,
            "sampling_config": payload.get("sampling_config", {}),
        }

    @staticmethod
    def _merge_completion_context(
        existing: dict[str, Any] | None,
        incoming: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(existing, dict) and not isinstance(incoming, dict):
            return {"stages": [], "sampling_config": {}}
        base = existing if isinstance(existing, dict) else {}
        update = incoming if isinstance(incoming, dict) else {}

        stages_by_index: dict[int, dict[str, Any]] = {}
        base_stages = base.get("stages")
        if isinstance(base_stages, list):
            for idx, stage in enumerate(base_stages, start=1):
                if isinstance(stage, dict):
                    stages_by_index[idx] = stage
        update_stages = update.get("stages")
        if isinstance(update_stages, list):
            for idx, stage in enumerate(update_stages, start=1):
                if isinstance(stage, dict):
                    stages_by_index[idx] = stage

        merged_stages = [stages_by_index[idx] for idx in sorted(stages_by_index)]

        sampling_config: dict[str, Any] = {}
        base_sampling = base.get("sampling_config")
        if isinstance(base_sampling, dict):
            sampling_config.update(base_sampling)
        update_sampling = update.get("sampling_config")
        if isinstance(update_sampling, dict):
            sampling_config.update(update_sampling)

        return {"stages": merged_stages, "sampling_config": sampling_config}

    @staticmethod
    def _parse_index(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

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
        env_sha = os.environ.get("RWKV_GIT_SHA", "").strip()
        if env_sha:
            return env_sha
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
        return sha
