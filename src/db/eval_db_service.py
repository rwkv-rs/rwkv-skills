from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from zoneinfo import ZoneInfo
from src.db.database import DatabaseManager
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.results.schema import iter_stage_indices
from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import split_benchmark_and_split
from src.eval.scheduler.models import _normalize_model_identifier, _parse_model_tags

from .eval_db_repo import EvalDbRepository


class EvalDbService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._repo = EvalDbRepository()

    @staticmethod
    def _now_cn() -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)
    def get_or_create_version(
        self,
        *,
        job_name: str | None,
        job_id: str | None,
        dataset: str,
        model: str,
        is_param_search: bool,
        allow_resume: bool = True,
    ) -> str:
        return self.get_or_create_task(
            job_name=job_name,
            job_id=job_id,
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            allow_resume=allow_resume,
        )

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
        with self._db.get_connection() as conn:
            benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
            benchmark_id = self._repo.get_benchmark_id(
                conn, benchmark_name=benchmark_name, benchmark_split=benchmark_split
            )
            if benchmark_id is None:
                benchmark_id = self._repo.insert_benchmark(
                    conn,
                    benchmark_name=benchmark_name,
                    benchmark_split=benchmark_split,
                    url=None,
                    status="Todo",
                    num_samples="0",
                )

            normalized = _normalize_model_identifier(model)
            arch, data_version, num_params = _parse_model_tags(normalized)
            arch_version = arch or "unknown"
            data_version = data_version or "unknown"
            num_params = num_params or "unknown"
            model_id = self._repo.get_model_id(
                conn,
                model_name=model,
                arch_version=arch_version,
                data_version=data_version,
                num_params=num_params,
            )
            if model_id is None:
                model_id = self._repo.insert_model(
                    conn,
                    model_name=model,
                    arch_version=arch_version,
                    data_version=data_version,
                    num_params=num_params,
                )

            if allow_resume:
                latest = self._repo.get_latest_task_id(
                    conn,
                    benchmark_id=benchmark_id,
                    model_id=model_id,
                    is_param_search=is_param_search,
                )
                if latest and not self._repo.task_has_score(conn, task_id=latest):
                    self._repo.update_task_status(conn, task_id=latest, status="running")
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
                conn,
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

    def ingest_completions(
        self,
        *,
        completions_path: str | Path,
        task_id: str,
    ) -> int:
        path = Path(completions_path)
        if not path.exists():
            return 0
        inserted = 0
        with path.open("r", encoding="utf-8") as fh, self._db.get_connection() as conn:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                context = self._build_completion_context(payload)
                self._repo.insert_completion(
                    conn,
                    task_id=int(task_id),
                    payload=payload,
                    context=context,
                    created_at=self._now_cn(),
                    status="final_answer",
                )
                inserted += 1
        return inserted

    def ingest_completion_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        task_id: str,
    ) -> int:
        inserted = 0
        with self._db.get_connection() as conn:
            for payload in payloads:
                context = self._build_completion_context(payload)
                self._repo.insert_completion(
                    conn,
                    task_id=int(task_id),
                    payload=payload,
                    context=context,
                    created_at=self._now_cn(),
                    status="final_answer",
                )
                inserted += 1
        return inserted

    def insert_completion_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        with self._db.get_connection() as conn:
            context = self._build_completion_context(payload)
            self._repo.insert_completion(
                conn,
                task_id=int(task_id),
                payload=payload,
                context=context,
                created_at=self._now_cn(),
                status="final_answer",
            )

    def ingest_eval(
        self,
        *,
        eval_path: str | Path,
        task_id: str,
    ) -> int:
        path = Path(eval_path)
        if not path.exists():
            return 0
        inserted = 0
        with path.open("r", encoding="utf-8") as fh, self._db.get_connection() as conn:
            mapping = self._repo.fetch_completion_id_map(conn, task_id=int(task_id))
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                completions_id = mapping.get(
                    (int(payload.get("sample_index", 0)), int(payload.get("repeat_index", 0)))
                )
                if completions_id is None:
                    continue
                self._repo.insert_eval(
                    conn,
                    completions_id=completions_id,
                    payload=payload,
                    created_at=self._now_cn(),
                )
                inserted += 1
        return inserted

    def ingest_eval_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        task_id: str,
    ) -> int:
        inserted = 0
        with self._db.get_connection() as conn:
            mapping = self._repo.fetch_completion_id_map(conn, task_id=int(task_id))
            for payload in payloads:
                completions_id = mapping.get(
                    (int(payload.get("sample_index", 0)), int(payload.get("repeat_index", 0)))
                )
                if completions_id is None:
                    continue
                self._repo.insert_eval(
                    conn,
                    completions_id=completions_id,
                    payload=payload,
                    created_at=self._now_cn(),
                )
                inserted += 1
        return inserted

    def insert_eval_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        with self._db.get_connection() as conn:
            mapping = self._repo.fetch_completion_id_map(conn, task_id=int(task_id))
            completions_id = mapping.get(
                (int(payload.get("sample_index", 0)), int(payload.get("repeat_index", 0)))
            )
            if completions_id is None:
                return
            self._repo.insert_eval(
                conn,
                completions_id=completions_id,
                payload=payload,
                created_at=self._now_cn(),
            )

    def record_score(
        self,
        *,
        score_path: str | Path,
        task_id: str,
    ) -> None:
        path = Path(score_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self._db.get_connection() as conn:
            self._repo.insert_score(
                conn,
                task_id=int(task_id),
                payload=payload,
            )
            samples = payload.get("samples")
            if samples is not None:
                self._repo.update_benchmark_num_samples_for_task(
                    conn,
                    task_id=int(task_id),
                    num_samples=str(samples),
                )
            self._repo.update_task_status(conn, task_id=int(task_id), status="completed")

    def record_score_payload(
        self,
        *,
        payload: dict[str, Any],
        task_id: str,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.insert_score(
                conn,
                task_id=int(task_id),
                payload=payload,
            )
            samples = payload.get("samples")
            if samples is not None:
                self._repo.update_benchmark_num_samples_for_task(
                    conn,
                    task_id=int(task_id),
                    num_samples=str(samples),
                )
            self._repo.update_task_status(conn, task_id=int(task_id), status="completed")

    def record_log_event(
        self,
        *,
        event: str,
        job_id: str,
        payload: dict[str, Any],
        version_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        return

    def list_latest_scores(self) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_latest_scores(conn)

    def list_scores_by_dataset(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        benchmark_name, benchmark_split = split_benchmark_and_split(dataset)
        with self._db.get_connection() as conn:
            return self._repo.fetch_scores_by_benchmark(
                conn,
                benchmark_name=benchmark_name,
                benchmark_split=benchmark_split,
                model_name=model,
                is_param_search=is_param_search,
            )

    def count_completions(
        self,
        *,
        task_id: str,
    ) -> int:
        with self._db.get_connection() as conn:
            return self._repo.count_completions(
                conn,
                task_id=int(task_id),
            )

    def list_completion_payloads(
        self,
        *,
        task_id: str,
    ) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            rows = self._repo.fetch_completions(
                conn,
                task_id=int(task_id),
            )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            context = row.get("context") if isinstance(row, dict) else None
            sampling_cfg = None
            if isinstance(context, dict):
                sampling_cfg = context.get("sampling_config")
            payload: dict[str, Any] = {
                "benchmark_name": row.get("benchmark_name", ""),
                "dataset_split": row.get("dataset_split", ""),
                "sample_index": int(row.get("sample_index", 0)),
                "repeat_index": int(row.get("repeat_index", 0)),
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
        with self._db.get_connection() as conn:
            rows = self._repo.fetch_completion_keys(
                conn,
                task_id=int(task_id),
            )
        return set(rows)

    def list_eval_payloads(
        self,
        *,
        task_id: str,
    ) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_eval_payloads(conn, task_id=int(task_id))

    def get_score_payload(
        self,
        *,
        task_id: str,
    ) -> dict[str, Any] | None:
        with self._db.get_connection() as conn:
            return self._repo.fetch_score_by_task(conn, task_id=int(task_id))

    def list_log_payloads(
        self,
        *,
        task_id: str,
    ) -> list[dict[str, Any]]:
        return []

    def get_task_payload(self, *, task_id: str) -> dict[str, Any] | None:
        with self._db.get_connection() as conn:
            return self._repo.fetch_task(conn, task_id=int(task_id))

    def get_task_bundle(self, *, task_id: str) -> dict[str, Any] | None:
        with self._db.get_connection() as conn:
            task = self._repo.fetch_task(conn, task_id=int(task_id))
            if not task:
                return None
            model_id = task.get("model_id")
            benchmark_id = task.get("benchmark_id")
            model = self._repo.fetch_model(conn, model_id=int(model_id)) if model_id else None
            benchmark = (
                self._repo.fetch_benchmark(conn, benchmark_id=int(benchmark_id)) if benchmark_id else None
            )
            return {"task": task, "model": model, "benchmark": benchmark}

    def list_completions_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_completions_rows(conn, task_id=int(task_id))

    def list_eval_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_eval_rows(conn, task_id=int(task_id))

    def list_scores_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_scores_rows(conn, task_id=int(task_id))

    def update_task_status(self, *, task_id: str, status: str) -> None:
        with self._db.get_connection() as conn:
            self._repo.update_task_status(conn, task_id=int(task_id), status=status)

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
    def _resolve_git_sha() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        sha = result.stdout.strip()
        return sha or None
