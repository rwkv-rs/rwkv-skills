from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from src.db.database import DatabaseManager
from src.eval.results.schema import iter_stage_indices
from src.eval.scheduler.config import REPO_ROOT

from .eval_db_repo import EvalDbRepository


class EvalDbService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._repo = EvalDbRepository()

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
        with self._db.get_connection() as conn:
            if allow_resume:
                latest = self._repo.get_latest_version_id(
                    conn,
                    dataset=dataset,
                    model=model,
                    is_param_search=is_param_search,
                )
                if latest and not self._repo.version_has_score(conn, version_id=latest):
                    return latest
            git_sha = self._resolve_git_sha()
            return self._repo.insert_version(
                conn,
                job_name=job_name,
                job_id=job_id,
                dataset=dataset,
                model=model,
                git_sha=git_sha,
                is_param_search=is_param_search,
            )

    def ingest_completions(
        self,
        *,
        completions_path: str | Path,
        version_id: str,
        is_param_search: bool,
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
                    version_id=version_id,
                    is_param_search=is_param_search,
                    payload=payload,
                    context=context,
                )
                inserted += 1
        return inserted

    def ingest_completion_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        version_id: str,
        is_param_search: bool,
    ) -> int:
        inserted = 0
        with self._db.get_connection() as conn:
            for payload in payloads:
                context = self._build_completion_context(payload)
                self._repo.insert_completion(
                    conn,
                    version_id=version_id,
                    is_param_search=is_param_search,
                    payload=payload,
                    context=context,
                )
                inserted += 1
        return inserted

    def insert_completion_payload(
        self,
        *,
        payload: dict[str, Any],
        version_id: str,
        is_param_search: bool,
    ) -> None:
        with self._db.get_connection() as conn:
            context = self._build_completion_context(payload)
            self._repo.insert_completion(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
                payload=payload,
                context=context,
            )

    def ingest_eval(
        self,
        *,
        eval_path: str | Path,
        version_id: str,
        is_param_search: bool,
    ) -> int:
        path = Path(eval_path)
        if not path.exists():
            return 0
        inserted = 0
        with path.open("r", encoding="utf-8") as fh, self._db.get_connection() as conn:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                self._repo.insert_eval(
                    conn,
                    version_id=version_id,
                    is_param_search=is_param_search,
                    payload=payload,
                )
                inserted += 1
        return inserted

    def ingest_eval_payloads(
        self,
        *,
        payloads: Iterable[dict[str, Any]],
        version_id: str,
        is_param_search: bool,
    ) -> int:
        inserted = 0
        with self._db.get_connection() as conn:
            for payload in payloads:
                self._repo.insert_eval(
                    conn,
                    version_id=version_id,
                    is_param_search=is_param_search,
                    payload=payload,
                )
                inserted += 1
        return inserted

    def insert_eval_payload(
        self,
        *,
        payload: dict[str, Any],
        version_id: str,
        is_param_search: bool,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.insert_eval(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
                payload=payload,
            )

    def record_score(
        self,
        *,
        score_path: str | Path,
        version_id: str,
        is_param_search: bool,
    ) -> None:
        path = Path(score_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self._db.get_connection() as conn:
            self._repo.insert_score(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
                payload=payload,
            )

    def record_score_payload(
        self,
        *,
        payload: dict[str, Any],
        version_id: str,
        is_param_search: bool,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.insert_score(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
                payload=payload,
            )

    def record_log_event(
        self,
        *,
        event: str,
        job_id: str,
        payload: dict[str, Any],
        version_id: str | None,
    ) -> None:
        with self._db.get_connection() as conn:
            self._repo.insert_log_event(
                conn,
                event=event,
                job_id=job_id,
                payload=payload,
                version_id=version_id,
            )

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
        with self._db.get_connection() as conn:
            return self._repo.fetch_scores_by_dataset(
                conn,
                dataset=dataset,
                model=model,
                is_param_search=is_param_search,
            )

    def count_completions(
        self,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> int:
        with self._db.get_connection() as conn:
            return self._repo.count_completions(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
            )

    def list_completion_payloads(
        self,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            rows = self._repo.fetch_completions(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
            )
        payloads: list[dict[str, Any]] = []
        for row in rows:
            context = row.get("context") if isinstance(row, dict) else None
            sampling_cfg = row.get("sampling_config") if isinstance(row, dict) else None
            payload: dict[str, Any] = {
                "benchmark_name": row.get("benchmark_name", ""),
                "dataset_split": row.get("dataset_split", ""),
                "sample_index": int(row.get("sample_index", 0)),
                "repeat_index": int(row.get("repeat_index", 0)),
                "sampling_config": sampling_cfg if isinstance(sampling_cfg, dict) else {},
                "context": context if isinstance(context, dict) else None,
            }
            payloads.append(payload)
        return payloads

    def list_completion_keys(
        self,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> set[tuple[int, int]]:
        with self._db.get_connection() as conn:
            rows = self._repo.fetch_completion_keys(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
            )
        return set(rows)

    def list_eval_payloads(
        self,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_eval_payloads(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
            )

    def get_score_payload(
        self,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> dict[str, Any] | None:
        with self._db.get_connection() as conn:
            return self._repo.fetch_score_by_version(
                conn,
                version_id=version_id,
                is_param_search=is_param_search,
            )

    def list_log_payloads(
        self,
        *,
        version_id: str,
    ) -> list[dict[str, Any]]:
        with self._db.get_connection() as conn:
            return self._repo.fetch_logs_by_version(conn, version_id=version_id)

    def get_version_payload(self, *, version_id: str) -> dict[str, Any] | None:
        with self._db.get_connection() as conn:
            return self._repo.fetch_version(conn, version_id=version_id)

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
