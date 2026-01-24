from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Json


class EvalDbRepository:
    @staticmethod
    def _json(value: dict[str, Any] | None) -> Json | None:
        return Json(value) if value is not None else None

    def insert_version(
        self,
        conn: psycopg.Connection,
        *,
        job_name: str | None,
        job_id: str | None,
        dataset: str | None,
        model: str | None,
        git_sha: str | None,
        is_param_search: bool,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO version (job_name, job_id, dataset, model, git_sha, is_param_search)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (job_name, job_id, dataset, model, git_sha, is_param_search),
        ).fetchone()
        return str(row["id"])

    def get_latest_version_id(
        self,
        conn: psycopg.Connection,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
    ) -> str | None:
        row = conn.execute(
            """
            SELECT id
            FROM version
            WHERE dataset = %s AND model = %s AND is_param_search = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset, model, is_param_search),
        ).fetchone()
        return str(row["id"]) if row else None

    def version_has_score(self, conn: psycopg.Connection, *, version_id: str) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM score
            WHERE version_id = %s
            LIMIT 1
            """,
            (version_id,),
        ).fetchone()
        return bool(row)

    def fetch_latest_scores(self, conn: psycopg.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT s.*, v.job_name, v.job_id
            FROM view_score_latest s
            LEFT JOIN version v ON v.id = s.version_id
            """
        ).fetchall()
        return list(rows or [])

    def fetch_scores_by_dataset(
        self,
        conn: psycopg.Connection,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT s.*, v.job_name, v.job_id
            FROM score s
            LEFT JOIN version v ON v.id = s.version_id
            WHERE s.dataset = %s AND s.model = %s AND s.is_param_search = %s
            """,
            (dataset, model, is_param_search),
        ).fetchall()
        return list(rows or [])

    def count_completions(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM completions
            WHERE version_id = %s AND is_param_search = %s
            """,
            (version_id, is_param_search),
        ).fetchone()
        return int(row["count"]) if row and row.get("count") is not None else 0

    def fetch_completions(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT benchmark_name, dataset_split, sample_index, repeat_index, sampling_config, context
            FROM completions
            WHERE version_id = %s AND is_param_search = %s
            ORDER BY sample_index ASC, repeat_index ASC
            """,
            (version_id, is_param_search),
        ).fetchall()
        return list(rows or [])

    def fetch_completion_keys(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> list[tuple[int, int]]:
        rows = conn.execute(
            """
            SELECT sample_index, repeat_index
            FROM completions
            WHERE version_id = %s AND is_param_search = %s
            """,
            (version_id, is_param_search),
        ).fetchall()
        keys: list[tuple[int, int]] = []
        for row in rows or []:
            keys.append((int(row.get("sample_index", 0)), int(row.get("repeat_index", 0))))
        return keys

    def fetch_eval_payloads(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT benchmark_name, dataset_split, sample_index, repeat_index,
                   context, answer, ref_answer, is_passed, fail_reason
            FROM eval
            WHERE version_id = %s AND is_param_search = %s
            ORDER BY sample_index ASC, repeat_index ASC
            """,
            (version_id, is_param_search),
        ).fetchall()
        return list(rows or [])

    def fetch_score_by_version(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM score
            WHERE version_id = %s AND is_param_search = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (version_id, is_param_search),
        ).fetchone()
        return dict(row) if row else None

    def fetch_logs_by_version(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT event, job_id, payload, created_at
            FROM logs
            WHERE version_id = %s
            ORDER BY created_at ASC
            """,
            (version_id,),
        ).fetchall()
        return list(rows or [])

    def fetch_version(self, conn: psycopg.Connection, *, version_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM version
            WHERE id = %s
            """,
            (version_id,),
        ).fetchone()
        return dict(row) if row else None

    def insert_completion(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
        payload: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO completions (
                version_id, is_param_search,
                benchmark_name, dataset_split, sample_index, repeat_index, sampling_config, context
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, benchmark_name, dataset_split, sample_index, repeat_index)
            DO NOTHING
            """,
            (
                version_id,
                is_param_search,
                str(payload.get("benchmark_name", "")),
                str(payload.get("dataset_split", "")),
                int(payload.get("sample_index", 0)),
                int(payload.get("repeat_index", 0)),
                self._json(payload.get("sampling_config") if isinstance(payload.get("sampling_config"), dict) else {}),
                self._json(context),
            ),
        )

    def insert_eval(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
        payload: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO eval (
                version_id, is_param_search,
                benchmark_name, dataset_split, sample_index, repeat_index,
                context, answer, ref_answer, is_passed, fail_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, benchmark_name, dataset_split, sample_index, repeat_index)
            DO NOTHING
            """,
            (
                version_id,
                is_param_search,
                str(payload.get("benchmark_name", "")),
                str(payload.get("dataset_split", "")),
                int(payload.get("sample_index", 0)),
                int(payload.get("repeat_index", 0)),
                payload.get("context"),
                payload.get("answer"),
                payload.get("ref_answer"),
                bool(payload.get("is_passed", False)),
                payload.get("fail_reason"),
            ),
        )

    def insert_score(
        self,
        conn: psycopg.Connection,
        *,
        version_id: str,
        is_param_search: bool,
        payload: dict[str, Any],
    ) -> None:
        created_at = payload.get("created_at")
        if not created_at:
            created_at = datetime.utcnow().replace(microsecond=False).isoformat() + "Z"
        conn.execute(
            """
            INSERT INTO score (
                version_id, is_param_search,
                dataset, model, cot, metrics, samples, problems, created_at,
                log_path, task, task_details
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                version_id,
                is_param_search,
                str(payload.get("dataset", "")),
                str(payload.get("model", "")),
                bool(payload.get("cot", False)),
                self._json(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}),
                int(payload.get("samples", 0)),
                int(payload["problems"]) if payload.get("problems") is not None else None,
                created_at,
                payload.get("log_path"),
                payload.get("task"),
                self._json(payload.get("task_details") if isinstance(payload.get("task_details"), dict) else None),
            ),
        )

    def insert_log_event(
        self,
        conn: psycopg.Connection,
        *,
        event: str,
        job_id: str,
        payload: dict[str, Any],
        version_id: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO logs (version_id, event, job_id, payload)
            VALUES (%s, %s, %s, %s)
            """,
            (version_id, event, job_id, self._json(payload)),
        )
