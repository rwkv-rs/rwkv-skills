from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import psycopg
from psycopg.types.json import Json


class EvalDbRepository:
    @staticmethod
    def _json(value: dict[str, Any] | None) -> Json | None:
        return Json(value) if value is not None else None

    def get_benchmark_id(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        benchmark_split: str,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT benchmark_id
            FROM benchmark
            WHERE benchmark_name = %s AND benchmark_split = %s
            """,
            (benchmark_name, benchmark_split),
        ).fetchone()
        return int(row["benchmark_id"]) if row else None

    def get_benchmark_num_samples(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_id: int,
    ) -> str | None:
        row = conn.execute(
            """
            SELECT num_samples
            FROM benchmark
            WHERE benchmark_id = %s
            """,
            (benchmark_id,),
        ).fetchone()
        if not row:
            return None
        value = row.get("num_samples")
        return str(value) if value is not None else None

    def update_benchmark_num_samples(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_id: int,
        num_samples: str,
    ) -> None:
        conn.execute(
            """
            UPDATE benchmark
            SET num_samples = %s
            WHERE benchmark_id = %s
            """,
            (num_samples, benchmark_id),
        )

    def insert_benchmark(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        benchmark_split: str,
        url: str | None,
        status: str,
        num_samples: str,
    ) -> int:
        row = conn.execute(
            """
            INSERT INTO benchmark (benchmark_name, benchmark_split, url, status, num_samples)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING benchmark_id
            """,
            (benchmark_name, benchmark_split, url, status, num_samples),
        ).fetchone()
        return int(row["benchmark_id"])

    def get_model_id(
        self,
        conn: psycopg.Connection,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT model_id
            FROM model
            WHERE model_name = %s AND arch_version = %s AND data_version = %s AND num_params = %s
            """,
            (model_name, arch_version, data_version, num_params),
        ).fetchone()
        return int(row["model_id"]) if row else None

    def insert_model(
        self,
        conn: psycopg.Connection,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int:
        row = conn.execute(
            """
            INSERT INTO model (model_name, arch_version, data_version, num_params)
            VALUES (%s, %s, %s, %s)
            RETURNING model_id
            """,
            (model_name, arch_version, data_version, num_params),
        ).fetchone()
        return int(row["model_id"])

    def insert_task(
        self,
        conn: psycopg.Connection,
        *,
        config_path: str | None,
        evaluator: str,
        is_param_search: bool,
        created_at: datetime,
        status: str,
        git_hash: str | None,
        model_id: int,
        benchmark_id: int,
        desc: str | None,
        sampling_config: dict[str, Any] | None,
        log_path: str,
    ) -> int:
        row = conn.execute(
            """
            INSERT INTO task (
                config_path, evaluator, is_param_search, created_at, status, git_hash,
                model_id, benchmark_id, "desc", sampling_config, log_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING task_id
            """,
            (
                config_path,
                evaluator,
                is_param_search,
                created_at,
                status,
                git_hash,
                model_id,
                benchmark_id,
                desc,
                self._json(sampling_config) if sampling_config is not None else None,
                log_path,
            ),
        ).fetchone()
        return int(row["task_id"])

    def update_task_status(self, conn: psycopg.Connection, *, task_id: int, status: str) -> None:
        conn.execute(
            """
            UPDATE task
            SET status = %s
            WHERE task_id = %s
            """,
            (status, task_id),
        )

    def update_benchmark_num_samples_for_task(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
        num_samples: str,
    ) -> None:
        conn.execute(
            """
            UPDATE benchmark
            SET num_samples = %s
            WHERE benchmark_id = (
                SELECT benchmark_id FROM task WHERE task_id = %s
            )
            """,
            (num_samples, task_id),
        )

    def get_latest_task_id(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_id: int,
        model_id: int,
        is_param_search: bool,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT task_id
            FROM task
            WHERE benchmark_id = %s AND model_id = %s AND is_param_search = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (benchmark_id, model_id, is_param_search),
        ).fetchone()
        return int(row["task_id"]) if row else None

    def task_has_score(self, conn: psycopg.Connection, *, task_id: int) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM scores
            WHERE task_id = %s
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        return bool(row)

    def fetch_latest_scores(self, conn: psycopg.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT
                s.task_id,
                s.is_cot AS cot,
                s.metrics,
                s.created_at,
                t.is_param_search,
                m.model_name AS model,
                CASE
                    WHEN b.benchmark_split <> '' THEN b.benchmark_name || '_' || b.benchmark_split
                    ELSE b.benchmark_name
                END AS dataset,
                NULL::INT AS samples,
                NULL::INT AS problems
            FROM (
                SELECT
                    s.*,
                    t.model_id,
                    t.benchmark_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.model_id, t.benchmark_id, s.is_cot
                        ORDER BY s.created_at DESC
                    ) AS rn
                FROM scores s
                JOIN task t ON t.task_id = s.task_id
            ) s
            JOIN task t ON t.task_id = s.task_id
            JOIN model m ON m.model_id = t.model_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE s.rn = 1 AND t.is_param_search = FALSE
            """
        ).fetchall()
        return list(rows or [])

    def fetch_scores_by_benchmark(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        benchmark_split: str,
        model_name: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT
                s.task_id,
                s.is_cot AS cot,
                s.metrics,
                s.created_at,
                t.is_param_search,
                m.model_name AS model,
                CASE
                    WHEN b.benchmark_split <> '' THEN b.benchmark_name || '_' || b.benchmark_split
                    ELSE b.benchmark_name
                END AS dataset,
                NULL::INT AS samples,
                NULL::INT AS problems
            FROM scores s
            JOIN task t ON t.task_id = s.task_id
            JOIN model m ON m.model_id = t.model_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE b.benchmark_name = %s AND b.benchmark_split = %s
              AND m.model_name = %s AND t.is_param_search = %s
            ORDER BY s.created_at DESC
            """,
            (benchmark_name, benchmark_split, model_name, is_param_search),
        ).fetchall()
        return list(rows or [])

    def count_completions(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM completions
            WHERE task_id = %s
            """,
            (task_id,),
        ).fetchone()
        return int(row["count"]) if row and row.get("count") is not None else 0

    def fetch_completions(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT b.benchmark_name, b.benchmark_split, c.sample_index, c.repeat_index, c.context
            FROM completions c
            JOIN task t ON t.task_id = c.task_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE c.task_id = %s
            ORDER BY c.sample_index ASC, c.repeat_index ASC
            """,
            (task_id,),
        ).fetchall()
        return list(rows or [])

    def fetch_completion_keys(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> list[tuple[int, int]]:
        rows = conn.execute(
            """
            SELECT sample_index, repeat_index
            FROM completions
            WHERE task_id = %s
            """,
            (task_id,),
        ).fetchall()
        keys: list[tuple[int, int]] = []
        for row in rows or []:
            keys.append((int(row.get("sample_index", 0)), int(row.get("repeat_index", 0))))
        return keys

    def fetch_completion_id_map(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> dict[tuple[int, int], int]:
        rows = conn.execute(
            """
            SELECT completions_id, sample_index, repeat_index
            FROM completions
            WHERE task_id = %s
            """,
            (task_id,),
        ).fetchall()
        mapping: dict[tuple[int, int], int] = {}
        for row in rows or []:
            key = (int(row.get("sample_index", 0)), int(row.get("repeat_index", 0)))
            mapping[key] = int(row.get("completions_id"))
        return mapping

    def fetch_eval_payloads(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT b.benchmark_name, b.benchmark_split, c.sample_index, c.repeat_index,
                   c.context, e.answer, e.ref_answer, e.is_passed, e.fail_reason
            FROM eval e
            JOIN completions c ON c.completions_id = e.completions_id
            JOIN task t ON t.task_id = c.task_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE t.task_id = %s
            ORDER BY c.sample_index ASC, c.repeat_index ASC
            """,
            (task_id,),
        ).fetchall()
        return list(rows or [])

    def fetch_score_by_task(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT
                s.task_id,
                s.is_cot AS cot,
                s.metrics,
                s.created_at,
                m.model_name AS model,
                CASE
                    WHEN b.benchmark_split <> '' THEN b.benchmark_name || '_' || b.benchmark_split
                    ELSE b.benchmark_name
                END AS dataset
            FROM scores s
            JOIN task t ON t.task_id = s.task_id
            JOIN model m ON m.model_id = t.model_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE s.task_id = %s
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_task(self, conn: psycopg.Connection, *, task_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM task
            WHERE task_id = %s
            """,
            (task_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_latest_task_by_names(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        benchmark_split: str,
        model_name: str,
        is_param_search: bool,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT t.*
            FROM task t
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            JOIN model m ON m.model_id = t.model_id
            WHERE b.benchmark_name = %s
              AND b.benchmark_split = %s
              AND m.model_name = %s
              AND t.is_param_search = %s
            ORDER BY t.created_at DESC
            LIMIT 1
            """,
            (benchmark_name, benchmark_split, model_name, is_param_search),
        ).fetchone()
        return dict(row) if row else None

    def fetch_model(self, conn: psycopg.Connection, *, model_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM model
            WHERE model_id = %s
            """,
            (model_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_benchmark(self, conn: psycopg.Connection, *, benchmark_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM benchmark
            WHERE benchmark_id = %s
            """,
            (benchmark_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_completions_rows(self, conn: psycopg.Connection, *, task_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM completions
            WHERE task_id = %s
            ORDER BY completions_id ASC
            """,
            (task_id,),
        ).fetchall()
        return list(rows or [])

    def fetch_eval_rows(self, conn: psycopg.Connection, *, task_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT e.*
            FROM eval e
            JOIN completions c ON c.completions_id = e.completions_id
            WHERE c.task_id = %s
            ORDER BY e.eval_id ASC
            """,
            (task_id,),
        ).fetchall()
        return list(rows or [])

    def fetch_scores_rows(self, conn: psycopg.Connection, *, task_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM scores
            WHERE task_id = %s
            ORDER BY created_at DESC
            """,
            (task_id,),
        ).fetchall()
        return list(rows or [])

    def insert_completion(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
        payload: dict[str, Any],
        context: dict[str, Any] | None,
        created_at: datetime,
        status: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO completions (
                task_id, context, sample_index, repeat_index, created_at, status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_id, sample_index, repeat_index)
            DO NOTHING
            """,
            (
                task_id,
                self._json(context or {}),
                str(payload.get("sample_index", "")),
                str(payload.get("repeat_index", "")),
                created_at,
                status,
            ),
        )

    def insert_eval(
        self,
        conn: psycopg.Connection,
        *,
        completions_id: int,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO eval (
                completions_id, answer, ref_answer, is_passed, fail_reason, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                completions_id,
                payload.get("answer"),
                payload.get("ref_answer"),
                bool(payload.get("is_passed", False)),
                payload.get("fail_reason"),
                created_at,
            ),
        )

    def insert_score(
        self,
        conn: psycopg.Connection,
        *,
        task_id: int,
        payload: dict[str, Any],
    ) -> None:
        created_at = payload.get("created_at")
        if not created_at:
            created_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)
        conn.execute(
            """
            INSERT INTO scores (
                task_id, is_cot, metrics, created_at
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                task_id,
                bool(payload.get("cot", False)),
                self._json(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}),
                created_at,
            ),
        )
