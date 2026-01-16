from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Json


@dataclass(slots=True)
class StageOutputRow:
    prompt: str | None
    completion: str | None
    finish_reason: str | None


class EvalDbRepository:
    @staticmethod
    def _json(value: dict[str, Any] | None) -> Json | None:
        return Json(value) if value is not None else None

    def upsert_dataset(
        self,
        conn: psycopg.Connection,
        *,
        dataset_slug: str,
        domain: str | None,
        dataset_version: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_dataset (dataset_slug, domain, dataset_version, meta)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (dataset_slug)
            DO UPDATE SET
                domain = EXCLUDED.domain,
                dataset_version = EXCLUDED.dataset_version,
                meta = EXCLUDED.meta,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (dataset_slug, domain, dataset_version, self._json(meta)),
        ).fetchone()
        return str(row["id"])

    def upsert_split(
        self,
        conn: psycopg.Connection,
        *,
        dataset_id: str,
        split_name: str,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_split (dataset_id, split_name)
            VALUES (%s, %s)
            ON CONFLICT (dataset_id, split_name)
            DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (dataset_id, split_name),
        ).fetchone()
        return str(row["id"])

    def upsert_sample(
        self,
        conn: psycopg.Connection,
        *,
        dataset_id: str,
        split_id: str,
        sample_index: int,
        question: str | None,
        reference_answer: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_sample (
                dataset_id, split_id, sample_index, question, reference_answer, meta
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id, split_id, sample_index)
            DO UPDATE SET
                question = EXCLUDED.question,
                reference_answer = EXCLUDED.reference_answer,
                meta = EXCLUDED.meta,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (dataset_id, split_id, sample_index, question, reference_answer, self._json(meta)),
        ).fetchone()
        return str(row["id"])

    def upsert_model(
        self,
        conn: psycopg.Connection,
        *,
        model_slug: str,
        model_name: str | None,
        model_revision: str | None,
        provider: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        row = conn.execute(
            """
            SELECT id
            FROM eval_model
            WHERE model_slug = %s AND COALESCE(model_revision, '') = COALESCE(%s, '')
            """,
            (model_slug, model_revision),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE eval_model
                SET model_name = %s,
                    model_revision = %s,
                    provider = %s,
                    meta = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (model_name, model_revision, provider, self._json(meta), row["id"]),
            )
            return str(row["id"])
        row = conn.execute(
            """
            INSERT INTO eval_model (model_slug, model_name, model_revision, provider, meta)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (model_slug, model_name, model_revision, provider, self._json(meta)),
        ).fetchone()
        return str(row["id"])

    def upsert_task(
        self,
        conn: psycopg.Connection,
        *,
        task_id: str,
        dataset_id: str,
        model_id: str,
        task_tag: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_task (task_id, dataset_id, model_id, task_tag, meta)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (task_id)
            DO UPDATE SET
                dataset_id = EXCLUDED.dataset_id,
                model_id = EXCLUDED.model_id,
                task_tag = EXCLUDED.task_tag,
                meta = EXCLUDED.meta,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (task_id, dataset_id, model_id, task_tag, self._json(meta)),
        ).fetchone()
        return str(row["id"])

    def get_run_by_tag(
        self,
        conn: psycopg.Connection,
        *,
        task_id: str,
        run_tag: str | None,
    ) -> str | None:
        row = conn.execute(
            """
            SELECT id
            FROM eval_run
            WHERE task_id = %s AND run_tag IS NOT DISTINCT FROM %s
            """,
            (task_id, run_tag),
        ).fetchone()
        return str(row["id"]) if row else None

    def delete_run_by_tag(
        self,
        conn: psycopg.Connection,
        *,
        task_id: str,
        run_tag: str | None,
    ) -> None:
        conn.execute(
            """
            DELETE FROM eval_run
            WHERE task_id = %s AND run_tag IS NOT DISTINCT FROM %s
            """,
            (task_id, run_tag),
        )

    def insert_run(
        self,
        conn: psycopg.Connection,
        *,
        task_id: str,
        run_tag: str | None,
        sampling_config: dict[str, Any] | None,
        runtime_config: dict[str, Any] | None,
        code_version: str | None,
        status: str,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_run (
                task_id, run_tag, sampling_config, runtime_config, code_version, status, started_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (
                task_id,
                run_tag,
                self._json(sampling_config),
                self._json(runtime_config),
                code_version,
                status,
            ),
        ).fetchone()
        return str(row["id"])

    def update_run_status(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str,
        status: str,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run
            SET status = %s,
                error_msg = %s,
                updated_at = CURRENT_TIMESTAMP,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, error_msg, finished, run_id),
        )

    def upsert_run_sample(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str,
        sample_id: str,
        repeat_index: int,
        status: str,
        current_stage: str | None,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_run_sample (
                run_id, sample_id, repeat_index, status, current_stage, started_at
            )
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (run_id, sample_id, repeat_index)
            DO UPDATE SET
                status = EXCLUDED.status,
                current_stage = EXCLUDED.current_stage,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (run_id, sample_id, repeat_index, status, current_stage),
        ).fetchone()
        return str(row["id"])

    def update_run_sample_status(
        self,
        conn: psycopg.Connection,
        *,
        run_sample_id: str,
        status: str,
        current_stage: str | None,
        latest_attempt_index: int | None = None,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run_sample
            SET status = %s,
                current_stage = %s,
                latest_attempt_index = COALESCE(%s, latest_attempt_index),
                error_msg = %s,
                updated_at = CURRENT_TIMESTAMP,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, current_stage, latest_attempt_index, error_msg, finished, run_sample_id),
        )

    def get_latest_attempt_index(
        self,
        conn: psycopg.Connection,
        *,
        run_sample_id: str,
    ) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(attempt_index), -1) AS max_attempt
            FROM eval_attempt
            WHERE run_sample_id = %s
            """,
            (run_sample_id,),
        ).fetchone()
        return int(row["max_attempt"])

    def insert_attempt(
        self,
        conn: psycopg.Connection,
        *,
        run_sample_id: str,
        attempt_index: int,
        worker_id: str | None,
        shard_id: int | None,
        shard_count: int | None,
        seed: int | None,
        status: str,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_attempt (
                run_sample_id, attempt_index, worker_id, shard_id, shard_count, seed, status, started_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (run_sample_id, attempt_index, worker_id, shard_id, shard_count, seed, status),
        ).fetchone()
        return str(row["id"])

    def update_attempt_status(
        self,
        conn: psycopg.Connection,
        *,
        attempt_id: str,
        status: str,
        error_msg: str | None = None,
        finished: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_attempt
            SET status = %s,
                error_msg = %s,
                updated_at = CURRENT_TIMESTAMP,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, error_msg, finished, attempt_id),
        )

    def fetch_latest_final_stage(
        self,
        conn: psycopg.Connection,
        *,
        run_sample_id: str,
        stage: str,
    ) -> StageOutputRow | None:
        row = conn.execute(
            """
            SELECT s.prompt, s.completion, s.finish_reason
            FROM eval_stage_output s
            JOIN eval_attempt a ON a.id = s.attempt_id
            WHERE a.run_sample_id = %s AND s.stage = %s AND s.is_final = TRUE
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            (run_sample_id, stage),
        ).fetchone()
        if not row:
            return None
        return StageOutputRow(
            prompt=row.get("prompt"),
            completion=row.get("completion"),
            finish_reason=row.get("finish_reason"),
        )

    def get_run_sample_id(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str,
        sample_index: int,
        repeat_index: int,
    ) -> str | None:
        row = conn.execute(
            """
            SELECT rs.id
            FROM eval_run_sample rs
            JOIN eval_sample s ON s.id = rs.sample_id
            WHERE rs.run_id = %s AND s.sample_index = %s AND rs.repeat_index = %s
            """,
            (run_id, sample_index, repeat_index),
        ).fetchone()
        return str(row["id"]) if row else None

    def insert_stage_output(
        self,
        conn: psycopg.Connection,
        *,
        attempt_id: str,
        stage: str,
        seq: int,
        prompt: str | None,
        completion: str | None,
        finish_reason: str | None,
        provider_request_id: str | None,
        raw_response: dict[str, Any] | None,
        token_count_prompt: int | None,
        token_count_response: int | None,
        latency_ms: int | None,
        cost_usd: float | None,
        is_partial: bool,
        is_final: bool,
    ) -> None:
        conn.execute(
            """
            INSERT INTO eval_stage_output (
                attempt_id, stage, seq, prompt, completion, finish_reason,
                provider_request_id, raw_response, token_count_prompt, token_count_response,
                latency_ms, cost_usd, is_partial, is_final
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                attempt_id,
                stage,
                seq,
                prompt,
                completion,
                finish_reason,
                provider_request_id,
                self._json(raw_response),
                token_count_prompt,
                token_count_response,
                latency_ms,
                cost_usd,
                is_partial,
                is_final,
            ),
        )

    def insert_metric(
        self,
        conn: psycopg.Connection,
        *,
        run_sample_id: str,
        name: str,
        value_num: float | None,
        value_text: str | None,
        meta: dict[str, Any] | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO eval_metric (run_sample_id, name, value_num, value_text, meta)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_sample_id, name, value_num, value_text, self._json(meta)),
        )

    def insert_run_event(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str | None,
        run_sample_id: str | None,
        event_type: str,
        message: str | None,
        meta: dict[str, Any] | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO eval_run_event (run_id, run_sample_id, event_type, message, meta)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, run_sample_id, event_type, message, self._json(meta)),
        )

