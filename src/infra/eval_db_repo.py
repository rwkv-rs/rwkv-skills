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

    def get_run_by_tag(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        dataset: str,
        dataset_split: str,
        model_name: str,
        cot: bool,
        run_tag: str | None,
    ) -> str | None:
        row = conn.execute(
            """
            SELECT id
            FROM eval_run
            WHERE benchmark_name = %s
              AND dataset = %s
              AND dataset_split = %s
              AND model_name = %s
              AND cot = %s
              AND run_tag IS NOT DISTINCT FROM %s
            """,
            (benchmark_name, dataset, dataset_split, model_name, cot, run_tag),
        ).fetchone()
        return str(row["id"]) if row else None

    def delete_run_by_tag(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        dataset: str,
        dataset_split: str,
        model_name: str,
        cot: bool,
        run_tag: str | None,
    ) -> None:
        conn.execute(
            """
            DELETE FROM eval_run
            WHERE benchmark_name = %s
              AND dataset = %s
              AND dataset_split = %s
              AND model_name = %s
              AND cot = %s
              AND run_tag IS NOT DISTINCT FROM %s
            """,
            (benchmark_name, dataset, dataset_split, model_name, cot, run_tag),
        )

    def insert_run(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        dataset: str,
        dataset_split: str,
        model_name: str,
        model_slug: str | None,
        model_revision: str | None,
        model_path: str | None,
        cot: bool,
        run_tag: str | None,
        sampling_config: dict[str, Any] | None,
        runtime_config: dict[str, Any] | None,
        code_version: str | None,
        task: str | None,
        task_details: dict[str, Any] | None,
        status: str,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_run (
                benchmark_name, dataset, dataset_split, model_name, model_slug,
                model_revision, model_path, cot, run_tag, sampling_config, runtime_config,
                code_version, task, task_details, status, started_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (
                benchmark_name,
                dataset,
                dataset_split,
                model_name,
                model_slug,
                model_revision,
                model_path,
                cot,
                run_tag,
                self._json(sampling_config),
                self._json(runtime_config),
                code_version,
                task,
                self._json(task_details),
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
        start_now: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run
            SET status = %s,
                error_msg = %s,
                started_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE started_at END,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, error_msg, start_now, finished, run_id),
        )

    def update_run_summary(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str,
        metrics: dict[str, Any],
        samples: int,
        problems: int | None,
        log_path: str | None,
        eval_details_path: str | None,
        task: str | None,
        task_details: dict[str, Any] | None,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run
            SET metrics = %s,
                samples = %s,
                problems = %s,
                log_path = %s,
                eval_details_path = %s,
                task = %s,
                task_details = %s
            WHERE id = %s
            """,
            (
                self._json(metrics),
                samples,
                problems,
                log_path,
                eval_details_path,
                task,
                self._json(task_details),
                run_id,
            ),
        )

    def upsert_sample(
        self,
        conn: psycopg.Connection,
        *,
        benchmark_name: str,
        dataset_split: str,
        sample_index: int,
        question: str | None,
        reference_answer: str | None,
        meta: dict[str, Any] | None,
    ) -> str:
        row = conn.execute(
            """
            INSERT INTO eval_sample (
                benchmark_name, dataset_split, sample_index, question, ref_answer, meta
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (benchmark_name, dataset_split, sample_index)
            DO UPDATE SET
                question = COALESCE(EXCLUDED.question, eval_sample.question),
                ref_answer = COALESCE(EXCLUDED.ref_answer, eval_sample.ref_answer),
                meta = COALESCE(EXCLUDED.meta, eval_sample.meta)
            RETURNING id
            """,
            (benchmark_name, dataset_split, sample_index, question, reference_answer, self._json(meta)),
        ).fetchone()
        return str(row["id"])

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
            VALUES (%s, %s, %s, %s, %s, CASE WHEN %s <> 'pending' THEN CURRENT_TIMESTAMP ELSE NULL END)
            ON CONFLICT (run_id, sample_id, repeat_index)
            DO UPDATE SET
                status = EXCLUDED.status,
                current_stage = EXCLUDED.current_stage
            RETURNING id
            """,
            (run_id, sample_id, repeat_index, status, current_stage, status),
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
        start_now: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run_sample
            SET status = %s,
                current_stage = %s,
                latest_attempt_index = COALESCE(%s, latest_attempt_index),
                error_msg = %s,
                started_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE started_at END,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, current_stage, latest_attempt_index, error_msg, start_now, finished, run_sample_id),
        )

    def update_run_sample_result(
        self,
        conn: psycopg.Connection,
        *,
        run_id: str,
        sample_index: int,
        repeat_index: int,
        answer: str | None,
        ref_answer: str | None,
        is_passed: bool,
        fail_reason: str | None,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_run_sample rs
            SET answer = %s,
                is_passed = %s,
                fail_reason = %s,
                status = 'succeeded',
                finished_at = CURRENT_TIMESTAMP
            FROM eval_sample s
            WHERE rs.sample_id = s.id
              AND rs.run_id = %s
              AND s.sample_index = %s
              AND rs.repeat_index = %s
            """,
            (answer, is_passed, fail_reason, run_id, sample_index, repeat_index),
        )
        if ref_answer is not None:
            conn.execute(
                """
                UPDATE eval_sample s
                SET ref_answer = %s
                FROM eval_run_sample rs
                WHERE rs.sample_id = s.id
                  AND rs.run_id = %s
                  AND s.sample_index = %s
                  AND rs.repeat_index = %s
                """,
                (ref_answer, run_id, sample_index, repeat_index),
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
        start_now: bool = False,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_attempt
            SET status = %s,
                error_msg = %s,
                started_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE started_at END,
                finished_at = CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE finished_at END
            WHERE id = %s
            """,
            (status, error_msg, start_now, finished, attempt_id),
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

    def insert_cot_checkpoint(
        self,
        conn: psycopg.Connection,
        *,
        attempt_id: str,
        stage: str,
        token_offset: int | None,
        partial_completion: str | None,
        kv_cache_ref: str | None,
        rng_state: dict[str, Any] | None,
        status: str | None,
    ) -> None:
        conn.execute(
            """
            UPDATE eval_cot_checkpoint
            SET latest = FALSE
            WHERE attempt_id = %s AND stage = %s AND latest = TRUE
            """,
            (attempt_id, stage),
        )
        conn.execute(
            """
            INSERT INTO eval_cot_checkpoint (
                attempt_id, stage, token_offset, partial_completion, kv_cache_ref, rng_state, status, latest
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
            """,
            (
                attempt_id,
                stage,
                token_offset,
                partial_completion,
                kv_cache_ref,
                self._json(rng_state),
                status,
            ),
        )
