from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.db.pool import Db, init_db_pool
from src.eval.results.schema import strict_nonneg_int


def _canonical_task_status(status: str) -> str:
    raw = str(status or "").strip().lower()
    mapping = {
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
    }
    return mapping.get(raw, status)


def _canonical_completion_status(status: str | None) -> str:
    raw = str(status or "").strip().lower()
    mapping = {
        "completed": "Completed",
        "running": "Running",
        "failed": "Failed",
    }
    if raw in mapping:
        return mapping[raw]
    raise ValueError(f"unsupported completion status: {status!r}")


def _parse_cot_mode(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    mapping = {
        "nocot": "NoCoT",
        "no_cot": "NoCoT",
        "no-cot": "NoCoT",
        "fakecot": "FakeCoT",
        "fake_cot": "FakeCoT",
        "fake-cot": "FakeCoT",
        "cot": "CoT",
    }
    return mapping.get(raw)


def _canonical_score_cot_mode(payload: dict[str, Any]) -> str:
    task_details = payload.get("task_details")
    sampling_config = payload.get("sampling_config")
    for candidate in (
        payload.get("cot_mode"),
        task_details.get("cot_mode") if isinstance(task_details, dict) else None,
        sampling_config.get("cot_mode") if isinstance(sampling_config, dict) else None,
    ):
        parsed = _parse_cot_mode(candidate)
        if parsed is not None:
            return parsed
    return "CoT" if bool(payload.get("cot", False)) else "NoCoT"


def _jsonb_param(value: Any) -> Jsonb | None:
    if value is None:
        return None
    return Jsonb(value)


class SqlEvalDbRepository:
    def __init__(self, db: Db | None = None) -> None:
        self.db = db or init_db_pool()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self.db.pool.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_benchmark_id(
        self,
        *,
        benchmark_name: str,
        benchmark_split: str,
    ) -> int | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT benchmark_id
                    FROM benchmark
                    WHERE benchmark_name = %s
                      AND benchmark_split = %s
                    """,
                    (benchmark_name, benchmark_split),
                )
                row = cur.fetchone()
        return int(row[0]) if row else None

    def get_benchmark_num_samples(self, *, benchmark_id: int) -> int | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT num_samples
                    FROM benchmark
                    WHERE benchmark_id = %s
                    """,
                    (int(benchmark_id),),
                )
                row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def update_benchmark_num_samples(self, *, benchmark_id: int, num_samples: int) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE benchmark
                    SET num_samples = %s
                    WHERE benchmark_id = %s
                    """,
                    (int(num_samples), int(benchmark_id)),
                )

    def insert_benchmark(
        self,
        *,
        benchmark_name: str,
        benchmark_split: str,
        url: str | None,
        status: str,
        num_samples: int,
    ) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark (
                        benchmark_name,
                        benchmark_split,
                        url,
                        status,
                        num_samples
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (benchmark_name, benchmark_split) DO UPDATE
                    SET benchmark_name = EXCLUDED.benchmark_name
                    RETURNING benchmark_id
                    """,
                    (
                        benchmark_name,
                        benchmark_split,
                        url,
                        str(status),
                        int(num_samples),
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to insert benchmark")
        return int(row[0])

    def get_model_id(
        self,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT model_id
                    FROM model
                    WHERE model_name = %s
                      AND arch_version = %s
                      AND data_version = %s
                      AND num_params = %s
                    """,
                    (
                        model_name,
                        arch_version,
                        data_version,
                        num_params,
                    ),
                )
                row = cur.fetchone()
        return int(row[0]) if row else None

    def insert_model(
        self,
        *,
        model_name: str,
        arch_version: str,
        data_version: str,
        num_params: str,
    ) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model (
                        data_version,
                        arch_version,
                        num_params,
                        model_name
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (arch_version, data_version, num_params, model_name) DO UPDATE
                    SET model_name = EXCLUDED.model_name
                    RETURNING model_id
                    """,
                    (
                        data_version,
                        arch_version,
                        num_params,
                        model_name,
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to insert model")
        return int(row[0])

    def insert_task(
        self,
        *,
        config_path: str | None,
        evaluator: str,
        is_param_search: bool,
        is_tmp: bool,
        created_at: datetime,
        status: str,
        git_hash: str,
        model_id: int,
        benchmark_id: int,
        desc: str | None,
        sampling_config: dict[str, Any] | None,
        log_path: str,
    ) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO task (
                        config_path,
                        evaluator,
                        is_param_search,
                        is_tmp,
                        created_at,
                        status,
                        git_hash,
                        model_id,
                        benchmark_id,
                        "desc",
                        sampling_config,
                        log_path
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING task_id
                    """,
                    (
                        config_path,
                        evaluator,
                        bool(is_param_search),
                        bool(is_tmp),
                        created_at,
                        _canonical_task_status(status),
                        git_hash,
                        int(model_id),
                        int(benchmark_id),
                        desc,
                        _jsonb_param(sampling_config),
                        log_path,
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to insert task")
        return int(row[0])

    def update_task_status(self, *, task_id: int, status: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE task
                    SET status = %s
                    WHERE task_id = %s
                    """,
                    (_canonical_task_status(status), int(task_id)),
                )

    def find_tasks_by_identity(
        self,
        *,
        config_path: str | None,
        evaluator: str,
        git_hash: str,
        model_id: int,
        benchmark_id: int,
        sampling_config: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT task_id, status
                    FROM task
                    WHERE evaluator = %s
                      AND git_hash = %s
                      AND model_id = %s
                      AND benchmark_id = %s
                      AND is_tmp = FALSE
                      AND config_path IS NOT DISTINCT FROM %s
                      AND sampling_config IS NOT DISTINCT FROM %s::jsonb
                    ORDER BY task_id ASC
                    """,
                    (
                        evaluator,
                        git_hash,
                        int(model_id),
                        int(benchmark_id),
                        config_path,
                        _jsonb_param(sampling_config),
                    ),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def task_has_score(self, *, task_id: int) -> bool:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM scores
                    WHERE task_id = %s
                    LIMIT 1
                    """,
                    (int(task_id),),
                )
                row = cur.fetchone()
        return row is not None

    def delete_scores_by_task_id(self, *, task_id: int) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM scores
                    WHERE task_id = %s
                    """,
                    (int(task_id),),
                )

    def count_completions(self, *, task_id: int, status: str | None = None) -> int:
        query = """
            SELECT COUNT(*)
            FROM completions
            WHERE task_id = %s
        """
        params: list[Any] = [int(task_id)]
        if status:
            query += " AND status = %s"
            params.append(_canonical_completion_status(status))
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def fetch_completions(
        self,
        *,
        task_id: int,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                b.benchmark_name AS benchmark_name,
                b.benchmark_split AS benchmark_split,
                c.sample_index AS sample_index,
                c.avg_repeat_index AS repeat_index,
                c.pass_index AS pass_index,
                c.context AS context
            FROM completions c
            JOIN task t ON t.task_id = c.task_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE c.task_id = %s
        """
        params: list[Any] = [int(task_id)]
        if status:
            query += " AND c.status = %s"
            params.append(_canonical_completion_status(status))
        query += " ORDER BY c.sample_index ASC, c.avg_repeat_index ASC, c.pass_index ASC"
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_completion_keys(
        self,
        *,
        task_id: int,
        status: str | None = None,
    ) -> list[tuple[int, int, int]]:
        query = """
            SELECT sample_index, avg_repeat_index, pass_index
            FROM completions
            WHERE task_id = %s
        """
        params: list[Any] = [int(task_id)]
        if status:
            query += " AND status = %s"
            params.append(_canonical_completion_status(status))
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]

    def fetch_completion_id_map(
        self,
        *,
        task_id: int,
        status: str | None = None,
    ) -> dict[tuple[int, int, int], int]:
        query = """
            SELECT completions_id, sample_index, avg_repeat_index, pass_index
            FROM completions
            WHERE task_id = %s
        """
        params: list[Any] = [int(task_id)]
        if status:
            query += " AND status = %s"
            params.append(_canonical_completion_status(status))
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        mapping: dict[tuple[int, int, int], int] = {}
        for completions_id, sample_index, repeat_index, pass_index in rows:
            mapping[(int(sample_index), int(repeat_index), int(pass_index))] = int(completions_id)
        return mapping

    def fetch_existing_eval_completion_ids(self, *, task_id: int) -> set[int]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.completions_id
                    FROM eval e
                    JOIN completions c ON c.completions_id = e.completions_id
                    WHERE c.task_id = %s
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return {int(row[0]) for row in rows}

    def insert_completion(
        self,
        *,
        task_id: int,
        payload: dict[str, Any],
        context: dict[str, Any] | None,
        created_at: datetime,
        status: str,
    ) -> int:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        pass_index = strict_nonneg_int(payload.get("pass_index", 0), "pass_index")
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO completions (
                        task_id,
                        context,
                        sample_index,
                        avg_repeat_index,
                        pass_index,
                        created_at,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id, sample_index, avg_repeat_index, pass_index) DO UPDATE
                    SET context = EXCLUDED.context,
                        created_at = EXCLUDED.created_at,
                        status = EXCLUDED.status
                    RETURNING completions_id
                    """,
                    (
                        int(task_id),
                        _jsonb_param(context or {}),
                        int(sample_index),
                        int(repeat_index),
                        int(pass_index),
                        created_at,
                        _canonical_completion_status(status),
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to upsert completion")
        return int(row[0])

    def insert_eval(
        self,
        *,
        completions_id: int,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval (
                        completions_id,
                        answer,
                        ref_answer,
                        is_passed,
                        fail_reason,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (completions_id) DO UPDATE
                    SET answer = EXCLUDED.answer,
                        ref_answer = EXCLUDED.ref_answer,
                        is_passed = EXCLUDED.is_passed,
                        fail_reason = EXCLUDED.fail_reason,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        int(completions_id),
                        str(payload.get("answer") or ""),
                        str(payload.get("ref_answer") or ""),
                        bool(payload.get("is_passed", False)),
                        str(payload.get("fail_reason") or ""),
                        created_at,
                    ),
                )

    def insert_score(self, *, task_id: int, payload: dict[str, Any]) -> None:
        created_at = payload.get("created_at")
        if not created_at:
            created_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scores (
                        task_id,
                        cot_mode,
                        metrics,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE
                    SET cot_mode = EXCLUDED.cot_mode,
                        metrics = EXCLUDED.metrics,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        int(task_id),
                        _canonical_score_cot_mode(payload),
                        _jsonb_param(metrics),
                        created_at,
                    ),
                )

    def fetch_existing_checker_completion_ids(self, *, task_id: int) -> set[int]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT k.completions_id
                    FROM checker k
                    JOIN completions c ON c.completions_id = k.completions_id
                    WHERE c.task_id = %s
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return {int(row[0]) for row in rows}

    def insert_checker(
        self,
        *,
        completions_id: int,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO checker (
                        completions_id,
                        answer_correct,
                        instruction_following_error,
                        world_knowledge_error,
                        math_error,
                        reasoning_logic_error,
                        thought_contains_correct_answer,
                        needs_human_review,
                        reason,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (completions_id) DO UPDATE
                    SET answer_correct = EXCLUDED.answer_correct,
                        instruction_following_error = EXCLUDED.instruction_following_error,
                        world_knowledge_error = EXCLUDED.world_knowledge_error,
                        math_error = EXCLUDED.math_error,
                        reasoning_logic_error = EXCLUDED.reasoning_logic_error,
                        thought_contains_correct_answer = EXCLUDED.thought_contains_correct_answer,
                        needs_human_review = EXCLUDED.needs_human_review,
                        reason = EXCLUDED.reason,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        int(completions_id),
                        bool(payload.get("answer_correct", False)),
                        bool(payload.get("instruction_following_error", False)),
                        bool(payload.get("world_knowledge_error", False)),
                        bool(payload.get("math_error", False)),
                        bool(payload.get("reasoning_logic_error", False)),
                        bool(payload.get("thought_contains_correct_answer", False)),
                        bool(payload.get("needs_human_review", False)),
                        str(payload.get("reason") or ""),
                        created_at,
                    ),
                )

    def fetch_latest_scores(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT
                            s.task_id AS task_id,
                            CASE WHEN s.cot_mode = 'NoCoT' THEN FALSE ELSE TRUE END AS cot,
                            s.cot_mode AS cot_mode,
                            s.metrics AS metrics,
                            s.created_at AS created_at,
                            t.is_param_search AS is_param_search,
                            t.model_id AS model_id,
                            t.benchmark_id AS benchmark_id,
                            t.evaluator AS task,
                            t.sampling_config AS sampling_config,
                            ROW_NUMBER() OVER (
                                PARTITION BY t.model_id, t.benchmark_id, t.evaluator, t.sampling_config
                                ORDER BY s.created_at DESC
                            ) AS rn
                        FROM scores s
                        JOIN task t ON t.task_id = s.task_id
                    )
                    SELECT
                        r.task_id AS task_id,
                        r.cot AS cot,
                        r.cot_mode AS cot_mode,
                        r.metrics AS metrics,
                        r.created_at AS created_at,
                        r.is_param_search AS is_param_search,
                        m.model_name AS model,
                        CASE
                            WHEN b.benchmark_split <> '' THEN CONCAT(b.benchmark_name, '_', b.benchmark_split)
                            ELSE b.benchmark_name
                        END AS dataset,
                        NULL::INT AS samples,
                        NULL::INT AS problems,
                        r.task AS task,
                        r.sampling_config AS sampling_config
                    FROM ranked r
                    JOIN task t ON t.task_id = r.task_id
                    JOIN model m ON m.model_id = t.model_id
                    JOIN benchmark b ON b.benchmark_id = t.benchmark_id
                    WHERE r.rn = 1
                      AND t.is_param_search = FALSE
                      AND t.is_tmp = FALSE
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_latest_scores_for_space(self, *, include_param_search: bool) -> list[dict[str, Any]]:
        query = """
            WITH ranked AS (
                SELECT
                    s.score_id AS score_id,
                    s.task_id AS task_id,
                    CASE WHEN s.cot_mode = 'NoCoT' THEN FALSE ELSE TRUE END AS cot,
                    s.cot_mode AS cot_mode,
                    s.metrics AS metrics,
                    s.created_at AS created_at,
                    t.is_param_search AS is_param_search,
                    t.evaluator AS task,
                    t.sampling_config AS sampling_config,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.model_id, t.benchmark_id, t.evaluator, t.sampling_config
                        ORDER BY s.created_at DESC, s.score_id DESC
                    ) AS rn
                FROM scores s
                JOIN task t ON t.task_id = s.task_id
            )
            SELECT
                r.task_id AS task_id,
                r.cot AS cot,
                r.cot_mode AS cot_mode,
                r.metrics AS metrics,
                r.created_at AS created_at,
                r.is_param_search AS is_param_search,
                m.model_name AS model,
                CASE
                    WHEN b.benchmark_split <> '' THEN CONCAT(b.benchmark_name, '_', b.benchmark_split)
                    ELSE b.benchmark_name
                END AS dataset,
                b.num_samples AS samples,
                b.num_samples AS problems,
                r.task AS task,
                NULL::JSONB AS task_details,
                r.sampling_config AS sampling_config,
                t.log_path AS log_path
            FROM ranked r
            JOIN task t ON t.task_id = r.task_id
            JOIN model m ON m.model_id = t.model_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            WHERE r.rn = 1
              AND t.is_tmp = FALSE
        """
        if not include_param_search:
            query += " AND t.is_param_search = FALSE"
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_scores_by_benchmark(
        self,
        *,
        benchmark_name: str,
        benchmark_split: str,
        model_name: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        s.task_id AS task_id,
                        CASE WHEN s.cot_mode = 'NoCoT' THEN FALSE ELSE TRUE END AS cot,
                        s.cot_mode AS cot_mode,
                        s.metrics AS metrics,
                        s.created_at AS created_at,
                        t.is_param_search AS is_param_search,
                        m.model_name AS model,
                        CASE
                            WHEN b.benchmark_split <> '' THEN CONCAT(b.benchmark_name, '_', b.benchmark_split)
                            ELSE b.benchmark_name
                        END AS dataset,
                        NULL::INT AS samples,
                        NULL::INT AS problems
                    FROM scores s
                    JOIN task t ON t.task_id = s.task_id
                    JOIN model m ON m.model_id = t.model_id
                    JOIN benchmark b ON b.benchmark_id = t.benchmark_id
                    WHERE b.benchmark_name = %s
                      AND b.benchmark_split = %s
                      AND m.model_name = %s
                      AND t.is_param_search = %s
                      AND t.is_tmp = FALSE
                    ORDER BY s.created_at DESC
                    """,
                    (
                        benchmark_name,
                        benchmark_split,
                        model_name,
                        bool(is_param_search),
                    ),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_score_by_task(self, *, task_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        s.task_id AS task_id,
                        CASE WHEN s.cot_mode = 'NoCoT' THEN FALSE ELSE TRUE END AS cot,
                        s.cot_mode AS cot_mode,
                        s.metrics AS metrics,
                        s.created_at AS created_at,
                        m.model_name AS model,
                        CASE
                            WHEN b.benchmark_split <> '' THEN CONCAT(b.benchmark_name, '_', b.benchmark_split)
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
                    (int(task_id),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def fetch_task(self, *, task_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM task WHERE task_id = %s",
                    (int(task_id),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def fetch_model(self, *, model_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM model WHERE model_id = %s",
                    (int(model_id),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def fetch_benchmark(self, *, benchmark_id: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM benchmark WHERE benchmark_id = %s",
                    (int(benchmark_id),),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def fetch_completions_rows(self, *, task_id: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM completions
                    WHERE task_id = %s
                    ORDER BY completions_id ASC
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_eval_rows(self, *, task_id: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT e.*
                    FROM eval e
                    JOIN completions c ON c.completions_id = e.completions_id
                    WHERE c.task_id = %s
                    ORDER BY e.eval_id ASC
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_eval_with_completions_by_task(
        self,
        *,
        task_id: int,
        only_wrong: bool,
        limit: int | None = None,
        offset: int | None = None,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        select_context = ", c.context AS context" if include_context else ""
        query = f"""
            SELECT
                c.sample_index AS sample_index,
                c.avg_repeat_index AS repeat_index,
                c.pass_index AS pass_index,
                e.is_passed AS is_passed,
                e.answer AS answer,
                e.ref_answer AS ref_answer,
                e.fail_reason AS fail_reason,
                LEFT(c.context::TEXT, 80) AS context_preview
                {select_context}
            FROM completions c
            JOIN eval e ON e.completions_id = c.completions_id
            WHERE c.task_id = %s
        """
        params: list[Any] = [int(task_id)]
        if only_wrong:
            query += " AND e.is_passed = FALSE"
        query += " ORDER BY c.sample_index ASC, c.avg_repeat_index ASC, c.pass_index ASC, e.eval_id ASC"
        if offset and offset > 0:
            query += " OFFSET %s"
            params.append(int(offset))
        if limit is not None and limit > 0:
            query += " LIMIT %s"
            params.append(int(limit))
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_eval_context_by_task_attempt(
        self,
        *,
        task_id: int,
        sample_index: int,
        repeat_index: int,
        pass_index: int,
    ) -> Any | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.context
                    FROM completions c
                    JOIN eval e ON e.completions_id = c.completions_id
                    WHERE c.task_id = %s
                      AND c.sample_index = %s
                      AND c.avg_repeat_index = %s
                      AND c.pass_index = %s
                    ORDER BY e.eval_id DESC
                    LIMIT 1
                    """,
                    (
                        int(task_id),
                        int(sample_index),
                        int(repeat_index),
                        int(pass_index),
                    ),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def fetch_scores_rows(self, *, task_id: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM scores
                    WHERE task_id = %s
                    ORDER BY created_at DESC
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_checker_rows(self, *, task_id: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT k.*
                    FROM checker k
                    JOIN completions c ON c.completions_id = k.completions_id
                    WHERE c.task_id = %s
                    ORDER BY k.checker_id ASC
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_checker_keys(self, *, task_id: int) -> set[tuple[int, int, int]]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.sample_index, c.avg_repeat_index, c.pass_index
                    FROM completions c
                    JOIN checker k ON k.completions_id = c.completions_id
                    WHERE c.task_id = %s
                    """,
                    (int(task_id),),
                )
                rows = cur.fetchall()
        return {(int(row[0]), int(row[1]), int(row[2])) for row in rows}


__all__ = ["SqlEvalDbRepository"]
