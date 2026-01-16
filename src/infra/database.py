from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from ..eval.scheduler.config import DBConfig

class DatabaseManager:
    _instance = None
    _pool = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def initialize(self, config: DBConfig) -> None:
        if not config.enabled:
            return
            
        # 1. Connect to default 'postgres' db to check/create target db
        try:
            admin_conn_str = (
                f"host={config.host} port={config.port} "
                f"dbname=postgres user={config.user} password={config.password}"
            )
            # autocommit=True needed for CREATE DATABASE
            with psycopg.connect(admin_conn_str, autocommit=True) as conn:
                # Check if db exists
                res = conn.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", 
                    (config.dbname,)
                ).fetchone()
                if not res:
                    print(f"📦 Database '{config.dbname}' does not exist. Creating...")
                    conn.execute(f'CREATE DATABASE "{config.dbname}"')
        except Exception as e:
            # If we can't connect to postgres db or create db, warn but try proceeding 
            print(f"⚠️  Database creation check failed (proceeding anyway): {e}")

        # 2. Connect to target db
        conn_str = (
            f"host={config.host} port={config.port} "
            f"dbname={config.dbname} user={config.user} password={config.password}"
        )
        
        self._pool = ConnectionPool(conn_str, min_size=1, max_size=10)
        self._pool.wait()
        self._init_schema()

    def _init_schema(self) -> None:
        if self._pool is None:
            return
            
        with self.get_connection() as conn:
            # Enable UUID generator (pgcrypto or uuid-ossp). pgcrypto is preferred.
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            except Exception:
                # Fallback: if pgcrypto unavailable, try uuid-ossp
                conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

            # Create tables if missing (no destructive operations).

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_dataset (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    dataset_slug VARCHAR(255) NOT NULL,
                    domain VARCHAR(128),
                    dataset_version VARCHAR(128),
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_dataset_slug ON eval_dataset(dataset_slug);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_dataset_domain ON eval_dataset(domain);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_split (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    dataset_id UUID NOT NULL REFERENCES eval_dataset(id) ON DELETE CASCADE,
                    split_name VARCHAR(128) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_split_dataset_name ON eval_split(dataset_id, split_name);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_split_dataset ON eval_split(dataset_id);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_sample (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    dataset_id UUID NOT NULL REFERENCES eval_dataset(id) ON DELETE CASCADE,
                    split_id UUID NOT NULL REFERENCES eval_split(id) ON DELETE CASCADE,
                    sample_index INT NOT NULL,
                    question TEXT,
                    reference_answer TEXT,
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_sample_lookup ON eval_sample(dataset_id, split_id, sample_index);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_sample_lookup ON eval_sample(dataset_id, split_id, sample_index);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_model (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_slug VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255),
                    model_revision VARCHAR(255),
                    provider VARCHAR(128),
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_model_slug ON eval_model(model_slug);"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_model_slug_revision ON eval_model(model_slug, COALESCE(model_revision, ''));"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_task (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id VARCHAR(255) NOT NULL,
                    dataset_id UUID NOT NULL REFERENCES eval_dataset(id) ON DELETE CASCADE,
                    model_id UUID NOT NULL REFERENCES eval_model(id) ON DELETE CASCADE,
                    task_tag VARCHAR(255),
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_task_id ON eval_task(task_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_task_dataset ON eval_task(dataset_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_task_model ON eval_task(model_id);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_run (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id UUID NOT NULL REFERENCES eval_task(id) ON DELETE CASCADE,
                    run_tag VARCHAR(255),
                    sampling_config JSONB,
                    runtime_config JSONB,
                    code_version VARCHAR(255),
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    error_msg TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_run_task_status ON eval_run(task_id, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_run_run_tag ON eval_run(run_tag);"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_task_tag ON eval_run(task_id, run_tag);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_run_sample (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_id UUID NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
                    sample_id UUID NOT NULL REFERENCES eval_sample(id) ON DELETE CASCADE,
                    repeat_index INT NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    current_stage VARCHAR(32),
                    latest_attempt_index INT DEFAULT 0,
                    error_msg TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_run_sample_unique ON eval_run_sample(run_id, sample_id, repeat_index);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_run_sample_status ON eval_run_sample(run_id, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_run_sample_sample ON eval_run_sample(sample_id);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_attempt (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_sample_id UUID NOT NULL REFERENCES eval_run_sample(id) ON DELETE CASCADE,
                    attempt_index INT NOT NULL,
                    worker_id VARCHAR(255),
                    shard_id INT,
                    shard_count INT,
                    seed BIGINT,
                    status VARCHAR(32) NOT NULL DEFAULT 'running',
                    error_msg TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_attempt_unique ON eval_attempt(run_sample_id, attempt_index);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_attempt_rs ON eval_attempt(run_sample_id, attempt_index DESC);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_stage_output (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
                    stage VARCHAR(32) NOT NULL,
                    seq INT NOT NULL DEFAULT 0,
                    prompt TEXT,
                    completion TEXT,
                    finish_reason VARCHAR(32),
                    provider_request_id VARCHAR(255),
                    raw_response JSONB,
                    token_count_prompt INT,
                    token_count_response INT,
                    latency_ms INT,
                    cost_usd NUMERIC,
                    is_partial BOOLEAN NOT NULL DEFAULT FALSE,
                    is_final BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_stage_latest ON eval_stage_output(attempt_id, stage, created_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_stage_finish ON eval_stage_output(stage, finish_reason);"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_stage_final ON eval_stage_output(attempt_id, stage) WHERE is_final;"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_cot_checkpoint (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    attempt_id UUID NOT NULL REFERENCES eval_attempt(id) ON DELETE CASCADE,
                    stage VARCHAR(32) NOT NULL,
                    token_offset INT,
                    partial_completion TEXT,
                    kv_cache_ref TEXT,
                    rng_state JSONB,
                    status VARCHAR(32),
                    latest BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_ckpt_latest ON eval_cot_checkpoint(attempt_id, stage) WHERE latest;"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_ckpt_lookup ON eval_cot_checkpoint(attempt_id, stage, created_at DESC);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_metric (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_sample_id UUID NOT NULL REFERENCES eval_run_sample(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    value_num DOUBLE PRECISION,
                    value_text TEXT,
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_metric_name ON eval_metric(name);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_metric_rs ON eval_metric(run_sample_id, name);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_run_event (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    run_id UUID REFERENCES eval_run(id) ON DELETE CASCADE,
                    run_sample_id UUID REFERENCES eval_run_sample(id) ON DELETE CASCADE,
                    event_type VARCHAR(255) NOT NULL,
                    message TEXT,
                    meta JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (run_id IS NOT NULL OR run_sample_id IS NOT NULL)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_event_run_time ON eval_run_event(run_id, created_at);"
            )

    @contextmanager
    def get_connection(self) -> Iterator[psycopg.Connection]:
        if self._pool is None:
            raise RuntimeError("Database not initialized")
        
        with self._pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            
    def close(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool = None

db_manager = DatabaseManager.instance()
