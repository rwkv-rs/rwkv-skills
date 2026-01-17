from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

import psycopg

from src.eval.scheduler.config import DBConfig, DEFAULT_DB_CONFIG


@dataclass(slots=True)
class MigrationConfig:
    drop_legacy: bool


def _conn_str(config: DBConfig) -> str:
    return (
        f"host={config.host} port={config.port} "
        f"dbname={config.dbname} user={config.user} password={config.password}"
    )


def _ensure_subject_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_subject (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_slug VARCHAR(255) NOT NULL,
            domain VARCHAR(128),
            dataset_version VARCHAR(128),
            dataset_meta JSONB,
            model_slug VARCHAR(255) NOT NULL,
            model_name VARCHAR(255),
            model_revision VARCHAR(255),
            provider VARCHAR(128),
            model_meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_subject_key "
        "ON eval_subject(dataset_slug, model_slug, COALESCE(model_revision, ''));"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_subject_dataset ON eval_subject(dataset_slug);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_subject_model ON eval_subject(model_slug);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_eval_subject_domain ON eval_subject(domain);"
    )


def _ensure_subject_columns(conn: psycopg.Connection) -> None:
    conn.execute("ALTER TABLE eval_task ADD COLUMN IF NOT EXISTS subject_id UUID;")
    conn.execute("ALTER TABLE eval_split ADD COLUMN IF NOT EXISTS subject_id UUID;")
    conn.execute("ALTER TABLE eval_sample ADD COLUMN IF NOT EXISTS subject_id UUID;")


def _seed_subjects(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        INSERT INTO eval_subject (
            dataset_slug, domain, dataset_version, dataset_meta,
            model_slug, model_name, model_revision, provider, model_meta
        )
        SELECT DISTINCT
            d.dataset_slug,
            d.domain,
            d.dataset_version,
            d.meta,
            m.model_slug,
            m.model_name,
            m.model_revision,
            m.provider,
            m.meta
        FROM eval_task t
        JOIN eval_dataset d ON d.id = t.dataset_id
        JOIN eval_model m ON m.id = t.model_id
        ON CONFLICT (dataset_slug, model_slug, COALESCE(model_revision, ''))
        DO UPDATE SET
            domain = EXCLUDED.domain,
            dataset_version = EXCLUDED.dataset_version,
            dataset_meta = EXCLUDED.dataset_meta,
            model_name = EXCLUDED.model_name,
            provider = EXCLUDED.provider,
            model_meta = EXCLUDED.model_meta,
            updated_at = CURRENT_TIMESTAMP
        """
    )


def _backfill_task_subjects(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        UPDATE eval_task t
        SET subject_id = s.id
        FROM eval_dataset d
        JOIN eval_model m ON m.id = t.model_id
        JOIN eval_subject s
          ON s.dataset_slug = d.dataset_slug
         AND s.model_slug = m.model_slug
         AND COALESCE(s.model_revision, '') = COALESCE(m.model_revision, '')
        WHERE t.dataset_id = d.id
        """
    )


def _backfill_splits_and_samples(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        INSERT INTO eval_split (subject_id, split_name)
        SELECT s.id, sp.split_name
        FROM eval_subject s
        JOIN eval_dataset d ON d.dataset_slug = s.dataset_slug
        JOIN eval_split sp ON sp.dataset_id = d.id
        ON CONFLICT (subject_id, split_name) DO NOTHING
        """
    )
    conn.execute(
        """
        INSERT INTO eval_sample (
            subject_id, split_id, sample_index, question, reference_answer, meta
        )
        SELECT
            s.id,
            sp_new.id,
            sm.sample_index,
            sm.question,
            sm.reference_answer,
            sm.meta
        FROM eval_subject s
        JOIN eval_dataset d ON d.dataset_slug = s.dataset_slug
        JOIN eval_split sp_old ON sp_old.dataset_id = d.id
        JOIN eval_sample sm ON sm.split_id = sp_old.id
        JOIN eval_split sp_new
          ON sp_new.subject_id = s.id
         AND sp_new.split_name = sp_old.split_name
        ON CONFLICT (subject_id, split_id, sample_index)
        DO UPDATE SET
            question = EXCLUDED.question,
            reference_answer = EXCLUDED.reference_answer,
            meta = EXCLUDED.meta,
            updated_at = CURRENT_TIMESTAMP
        """
    )


def _remap_run_samples(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        UPDATE eval_run_sample rs
        SET sample_id = sm_new.id
        FROM eval_run r
        JOIN eval_task t ON t.id = r.task_id
        JOIN eval_sample sm_old ON sm_old.id = rs.sample_id
        JOIN eval_split sp_old ON sp_old.id = sm_old.split_id
        JOIN eval_split sp_new
          ON sp_new.subject_id = t.subject_id
         AND sp_new.split_name = sp_old.split_name
        JOIN eval_sample sm_new
          ON sm_new.subject_id = t.subject_id
         AND sm_new.split_id = sp_new.id
         AND sm_new.sample_index = sm_old.sample_index
        WHERE rs.run_id = r.id
        """
    )


def _drop_legacy(conn: psycopg.Connection) -> None:
    conn.execute("ALTER TABLE eval_task DROP COLUMN IF EXISTS dataset_id;")
    conn.execute("ALTER TABLE eval_task DROP COLUMN IF EXISTS model_id;")
    conn.execute("ALTER TABLE eval_split DROP COLUMN IF EXISTS dataset_id;")
    conn.execute("ALTER TABLE eval_sample DROP COLUMN IF EXISTS dataset_id;")
    conn.execute("DROP TABLE IF EXISTS eval_model CASCADE;")
    conn.execute("DROP TABLE IF EXISTS eval_dataset CASCADE;")


def migrate(config: DBConfig, options: MigrationConfig) -> None:
    if not config.enabled:
        raise RuntimeError("RWKV_DB_ENABLED is false; aborting migration.")
    with psycopg.connect(_conn_str(config)) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        _ensure_subject_table(conn)
        _ensure_subject_columns(conn)
        _seed_subjects(conn)
        _backfill_task_subjects(conn)
        _backfill_splits_and_samples(conn)
        _remap_run_samples(conn)
        if options.drop_legacy:
            _drop_legacy(conn)
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate eval_dataset/eval_model into eval_subject")
    parser.add_argument("--drop-legacy", action="store_true", help="Drop legacy tables/columns after migration")
    args = parser.parse_args()
    migrate(DEFAULT_DB_CONFIG, MigrationConfig(drop_legacy=args.drop_legacy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
