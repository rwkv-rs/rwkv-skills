from __future__ import annotations

"""Distributed claim/lease coordination using explicit PostgreSQL queries."""

from dataclasses import dataclass
import os
import socket
import uuid
from typing import Any, Mapping, Protocol, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.db.pool import Db, get_db, init_db_pool


def default_scheduler_node_id() -> str:
    configured = str(os.environ.get("RWKV_SCHEDULER_NODE_ID", "") or "").strip()
    if configured:
        return configured
    return socket.gethostname() or "scheduler"


def default_scheduler_owner_id(node_id: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"{node_id}:{os.getpid()}:{suffix}"


@dataclass(frozen=True, slots=True)
class SchedulerLeaseRecord:
    job_id: str
    owner_id: str
    node_id: str
    claimed_at: object
    heartbeat_at: object
    lease_until: object
    lease_meta: Mapping[str, Any] | None = None


class SchedulerLeaseStore(Protocol):
    def ensure_schema(self) -> None: ...

    def claim(
        self,
        *,
        job_id: str,
        owner_id: str,
        node_id: str,
        lease_duration_s: int,
        lease_meta: Mapping[str, Any] | None = None,
    ) -> bool: ...

    def renew(
        self,
        *,
        job_ids: Sequence[str],
        owner_id: str,
        lease_duration_s: int,
    ) -> set[str]: ...

    def release(self, *, job_ids: Sequence[str], owner_id: str) -> int: ...

    def release_all(self, *, owner_id: str) -> int: ...

    def list_active(self) -> list[SchedulerLeaseRecord]: ...


class PgSchedulerLeaseStore:
    def __init__(self, db: Db | None = None) -> None:
        self.db = db or init_db_pool()

    def ensure_schema(self) -> None:
        with self.db.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scheduler_lease (
                        job_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        claimed_at TIMESTAMPTZ NOT NULL,
                        heartbeat_at TIMESTAMPTZ NOT NULL,
                        lease_until TIMESTAMPTZ NOT NULL,
                        lease_meta JSONB
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scheduler_lease_owner ON scheduler_lease(owner_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scheduler_lease_until ON scheduler_lease(lease_until)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_scheduler_lease_node ON scheduler_lease(node_id)"
                )
            conn.commit()

    def claim(
        self,
        *,
        job_id: str,
        owner_id: str,
        node_id: str,
        lease_duration_s: int,
        lease_meta: Mapping[str, Any] | None = None,
    ) -> bool:
        payload = Jsonb(dict(lease_meta)) if lease_meta is not None else None
        with self.db.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scheduler_lease (
                        job_id,
                        owner_id,
                        node_id,
                        claimed_at,
                        heartbeat_at,
                        lease_until,
                        lease_meta
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        NOW(),
                        NOW(),
                        NOW() + make_interval(secs => %s),
                        %s
                    )
                    ON CONFLICT (job_id) DO UPDATE
                    SET owner_id = EXCLUDED.owner_id,
                        node_id = EXCLUDED.node_id,
                        claimed_at = NOW(),
                        heartbeat_at = NOW(),
                        lease_until = NOW() + make_interval(secs => %s),
                        lease_meta = EXCLUDED.lease_meta
                    WHERE scheduler_lease.owner_id = EXCLUDED.owner_id
                       OR scheduler_lease.lease_until <= NOW()
                    RETURNING job_id
                    """,
                    (
                        str(job_id),
                        str(owner_id),
                        str(node_id),
                        int(lease_duration_s),
                        payload,
                        int(lease_duration_s),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return row is not None

    def renew(
        self,
        *,
        job_ids: Sequence[str],
        owner_id: str,
        lease_duration_s: int,
    ) -> set[str]:
        normalized = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not normalized:
            return set()
        with self.db.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE scheduler_lease
                    SET heartbeat_at = NOW(),
                        lease_until = NOW() + make_interval(secs => %s)
                    WHERE owner_id = %s
                      AND job_id = ANY(%s::text[])
                      AND lease_until > NOW()
                    RETURNING job_id
                    """,
                    (
                        int(lease_duration_s),
                        str(owner_id),
                        normalized,
                    ),
                )
                rows = cur.fetchall()
            conn.commit()
        return {str(row[0]) for row in rows}

    def release(self, *, job_ids: Sequence[str], owner_id: str) -> int:
        normalized = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not normalized:
            return 0
        with self.db.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM scheduler_lease
                    WHERE owner_id = %s
                      AND job_id = ANY(%s::text[])
                    """,
                    (str(owner_id), normalized),
                )
                count = cur.rowcount or 0
            conn.commit()
        return int(count)

    def release_all(self, *, owner_id: str) -> int:
        with self.db.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM scheduler_lease WHERE owner_id = %s",
                    (str(owner_id),),
                )
                count = cur.rowcount or 0
            conn.commit()
        return int(count)

    def list_active(self) -> list[SchedulerLeaseRecord]:
        with self.db.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        job_id,
                        owner_id,
                        node_id,
                        claimed_at,
                        heartbeat_at,
                        lease_until,
                        lease_meta
                    FROM scheduler_lease
                    WHERE lease_until > NOW()
                    """
                )
                rows = cur.fetchall()
        return [
            SchedulerLeaseRecord(
                job_id=str(row["job_id"]),
                owner_id=str(row["owner_id"]),
                node_id=str(row["node_id"]),
                claimed_at=row["claimed_at"],
                heartbeat_at=row["heartbeat_at"],
                lease_until=row["lease_until"],
                lease_meta=(row.get("lease_meta") if isinstance(row.get("lease_meta"), dict) else None),
            )
            for row in rows
        ]


class SchedulerLeaseManager:
    def __init__(
        self,
        store: SchedulerLeaseStore | None = None,
        *,
        node_id: str | None = None,
        owner_id: str | None = None,
        lease_duration_s: int = 120,
    ) -> None:
        self.store = store or PgSchedulerLeaseStore(get_db())
        self.node_id = str(node_id or default_scheduler_node_id())
        self.owner_id = str(owner_id or default_scheduler_owner_id(self.node_id))
        self.lease_duration_s = max(5, int(lease_duration_s))
        self.store.ensure_schema()

    def claim(self, job_id: str, *, lease_meta: Mapping[str, Any] | None = None) -> bool:
        return bool(
            self.store.claim(
                job_id=str(job_id),
                owner_id=self.owner_id,
                node_id=self.node_id,
                lease_duration_s=self.lease_duration_s,
                lease_meta=lease_meta,
            )
        )

    def renew(self, job_ids: Sequence[str]) -> set[str]:
        return set(
            self.store.renew(
                job_ids=job_ids,
                owner_id=self.owner_id,
                lease_duration_s=self.lease_duration_s,
            )
        )

    def release(self, job_ids: Sequence[str]) -> int:
        return int(self.store.release(job_ids=job_ids, owner_id=self.owner_id))

    def release_all(self) -> int:
        return int(self.store.release_all(owner_id=self.owner_id))

    def active_foreign_job_ids(self) -> set[str]:
        return {
            lease.job_id
            for lease in self.store.list_active()
            if lease.owner_id != self.owner_id
        }


__all__ = [
    "PgSchedulerLeaseStore",
    "SchedulerLeaseManager",
    "SchedulerLeaseRecord",
    "default_scheduler_node_id",
    "default_scheduler_owner_id",
]
