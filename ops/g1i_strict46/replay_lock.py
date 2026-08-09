"""Shared PostgreSQL advisory locks for append-only completion replays.

Every replay monitor must take the same source and logical-cell locks before
it rechecks database state and launches a replay subprocess.  Session-level
locks intentionally survive transaction boundaries and are released when the
dedicated lock-holding connection closes, including after an exception.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Iterator, Sequence

import psycopg


LOCK_NAMESPACE = "rwkv-skills:completion-replay:v1"


def _signed_lock_key(material: str) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=True)


def replay_advisory_lock_keys(
    *,
    dbname: str,
    source_task_id: int,
    model_name: str,
    benchmark_name: str,
    benchmark_split: str,
) -> tuple[int, ...]:
    """Return stable, globally ordered source and cell lock keys."""

    source = _signed_lock_key(
        f"{LOCK_NAMESPACE}:db={dbname}:source={int(source_task_id)}"
    )
    cell = _signed_lock_key(
        f"{LOCK_NAMESPACE}:db={dbname}:cell="
        f"{model_name}::{benchmark_name}::{benchmark_split}"
    )
    return tuple(sorted({source, cell}))


def _row_scalar(row: object, key: str) -> bool:
    if isinstance(row, dict):
        return bool(row.get(key))
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        return bool(row[0]) if row else False
    return False


def try_acquire_replay_advisory_locks(
    connection: psycopg.Connection[Any], keys: Sequence[int]
) -> tuple[int, ...]:
    """Acquire every key or release the partial set and return ``()``."""

    acquired: list[int] = []
    try:
        for key in sorted(set(int(item) for item in keys)):
            row = connection.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired",
                (key,),
            ).fetchone()
            if row is not None and _row_scalar(row, "acquired"):
                acquired.append(key)
                continue
            release_replay_advisory_locks(connection, acquired)
            return ()
    except Exception:
        release_replay_advisory_locks(connection, acquired)
        raise
    return tuple(acquired)


def release_replay_advisory_locks(
    connection: psycopg.Connection[Any], keys: Sequence[int]
) -> None:
    for key in reversed(tuple(keys)):
        connection.execute("SELECT pg_advisory_unlock(%s)", (int(key),))


@contextmanager
def held_replay_advisory_locks(
    connection: psycopg.Connection[Any], keys: Sequence[int]
) -> Iterator[bool]:
    """Yield whether all locks were acquired and release only owned keys."""

    acquired = try_acquire_replay_advisory_locks(connection, keys)
    try:
        yield bool(acquired)
    finally:
        if acquired:
            release_replay_advisory_locks(connection, acquired)


__all__ = [
    "LOCK_NAMESPACE",
    "held_replay_advisory_locks",
    "release_replay_advisory_locks",
    "replay_advisory_lock_keys",
    "try_acquire_replay_advisory_locks",
]
