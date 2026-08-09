from __future__ import annotations

from dataclasses import dataclass

import pytest

from ops.g1i_strict46.replay_lock import (
    held_replay_advisory_locks,
    replay_advisory_lock_keys,
    try_acquire_replay_advisory_locks,
)


@dataclass
class _Result:
    row: dict[str, bool]

    def fetchone(self) -> dict[str, bool]:
        return self.row


class _Connection:
    def __init__(self, *, unavailable: set[int] | None = None) -> None:
        self.unavailable = unavailable or set()
        self.held: set[int] = set()
        self.calls: list[tuple[str, int]] = []

    def execute(self, query: str, params: tuple[int]) -> _Result:
        key = int(params[0])
        self.calls.append((query, key))
        if "pg_try_advisory_lock" in query:
            acquired = key not in self.unavailable
            if acquired:
                self.held.add(key)
            return _Result({"acquired": acquired})
        if "pg_advisory_unlock" in query:
            self.held.discard(key)
            return _Result({"released": True})
        raise AssertionError(query)


def test_lock_keys_are_stable_shared_source_and_cell_keys() -> None:
    kwargs = {
        "dbname": "strict46",
        "source_task_id": 28642,
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "minerva_math",
        "benchmark_split": "test",
    }
    first = replay_advisory_lock_keys(**kwargs)
    second = replay_advisory_lock_keys(**kwargs)

    assert first == second
    assert len(first) == 2
    assert first == tuple(sorted(first))
    assert all(-(2**63) <= key < 2**63 for key in first)

    same_cell_other_source = replay_advisory_lock_keys(
        **{**kwargs, "source_task_id": 28643}
    )
    assert len(set(first) & set(same_cell_other_source)) == 1


def test_partial_advisory_lock_acquisition_releases_owned_key() -> None:
    keys = (-10, 20)
    connection = _Connection(unavailable={20})

    assert try_acquire_replay_advisory_locks(connection, keys) == ()
    assert connection.held == set()
    assert any("pg_advisory_unlock" in query for query, _key in connection.calls)


def test_lock_context_holds_both_keys_then_releases_on_exit() -> None:
    keys = (-10, 20)
    connection = _Connection()

    with held_replay_advisory_locks(connection, keys) as acquired:
        assert acquired
        assert connection.held == set(keys)

    assert connection.held == set()


def test_acquisition_exception_releases_partial_lock() -> None:
    class FailingConnection(_Connection):
        def execute(self, query: str, params: tuple[int]) -> _Result:
            if "pg_try_advisory_lock" in query and int(params[0]) == 20:
                raise RuntimeError("database connection interrupted")
            return super().execute(query, params)

    connection = FailingConnection()
    with pytest.raises(RuntimeError, match="database connection interrupted"):
        try_acquire_replay_advisory_locks(connection, (-10, 20))

    assert connection.held == set()
