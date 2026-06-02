from __future__ import annotations

from src.db.eval_db_service import EvalDbService


def test_eval_ingest_dedupes_by_completion_and_eval_group(monkeypatch) -> None:
    service = object.__new__(EvalDbService)
    repo = _FakeRepo()
    service._repo = repo
    monkeypatch.setattr("src.db.eval_db_service.get_session", lambda: _FakeSession())

    inserted = service.ingest_eval_payloads(
        task_id="123",
        payloads=[
            _payload("strategy_a"),
            _payload("strategy_b"),
            _payload("strategy_a"),
        ],
    )

    assert inserted == 2
    assert repo.inserted == [(10, "strategy_a"), (10, "strategy_b")]


def _payload(eval_group: str) -> dict:
    return {
        "benchmark_name": "free",
        "dataset_split": "test",
        "sample_index": 0,
        "repeat_index": 0,
        "context": "p",
        "answer": "a",
        "ref_answer": "r",
        "is_passed": False,
        "fail_reason": "incorrect",
        "eval_group": eval_group,
    }


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeRepo:
    def __init__(self) -> None:
        self.known: set[tuple[int, str]] = set()
        self.inserted: list[tuple[int, str]] = []

    def fetch_completion_id_map(self, _session, *, task_id: int):
        assert task_id == 123
        return {(0, 0): 10}

    def fetch_existing_eval_keys(self, _session, *, task_id: int):
        assert task_id == 123
        return set(self.known)

    def insert_eval(self, _session, *, completions_id: int, payload: dict, created_at):
        _ = created_at
        eval_group = str(payload.get("eval_group") or "strategy_a")
        self.known.add((completions_id, eval_group))
        self.inserted.append((completions_id, eval_group))
