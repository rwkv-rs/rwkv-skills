from __future__ import annotations

from src.db.eval_db_service import EvalDbService


def test_eval_ingest_dedupes_by_completion() -> None:
    service = object.__new__(EvalDbService)
    repo = _FakeRepo()
    service._repo = repo

    inserted = service.ingest_eval_payloads(
        task_id="123",
        payloads=[
            _payload("first"),
            _payload("second"),
            _payload("third"),
        ],
    )

    assert inserted == 1
    assert repo.inserted == [10]


def test_completion_ingest_uses_batch_repo_method() -> None:
    service = object.__new__(EvalDbService)
    repo = _FakeRepo()
    service._repo = repo

    inserted = service.insert_completion_payloads_batch(
        task_id="123",
        payloads=[
            _completion_payload(),
            {**_completion_payload(), "sample_index": 1},
            {**_completion_payload(), "_stage": "draft", "sample_index": 2},
        ],
    )

    assert inserted == 2
    assert len(repo.completion_batches) == 1
    assert [row[0]["sample_index"] for row in repo.completion_batches[0]] == [0, 1]


def test_checker_ingest_uses_batch_repo_method() -> None:
    service = object.__new__(EvalDbService)
    repo = _FakeRepo()
    service._repo = repo

    inserted = service.ingest_checker_payloads(
        task_id="123",
        payloads=[
            _checker_payload(0),
            _checker_payload(1),
            _checker_payload(2),
        ],
    )

    assert inserted == 2
    assert repo.checker_batches == [[10, 11]]


def test_eval_strategy_groups_use_separate_tasks(monkeypatch) -> None:
    service = object.__new__(EvalDbService)
    service.calls = []  # type: ignore[attr-defined]

    def fake_ingest_eval_payloads(*, payloads, task_id):
        service.calls.append(("eval", task_id, [item["answer"] for item in payloads]))  # type: ignore[attr-defined]
        return len(payloads)

    def fake_create_eval_strategy_task(*, parent_task_id, strategy):
        service.calls.append(("create", parent_task_id, strategy))  # type: ignore[attr-defined]
        return {"strategy_b": 124, "strategy_c": 125}[strategy]

    def fake_insert_completion_payloads_batch(*, payloads, task_id):
        service.calls.append(("completion", task_id, len(payloads)))  # type: ignore[attr-defined]
        return len(payloads)

    def fake_update_task_status(*, task_id, status):
        service.calls.append(("status", task_id, status))  # type: ignore[attr-defined]

    monkeypatch.setattr(service, "ingest_eval_payloads", fake_ingest_eval_payloads)
    monkeypatch.setattr(service, "create_eval_strategy_task", fake_create_eval_strategy_task)
    monkeypatch.setattr(service, "insert_completion_payloads_batch", fake_insert_completion_payloads_batch)
    monkeypatch.setattr(service, "update_task_status", fake_update_task_status)

    task_ids = service.ingest_eval_payload_groups(
        task_id="123",
        completion_payloads=[_completion_payload()],
        payloads_by_group={
            "strategy_a": [_payload("a")],
            "strategy_b": [_payload("b")],
            "strategy_c": [_payload("c")],
        },
        primary_group="strategy_a",
    )

    assert task_ids == {"strategy_a": 123, "strategy_b": 124, "strategy_c": 125}
    assert service.calls == [  # type: ignore[attr-defined]
        ("eval", "123", ["a"]),
        ("create", 123, "strategy_b"),
        ("completion", "124", 1),
        ("eval", "124", ["b"]),
        ("status", "124", "completed"),
        ("create", 123, "strategy_c"),
        ("completion", "125", 1),
        ("eval", "125", ["c"]),
        ("status", "125", "completed"),
    ]


def _payload(answer: str) -> dict:
    return {
        "benchmark_name": "free",
        "dataset_split": "test",
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "context": "p",
        "answer": answer,
        "ref_answer": "r",
        "is_passed": False,
        "fail_reason": "incorrect",
    }


def _completion_payload() -> dict:
    return {
        "benchmark_name": "free",
        "dataset_split": "test",
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "prompt1": "p",
        "completion1": "c",
        "stop_reason1": "stop_token",
        "_stage": "answer",
    }


def _checker_payload(sample_index: int) -> dict:
    return {
        "sample_index": sample_index,
        "repeat_index": 0,
        "pass_index": 0,
        "answer_correct": sample_index == 0,
        "instruction_following_error": False,
        "world_knowledge_error": False,
        "math_error": False,
        "reasoning_logic_error": False,
        "thought_contains_correct_answer": False,
        "reason": "",
    }


class _FakeRepo:
    def __init__(self) -> None:
        self.known: set[int] = set()
        self.inserted: list[int] = []
        self.completion_batches: list[list[tuple[dict, dict]]] = []
        self.checker_batches: list[list[int]] = []

    def fetch_completion_id_map(self, *, task_id: int, status: str | None = None):
        assert task_id == 123
        assert status == "Completed"
        return {(0, 0, 0): 10, (1, 0, 0): 11}

    def fetch_existing_eval_completion_ids(self, *, task_id: int):
        assert task_id == 123
        return set(self.known)

    def fetch_existing_checker_completion_ids(self, *, task_id: int):
        assert task_id == 123
        return set()

    def insert_completions_batch(self, *, task_id: int, rows: list[tuple[dict, dict]], created_at, status: str):
        assert task_id == 123
        assert status == "Completed"
        _ = created_at
        self.completion_batches.append(list(rows))
        return len(rows)

    def insert_eval(self, *, completions_id: int, payload: dict, created_at):
        _ = created_at
        _ = payload
        self.known.add(completions_id)
        self.inserted.append(completions_id)

    def insert_eval_batch(self, *, rows: list[tuple[int, dict]], created_at):
        _ = created_at
        for completions_id, _payload in rows:
            self.known.add(completions_id)
            self.inserted.append(completions_id)
        return len(rows)

    def insert_checker_batch(self, *, rows: list[tuple[int, dict]], created_at):
        _ = created_at
        self.checker_batches.append([completions_id for completions_id, _payload in rows])
        return len(rows)
