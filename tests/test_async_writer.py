from __future__ import annotations

from typing import Any

from src.db.async_writer import CompletionWriteWorker


def test_completion_writer_flushes_payload_batches() -> None:
    service = _FakeService()
    writer = CompletionWriteWorker(
        service=service,
        task_id="123",
        flush_rows=2,
        flush_interval_s=0.01,
    )
    try:
        writer.enqueue({"sample_index": 0})
        writer.enqueue({"sample_index": 1})
        writer.enqueue({"sample_index": 2})
        assert writer.drain(timeout_s=2.0)
    finally:
        writer.close(timeout_s=2.0)

    assert service.batches == [[0, 1], [2]]


class _FakeService:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def insert_completion_payloads_batch(self, *, payloads: list[dict[str, Any]], task_id: str) -> int:
        assert task_id == "123"
        self.batches.append([int(payload["sample_index"]) for payload in payloads])
        return len(payloads)
