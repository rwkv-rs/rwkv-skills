from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import src.dashboard.web.api as api_module
from src.dashboard.core.data import ScoreEntry
from src.dashboard.core.domains import DOMAIN_MMLU
from src.dashboard.web.api import create_app
from src.dashboard.web.store import DashboardStore


def _entry() -> ScoreEntry:
    return ScoreEntry(
        task_id=123,
        dataset="mmlu_test",
        model="rwkv7-g1g-1.5b-test",
        metrics={"accuracy": 0.5},
        samples=2,
        problems=2,
        created_at=datetime(2026, 7, 1, 12, 0, 0),
        log_path="",
        cot=False,
        task="multi_choice_plain",
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain=DOMAIN_MMLU,
        extra={"sampling_config": {"avg_k": 1}, "cot_mode": "NoCoT"},
        arch_version="RWKV7",
        data_version="G1G",
        num_params="1_5b",
    )


class FakeDashboardStore(DashboardStore):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def rebuild_score_index_from_db(self, *, include_param_search: bool = False) -> int:
        self.calls.append(f"rebuild:{include_param_search}")
        return 1

    def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append(f"eval_records:{task_id}:{only_wrong}:{limit}:{offset}:{include_preview}")
        records = [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "is_passed": True,
                "answer": "A",
                "ref_answer": "A",
                "fail_reason": "",
                "context_preview": "prompt A",
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "is_passed": False,
                "answer": "B",
                "ref_answer": "C",
                "fail_reason": "wrong",
                "context_preview": "prompt B",
            },
        ]
        return [row for row in records if not only_wrong or not row["is_passed"]][offset : offset + (limit or len(records))]

    def get_eval_context_for_space(
        self,
        *,
        task_id: str,
        sample_index: int,
        repeat_index: int,
        pass_index: int = 0,
    ) -> Any | None:
        self.calls.append(f"eval_context:{task_id}:{sample_index}:{repeat_index}:{pass_index}")
        return {
            "stages": [{"prompt": "What is 1+1?", "completion": "2", "stop_reason": "stop"}],
            "sampling_config": {"answer": {"temperature": 0.2, "stop_tokens": [0]}},
        }

    def list_score_history(self, *, model: str, dataset: str) -> list[dict[str, Any]]:
        self.calls.append(f"history:{model}:{dataset}")
        return [
            {
                "score_id": 1,
                "task_id": 123,
                "cot_mode": "NoCoT",
                "evaluator": "multi_choice_plain",
                "metrics": {"avg@1": 0.5},
                "sampling_config": {"avg_k": 1},
                "created_at": datetime(2026, 7, 1, 12, 0, 0),
                "model": model,
                "dataset": dataset,
            },
            {
                "score_id": 2,
                "task_id": 124,
                "cot_mode": "NoCoT",
                "evaluator": "multi_choice_plain",
                "metrics": {"avg@1": 0.6},
                "sampling_config": {"avg_k": 1},
                "created_at": datetime(2026, 7, 1, 13, 0, 0),
                "model": model,
                "dataset": dataset,
            }
        ]

    def list_score_history_pairs(self) -> list[dict[str, Any]]:
        self.calls.append("history_pairs")
        return [{"model": "rwkv7-g1g-1.5b-test", "dataset": "mmlu_test"}]

    def get_score_history_detail(self, *, task_id: str) -> dict[str, Any] | None:
        self.calls.append(f"history_detail:{task_id}")
        return {
            "score": {
                "model": "rwkv7-g1g-1.5b-test",
                "dataset": "mmlu_test",
                "cot_mode": "NoCoT",
                "metrics": {"avg@1": 0.5},
            },
            "task": {
                "evaluator": "multi_choice_plain",
                "sampling_config": {
                    "avg_k": 1,
                    "sampling_config": {
                        "answer": {
                            "temperature": 0.2,
                            "top_p": 0.95,
                            "max_new_tokens": 128,
                            "stop_tokens": [0],
                        }
                    },
                },
            },
            "context": {
                "stages": [{"prompt": "What is 1+1?", "completion": "2", "stop_reason": "stop"}],
            },
        }


def test_dashboard_api_uses_injected_store(monkeypatch) -> None:
    fake_store = FakeDashboardStore()
    monkeypatch.setattr(api_module, "load_scores", lambda errors: [_entry()])
    app = create_app(store=fake_store)

    with TestClient(app) as client:
        meta = client.get("/api/meta").json()
        assert meta["entry_count"] == 1
        assert "rwkv7-g1g-1.5b-test" in meta["models"]

        leaderboard = client.get(
            "/api/leaderboard",
            params={"model": "rwkv7-g1g-1.5b-test", "view": "benchmark_detail_latest"},
        ).json()
        knowledge = next(domain for domain in leaderboard["domains"] if domain["key"] == "knowledge")
        assert knowledge["rows"][0]["benchmark_name"] == "mmlu_nocot"
        assert knowledge["rows"][0]["cells"][0]["meta"]["task_id"] == 123

        records = client.get("/api/eval-records", params={"task_id": 123, "limit": 1}).json()
        assert records["has_more"] is True
        assert records["records"][0]["answer"] == "A"

        wrong = client.get(
            "/api/eval-records",
            params={"task_id": 123, "only_wrong": "true", "limit": 10},
        ).json()
        assert [row["sample_index"] for row in wrong["records"]] == [1]

        context = client.get(
            "/api/eval-context",
            params={"task_id": 123, "sample_index": 0, "repeat_index": 0, "pass_index": 0},
        ).json()
        assert context["view"] == "structured"
        assert context["context"]["stages"][0]["prompt"] == "What is 1+1?"
        assert context["stop_tokens"]["answer"][0]["id"] == 0

        options = client.get("/api/score-history/options").json()
        assert options["pairs"] == [{"model": "rwkv7-g1g-1.5b-test", "dataset": "mmlu_test"}]

        history = client.get(
            "/api/score-history",
            params={"model": "rwkv7-g1g-1.5b-test", "benchmark": "mmlu_test"},
        ).json()
        assert history["total"] == 1
        assert history["raw_total"] == 2
        assert history["compact"] is True
        assert history["groups"][0]["points"][0]["percent"] == 60.0

        full_history = client.get(
            "/api/score-history",
            params={"model": "rwkv7-g1g-1.5b-test", "benchmark": "mmlu_test", "compact": "false"},
        ).json()
        assert full_history["total"] == 2
        assert full_history["compact"] is False

        detail = client.get("/api/score-history/detail", params={"task_id": 123}).json()
        assert detail["found"] is True
        assert detail["metric"] == "avg@1"
        assert detail["sampling"]["stages"]["answer"]["temperature"] == 0.2

    assert "history_pairs" in fake_store.calls
    assert "history:rwkv7-g1g-1.5b-test:mmlu_test" in fake_store.calls
    assert "history_detail:123" in fake_store.calls
