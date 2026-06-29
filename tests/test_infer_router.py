from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi.responses import Response

import src.bin.run_infer_router as infer_router
from src.bin.run_infer_router import (
    _build_backpressure_payload,
    _build_batch_metrics_payload,
    _backend_url_for_path,
    create_app,
    parse_args,
    parse_routes,
)


def _close_router_app(app) -> None:
    app.state.forward_http_client.close()
    app.state.forward_executor.shutdown(wait=False, cancel_futures=True)


def test_parse_routes_normalizes_base_urls() -> None:
    routes = parse_routes(
        (
            "model-a=127.0.0.1:18081",
            "model-b=http://127.0.0.1:18082/v1",
            "model-c=http://127.0.0.1:18083/v2",
        )
    )

    assert routes == {
        "model-a": ("http://127.0.0.1:18081/v1",),
        "model-b": ("http://127.0.0.1:18082/v1",),
        "model-c": ("http://127.0.0.1:18083/v1",),
    }


def test_backend_url_for_path_switches_api_version() -> None:
    assert (
        _backend_url_for_path("http://127.0.0.1:18081/v1", "v2/chat/completions")
        == "http://127.0.0.1:18081/v2/chat/completions"
    )
    assert (
        _backend_url_for_path("http://127.0.0.1:18081/v1", "high_throughput/chat/completions")
        == "http://127.0.0.1:18081/high_throughput/chat/completions"
    )
    assert (
        _backend_url_for_path("http://127.0.0.1:18081/openai/v1", "choice_logits")
        == "http://127.0.0.1:18081/openai/v1/choice_logits"
    )


def test_parse_routes_preserves_duplicate_model_backends() -> None:
    routes = parse_routes(
        (
            "model-a=http://127.0.0.1:18081",
            "model-a=http://127.0.0.1:18082/v1",
        )
    )

    assert routes == {
        "model-a": (
            "http://127.0.0.1:18081/v1",
            "http://127.0.0.1:18082/v1",
        )
    }


def test_create_app_registers_openai_routes() -> None:
    app = create_app({"model-b": "http://127.0.0.1:18082/v1", "model-a": "http://127.0.0.1:18081/v1"})
    try:
        paths = {route.path for route in app.routes}

        assert "/healthz" in paths
        assert "/v1/models" in paths
        assert "/v1/backpressure" in paths
        assert "/v1/batch-metrics" in paths
        assert "/v1/chat/completions" in paths
        assert "/v2/chat/completions" in paths
        assert "/high_throughput/chat/completions" in paths
        assert "/v1/choice_logits" in paths
        assert "/v1/completions" in paths
    finally:
        _close_router_app(app)


def test_router_forward_max_workers_is_configurable() -> None:
    args = parse_args(
        [
            "--route",
            "model-a=http://127.0.0.1:18081",
            "--forward-max-workers",
            "96",
        ]
    )
    app = create_app({"model-a": "http://127.0.0.1:18081"}, forward_max_workers=args.forward_max_workers)
    try:
        assert args.forward_max_workers == 96
        assert app.state.forward_executor._max_workers == 96
    finally:
        _close_router_app(app)


def test_choice_logits_timeout_is_configurable(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def _fake_post_bytes(url, body, authorization, content_type, timeout_s, http_client):  # noqa: ANN001
        del body, authorization, content_type, http_client
        calls.append((url, float(timeout_s)))
        return Response(
            content=json.dumps(
                {
                    "id": "backend-choice-logits",
                    "object": "choice.logits",
                    "model": "model-a",
                    "choice_logits": {"A": 1.0},
                    "best_choice": "A",
                }
            ).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    monkeypatch.setattr(infer_router, "_post_bytes", _fake_post_bytes)
    args = parse_args(
        [
            "--route",
            "model-a=http://127.0.0.1:18081",
            "--timeout-s",
            "900",
            "--choice-logits-timeout-s",
            "37",
        ]
    )
    app = create_app(
        {"model-a": "http://127.0.0.1:18081"},
        timeout_s=args.timeout_s,
        choice_logits_timeout_s=args.choice_logits_timeout_s,
    )
    try:
        async def _body() -> bytes:
            return json.dumps(
                {"model": "model-a", "content": "question", "choices": {"A": " A"}}
            ).encode("utf-8")

        response = asyncio.run(
            infer_router._forward_json_request(
                SimpleNamespace(
                    app=SimpleNamespace(state=app.state),
                    headers={"content-type": "application/json"},
                    body=_body,
                ),
                routes=infer_router.normalize_routes({"model-a": "http://127.0.0.1:18081"}),
                route_offsets={"model-a": 0},
                backend_path="choice_logits",
                timeout_s=float(args.choice_logits_timeout_s),
            )
        )
    finally:
        _close_router_app(app)

    assert response.status_code == 200
    assert calls == [("http://127.0.0.1:18081/v1/choice_logits", 37.0)]


def test_choice_logits_contents_batch_is_sharded_across_replicas() -> None:
    payload = {
        "model": "model-a",
        "contents": ["q0", "q1", "q2", "q3"],
        "choices": {"A": " A", "B": " B"},
    }
    urls = infer_router.normalize_routes(
        {
            "model-a": (
                "http://127.0.0.1:18081",
                "http://127.0.0.1:18082",
            )
        }
    )["model-a"]
    subrequests = infer_router._split_contents_payload(payload, urls=urls, start_offset=0)
    calls = [(base_url, subpayload) for base_url, _indices, subpayload in subrequests]
    results = []
    for _base_url, _indices, subpayload in subrequests:
        local_results = [
            {
                "index": index,
                "choice_logits": {"A": float(index), "B": -float(index)},
                "choice_probabilities": {"A": 0.75, "B": 0.25},
                "best_choice": "A",
            }
            for index, _content in enumerate(subpayload["contents"])
        ]
        results.append(
            (
                200,
                json.dumps(
                    {
                        "id": "backend-choice-logits",
                        "object": "choice.logits.batch",
                        "model": subpayload["model"],
                        "results": local_results,
                    }
                ).encode("utf-8"),
                "application/json",
            )
        )

    body = infer_router._merge_choice_logits_batch_responses(
        payload,
        subrequests=subrequests,
        results=results,
    )

    assert [call[0] for call in calls] == [
        "http://127.0.0.1:18081/v1",
        "http://127.0.0.1:18082/v1",
    ]
    assert [call[1]["contents"] for call in calls] == [["q0", "q1"], ["q2", "q3"]]
    assert body["object"] == "choice.logits.batch"
    assert body["model"] == "model-a"
    assert [result["index"] for result in body["results"]] == [0, 1, 2, 3]


def test_router_forward_http_client_is_shared(monkeypatch) -> None:
    created_clients = []

    class _FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created_clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(infer_router.httpx, "Client", _FakeHTTPClient)
    app = create_app({"model-a": "http://127.0.0.1:18081"}, forward_max_workers=96, timeout_s=12.0)

    assert len(created_clients) == 1
    assert app.state.forward_http_client is created_clients[0]
    assert created_clients[0].kwargs["follow_redirects"] is True

    _close_router_app(app)

    assert created_clients[0].closed is True


def test_build_backpressure_payload_aggregates_backend_metrics() -> None:
    routes = {
        "model-a": (
            "http://127.0.0.1:18081/v1",
            "http://127.0.0.1:18082/v1",
        )
    }
    payload = _build_backpressure_payload(
        routes,
        {
            "model-a": [
                (
                    "http://127.0.0.1:18081/v1",
                    200,
                    {
                        "model_name": "model-a",
                        "max_batch_size": 16,
                        "batch_collect_ms": 5,
                        "pending": {
                            "pending_queue": 5,
                            "service_queue": 2,
                            "engine_inbox": 1,
                            "active_records": 2,
                            "scheduler_waiting": 1,
                            "scheduler_running": 1,
                            "prefill_reserved_bsz": 3,
                        },
                        "totals": {
                            "total_batches": 3,
                            "total_requests": 12,
                            "completed_requests": 11,
                            "error_requests": 1,
                            "failed_batches": 0,
                            "last_total_tok_s": 40.0,
                        },
                    },
                ),
                (
                    "http://127.0.0.1:18082/v1",
                    200,
                    {
                        "model_name": "model-a",
                        "max_batch_size": 8,
                        "pending": {
                            "pending_queue": 3,
                            "service_queue": 1,
                            "engine_inbox": 1,
                            "active_records": 1,
                            "scheduler_waiting": 1,
                            "scheduler_running": 0,
                            "prefill_reserved_bsz": 2,
                        },
                        "totals": {
                            "total_batches": 2,
                            "total_requests": 8,
                            "completed_requests": 7,
                            "error_requests": 1,
                            "failed_batches": 1,
                            "last_total_tok_s": 20.0,
                        },
                    },
                ),
            ],
        },
    )

    aggregate = payload["models"]["model-a"]["aggregate"]
    assert aggregate["status"] == "ok"
    assert aggregate["ok_route_count"] == 2
    assert aggregate["pending_queue"] == 8
    assert aggregate["service_queue"] == 3
    assert aggregate["engine_inbox"] == 2
    assert aggregate["active_records"] == 3
    assert aggregate["scheduler_waiting"] == 2
    assert aggregate["scheduler_running"] == 1
    assert aggregate["prefill_reserved_bsz"] == 5
    assert aggregate["max_batch_size"] == 24
    assert aggregate["failed_batches"] == 1
    assert aggregate["completed_requests"] == 18
    assert aggregate["error_requests"] == 2
    assert aggregate["last_total_tok_s"] == 60.0


def test_build_batch_metrics_payload_preserves_backend_live_metrics() -> None:
    routes = {"model-a": ("http://127.0.0.1:18081/v1",)}
    payload = _build_batch_metrics_payload(
        routes,
        {
            "model-a": [
                (
                    "http://127.0.0.1:18081/v1",
                    200,
                    {
                        "model_name": "model-a",
                        "max_batch_size": 16,
                        "pending": {
                            "pending_queue": 4,
                            "service_queue": 0,
                            "engine_inbox": 1,
                            "active_records": 3,
                            "scheduler_waiting": 2,
                            "scheduler_running": 1,
                            "prefill_reserved_bsz": 7,
                        },
                        "totals": {
                            "total_batches": 5,
                            "total_requests": 10,
                            "completed_requests": 9,
                            "error_requests": 1,
                            "failed_batches": 1,
                        },
                        "backend_live": {
                            "engine_inbox": 1,
                            "active_records": 3,
                            "state_cache": {"hits": 2, "misses": 1},
                        },
                    },
                )
            ],
        },
    )

    model = payload["models"]["model-a"]
    assert model["status"] == "ok"
    assert model["aggregate"]["pending_queue"] == 4
    assert model["aggregate"]["prefill_reserved_bsz"] == 7
    assert model["aggregate"]["completed_requests"] == 9
    assert model["backends"][0]["backend_live"]["state_cache"]["hits"] == 2
