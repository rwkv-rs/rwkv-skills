from __future__ import annotations

from src.bin.run_infer_router import create_app, parse_routes


def test_parse_routes_normalizes_base_urls() -> None:
    routes = parse_routes(("model-a=127.0.0.1:18081", "model-b=http://127.0.0.1:18082/v1"))

    assert routes == {
        "model-a": "http://127.0.0.1:18081/v1",
        "model-b": "http://127.0.0.1:18082/v1",
    }


def test_create_app_registers_openai_routes() -> None:
    app = create_app({"model-b": "http://127.0.0.1:18082/v1", "model-a": "http://127.0.0.1:18081/v1"})

    paths = {route.path for route in app.routes}

    assert "/healthz" in paths
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
