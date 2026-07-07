from __future__ import annotations

import argparse

import pytest

from src.infer.backend import (
    DEFAULT_REMOTE_MAX_WORKERS,
    RemoteInferenceBackend,
    RemoteInferenceConfig,
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    require_completion_style_remote_protocol,
    validate_inference_backend_args,
)
from src.infer.sampling import SamplingConfig


def test_validate_args_requires_remote_base_url() -> None:
    args = argparse.Namespace(model_path="/models/rwkv.pth", infer_base_url="", infer_model="")

    with pytest.raises(ValueError, match="缺少 --infer-base-url"):
        validate_inference_backend_args(args)


def test_completion_style_converts_vllm_to_completions() -> None:
    args = argparse.Namespace(
        model_path="",
        infer_base_url="http://127.0.0.1:19082",
        infer_model="demo",
        infer_protocol="vllm",
    )

    changed = require_completion_style_remote_protocol(args, benchmark_name="demo_bench")

    assert changed is True
    assert args.infer_protocol == "completions"


def test_inference_backend_cli_defaults_to_remote_config_workers() -> None:
    parser = argparse.ArgumentParser()
    add_inference_backend_arguments(parser)

    args = parser.parse_args(["--infer-base-url", "http://127.0.0.1:19082", "--infer-model", "demo"])
    backend = build_inference_backend_from_args(args)

    assert args.infer_max_workers == DEFAULT_REMOTE_MAX_WORKERS
    assert isinstance(backend, RemoteInferenceBackend)
    assert backend.config.max_workers == DEFAULT_REMOTE_MAX_WORKERS


def test_vllm_generation_uses_chat_completions(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19082",
            model="demo",
            protocol="vllm",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
        calls.append((url, payload))
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", _fake_post_json)

    outputs = backend.generate(
        ["hello"],
        sampling=SamplingConfig(max_generate_tokens=3, temperature=0.0, top_p=1.0, top_k=0),
        batch_size=1,
        show_progress=False,
    )

    assert outputs[0].text == "ok"
    assert calls[0][0] == "http://127.0.0.1:19082/v1/chat/completions"
    assert calls[0][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert "top_k" not in calls[0][1]


def test_vllm_tool_call_generation_preserves_native_tool_calls(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19082",
            model="demo",
            protocol="vllm",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
        calls.append((url, payload))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\":\"Paris\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", _fake_post_json)

    outputs = backend.generate_tool_calls(
        [[{"role": "user", "content": "weather?"}]],
        [
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ]
        ],
        sampling=SamplingConfig(max_generate_tokens=64, temperature=0.0, top_p=1.0, top_k=0),
        batch_size=1,
        show_progress=False,
    )

    assert calls[0][0] == "http://127.0.0.1:19082/v1/chat/completions"
    assert calls[0][1]["tool_choice"] == "auto"
    assert "seed" not in calls[0][1]
    assert outputs[0].content == ""
    assert outputs[0].finish_reason == "tool_calls"
    assert outputs[0].response_source == "tool_calls"
    assert [(call.id, call.name, call.arguments) for call in outputs[0].tool_calls] == [
        ("call_1", "get_weather", {"city": "Paris"})
    ]


def test_completions_protocol_rejects_native_tool_calls() -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19082",
            model="demo",
            protocol="completions",
        )
    )

    with pytest.raises(NotImplementedError, match="native chat tool calls"):
        backend.generate_tool_calls(
            [[{"role": "user", "content": "weather?"}]],
            [[]],
            sampling=SamplingConfig(),
            batch_size=1,
            show_progress=False,
        )


def test_completions_generation_uses_raw_prompt_with_private_sampling(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19082/v1",
            model="demo",
            protocol="completions",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:  # noqa: ANN001
        calls.append((url, payload))
        return {"choices": [{"text": "ok", "finish_reason": "stop"}]}

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", _fake_post_json)

    outputs = backend.generate(
        ["hello"],
        sampling=SamplingConfig(max_generate_tokens=3, temperature=0.0, top_p=1.0, top_k=17),
        batch_size=1,
        show_progress=False,
    )

    assert outputs[0].text == "ok"
    assert calls[0][0] == "http://127.0.0.1:19082/v1/completions"
    assert calls[0][1]["prompt"] == "hello"
    assert calls[0][1]["top_k"] == 17
