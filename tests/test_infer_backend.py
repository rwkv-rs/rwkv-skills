from __future__ import annotations

import argparse

import pytest

from src.infer.backend import (
    RemoteInferenceBackend,
    RemoteInferenceConfig,
    require_completion_style_remote_protocol,
    validate_inference_backend_args,
)
from src.infer.sampling import SamplingConfig


def test_validate_args_rejects_removed_local_backend() -> None:
    args = argparse.Namespace(model_path="/models/rwkv.pth", infer_base_url="", infer_model="")

    with pytest.raises(ValueError, match="已移除本地推理"):
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
