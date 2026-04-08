from __future__ import annotations

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.infer.api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)
from src.infer.backend import (
    LocalInferenceBackend,
    RemoteHTTPError,
    RemoteInferenceBackend,
    RemoteInferenceConfig,
    normalize_api_base,
    resolve_backend_model_name,
    validate_inference_backend_args,
)
from src.infer.sampling import GenerationOutput, SamplingConfig
from src.infer.service import InferenceService


class _FakeBackend:
    def __init__(self) -> None:
        self.model_name = "demo-model"
        self.generate_calls: list[dict[str, object]] = []
        self.score_calls: list[tuple[str, list[str]]] = []
        self.shutdown_calls = 0

    def generate(
        self,
        prompts,
        *,
        sampling,
        batch_size,
        progress_desc="Generating",
        probe_only=False,
        on_complete=None,
        prompt_seeds=None,
        prefill_chunk_size=16,
        show_progress=True,
    ):
        self.generate_calls.append(
            {
                "prompts": list(prompts),
                "batch_size": batch_size,
                "prompt_seeds": None if prompt_seeds is None else list(prompt_seeds),
                "prefill_chunk_size": prefill_chunk_size,
                "show_progress": show_progress,
            }
        )
        outputs = []
        for index, prompt in enumerate(prompts):
            output = GenerationOutput(
                prompt_index=index,
                prompt=prompt,
                token_ids=[],
                text=f"gen:{prompt}",
                finish_reason="stop_token",
            )
            outputs.append(output)
            if on_complete is not None and not probe_only:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        choices = list(choice_token_texts)
        self.score_calls.append((prompt, choices))
        return {choice: float(index) for index, choice in enumerate(choices)}, choices[-1]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_completion_request_to_sampling_config_preserves_custom_fields() -> None:
    request = CompletionRequest(
        model="demo-model",
        prompt="hello",
        max_tokens=12,
        temperature=0.7,
        top_k=42,
        top_p=0.9,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        penalty_decay=0.95,
        stop_tokens=[1, 2],
        ban_tokens=[3],
        pad_zero=False,
        no_penalty_token_ids=[4, 5],
        prefill_chunk_size=32,
    )
    sampling = request.to_sampling_config()
    assert sampling.max_generate_tokens == 12
    assert sampling.temperature == 0.7
    assert sampling.top_k == 42
    assert sampling.top_p == 0.9
    assert sampling.alpha_presence == 0.2
    assert sampling.alpha_frequency == 0.3
    assert sampling.alpha_decay == 0.95
    assert sampling.stop_tokens == (1, 2)
    assert sampling.ban_tokens == (3,)
    assert sampling.pad_zero is False
    assert sampling.no_penalty_token_ids == (4, 5)
    assert request.effective_prefill_chunk_size() == 32


def test_inference_backend_arg_validation_and_model_name_resolution() -> None:
    local_args = argparse.Namespace(
        model_path="/tmp/model.pth",
        infer_base_url="",
        infer_model="",
    )
    validate_inference_backend_args(local_args)
    assert resolve_backend_model_name(local_args) == "model"

    remote_args = argparse.Namespace(
        model_path="",
        infer_base_url="127.0.0.1:8081",
        infer_model="remote-demo",
    )
    validate_inference_backend_args(remote_args)
    assert resolve_backend_model_name(remote_args) == "remote-demo"
    assert normalize_api_base("127.0.0.1:8081") == "http://127.0.0.1:8081/v1"


def test_inference_service_batches_generation_and_handles_choice_scoring() -> None:
    backend = _FakeBackend()
    service = InferenceService(backend, max_batch_size=4, batch_collect_ms=10)
    try:
        future_one = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="prompt-one",
                max_tokens=8,
                temperature=0.3,
                seed=11,
            )
        )
        future_two = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="prompt-two",
                max_tokens=8,
                temperature=0.3,
                seed=22,
            )
        )
        future_score = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="question",
                max_tokens=1,
                logprobs=1,
                candidate_token_texts=[" A", " B"],
            )
        )

        response_one = future_one.result(timeout=2.0)
        response_two = future_two.result(timeout=2.0)
        response_score = future_score.result(timeout=2.0)
    finally:
        service.shutdown()

    assert [call["prompts"] for call in backend.generate_calls] == [["prompt-one", "prompt-two"]]
    assert backend.generate_calls[0]["prompt_seeds"] == [11, 22]
    assert backend.generate_calls[0]["show_progress"] is False

    assert response_one.choices[0].text == "gen:prompt-one"
    assert response_two.choices[0].text == "gen:prompt-two"

    assert backend.score_calls == [("question", [" A", " B"])]
    top_logprobs = response_score.choices[0].logprobs.top_logprobs
    assert top_logprobs is not None
    assert top_logprobs[0][" B"] > top_logprobs[0][" A"]
    assert backend.shutdown_calls == 1


def test_chat_completion_models_bridge_to_completion_shapes() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[
            ChatCompletionMessage(role="user", content="hello"),
            ChatCompletionMessage(role="assistant", content="prefill"),
        ],
        max_tokens=7,
        temperature=0.2,
    )

    completion_request = request.to_completion_request()

    assert completion_request.prompt == "User: hello\n\nAssistant: prefill"
    assert completion_request.max_tokens == 7
    assert completion_request.temperature == 0.2

    response = ChatCompletionResponse.from_completion_response(
        CompletionResponse(
            id="cmpl-demo",
            created=123,
            model="demo-model",
            choices=[CompletionChoice(text="world", finish_reason="stop_token")],
        )
    )

    assert response.id == "chatcmpl-demo"
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "world"
    assert response.choices[0].finish_reason == "stop"


def test_remote_backend_uses_chat_completions_and_caches_unsupported_choice_scoring(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/chat/completions"):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ]
            }
        raise RemoteHTTPError(404, "missing")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(
            max_generate_tokens=4,
            temperature=0.3,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
        ),
        batch_size=1,
        show_progress=False,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "answer"
    assert outputs[0].finish_reason == "stop_token"
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["messages"] == [{"role": "user", "content": "prompt"}]
    assert "top_k" not in calls[0][1]
    assert "penalty_decay" not in calls[0][1]

    with pytest.raises(NotImplementedError):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])
    with pytest.raises(NotImplementedError):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    completion_calls = [url for url, _payload in calls if url.endswith("/v1/completions")]
    assert len(completion_calls) == 1


def test_local_inference_backend_can_select_lightning_engine(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, token_ids) -> str:
            return ""

    class _FakeEngine:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def generate(self, prompts, **_kwargs):
            return []

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    fake_engine = _FakeEngine()

    def _fake_load_rwkv_model(config):
        captured["config"] = config
        return object(), _FakeTokenizer()

    def _fake_build_local_engine(model, tokenizer, *, mode, state_db_path):
        captured["mode"] = mode
        captured["state_db_path"] = state_db_path
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        return fake_engine

    fake_model_module = ModuleType("src.infer.model")
    fake_model_module.load_rwkv_model = _fake_load_rwkv_model
    monkeypatch.setitem(sys.modules, "src.infer.model", fake_model_module)
    monkeypatch.setattr("src.infer.backend.build_local_engine", _fake_build_local_engine)

    backend = LocalInferenceBackend.from_model_config(
        SimpleNamespace(weights_path="/tmp/demo-model.pth", device="cpu"),
        engine_mode="lightning",
        state_db_path=str(tmp_path / "state-cache.db"),
    )

    assert backend.engine_mode == "lightning"
    assert captured["mode"] == "lightning"
    assert captured["state_db_path"] == str(tmp_path / "state-cache.db")
    assert backend.engine is fake_engine

    backend.shutdown()
    assert fake_engine.shutdown_calls == 1
