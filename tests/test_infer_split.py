from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from src.infer.api import (
    ChatCompletionRequest,
    ChatCompletionMessage,
    ChatNamedToolChoice,
    ChatNamedToolChoiceFunction,
    ChatResponseFormat,
    ChatTool,
    ChatToolFunction,
    ChatCompletionToolCall,
    ChatCompletionToolCallFunction,
    ChoiceLogitsRequest,
    CompletionChoice,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
)
from src.infer.backend import (
    LocalInferenceBackend,
    RemoteHTTPError,
    RemoteInferenceBackend,
    RemoteInferenceConfig,
    normalize_api_base,
    normalize_local_device,
    remote_contents_inflight_batches,
    require_completion_style_remote_protocol,
    resolve_generation_prompt_batch_size,
    resolve_backend_model_name,
    validate_inference_backend_args,
)
from src.infer.constraints import LiteralChoiceConstraint
from src.infer.openai_service import build_chat_completion_response, prepare_chat_completion_request
from src.infer.openai_service import (
    build_chat_completion_stream_responses,
    build_completion_stream_responses,
)
from src.infer.sampling import GeneratedTextDelta, GeneratedToken, GeneratedTokenCandidate
from src.infer.sampling import GenerationOutput, SamplingConfig
from src.infer.server import create_app as create_infer_app
from src.infer.service import InferenceService
from src.infer.sse import encode_sse_comment, iter_sse_payloads


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
        on_token=None,
        prompt_stop_suffixes=None,
        constraints=None,
        constraint_mode="off",
        prompt_seeds=None,
        top_logprobs=0,
        prefill_chunk_size=16,
        show_progress=True,
    ):
        self.generate_calls.append(
            {
                "prompts": list(prompts),
                "batch_size": batch_size,
                "prompt_stop_suffixes": None if prompt_stop_suffixes is None else [list(item or ()) for item in prompt_stop_suffixes],
                "constraints": constraints,
                "constraint_mode": constraint_mode,
                "prompt_seeds": None if prompt_seeds is None else list(prompt_seeds),
                "prefill_chunk_size": prefill_chunk_size,
                "show_progress": show_progress,
            }
        )
        outputs = []
        for index, prompt in enumerate(prompts):
            generated_tokens = [
                GeneratedToken(
                    token_id=100 + index * 2,
                    text="gen:",
                    logprob=-0.1 if top_logprobs else None,
                    top_logprobs=[
                        GeneratedTokenCandidate(token_id=100 + index * 2, text="gen:", logprob=-0.1),
                        GeneratedTokenCandidate(token_id=200 + index * 2, text="alt:", logprob=-1.0),
                    ]
                    if top_logprobs
                    else [],
                ),
                GeneratedToken(
                    token_id=101 + index * 2,
                    text=prompt,
                    logprob=-0.2 if top_logprobs else None,
                    top_logprobs=[
                        GeneratedTokenCandidate(token_id=101 + index * 2, text=prompt, logprob=-0.2),
                    ]
                    if top_logprobs
                    else [],
                ),
            ]
            output = GenerationOutput(
                prompt_index=index,
                prompt=prompt,
                token_ids=[],
                text=f"gen:{prompt}",
                finish_reason="stop_token",
                tokens=generated_tokens,
            )
            outputs.append(output)
            if on_token is not None and not probe_only:
                for token in generated_tokens:
                    on_token(index, GeneratedTextDelta(text=token.text, tokens=[token]))
            if on_complete is not None and not probe_only:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        choices = list(choice_token_texts)
        self.score_calls.append((prompt, choices))
        return {choice: float(index) for index, choice in enumerate(choices)}, choices[-1]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _Utf8StreamingBackend(_FakeBackend):
    def generate(
        self,
        prompts,
        *,
        sampling,
        batch_size,
        progress_desc="Generating",
        probe_only=False,
        on_complete=None,
        on_token=None,
        prompt_stop_suffixes=None,
        constraints=None,
        constraint_mode="off",
        prompt_seeds=None,
        top_logprobs=0,
        prefill_chunk_size=16,
        show_progress=True,
    ):
        del (
            sampling,
            batch_size,
            progress_desc,
            prompt_stop_suffixes,
            constraints,
            constraint_mode,
            prompt_seeds,
            top_logprobs,
            prefill_chunk_size,
            show_progress,
        )
        tokens = [
            GeneratedToken(token_id=1, text="\ufffd", bytes=b"\xe4"),
            GeneratedToken(token_id=2, text="\ufffd", bytes=b"\xb8"),
            GeneratedToken(token_id=3, text="\ufffd", bytes=b"\x96"),
        ]
        output = GenerationOutput(
            prompt_index=0,
            prompt=str(prompts[0]),
            token_ids=[1, 2, 3],
            text="世",
            finish_reason="stop_token",
            tokens=tokens,
        )
        if on_token is not None and not probe_only:
            on_token(0, GeneratedTextDelta(text="世", tokens=tokens))
        if on_complete is not None and not probe_only:
            on_complete(output)
        return [output]


class _EarlyCompleteBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.first_completed = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        prompts,
        *,
        sampling,
        batch_size,
        progress_desc="Generating",
        probe_only=False,
        on_complete=None,
        on_token=None,
        prompt_stop_suffixes=None,
        constraints=None,
        constraint_mode="off",
        prompt_seeds=None,
        top_logprobs=0,
        prefill_chunk_size=16,
        show_progress=True,
    ):
        del (
            sampling,
            batch_size,
            progress_desc,
            probe_only,
            on_token,
            prompt_stop_suffixes,
            constraints,
            constraint_mode,
            prompt_seeds,
            top_logprobs,
            prefill_chunk_size,
            show_progress,
        )
        outputs = [
            GenerationOutput(
                prompt_index=index,
                prompt=prompt,
                token_ids=[index + 1],
                text=f"early:{prompt}",
                finish_reason="stop_token",
            )
            for index, prompt in enumerate(prompts)
        ]
        if on_complete is not None:
            on_complete(outputs[0])
        self.first_completed.set()
        if len(outputs) > 1:
            self.release.wait(timeout=2.0)
            if on_complete is not None:
                on_complete(outputs[1])
        return outputs


class _SubmitCapableBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.submit_calls: list[dict[str, object]] = []
        self.slow_started = threading.Event()
        self.release_slow = threading.Event()
        self.completed_order: list[str] = []

    def submit(
        self,
        prompt,
        *,
        sampling,
        prompt_index=0,
        prompt_stop_suffixes=None,
        prompt_seed=None,
        top_logprobs=0,
        prefill_chunk_size=16,
        on_token=None,
    ):
        self.submit_calls.append(
            {
                "prompt": prompt,
                "max_tokens": sampling.max_generate_tokens,
                "prompt_seed": prompt_seed,
                "prompt_stop_suffixes": prompt_stop_suffixes,
                "top_logprobs": top_logprobs,
                "prefill_chunk_size": prefill_chunk_size,
            }
        )
        future: Future[GenerationOutput] = Future()

        def _run() -> None:
            if int(sampling.max_generate_tokens) > 10:
                self.slow_started.set()
                self.release_slow.wait(timeout=2.0)
            else:
                time.sleep(0.01)
            output = GenerationOutput(
                prompt_index=int(prompt_index),
                prompt=str(prompt),
                token_ids=[int(sampling.max_generate_tokens)],
                text=f"submit:{prompt}",
                finish_reason="stop_token",
            )
            if on_token is not None:
                on_token(int(prompt_index), GeneratedTextDelta(text=output.text))
            self.completed_order.append(str(prompt))
            future.set_result(output)

        threading.Thread(target=_run, daemon=True).start()
        return future


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
    assert normalize_api_base("http://127.0.0.1:8081/v2") == "http://127.0.0.1:8081/v1"


def test_completion_style_remote_protocol_normalizes_openai_to_completions() -> None:
    args = argparse.Namespace(
        model_path="",
        infer_base_url="127.0.0.1:8081",
        infer_model="remote-demo",
        infer_protocol="openai",
    )

    assert require_completion_style_remote_protocol(args, benchmark_name="legacy code") is True
    assert args.infer_protocol == "completions"


def test_completion_style_remote_protocol_rejects_vllm() -> None:
    args = argparse.Namespace(
        model_path="",
        infer_base_url="127.0.0.1:8081",
        infer_model="remote-demo",
        infer_protocol="vllm",
    )

    with pytest.raises(ValueError, match="requires completion-style"):
        require_completion_style_remote_protocol(args, benchmark_name="legacy code")


def test_completion_style_remote_protocol_allows_lightning_contents() -> None:
    args = argparse.Namespace(
        model_path="",
        infer_base_url="127.0.0.1:8081",
        infer_model="remote-demo",
        infer_protocol="lightning",
    )

    assert require_completion_style_remote_protocol(args, benchmark_name="legacy code") is True
    assert args.infer_protocol == "lightning"


def test_infer_server_registers_lightning_generation_and_choice_logits_routes() -> None:
    service = InferenceService(_FakeBackend(), max_batch_size=4, batch_collect_ms=0)
    try:
        app = create_infer_app(service)
        paths = {route.path for route in app.routes}
    finally:
        service.shutdown()

    assert "/v2/chat/completions" in paths
    assert "/openai/v2/chat/completions" in paths
    assert "/v1/choice_logits" in paths
    assert "/openai/v1/choice_logits" in paths


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
        future_logits = service.submit_choice_logits(
            ChoiceLogitsRequest(
                model="demo-model",
                prompt="question2",
                choices={"C": " C", "D": " D"},
            )
        )

        response_one = future_one.result(timeout=2.0)
        response_two = future_two.result(timeout=2.0)
        response_score = future_score.result(timeout=2.0)
        response_logits = future_logits.result(timeout=2.0)
    finally:
        service.shutdown()

    assert [call["prompts"] for call in backend.generate_calls] == [["prompt-one", "prompt-two"]]
    assert backend.generate_calls[0]["prompt_seeds"] == [11, 22]
    assert backend.generate_calls[0]["show_progress"] is False

    assert response_one.choices[0].text == "gen:prompt-one"
    assert response_two.choices[0].text == "gen:prompt-two"

    assert backend.score_calls == [("question", [" A", " B"]), ("question2", [" C", " D"])]
    top_logprobs = response_score.choices[0].logprobs.top_logprobs
    assert top_logprobs is not None
    assert top_logprobs[0][" B"] > top_logprobs[0][" A"]
    assert response_logits.best_choice == "D"
    assert response_logits.choice_logits["D"] > response_logits.choice_logits["C"]
    assert response_logits.choice_probabilities["D"] > response_logits.choice_probabilities["C"]
    assert backend.shutdown_calls == 1


def test_inference_service_completes_items_before_full_batch_returns() -> None:
    backend = _EarlyCompleteBackend()
    service = InferenceService(backend, max_batch_size=2, batch_collect_ms=10)
    try:
        future_one = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="prompt-one",
                max_tokens=8,
                temperature=0.3,
            )
        )
        future_two = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="prompt-two",
                max_tokens=8,
                temperature=0.3,
            )
        )

        assert backend.first_completed.wait(timeout=2.0)
        response_one = future_one.result(timeout=1.0)
        assert response_one.choices[0].text == "early:prompt-one"
        assert not future_two.done()

        backend.release.set()
        response_two = future_two.result(timeout=2.0)
    finally:
        backend.release.set()
        service.shutdown()

    assert response_two.choices[0].text == "early:prompt-two"


def test_inference_service_submit_backend_allows_fast_request_to_finish_before_slow_request() -> None:
    backend = _SubmitCapableBackend()
    service = InferenceService(backend, max_batch_size=4, batch_collect_ms=10)
    try:
        slow_future = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="slow",
                max_tokens=64,
                temperature=0.3,
            )
        )
        fast_future = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="fast",
                max_tokens=1,
                temperature=0.9,
            )
        )

        assert backend.slow_started.wait(timeout=2.0)
        fast_response = fast_future.result(timeout=2.0)
        assert fast_response.choices[0].text == "submit:fast"
        assert not slow_future.done()

        backend.release_slow.set()
        slow_response = slow_future.result(timeout=2.0)
    finally:
        backend.release_slow.set()
        service.shutdown()

    assert slow_response.choices[0].text == "submit:slow"
    assert backend.generate_calls == []
    assert [call["prompt"] for call in backend.submit_calls] == ["slow", "fast"]
    assert backend.completed_order[0] == "fast"


def test_inference_service_non_submit_backend_keeps_different_sampling_in_separate_batches() -> None:
    backend = _FakeBackend()
    service = InferenceService(backend, max_batch_size=4, batch_collect_ms=10)
    try:
        first = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="first",
                max_tokens=1,
                temperature=0.3,
            )
        )
        second = service.submit_completion(
            CompletionRequest(
                model="demo-model",
                prompt="second",
                max_tokens=2,
                temperature=0.3,
            )
        )
        assert first.result(timeout=2.0).choices[0].text == "gen:first"
        assert second.result(timeout=2.0).choices[0].text == "gen:second"
    finally:
        service.shutdown()

    assert [call["prompts"] for call in backend.generate_calls] == [["first"], ["second"]]


def test_inference_service_streams_local_token_events_and_builds_logprobs() -> None:
    backend = _FakeBackend()
    service = InferenceService(backend, max_batch_size=4, batch_collect_ms=0)
    try:
        handle = service.submit_streaming_completion(
            CompletionRequest(
                model="demo-model",
                prompt="stream-me",
                max_tokens=8,
                temperature=0.3,
                stream=True,
                logprobs=2,
            )
        )
        first = handle.token_queue.get(timeout=2.0)
        second = handle.token_queue.get(timeout=2.0)
        sentinel = handle.token_queue.get(timeout=2.0)
        response = handle.future.result(timeout=2.0)
    finally:
        service.shutdown()

    assert first is not None
    assert second is not None
    assert first.text == "gen:"
    assert second.text == "stream-me"
    assert sentinel is None
    assert response.id == handle.response_id
    assert response.created == handle.created
    assert response.choices[0].text == "gen:stream-me"
    assert response.choices[0].logprobs is not None
    assert response.choices[0].logprobs.tokens == ["gen:", "stream-me"]
    assert response.choices[0].logprobs.top_logprobs[0]["gen:"] == -0.1


def test_inference_service_stream_queue_waits_for_stable_utf8_text() -> None:
    backend = _Utf8StreamingBackend()
    service = InferenceService(backend, max_batch_size=1, batch_collect_ms=0)
    try:
        handle = service.submit_streaming_completion(
            CompletionRequest(
                model="demo-model",
                prompt="utf8",
                max_tokens=8,
                stream=True,
            )
        )
        delta = handle.token_queue.get(timeout=2.0)
        sentinel = handle.token_queue.get(timeout=2.0)
        response = handle.future.result(timeout=2.0)
    finally:
        service.shutdown()

    assert delta is not None
    assert delta.text == "世"
    assert [token.bytes for token in delta.tokens] == [b"\xe4", b"\xb8", b"\x96"]
    assert sentinel is None
    assert response.choices[0].text == "世"


def test_chat_completion_request_preparation_preserves_chat_and_sampling_fields() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[
            ChatCompletionMessage(role="user", content="hello"),
            ChatCompletionMessage(role="assistant", content="prefill"),
        ],
        max_tokens=7,
        temperature=0.2,
        repetition_penalty=0.4,
    )

    prepared = prepare_chat_completion_request(request)
    completion_request = prepared.completion_request

    assert completion_request.prompt == "User:hello\n\nAssistant: prefill"
    assert completion_request.max_tokens == 7
    assert completion_request.temperature == 0.2
    assert completion_request.repetition_penalty == 0.4
    assert prepared.response_mode == "plain_text"

    response = build_chat_completion_response(
        request,
        prepared,
        CompletionResponse(
            id="cmpl-demo",
            created=123,
            model="demo-model",
            choices=[
                CompletionChoice(
                    text="world",
                    finish_reason="stop_token",
                    logprobs=CompletionLogprobs(
                        tokens=["world"],
                        token_logprobs=[-0.25],
                        top_logprobs=[{"world": -0.25}],
                    ),
                )
            ],
        ),
    )

    assert response.id == "chatcmpl-demo"
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "world"
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].logprobs is None


def test_chat_completion_request_supports_json_response_format() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[ChatCompletionMessage(role="user", content="Return a JSON object.")],
        response_format=ChatResponseFormat(type="json_object"),
        max_tokens=16,
    )

    prepared = prepare_chat_completion_request(request)

    assert prepared.response_mode == "json_text"
    assert prepared.completion_request.prompt.startswith("System: Return only a valid JSON object.")
    assert prepared.completion_request.prompt.endswith("\n\nAssistant:")


def test_chat_completion_request_supports_openai_tool_prompting_and_parsing() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[ChatCompletionMessage(role="user", content="Lookup weather for Hangzhou")],
        tools=[
            ChatTool(
                function=ChatToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            )
        ],
        tool_choice=ChatNamedToolChoice(
            function=ChatNamedToolChoiceFunction(name="get_weather"),
        ),
        parallel_tool_calls=False,
    )

    prepared = prepare_chat_completion_request(request)

    assert prepared.response_mode == "tool_call"
    assert "OpenAI tool-calling interface" in prepared.completion_request.prompt
    assert "get_weather" in prepared.completion_request.prompt
    assert prepared.completion_request.prompt.endswith("\n\nAssistant: ```json\n{")

    response = build_chat_completion_response(
        request,
        prepared,
        CompletionResponse(
            id="cmpl-demo",
            created=123,
            model="demo-model",
            choices=[
                CompletionChoice(
                    text=(
                        "<think>\n</think>\n```json\n"
                        + json.dumps(
                            {
                                "type": "tool_calls",
                                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Hangzhou"}}],
                            },
                            ensure_ascii=False,
                        )
                        + "\n```"
                    ),
                    finish_reason="stop_token",
                )
            ],
        ),
    )

    assert response.choices[0].message.content is None
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "Hangzhou"}
    assert response.choices[0].finish_reason == "tool_calls"


def test_chat_completion_tool_prompt_collapses_history_to_single_turn() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[
            ChatCompletionMessage(role="user", content="Lookup weather for Hangzhou"),
            ChatCompletionMessage(
                role="assistant",
                tool_calls=[
                    ChatCompletionToolCall(
                        id="call_0",
                        function=ChatCompletionToolCallFunction(
                            name="get_weather",
                            arguments='{"city":"Hangzhou"}',
                        ),
                    )
                ],
            ),
            ChatCompletionMessage(role="tool", tool_call_id="call_0", content='{"temp_c": 20}'),
            ChatCompletionMessage(role="user", content="Summarize it."),
        ],
        tools=[
            ChatTool(
                function=ChatToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            )
        ],
        tool_choice="required",
    )

    prepared = prepare_chat_completion_request(request)
    prompt = prepared.completion_request.prompt

    assert sum(1 for line in prompt.splitlines() if line.startswith("User:")) == 1
    assert sum(1 for line in prompt.splitlines() if line.startswith("Assistant:")) == 1
    assert "Conversation transcript JSON:" in prompt
    assert prompt.endswith("Assistant: ```json\n{")


def test_chat_completion_request_rejects_invalid_tool_configuration() -> None:
    with pytest.raises(ValueError, match="tool_choice requires tools"):
        prepare_chat_completion_request(
            ChatCompletionRequest(
                model="demo-model",
                messages=[ChatCompletionMessage(role="user", content="hello")],
                tool_choice="auto",
            )
        )


def test_completion_stream_builder_matches_openai_chunk_shape() -> None:
    response = CompletionResponse(
        id="cmpl-demo",
        created=123,
        model="demo-model",
        choices=[
            CompletionChoice(
                text="world",
                finish_reason="stop_token",
                logprobs=CompletionLogprobs(
                    tokens=["world"],
                    token_logprobs=[-0.25],
                    top_logprobs=[{"world": -0.25}],
                    text_offset=[0],
                ),
            )
        ],
    )

    chunks = build_completion_stream_responses(response)

    assert len(chunks) == 2
    assert chunks[0].object == "text_completion.chunk"
    assert chunks[0].choices[0].text == "world"
    assert chunks[0].choices[0].finish_reason is None
    assert chunks[0].choices[0].logprobs is not None
    assert chunks[1].choices[0].text == ""
    assert chunks[1].choices[0].finish_reason == "stop_token"


def test_chat_stream_builder_matches_openai_chunk_shape_for_text() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[ChatCompletionMessage(role="user", content="hello")],
        logprobs=True,
        top_logprobs=1,
    )
    prepared = prepare_chat_completion_request(request)

    chunks = build_chat_completion_stream_responses(
        request,
        prepared,
        CompletionResponse(
            id="cmpl-demo",
            created=123,
            model="demo-model",
            choices=[
                CompletionChoice(
                    text="world",
                    finish_reason="stop_token",
                    logprobs=CompletionLogprobs(
                        tokens=["world"],
                        token_logprobs=[-0.25],
                        top_logprobs=[{"world": -0.25}],
                        text_offset=[0],
                    ),
                )
            ],
        ),
    )

    assert len(chunks) == 3
    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.content == "world"
    assert chunks[1].choices[0].logprobs is not None
    assert chunks[2].choices[0].finish_reason == "stop"


def test_chat_stream_builder_matches_openai_chunk_shape_for_tool_calls() -> None:
    request = ChatCompletionRequest(
        model="demo-model",
        messages=[ChatCompletionMessage(role="user", content="Lookup weather for Hangzhou")],
        tools=[
            ChatTool(
                function=ChatToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            )
        ],
    )
    prepared = prepare_chat_completion_request(request)

    chunks = build_chat_completion_stream_responses(
        request,
        prepared,
        CompletionResponse(
            id="cmpl-demo",
            created=123,
            model="demo-model",
            choices=[
                CompletionChoice(
                    text=(
                        "```json\n"
                        + json.dumps(
                            {
                                "type": "tool_calls",
                                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Hangzhou"}}],
                            },
                            ensure_ascii=False,
                        )
                        + "\n```"
                    ),
                    finish_reason="stop_token",
                )
            ],
        ),
    )

    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.role == "assistant"
    tool_calls = chunks[1].choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Hangzhou"}'
    assert chunks[2].choices[0].finish_reason == "tool_calls"


def test_sse_payload_encoder_emits_done_marker() -> None:
    payloads = list(
        iter_sse_payloads(
            [
                CompletionResponse(
                    id="cmpl-demo",
                    created=123,
                    model="demo-model",
                    choices=[CompletionChoice(text="hello")],
                ),
                "[DONE]",
            ]
        )
    )

    assert payloads[0].decode("utf-8").startswith('data: {"id":"cmpl-demo"')
    assert payloads[1].decode("utf-8") == "data: [DONE]\n\n"
    assert encode_sse_comment("ping").decode("utf-8") == ": ping\n\n"


def test_remote_backend_uses_chat_completions_for_generate_and_caches_choice_scoring(monkeypatch) -> None:
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
        if url.endswith("/completions"):
            return {
                "choices": [
                    {
                        "text": "answer",
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
            top_k=42,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.95,
            stop_tokens=(0,),
            ban_tokens=(123,),
            pad_zero=False,
            no_penalty_token_ids=(33, 10),
        ),
        batch_size=1,
        prefill_chunk_size=64,
        show_progress=False,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "answer"
    assert outputs[0].finish_reason == "stop_token"
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["messages"] == [{"role": "user", "content": "prompt"}]
    assert "prompt" not in calls[0][1]
    assert calls[0][1]["temperature"] == 0.3
    assert calls[0][1]["top_p"] == 0.8
    assert calls[0][1]["presence_penalty"] == 0.1
    assert calls[0][1]["frequency_penalty"] == 0.2
    assert calls[0][1]["stream"] is False
    for private_key in (
        "top_k",
        "penalty_decay",
        "stop_tokens",
        "ban_tokens",
        "pad_zero",
        "no_penalty_token_ids",
        "prefill_chunk_size",
    ):
        assert private_key not in calls[0][1]

    with pytest.raises(NotImplementedError):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])
    with pytest.raises(NotImplementedError):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    choice_scoring_calls = [
        payload for _url, payload in calls if payload.get("candidate_token_texts") == [" A", " B"]
    ]
    assert len(choice_scoring_calls) == 1


def test_remote_backend_falls_back_to_text_completions_for_generate(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/chat/completions"):
            raise RemoteHTTPError(404, "missing")
        if url.endswith("/completions"):
            return {
                "choices": [
                    {
                        "text": "legacy answer",
                        "finish_reason": "stop",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(max_generate_tokens=4, temperature=0.0, top_p=0.8),
        batch_size=1,
        show_progress=False,
    )

    assert outputs[0].text == "legacy answer"
    assert calls[0][0] == "http://127.0.0.1:19081/openai/v1/chat/completions"
    assert calls[0][1]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0][1]["temperature"] == 0.0
    assert calls[0][1]["top_p"] == 0.8
    assert calls[1][0] == "http://127.0.0.1:19081/openai/v1/completions"
    assert calls[1][1]["prompt"] == "prompt"


def test_remote_backend_vllm_protocol_uses_standard_chat_without_completion_fallback(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            prefer_chat_completions=False,
            protocol="vllm",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/chat/completions"):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "vllm answer"},
                        "finish_reason": "stop",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(
            max_generate_tokens=4,
            temperature=0.0,
            top_k=42,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.95,
            stop_tokens=(0,),
            ban_tokens=(123,),
            pad_zero=False,
            no_penalty_token_ids=(33, 10),
        ),
        batch_size=1,
        prompt_stop_suffixes=[(" END",)],
        prompt_seeds=[123],
        prefill_chunk_size=64,
        show_progress=False,
    )

    assert outputs[0].text == "vllm answer"
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:19081/openai/v1/chat/completions"
    payload = calls[0][1]
    assert payload == {
        "model": "remote-demo",
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
        "top_p": 0.8,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "stop": [" END"],
    }


def test_remote_backend_openai_omit_seed_policy_drops_prompt_seed(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            protocol="openai",
            seed_policy="omit-for-contents",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/chat/completions"):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "seedless answer"},
                        "finish_reason": "stop",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(max_generate_tokens=4, temperature=0.0, top_p=0.8),
        batch_size=1,
        prompt_seeds=[123],
        show_progress=False,
    )

    assert outputs[0].text == "seedless answer"
    assert len(calls) == 1
    payload = calls[0][1]
    assert calls[0][0] == "http://127.0.0.1:19081/openai/v1/chat/completions"
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert "seed" not in payload


def test_remote_backend_completions_protocol_preserves_raw_prompt_private_fields_and_omits_seed(
    monkeypatch,
) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            prefer_chat_completions=True,
            protocol="completions",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/completions"):
            return {
                "choices": [
                    {
                        "text": "raw answer",
                        "finish_reason": "stop",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["User: solve\n\nAssistant: <think>"],
        sampling=SamplingConfig(
            max_generate_tokens=4,
            temperature=0.0,
            top_k=42,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.95,
            stop_tokens=(0,),
            ban_tokens=(123,),
            pad_zero=False,
            no_penalty_token_ids=(33, 10),
        ),
        batch_size=1,
        prompt_stop_suffixes=[(" END",)],
        prompt_seeds=[123],
        prefill_chunk_size=64,
        show_progress=False,
    )

    assert outputs[0].text == "raw answer"
    assert outputs[0].finish_reason == "stop_token"
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:19081/openai/v1/completions"
    payload = calls[0][1]
    assert payload["prompt"] == "User: solve\n\nAssistant: <think>"
    assert "messages" not in payload
    assert "seed" not in payload
    assert payload["top_k"] == 42
    assert payload["stop"] == [" END"]
    assert payload["stop_tokens"] == [0]
    assert payload["ban_tokens"] == [123]
    assert payload["pad_zero"] is False
    assert payload["no_penalty_token_ids"] == [33, 10]
    assert payload["prefill_chunk_size"] == 64


def test_remote_backend_vllm_protocol_rejects_private_choice_scoring(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            protocol="vllm",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        RemoteInferenceBackend,
        "_post_json",
        lambda self, url, payload: calls.append((url, payload)) or {},
    )

    with pytest.raises(NotImplementedError, match="candidate choice scoring"):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    assert calls == []


def test_remote_backend_choice_scoring_unsupported_500_falls_back(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        raise RemoteHTTPError(500, '{"detail":"nano-vLLM backend does not support candidate choice scoring"}')

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    with pytest.raises(NotImplementedError, match="candidate choice scoring"):
        backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    assert backend._legacy_choice_scoring_supported is False
    assert len(calls) == 1


def test_remote_backend_completions_protocol_supports_choice_scoring(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            protocol="completions",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        RemoteInferenceBackend,
        "_post_json",
        lambda self, url, payload: calls.append((url, payload))
        or {
            "choices": [
                {
                    "text": " B",
                    "logprobs": {
                        "tokens": [" B"],
                        "token_logprobs": [-0.1],
                        "top_logprobs": [{" A": -2.0, " B": -0.1}],
                    },
                }
            ]
        },
    )

    scores, best_text = backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    assert best_text == " B"
    assert scores[" B"] > scores[" A"]
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:19081/openai/v1/completions"
    assert calls[0][1]["candidate_token_texts"] == [" A", " B"]


def test_remote_backend_lightning_uses_v2_contents_and_choice_logits(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081/openai",
            model="remote-demo",
            api_key="pw",
            protocol="lightning",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/v2/chat/completions"):
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer-a"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "answer-b"},
                        "finish_reason": "stop",
                    },
                ]
            }
        if url.endswith("/v1/choice_logits"):
            return {
                "model": "remote-demo",
                "choice_logits": {
                    "choice_0": -2.0,
                    "choice_1": 3.0,
                },
                "choice_probabilities": {
                    "choice_0": 0.01,
                    "choice_1": 0.99,
                },
                "best_choice": "choice_1",
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["raw prompt A", "raw prompt B"],
        sampling=SamplingConfig(
            max_generate_tokens=4,
            temperature=0.0,
            top_k=42,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.95,
            stop_tokens=(0,),
            ban_tokens=(123,),
            pad_zero=False,
            no_penalty_token_ids=(33, 10),
        ),
        batch_size=2,
        prompt_seeds=[123, 456],
        prefill_chunk_size=64,
        show_progress=False,
    )
    scores, best_text = backend.score_choice_tokens(prompt="question", choice_token_texts=[" A", " B"])

    assert [output.text for output in outputs] == ["answer-a", "answer-b"]
    assert best_text == " B"
    assert scores == {" A": -2.0, " B": 3.0}
    assert calls[0][0] == "http://127.0.0.1:19081/v2/chat/completions"
    assert calls[0][1]["contents"] == ["raw prompt A", "raw prompt B"]
    assert calls[0][1]["top_k"] == 42
    assert calls[0][1]["chunk_size"] == 64
    assert calls[0][1]["ban_tokens"] == [123]
    assert calls[0][1]["password"] == "pw"
    assert "messages" not in calls[0][1]
    assert "seed" not in calls[0][1]
    assert calls[1][0] == "http://127.0.0.1:19081/v1/choice_logits"
    assert calls[1][1] == {
        "model": "remote-demo",
        "prompt": "question",
        "choices": {"choice_0": " A", "choice_1": " B"},
        "temperature": 1.0,
        "use_prefix_cache": False,
        "password": "pw",
    }


def test_remote_backend_lightning_high_throughput_uses_batch_endpoint(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="http://127.0.0.1:19081",
            model="remote-demo",
            protocol="lightning-high-throughput",
            seed_policy="omit-for-contents",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        if url.endswith("/high_throughput/chat/completions"):
            return {
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": f"answer-{index}"},
                        "finish_reason": "stop",
                    }
                    for index, _prompt in enumerate(payload["contents"])  # type: ignore[index]
                ]
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["raw prompt A", "raw prompt B"],
        sampling=SamplingConfig(max_generate_tokens=4, temperature=0.0),
        batch_size=2,
        show_progress=False,
    )

    assert [output.text for output in outputs] == ["answer-0", "answer-1"]
    assert calls[0][0] == "http://127.0.0.1:19081/high_throughput/chat/completions"
    assert calls[0][1]["contents"] == ["raw prompt A", "raw prompt B"]
    assert calls[0][1]["max_batch_size"] == 2


def test_remote_backend_legacy_nano_single_requests_keep_private_fields(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
            protocol="nano-vllm-contents",
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((url, payload))
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(
            max_generate_tokens=4,
            temperature=0.3,
            top_k=42,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.95,
            stop_tokens=(0,),
            ban_tokens=(123,),
            pad_zero=False,
            no_penalty_token_ids=(33, 10),
        ),
        batch_size=1,
        prompt_seeds=[123],
        prefill_chunk_size=64,
        show_progress=False,
    )

    assert outputs[0].text == "answer"
    payload = calls[0][1]
    assert payload["top_k"] == 42
    assert payload["penalty_decay"] == 0.95
    assert payload["stop_tokens"] == ["0"]
    assert payload["ban_tokens"] == [123]
    assert payload["pad_zero"] is False
    assert payload["no_penalty_token_ids"] == [33, 10]
    assert payload["prefill_chunk_size"] == 64


def test_remote_backend_contents_protocol_pipelines_batches_by_worker_budget(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:19081",
            model="remote-demo",
            api_key="pw",
            protocol="lightning",
            max_workers=4,
        )
    )
    calls: list[list[str]] = []

    def _fake_post_json(_url: str, payload: dict[str, object]) -> dict[str, object]:
        contents = [str(item) for item in payload["contents"]]  # type: ignore[index]
        calls.append(contents)
        return {
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": f"answer-{prompt}"},
                    "finish_reason": "stop",
                }
                for index, prompt in enumerate(contents)
            ]
        }

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))

    assert remote_contents_inflight_batches(backend, 2) == 2
    assert resolve_generation_prompt_batch_size(backend, 2) == 4

    outputs = backend.generate(
        ["a", "b", "c", "d"],
        sampling=SamplingConfig(max_generate_tokens=4, temperature=0.0),
        batch_size=2,
        show_progress=False,
    )

    assert [output.text for output in outputs] == ["answer-a", "answer-b", "answer-c", "answer-d"]
    assert sorted(calls) == [["a", "b"], ["c", "d"]]


def test_remote_backend_tqdm_cleanup_errors_do_not_fail_generation(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
        )
    )

    def _fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        del payload
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

    class _BrokenTqdm:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            del args, kwargs

        def update(self, amount: int) -> None:
            del amount
            raise AttributeError("'tqdm' object has no attribute 'sp'")

        def close(self) -> None:
            raise AttributeError("'tqdm' object has no attribute 'sp'")

    monkeypatch.setattr(RemoteInferenceBackend, "_post_json", lambda self, url, payload: _fake_post_json(url, payload))
    monkeypatch.setattr("src.infer.backend.tqdm", _BrokenTqdm)

    outputs = backend.generate(
        ["prompt"],
        sampling=SamplingConfig(max_generate_tokens=4, temperature=0.0, top_p=0.8),
        batch_size=1,
        show_progress=True,
    )

    assert outputs[0].text == "answer"


def test_remote_backend_rejects_prompt_constraints_in_strict_mode() -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
        )
    )

    with pytest.raises(NotImplementedError, match="does not support prompt constraints"):
        backend.generate(
            ["prompt"],
            sampling=SamplingConfig(max_generate_tokens=4),
            batch_size=1,
            constraints=[LiteralChoiceConstraint(("TOOL", "ASK", "HANDOFF"))],
            constraint_mode="strict",
            show_progress=False,
        )


def test_remote_backend_reuses_pooled_http_client(monkeypatch) -> None:
    created_clients = []

    class _FakeResponse:
        status_code = 200
        content = b'{"choices":[{"text":"ok","finish_reason":"stop"}]}'
        text = content.decode("utf-8")

    class _FakeHTTPClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.posts = []
            self.closed = False
            created_clients.append(self)

        def post(self, url, *, content, headers, timeout):
            self.posts.append((url, content, headers, timeout))
            return _FakeResponse()

        def close(self):
            self.closed = True

    monkeypatch.setattr("src.infer.backend.httpx.Client", _FakeHTTPClient)
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
            max_workers=8,
        )
    )

    first = backend._post_json("http://127.0.0.1:8081/v1/completions", {"model": "remote-demo"})
    second = backend._post_json("http://127.0.0.1:8081/v1/completions", {"model": "remote-demo"})

    assert first["choices"][0]["text"] == "ok"
    assert second["choices"][0]["text"] == "ok"
    assert len(created_clients) == 1
    assert len(created_clients[0].posts) == 2
    assert created_clients[0].kwargs["follow_redirects"] is True

    backend.shutdown()

    assert created_clients[0].closed is True
    assert backend._http_client is None


def test_remote_backend_retries_transient_disconnect(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
            max_retries=2,
            retry_initial_delay_s=0.0,
        )
    )
    calls = 0

    class _FakeResponse:
        status_code = 200
        content = b'{"choices":[{"text":"ok","finish_reason":"stop"}]}'
        text = content.decode("utf-8")

    class _FakeHTTPClient:
        def post(self, url, *, content, headers, timeout):
            nonlocal calls
            del url, content, headers, timeout
            calls += 1
            if calls == 1:
                raise httpx.TransportError("closed")
            return _FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr("src.infer.backend.httpx.Client", lambda **_kwargs: _FakeHTTPClient())

    response = backend._post_json("http://127.0.0.1:8081/v1/completions", {"model": "remote-demo"})

    assert calls == 2
    assert response["choices"][0]["text"] == "ok"


def test_remote_backend_retries_retryable_http_status(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
            max_retries=2,
            retry_initial_delay_s=0.0,
        )
    )
    calls = 0

    class _FakeResponse:
        def __init__(self, status_code: int, content: bytes) -> None:
            self.status_code = status_code
            self.content = content
            self.text = content.decode("utf-8")

    class _FakeHTTPClient:
        def post(self, url, *, content, headers, timeout):
            nonlocal calls
            del url, content, headers, timeout
            calls += 1
            if calls == 1:
                return _FakeResponse(
                    502,
                    b'{"detail":"backend request failed: Server disconnected without sending a response."}',
                )
            return _FakeResponse(200, b'{"choices":[{"text":"ok","finish_reason":"stop"}]}')

        def close(self):
            return None

    monkeypatch.setattr("src.infer.backend.httpx.Client", lambda **_kwargs: _FakeHTTPClient())

    response = backend._post_json("http://127.0.0.1:8081/v1/completions", {"model": "remote-demo"})

    assert calls == 2
    assert response["choices"][0]["text"] == "ok"


def test_remote_backend_does_not_retry_non_retryable_http_status(monkeypatch) -> None:
    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url="127.0.0.1:8081",
            model="remote-demo",
            max_retries=2,
            retry_initial_delay_s=0.0,
        )
    )
    calls = 0

    class _FakeResponse:
        status_code = 404
        content = b'{"detail":"missing"}'
        text = content.decode("utf-8")

    class _FakeHTTPClient:
        def post(self, url, *, content, headers, timeout):
            nonlocal calls
            del url, content, headers, timeout
            calls += 1
            return _FakeResponse()

        def close(self):
            return None

    monkeypatch.setattr("src.infer.backend.httpx.Client", lambda **_kwargs: _FakeHTTPClient())

    with pytest.raises(RemoteHTTPError, match="HTTP 404"):
        backend._post_json("http://127.0.0.1:8081/v1/completions", {"model": "remote-demo"})

    assert calls == 1


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


def test_normalize_local_device_promotes_bare_cuda_to_index_zero() -> None:
    assert normalize_local_device("cuda") == "cuda:0"
    assert normalize_local_device("cuda:1") == "cuda:1"
    assert normalize_local_device("cpu") == "cpu"


def test_local_inference_backend_normalizes_bare_cuda_before_model_load(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, token_ids) -> str:
            return ""

    class _FakeEngine:
        def generate(self, prompts, **_kwargs):
            return []

        def shutdown(self) -> None:
            return None

    def _fake_load_rwkv_model(config):
        captured["device"] = config.device
        return object(), _FakeTokenizer()

    fake_model_module = ModuleType("src.infer.model")
    fake_model_module.load_rwkv_model = _fake_load_rwkv_model
    monkeypatch.setitem(sys.modules, "src.infer.model", fake_model_module)
    monkeypatch.setattr("src.infer.backend.build_local_engine", lambda *_args, **_kwargs: _FakeEngine())

    _ = LocalInferenceBackend.from_model_config(
        SimpleNamespace(weights_path="/tmp/demo-model.pth", device="cuda"),
        engine_mode="classic",
    )

    assert captured["device"] == "cuda:0"
