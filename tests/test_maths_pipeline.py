from __future__ import annotations

from src.eval.tasks.maths.pipeline import FREE_RESPONSE_STOP_TOKENS, FreeResponsePipeline
from src.infer.sampling import GenerationOutput, SamplingConfig


def test_free_response_pipeline_generates_single_full_response_stage(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["sampling"].stop_tokens == FREE_RESPONSE_STOP_TOKENS
    assert call["prompt_stop_suffixes"] == [("\nUser:",)]
    assert call["prompts"] == ["User: solve\n2+5?\n\nAssistant: <think"]
    assert result.payloads == [
        {
            "benchmark_name": "math",
            "dataset_split": "test",
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "sampling_config": {
                "stage1": {
                    "max_new_tokens": 32,
                    "temperature": 0.3,
                    "top_k": 50,
                    "top_p": 0.3,
                    "presence_penalty": 0.5,
                    "repetition_penalty": 0.5,
                    "penalty_decay": 0.99,
                    "stop_tokens": [0],
                    "ban_tokens": None,
                    "pad_zero": True,
                    "no_penalty_token_ids": [33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
                }
            },
            "prompt1": "User: solve\n2+5?\n\nAssistant: <think",
            "completion1": "</think>\nTherefore, the final answer is \\(\\boxed{7}\\).",
            "stop_reason1": "stop_token",
            "stats": {
                "truncated": False,
                "stop_detail": "stop_token",
                "generated_token_count": 1,
            },
            "_stage": "answer",
        }
    ]


def test_free_response_pipeline_clamps_rendered_prompt_chars(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text(
        '{"question":"prefix ' + ("context " * 200) + ' final question?","answer":"7"}\n',
        encoding="utf-8",
    )
    backend = _FakeBackend()
    pipeline = FreeResponsePipeline(backend)

    pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
        prompt_max_chars=220,
    )

    prompt = backend.calls[0]["prompts"][0]
    assert len(prompt) <= 220
    assert "prefix context" in prompt
    assert "[...truncated...]" in prompt
    assert "final question?" in prompt
    assert prompt.endswith("Assistant: <think")


def test_free_response_pipeline_generates_cot_then_final_answer(tmp_path) -> None:
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"2+5?","answer":"7"}\n', encoding="utf-8")
    backend = _TwoStageFakeBackend()
    pipeline = FreeResponsePipeline(backend)

    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: solve\n<Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(max_generate_tokens=32, stop_tokens=(0, 261)),
        final_answer_template="<Q><COT>\nTherefore, the answer is \\(\\boxed{",
        final_sampling=SamplingConfig(max_generate_tokens=8, temperature=1.0, top_p=0.3, stop_tokens=(0, 2402)),
        batch_size=4,
        dataset_name="math_test",
        pass_k=(1,),
        samples_per_task=1,
    )

    assert len(backend.calls) == 2
    assert backend.calls[0]["sampling"].stop_tokens == FREE_RESPONSE_STOP_TOKENS
    assert backend.calls[1]["sampling"].stop_tokens == (0, 2402)
    assert backend.calls[1]["prompts"] == [
        "User: solve\n2+5?\n\nAssistant: <think</think>\nwork\nTherefore, the answer is \\(\\boxed{"
    ]
    assert result.payloads[0]["prompt1"] == "User: solve\n2+5?\n\nAssistant: <think"
    assert result.payloads[0]["completion1"] == "</think>\nwork"
    assert result.payloads[0]["prompt2"] == "\nTherefore, the answer is \\(\\boxed{"
    assert result.payloads[0]["completion2"] == "7}\\)."
    assert result.payloads[0]["sampling_config"]["stage2"]["max_new_tokens"] == 8
    assert result.payloads[0]["stats"]["stage1"]["generated_token_count"] == 1
    assert result.payloads[0]["stats"]["stage2"]["generated_token_count"] == 2
    assert result.payloads[0]["_stage"] == "answer"


class _FakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=[1],
                text="</think>\nTherefore, the final answer is \\(\\boxed{7}\\).\nUser: next",
                finish_reason="stop_token",
            )
            for idx, prompt in enumerate(prompts)
        ]
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError


class _TwoStageFakeBackend:
    model_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        is_final = "Therefore, the answer is" in prompts[0]
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=[2, 3] if is_final else [1],
                text="7}\\).\nUser: next" if is_final else "</think>\nwork",
                finish_reason="stop_token",
            )
            for idx, prompt in enumerate(prompts)
        ]
        on_complete = kwargs.get("on_complete")
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError
