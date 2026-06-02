from __future__ import annotations

from src.eval.evaluators.free_response import FreeResponsePipeline, USER_SENTINEL
from src.infer.sampling import GenerationOutput, SamplingConfig


def test_free_response_pipeline_generates_one_stage_and_clips_user_sentinel(tmp_path) -> None:
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"What is 6+1?","answer":"7"}\n', encoding="utf-8")
    pipeline = object.__new__(FreeResponsePipeline)
    pipeline.engine = _FakeEngine()

    records: list[dict] = []
    result = pipeline.run(
        dataset_path=str(dataset),
        prompt_template="User: <Q>\n\nAssistant: <think",
        generation_sampling=SamplingConfig(stop_tokens=(0, 261, 24281)),
        batch_size=4,
        pass_k=(1,),
        on_record=records.append,
    )

    assert result.sample_count == 1
    assert len(result.payloads) == 1
    assert records == result.payloads
    payload = result.payloads[0]
    assert payload["prompt1"].endswith("Assistant: <think")
    assert payload["completion1"] == "answer before sentinel"
    assert payload["stop_reason1"] == "stop_condition"
    assert payload["stats"] == {
        "truncated": False,
        "stop_detail": "user_sentinel",
        "generated_token_count": 3,
    }
    assert payload["sampling_config"]["stage1"]["stop_tokens"] == [0]
    assert "prompt2" not in payload
    assert "completion2" not in payload
    assert pipeline.engine.calls == [
        {
            "sampling_stop_tokens": (0,),
            "prompt_stop_suffixes": [(USER_SENTINEL,)],
            "probe_only": False,
        }
    ]


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompts, *, sampling, prompt_stop_suffixes=None, probe_only=False, on_complete=None, **_kwargs):
        self.calls.append(
            {
                "sampling_stop_tokens": sampling.stop_tokens,
                "prompt_stop_suffixes": prompt_stop_suffixes,
                "probe_only": probe_only,
            }
        )
        outputs = [
            GenerationOutput(
                prompt_index=idx,
                prompt=prompt,
                token_ids=[1, 2, 3],
                text=f"answer before sentinel{USER_SENTINEL} ignored",
                finish_reason="stop_condition",
                finish_detail="user_sentinel",
                truncated=False,
            )
            for idx, prompt in enumerate(prompts)
        ]
        if on_complete is not None:
            for output in outputs:
                on_complete(output)
        return outputs
