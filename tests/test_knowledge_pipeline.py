from __future__ import annotations

import json

from src.eval.knowledge.pipeline import MultipleChoicePipeline
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.infer.sampling import GenerationOutput
from src.infer.sampling import SamplingConfig


class _FallbackOnlyBackend:
    def __init__(self, *, text: str = " B") -> None:
        self.model_name = "remote-openai"
        self.text = text
        self.generate_calls: list[list[str]] = []
        self.generate_batch_sizes: list[int] = []

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
        self.generate_calls.append(list(prompts))
        self.generate_batch_sizes.append(int(batch_size))
        outputs = [
            GenerationOutput(
                prompt_index=index,
                prompt=str(prompt),
                token_ids=[],
                text=self.text,
                finish_reason="stop_token",
            )
            for index, prompt in enumerate(prompts)
        ]
        if on_complete is not None and not probe_only:
            for output in outputs:
                on_complete(output)
        return outputs

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise AssertionError("multiple-choice generation should not read logits")


def test_multiple_choice_pipeline_generates_choice_by_default(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path))

    assert result.sample_count == 1
    assert result.payloads[0]["completion1"] == " B"
    assert result.payloads[0]["stop_reason1"] == "generated_choice"
    assert len(backend.generate_calls) == 1


def test_multiple_choice_pipeline_batches_generation(tmp_path) -> None:
    dataset_path = tmp_path / "cmmlu_demo_test.jsonl"
    rows = [
        {
            "question": f"{index}+1=?",
            "A": str(index),
            "B": str(index + 1),
            "C": str(index + 2),
            "D": str(index + 3),
            "answer": "B",
            "subject": "math",
        }
        for index in range(5)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path), batch_size=3)

    assert result.sample_count == 5
    assert len(result.payloads) == 5
    assert [len(call) for call in backend.generate_calls] == [5]
    assert backend.generate_batch_sizes == [3]


def test_multiple_choice_pipeline_streams_generated_payloads_in_order(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_demo_test.jsonl"
    rows = [
        {
            "question": f"{index}+1=?",
            "A": str(index),
            "B": str(index + 1),
            "C": str(index + 2),
            "D": str(index + 3),
            "answer": "B",
            "subject": "math",
        }
        for index in range(4)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)
    streamed_payloads: list[dict] = []

    result = pipeline.run_direct(str(dataset_path), batch_size=4, on_record=streamed_payloads.append)

    assert result.sample_count == 4
    assert [payload["sample_index"] for payload in result.payloads] == [0, 1, 2, 3]
    assert [payload["sample_index"] for payload in streamed_payloads] == [0, 1, 2, 3]
    assert [payload["completion1"] for payload in result.payloads] == [" B"] * 4
    assert [len(call) for call in backend.generate_calls] == [4]


def test_multiple_choice_pipeline_marks_invalid_generation_wrong(tmp_path) -> None:
    dataset_path = tmp_path / "cmmlu_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=" (1)(2)(3).\n")
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(str(dataset_path))
    metrics = evaluate_multiple_choice(result.payloads, dataset_path=dataset_path)

    assert result.sample_count == 1
    assert result.payloads[0]["completion1"] == " "
    assert metrics.accuracy == 0.0


def test_multiple_choice_cot_generates_final_answer_by_default(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_pro_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
    )

    assert result.sample_count == 1
    assert result.payloads[0]["completion2"] == " B"
    assert result.payloads[0]["stop_reason2"] == "generated_choice"
    assert len(backend.generate_calls) == 2


def test_multiple_choice_cot_streams_generated_final_answer(tmp_path) -> None:
    dataset_path = tmp_path / "mmlu_pro_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "2+2=?",
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6",
                "answer": "B",
                "subject": "math",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)
    streamed_payloads: list[dict] = []

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
        on_record=streamed_payloads.append,
    )
    metrics = evaluate_multiple_choice(result.payloads, dataset_path=dataset_path)

    assert result.sample_count == 1
    assert len(streamed_payloads) == 2
    assert streamed_payloads[0]["_stage"] == "cot"
    assert streamed_payloads[1]["_stage"] == "answer"
    assert len(result.payloads) == 1
    assert result.payloads[0]["completion2"] == " B"
    assert len(backend.generate_calls) == 2
    assert metrics.accuracy == 1.0
