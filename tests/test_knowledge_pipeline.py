from __future__ import annotations

import json

from src.eval.tasks.knowledge.pipeline import MultipleChoicePipeline
from src.eval.metrics.multi_choice import (
    evaluate_multiple_choice,
    evaluate_multiple_choice_cascade,
    extract_answer_after_think,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig
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
        text_stop_detectors=None,
        prefill_chunk_size=16,
        show_progress=True,
    ):
        del text_stop_detectors
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


class _ScriptedBackend(_FallbackOnlyBackend):
    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self.texts = list(texts)

    def generate(self, prompts, **kwargs):
        if not self.texts:
            raise AssertionError("unexpected generation call")
        self.text = self.texts.pop(0)
        return super().generate(prompts, **kwargs)


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


def test_multiple_choice_pipeline_compacts_context_with_question_and_choices_query(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "Which color is tied to catalyst77?",
                "A": "red",
                "B": "blue",
                "C": "green",
                "D": "yellow",
                "answer": "B",
                "subject": "chemistry",
                "context": "\n".join(
                    [f"noise row {idx}" for idx in range(20)]
                    + ["catalyst77 blue pathway evidence"]
                    + [f"tail row {idx}" for idx in range(20)]
                ),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend()
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_direct(
        str(dataset_path),
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            max_chunk_chars=120,
            overlap_lines=0,
            min_long_text_chars=100,
            max_evidence_chunks=1,
            max_evidence_chars=180,
        ),
    )

    prompt = backend.generate_calls[0][0]
    assert "Context:" in prompt
    assert "Long document compacted" in prompt
    assert "catalyst77 blue pathway evidence" in prompt
    assert result.payloads[0]["long_doc"]["query_policy"] == "question_and_choices"
    assert result.payloads[0]["long_doc"]["compacted"] is True


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


def test_multiple_choice_cot_can_extract_answer_from_same_completion(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=">reasoning</think>\nFinal answer: B")
    pipeline = MultipleChoicePipeline(backend)

    result = pipeline.run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        batch_size=1,
        answer_strategy="cascade_a_b",
    )
    metrics = evaluate_multiple_choice_cascade(result.payloads, dataset_path=dataset_path)

    assert result.payloads[0]["strategy_a_completion"] == ">reasoning</think>\nFinal answer: B"
    assert "completion1" not in result.payloads[0]
    assert len(backend.generate_calls) == 1
    assert metrics.metrics_by_group["strategy_a"]["exact_accuracy"] == 1.0
    assert metrics.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0


def test_multiple_choice_cot_extracts_chinese_final_answer(tmp_path) -> None:
    dataset_path = tmp_path / "ceval_demo_test.jsonl"
    dataset_path.write_text(
        json.dumps({"question": "2+2=?", "A": "3", "B": "4", "answer": "B"}) + "\n",
        encoding="utf-8",
    )
    backend = _FallbackOnlyBackend(text=">推理过程</think>\n最终答案是 B。")

    result = MultipleChoicePipeline(backend).run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        answer_strategy="cascade_a_b",
    )

    assert result.payloads[0]["strategy_a_completion"] == ">推理过程</think>\n最终答案是 B。"
    assert len(backend.generate_calls) == 1


def test_answer_after_think_uses_first_formal_answer() -> None:
    text = ">reasoning</think>\nThe correct answer is **B. 4**\ncontinued text\nFinal answer: C"

    assert extract_answer_after_think(text, 4) == "B"
    assert extract_answer_after_think("Final answer: B", 4) == ""


def test_multiple_choice_cascade_routes_only_strategy_a_failure_to_b(tmp_path) -> None:
    dataset_path = tmp_path / "gpqa_diamond_test.jsonl"
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = _ScriptedBackend(
        [
            ">first attempt</think>\nFinal answer: A",
            ">fresh reasoning without a final answer",
            " B",
        ]
    )

    result = MultipleChoicePipeline(backend).run_chain_of_thought(
        str(dataset_path),
        cot_sampling=SamplingConfig(max_generate_tokens=32),
        answer_strategy="cascade_a_b",
    )
    metrics = evaluate_multiple_choice_cascade(result.payloads, dataset_path=dataset_path)

    assert len(backend.generate_calls) == 3
    assert metrics.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert metrics.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0
    assert metrics.metrics_by_group["strategy_b"]["rescued"] == 1.0
