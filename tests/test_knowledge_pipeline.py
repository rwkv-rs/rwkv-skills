from __future__ import annotations

import json

from src.eval.knowledge.pipeline import MultipleChoicePipeline
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.infer.sampling import GenerationOutput


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
        return [
            GenerationOutput(
                prompt_index=index,
                prompt=str(prompt),
                token_ids=[],
                text=self.text,
                finish_reason="stop_token",
            )
            for index, prompt in enumerate(prompts)
        ]

    def score_choice_tokens(self, *, prompt: str, choice_token_texts):
        raise NotImplementedError("standard chat backend has no choice logits")


def test_multiple_choice_pipeline_falls_back_to_generation_when_choice_scoring_is_unavailable(tmp_path) -> None:
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
    assert len(result.payloads) == 1
    assert result.payloads[0]["completion1"] == " B"
    assert result.payloads[0]["stop_reason1"] == "logits_only"
    assert backend.generate_calls and "Assistant: The answer is" in backend.generate_calls[0][0]


def test_multiple_choice_pipeline_batches_fallback_generation(tmp_path) -> None:
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
    assert [len(call) for call in backend.generate_calls] == [3, 2]
    assert backend.generate_batch_sizes == [3, 3]


def test_multiple_choice_pipeline_marks_invalid_fallback_generation_wrong(tmp_path) -> None:
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
