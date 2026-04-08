from __future__ import annotations

import json

from src.eval.knowledge.pipeline import MultipleChoicePipeline
from src.infer.sampling import GenerationOutput


class _FallbackOnlyBackend:
    def __init__(self) -> None:
        self.model_name = "remote-openai"
        self.generate_calls: list[list[str]] = []

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
        return [
            GenerationOutput(
                prompt_index=0,
                prompt=str(prompts[0]),
                token_ids=[],
                text=" B",
                finish_reason="stop_token",
            )
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
    assert backend.generate_calls and "Therefore, the answer is" in backend.generate_calls[0][0]
