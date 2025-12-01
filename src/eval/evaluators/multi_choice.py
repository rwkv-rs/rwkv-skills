from __future__ import annotations

"""Multiple-choice 评估流水线：负责读数据、跑模型、导出阶段化 JSONL。"""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.datasets.data_struct.multiple_choice import MultipleChoiceRecord
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_TOKEN_FORMAT = " <LETTER>"

EN_DIRECT_PROMPT_TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
<CHOICES>

Assistant: The answer is"""

EN_COT_PROMPT_TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
<CHOICES>

Assistant: <think"""

EN_FINAL_ANSWER_TEMPLATE = """<Q><COT>
Therefore, the answer is"""

ZH_DIRECT_PROMPT_TEMPLATE = """User: <Q>
<CHOICES>

Assistant: 正确答案是"""

ZH_COT_PROMPT_TEMPLATE = """User: <Q>
<CHOICES>

Assistant:"""

ZH_FINAL_ANSWER_TEMPLATE = """<Q><COT>
综上所述，答案是"""

COT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.5,
    top_k=50,
    top_p=0.3,
    alpha_presence=1.0,
    alpha_frequency=0.1,
    alpha_decay=0.99,
    stop_tokens=(0, 261, 24281),
)


@dataclass(frozen=True)
class PromptTemplates:
    direct: str
    cot: str
    final: str


def _select_prompt_templates(dataset_name: str | None) -> PromptTemplates:
    if dataset_name:
        stem = dataset_name.lower()
        if any(token in stem for token in ("ceval", "zh", "cn")):
            return PromptTemplates(
                ZH_DIRECT_PROMPT_TEMPLATE,
                ZH_COT_PROMPT_TEMPLATE,
                ZH_FINAL_ANSWER_TEMPLATE,
            )
    return PromptTemplates(
        EN_DIRECT_PROMPT_TEMPLATE,
        EN_COT_PROMPT_TEMPLATE,
        EN_FINAL_ANSWER_TEMPLATE,
    )


@dataclass(slots=True)
class MultipleChoicePipelineResult:
    dataset: str
    sample_count: int
    output_path: Path


class MultipleChoicePipeline:
    """Wraps direct & CoT 流程，导出阶段化 JSONL，方便后续 metrics 消费。"""

    def __init__(self, model_config: ModelLoadConfig, target_token_format: str = TARGET_TOKEN_FORMAT) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.target_token_format = target_token_format
        self._choice_token_cache: dict[int, list[int]] = {}

    def run_direct(
        self,
        dataset_path: str,
        output_path: str,
        *,
        prompt_template: str | None = None,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
    ) -> MultipleChoicePipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        templates = _select_prompt_templates(dataset_name)
        if prompt_template is None:
            prompt_template = templates.direct

        writer = JsonlStageWriter(output_path)
        for idx, record in enumerate(records):
            prompt = self._format_prompt(record, prompt_template)
            logits_map, pred_letter = self._score_prompt(record, prompt)
            stages = [
                StageRecord(
                    prompt=prompt,
                    logits=logits_map,
                    finish_reason="logits_only",
                )
            ]
            metadata = {
                "question": record.question,
                "choices": {ALPHABET[i]: text for i, text in enumerate(record.choices)},
                "answer": ALPHABET[record.answer_index],
                "predicted": pred_letter,
                "subject": record.subject,
                "correct": pred_letter == ALPHABET[record.answer_index],
            }
            writer.write(
                SampleRecord(
                    index=idx,
                    dataset=dataset_name,
                    stages=stages,
                    metadata=metadata,
                )
            )
        writer.close()
        return MultipleChoicePipelineResult(dataset_name, len(records), Path(output_path))

    def run_chain_of_thought(
        self,
        dataset_path: str,
        output_path: str,
        *,
        cot_prompt_template: str | None = None,
        final_answer_template: str | None = None,
        cot_sampling: SamplingConfig = COT_SAMPLING,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
    ) -> MultipleChoicePipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        templates = _select_prompt_templates(dataset_name)
        if cot_prompt_template is None:
            cot_prompt_template = templates.cot
        if final_answer_template is None:
            final_answer_template = templates.final

        cot_prompts = [self._format_prompt(record, cot_prompt_template) for record in records]
        if not cot_prompts:
            return MultipleChoicePipelineResult(dataset_name, 0, Path(output_path))

        outputs = self.engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=max(1, min(batch_size, len(cot_prompts))),
            progress_desc="Generating CoT",
        )
        cot_by_idx = {item.prompt_index: item for item in outputs}

        writer = JsonlStageWriter(output_path)
        for idx, record in enumerate(records):
            cot_seq = cot_by_idx.get(idx)
            if cot_seq is None:
                continue
            cot_prompt = cot_prompts[idx]
            cot_stage = StageRecord(
                prompt=cot_prompt,
                output=cot_seq.text,
                finish_reason=cot_seq.finish_reason,
            )
            final_prompt = (
                (final_answer_template or EN_FINAL_ANSWER_TEMPLATE)
                .replace("<Q>", cot_prompt)
                .replace("<COT>", cot_seq.text)
            )
            logits_map, pred_letter = self._score_prompt(record, final_prompt)
            final_stage = StageRecord(
                prompt=final_prompt,
                logits=logits_map,
                finish_reason="logits_only",
            )
            metadata = {
                "question": record.question,
                "choices": {ALPHABET[i]: text for i, text in enumerate(record.choices)},
                "answer": ALPHABET[record.answer_index],
                "predicted": pred_letter,
                "subject": record.subject,
                "correct": pred_letter == ALPHABET[record.answer_index],
            }
            writer.write(
                SampleRecord(
                    index=idx,
                    dataset=dataset_name,
                    stages=[cot_stage, final_stage],
                    metadata=metadata,
                )
            )
        writer.close()
        return MultipleChoicePipelineResult(dataset_name, len(records), Path(output_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[MultipleChoiceRecord], str]:
        loader = JsonlMultipleChoiceLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        dataset_name = infer_dataset_slug_from_path(dataset_path)
        return records, dataset_name

    def _format_prompt(self, record: MultipleChoiceRecord, template: str) -> str:
        subject = (record.subject or "unknown").replace("_", " ")
        choice_lines = [f"{ALPHABET[i]}. {choice}" for i, choice in enumerate(record.choices)]
        return (
            template.replace("<SUBJECT>", subject)
            .replace("<Q>", record.question)
            .replace("<CHOICES>", "\n".join(choice_lines))
        )

    def _choice_tokens(self, num_choices: int) -> list[int]:
        if num_choices not in self._choice_token_cache:
            tokens = []
            for letter in ALPHABET[:num_choices]:
                text = self.target_token_format.replace("<LETTER>", letter)
                token_ids = self.tokenizer.encode(text)
                if len(token_ids) != 1:
                    raise ValueError(
                        f"target token format '{self.target_token_format}' 未映射为单 token: {token_ids}"
                    )
                tokens.append(token_ids[0])
            self._choice_token_cache[num_choices] = tokens
        return self._choice_token_cache[num_choices]

    def _score_prompt(self, record: MultipleChoiceRecord, prompt: str) -> tuple[dict[str, float], str]:
        tokens = [0] + self.tokenizer.encode(prompt.strip())
        state = self._blank_state()
        logits = self.model.forward(tokens, state, full_output=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.to(torch.float32)
        choice_tokens = self._choice_tokens(len(record.choices))
        slice_values = logits[choice_tokens]
        logits_map = {
            ALPHABET[i]: float(value)
            for i, value in enumerate(slice_values.cpu())
        }
        pred_idx = torch.argmax(slice_values).item()
        return logits_map, ALPHABET[pred_idx]

    def _blank_state(self):
        try:
            return self.model.generate_zero_state(0)
        except TypeError:
            return self.model.generate_zero_state()


__all__ = ["MultipleChoicePipeline", "MultipleChoicePipelineResult"]
