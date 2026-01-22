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
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state

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
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        templates = _select_prompt_templates(dataset_name)
        if prompt_template is None:
            prompt_template = templates.direct

        resume = detect_resume_state(output_path, repeats=1)
        start_index = min(resume.next_index, len(records))
        if start_index and len(records):
            remaining = max(len(records) - start_index, 0)
            print(f"⏩ 多选 Direct 恢复运行：已完成 {start_index}/{len(records)}，剩余 {remaining}")
        indexed_records = list(enumerate(records[start_index:], start=start_index))
        if not indexed_records:
            return MultipleChoicePipelineResult(dataset_name, len(records), Path(output_path))

        writer = JsonlStageWriter(output_path, resume=resume.has_progress)
        for idx, record in indexed_records:
            prompt = self._format_prompt(record, prompt_template)
            _, pred_letter = self._score_prompt(record, prompt)
            token_text = self.target_token_format.replace("<LETTER>", pred_letter)
            stages = [
                StageRecord(
                    prompt=prompt,
                    completion=token_text,
                    stop_reason="logits_only",
                )
            ]
            writer.write(
                SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=idx,
                    repeat_index=0,
                    sampling_config={},
                    stages=stages,
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
        cot_sampling: SamplingConfig,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        min_prompt_count: int | None = None,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> MultipleChoicePipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        templates = _select_prompt_templates(dataset_name)
        write_output = write_output and (not probe_only)
        batch_size = max(1, int(batch_size))
        if min_prompt_count and min_prompt_count > len(records) and records:
            repeats = (min_prompt_count + len(records) - 1) // len(records)
            records = (records * repeats)[:min_prompt_count]
        if cot_prompt_template is None:
            cot_prompt_template = templates.cot
        if final_answer_template is None:
            final_answer_template = templates.final

        if probe_only and records:
            if len(records) >= batch_size:
                records = records[:batch_size]
            else:
                repeat = (batch_size + len(records) - 1) // len(records)
                records = (records * repeat)[:batch_size]

        if write_output:
            resume = detect_resume_state(output_path, repeats=1)
            start_index = min(resume.next_index, len(records))
            if start_index and len(records):
                remaining = max(len(records) - start_index, 0)
                print(f"⏩ 多选 CoT 恢复运行：已完成 {start_index}/{len(records)}，剩余 {remaining}")
        else:
            start_index = 0
            resume = None
        remaining_records = records[start_index:]
        if not remaining_records:
            return MultipleChoicePipelineResult(dataset_name, len(records), Path(output_path))

        prompts = [self._format_prompt(record, cot_prompt_template) for record in remaining_records]
        outputs = self.engine.generate(
            prompts,
            sampling=cot_sampling,
            batch_size=batch_size,
            progress_desc="Generating CoT" if not probe_only else "Probing CoT",
            probe_only=probe_only,
        )
        if probe_only or not write_output:
            return MultipleChoicePipelineResult(dataset_name, len(records), Path(output_path))

        writer = JsonlStageWriter(output_path, resume=bool(resume and resume.has_progress))
        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling)])
        cot_by_idx = {item.prompt_index: item for item in outputs}
        for local_idx, record in enumerate(remaining_records):
            global_idx = start_index + local_idx
            cot_seq = cot_by_idx.get(local_idx)
            if cot_seq is None:
                continue
            cot_prompt = prompts[local_idx]
            cot_stage = StageRecord(
                prompt=cot_prompt,
                completion=cot_seq.text,
                stop_reason=cot_seq.finish_reason,
            )
            final_prompt = (
                (final_answer_template or EN_FINAL_ANSWER_TEMPLATE)
                .replace("<Q>", cot_prompt)
                .replace("<COT>", cot_seq.text)
            )
            _, pred_letter = self._score_prompt(record, final_prompt)
            prior_context = f"{cot_prompt}{cot_seq.text}"
            delta_prompt = prompt_delta(final_prompt, prior_context)
            token_text = self.target_token_format.replace("<LETTER>", pred_letter)
            final_stage = StageRecord(
                prompt=delta_prompt,
                completion=token_text,
                stop_reason="logits_only",
            )
            writer.write(
                SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=global_idx,
                    repeat_index=0,
                    sampling_config=sampling_config,
                    stages=[cot_stage, final_stage],
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
