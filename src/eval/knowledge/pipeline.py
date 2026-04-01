from __future__ import annotations

"""Knowledge benchmark pipeline for multiple-choice datasets."""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.datasets.data_struct.multiple_choice import MultipleChoiceRecord
from src.eval.execution_plan import AttemptKey
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGET_TOKEN_FORMAT = " <LETTER>"

EN_DIRECT_PROMPT_TEMPLATE = """User: You are a very talented expert in <SUBJECT>.
Answer this question and finish with a single option letter.
Question: <Q>
Choices:
<CHOICES>

Assistant: Therefore, the answer is"""

EN_FAKE_COT_PROMPT_TEMPLATE = """User: You are a very talented expert in <SUBJECT>.
Answer this question and finish with a single option letter.
Question: <Q>
Choices:
<CHOICES>

Assistant: <think>
</think>
Therefore, the answer is"""

EN_COT_PROMPT_TEMPLATE = """User: You are a very talented expert in <SUBJECT>.
Answer this question and finish with a single option letter.
Question: <Q>
Choices:
<CHOICES>

Assistant: <think>"""

EN_FINAL_ANSWER_TEMPLATE = """<Q><COT></think>
Therefore, the answer is"""


@dataclass(frozen=True)
class PromptTemplates:
    direct: str
    fake_cot: str
    cot: str
    final: str


def _select_prompt_templates(_dataset_name: str | None) -> PromptTemplates:
    return PromptTemplates(
        EN_DIRECT_PROMPT_TEMPLATE,
        EN_FAKE_COT_PROMPT_TEMPLATE,
        EN_COT_PROMPT_TEMPLATE,
        EN_FINAL_ANSWER_TEMPLATE,
    )


@dataclass(slots=True)
class MultipleChoicePipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict]


class MultipleChoicePipeline:
    """Wrap direct and CoT multiple-choice execution into canonical payloads."""

    def __init__(self, model_config: ModelLoadConfig, target_token_format: str = TARGET_TOKEN_FORMAT) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.target_token_format = target_token_format
        self._choice_token_cache: dict[int, list[int]] = {}
        self.model_path = model_config.weights_path

    def run_direct(
        self,
        dataset_path: str,
        *,
        prompt_template: str | None = None,
        cot_mode: CoTMode = CoTMode.NO_COT,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        samples_per_task: int = 1,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> MultipleChoicePipelineResult:
        records, resolved_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        templates = _select_prompt_templates(dataset_name)
        if prompt_template is None:
            prompt_template = templates.fake_cot if cot_mode is CoTMode.FAKE_COT else templates.direct

        skip_keys = skip_keys or set()
        if resume_start_index < 0:
            resume_start_index = 0
        record_map = {int(idx): record for idx, record in records}
        expanded: list[tuple[AttemptKey, MultipleChoiceRecord]] = []
        if attempt_keys is not None:
            for key in attempt_keys:
                record = record_map.get(int(key.sample_index))
                if record is None or key.as_tuple() in skip_keys:
                    continue
                expanded.append((key, record))
        else:
            for idx, record in records:
                for sample_id in range(max(1, int(samples_per_task))):
                    key = AttemptKey(sample_index=int(idx), repeat_index=int(sample_id), pass_index=0)
                    if key.as_tuple() in skip_keys:
                        continue
                    expanded.append((key, record))
        expanded = expanded[resume_start_index:]
        if not expanded:
            return MultipleChoicePipelineResult(dataset_name, 0, [])

        payloads: list[dict] = []
        for key, record in expanded:
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
            payload = SampleRecord(
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                sample_index=key.sample_index,
                repeat_index=key.repeat_index,
                pass_index=key.pass_index,
                sampling_config={},
                stages=stages,
            ).as_payload()
            if on_record is not None:
                on_record(payload)
            payloads.append(payload)
        return MultipleChoicePipelineResult(dataset_name, len(expanded), payloads)

    def run_chain_of_thought(
        self,
        dataset_path: str,
        *,
        cot_prompt_template: str | None = None,
        final_answer_template: str | None = None,
        cot_sampling: SamplingConfig,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        min_prompt_count: int | None = None,
        samples_per_task: int = 1,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> MultipleChoicePipelineResult:
        records, resolved_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        templates = _select_prompt_templates(dataset_name)
        batch_size = max(1, int(batch_size))
        if cot_prompt_template is None:
            cot_prompt_template = templates.cot
        if final_answer_template is None:
            final_answer_template = templates.final

        repeats = max(1, int(samples_per_task)) if not probe_only else 1
        skip_keys = skip_keys or set()
        record_map = {int(idx): record for idx, record in records}
        expanded: list[tuple[AttemptKey, MultipleChoiceRecord]] = []
        if attempt_keys is not None:
            for key in attempt_keys:
                record = record_map.get(int(key.sample_index))
                if record is None or key.as_tuple() in skip_keys:
                    continue
                expanded.append((key, record))
        else:
            for idx, record in records:
                for sample_id in range(repeats):
                    key = AttemptKey(sample_index=int(idx), repeat_index=int(sample_id), pass_index=0)
                    if key.as_tuple() in skip_keys:
                        continue
                    expanded.append((key, record))

        if min_prompt_count and min_prompt_count > len(expanded) and expanded:
            repeat = (min_prompt_count + len(expanded) - 1) // len(expanded)
            expanded = (expanded * repeat)[:min_prompt_count]

        if probe_only and expanded:
            if len(expanded) >= batch_size:
                expanded = expanded[:batch_size]
            else:
                repeat = (batch_size + len(expanded) - 1) // len(expanded)
                expanded = (expanded * repeat)[:batch_size]

        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return MultipleChoicePipelineResult(dataset_name, len(expanded), [])
            remaining_entries = [
                (key, record)
                for key, record in expanded[resume_start_index:]
            ]
            print(
                f"⏩ 多选 CoT 恢复运行：已完成 {resume_start_index}/{len(expanded)}，剩余 {len(remaining_entries)}"
            )
        else:
            remaining_entries = expanded
        if not remaining_entries:
            return MultipleChoicePipelineResult(dataset_name, 0, [])

        payloads: list[dict] = []
        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling)])
        chunk_size = max(1, batch_size)
        for start in range(0, len(remaining_entries), chunk_size):
            chunk = remaining_entries[start : start + chunk_size]
            prompts = [self._format_prompt(record, cot_prompt_template) for _key, record in chunk]

            def _on_cot_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                key, record = chunk[local_idx]
                cot_prompt = prompts[local_idx]
                cot_stage = StageRecord(
                    prompt=cot_prompt,
                    completion=output.text,
                    stop_reason=output.finish_reason,
                )
                cot_payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=key.sample_index,
                    repeat_index=key.repeat_index,
                    pass_index=key.pass_index,
                    sampling_config=sampling_config,
                    stages=[cot_stage],
                ).as_payload()
                cot_payload["_stage"] = "cot"
                if on_record is not None:
                    on_record(cot_payload)
                final_prompt = (
                    (final_answer_template or EN_FINAL_ANSWER_TEMPLATE)
                    .replace("<Q>", cot_prompt)
                    .replace("<COT>", output.text)
                )
                _, pred_letter = self._score_prompt(record, final_prompt)
                prior_context = f"{cot_prompt}{output.text}"
                delta_prompt = prompt_delta(final_prompt, prior_context)
                token_text = self.target_token_format.replace("<LETTER>", pred_letter)
                final_stage = StageRecord(
                    prompt=delta_prompt,
                    completion=token_text,
                    stop_reason="logits_only",
                )
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=key.sample_index,
                    repeat_index=key.repeat_index,
                    pass_index=key.pass_index,
                    sampling_config=sampling_config,
                    stages=[cot_stage, final_stage],
                ).as_payload()
                payload["_stage"] = "answer"
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.engine.generate(
                prompts,
                sampling=cot_sampling,
                batch_size=batch_size,
                progress_desc="Generating CoT" if not probe_only else "Probing CoT",
                probe_only=probe_only,
                on_complete=None if probe_only else _on_cot_complete,
                prompt_seeds=[
                    sample_repeat_seed(
                        key.sample_index,
                        key.repeat_index,
                        pass_index=key.pass_index,
                        stage=1,
                    )
                    for key, _record in chunk
                ],
            )
            if probe_only:
                return MultipleChoicePipelineResult(dataset_name, len(expanded), [])
        return MultipleChoicePipelineResult(dataset_name, len(expanded), payloads)

    def _load_records(
        self,
        dataset_path: str,
        sample_limit: int | None,
        *,
        record_indices: Sequence[int] | None = None,
    ) -> tuple[list[tuple[int, MultipleChoiceRecord]], str]:
        loader = JsonlMultipleChoiceLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if record_indices is not None:
            indexed_records = [(int(index), records[int(index)]) for index in record_indices]
        else:
            indexed_records = list(enumerate(records))
            if sample_limit is not None and sample_limit > 0:
                indexed_records = indexed_records[: min(sample_limit, len(indexed_records))]
        dataset_name = infer_dataset_slug_from_path(dataset_path)
        return indexed_records, dataset_name

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
