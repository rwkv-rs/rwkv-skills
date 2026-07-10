from __future__ import annotations

"""Knowledge benchmark pipeline for multiple-choice datasets."""

from dataclasses import dataclass
import re
from typing import Callable, Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.datasets.data_struct.multiple_choice import MultipleChoiceRecord
from src.eval.execution_plan import AttemptKey
from src.eval.prompt_builders import (
    ALPHABET,
    concat_choices,
)
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.infer.backend import InferenceBackend
from src.infer.sampling import GenerationOutput, SamplingConfig

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

Assistant: <think"""

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
    payloads: list[dict]


class MultipleChoicePipeline:
    """Wrap direct and CoT multiple-choice execution into canonical payloads."""

    def __init__(
        self,
        backend: InferenceBackend,
        target_token_format: str = TARGET_TOKEN_FORMAT,
        *,
        allow_generation_fallback: bool = False,
    ) -> None:
        del allow_generation_fallback
        self.backend = backend
        self.target_token_format = target_token_format

    def run_direct(
        self,
        dataset_path: str,
        *,
        prompt_template: str | None = None,
        cot_mode: CoTMode = CoTMode.NO_COT,
        batch_size: int = 64,
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
            prompt_template = templates.direct
        skip_keys = skip_keys or set()
        batch_size = max(1, int(batch_size))
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

        payloads = self._run_direct_generation_batches(
            expanded,
            prompt_template=prompt_template,
            benchmark_name=benchmark_name,
            dataset_split=dataset_split,
            batch_size=batch_size,
            on_record=on_record,
        )
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
        remaining_entries = list(remaining_entries)

        # 整段一次性构造，去掉分块屏障。每条的 stage-1 seed 由 sample/repeat/pass 决定，
        # 与分块边界无关；prompt_index→entry 映射保持不变。
        cot_prompts = [self._format_prompt(record, cot_prompt_template) for _key, record in remaining_entries]
        cot_seeds = [
            sample_repeat_seed(
                key.sample_index,
                key.repeat_index,
                pass_index=key.pass_index,
                stage=1,
            )
            for key, _record in remaining_entries
        ]
        cot_outputs: list[GenerationOutput | None] = [None] * len(remaining_entries)

        def _on_cot_complete(output: GenerationOutput) -> None:
            idx = output.prompt_index
            if idx < 0 or idx >= len(remaining_entries):
                return
            cot_outputs[idx] = output
            key, _record = remaining_entries[idx]
            cot_stage = StageRecord(
                prompt=cot_prompts[idx],
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

        # 阶段一：流式提交所有 CoT，在飞并发上限用本 benchmark 的 batch_size（去屏障、保档位）
        _ = self.backend.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=batch_size,
            progress_desc="Generating CoT" if not probe_only else "Probing CoT",
            probe_only=probe_only,
            on_complete=None if probe_only else _on_cot_complete,
            prompt_seeds=cot_seeds,
        )
        if probe_only:
            return MultipleChoicePipelineResult(dataset_name, len(expanded), [])

        # 阶段二：用全部 CoT 输出构造答案段 prompt，单次并发生成。
        # 贪心采样、无 seed，解析函数（_extract_generated_choice_letter）不变 ⇒ 逐条等价。
        answer_indices: list[int] = []
        answer_prompts: list[str] = []
        final_prompt_by_idx: dict[int, str] = {}
        for idx, output in enumerate(cot_outputs):
            if output is None:
                continue
            final_prompt = (
                final_answer_template.replace("<Q>", cot_prompts[idx]).replace("<COT>", output.text)
            )
            final_prompt_by_idx[idx] = final_prompt
            answer_indices.append(idx)
            answer_prompts.append(final_prompt)

        pred_letter_by_idx: dict[int, str] = {}
        if answer_prompts:
            answer_outputs = self.backend.generate(
                answer_prompts,
                sampling=_multiple_choice_answer_sampling(),
                batch_size=batch_size,
                progress_desc="Generating MC answer",
                show_progress=False,
            )
            by_index = {int(o.prompt_index): o for o in answer_outputs}
            for local_index, idx in enumerate(answer_indices):
                ans_output = by_index.get(local_index)
                if ans_output is None:
                    raise RuntimeError("backend returned incomplete multiple-choice answer batch")
                _key, record = remaining_entries[idx]
                pred_letter_by_idx[idx] = self._extract_generated_choice_letter(
                    ans_output.text, len(record.choices)
                )

        for idx, output in enumerate(cot_outputs):
            if output is None:
                continue
            key, _record = remaining_entries[idx]
            cot_prompt = cot_prompts[idx]
            cot_stage = StageRecord(
                prompt=cot_prompt,
                completion=output.text,
                stop_reason=output.finish_reason,
            )
            final_prompt = final_prompt_by_idx[idx]
            pred_letter = pred_letter_by_idx.get(idx, "")
            prior_context = f"{cot_prompt}{output.text}"
            delta_prompt = prompt_delta(final_prompt, prior_context)
            token_text = self.target_token_format.replace("<LETTER>", pred_letter)
            final_stage = StageRecord(
                prompt=delta_prompt,
                completion=token_text,
                stop_reason="generated_choice",
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
        question = record.question.lstrip()
        return (
            template.replace("<SUBJECT>", subject)
            .replace("<Q>", question)
            .replace("<CHOICES>", concat_choices(record.choices))
        ).rstrip(" ")

    def _build_direct_payload(
        self,
        *,
        benchmark_name: str,
        dataset_split: str,
        key: AttemptKey,
        prompt: str,
        pred_letter: str,
    ) -> dict:
        token_text = self.target_token_format.replace("<LETTER>", pred_letter)
        stages = [
            StageRecord(
                prompt=prompt,
                completion=token_text,
                stop_reason="generated_choice",
            )
        ]
        return SampleRecord(
            benchmark_name=benchmark_name,
            dataset_split=dataset_split,
            sample_index=key.sample_index,
            repeat_index=key.repeat_index,
            pass_index=key.pass_index,
            sampling_config={},
            stages=stages,
        ).as_payload()

    def _run_direct_generation_batches(
        self,
        entries: Sequence[tuple[AttemptKey, MultipleChoiceRecord]],
        *,
        prompt_template: str,
        benchmark_name: str,
        dataset_split: str,
        batch_size: int,
        on_record: Callable[[dict], None] | None,
    ) -> list[dict]:
        payloads: list[dict] = []
        sampling = _multiple_choice_answer_sampling()
        entries = list(entries)
        if not entries:
            return payloads
        # 一次性提交整段 prompts，去掉分块屏障；但在飞并发上限仍用本 benchmark 自己的
        # batch_size（ThreadPoolExecutor 会持续填充，长尾不阻塞、又不超过该档并发）。
        # prompt_index→record 映射不变，贪心采样 + 原解析逻辑保留 ⇒ 逐条结果等价。
        prompts = [self._format_prompt(record, prompt_template) for _key, record in entries]
        outputs = self.backend.generate(
            prompts,
            sampling=sampling,
            batch_size=batch_size,
            progress_desc="Generating MC answer",
            show_progress=False,
        )
        by_index = {int(output.prompt_index): output for output in outputs}
        for index, ((key, record), prompt) in enumerate(zip(entries, prompts, strict=True)):
            output = by_index.get(index)
            if output is None:
                raise RuntimeError("backend returned incomplete multiple-choice generation batch")
            pred_letter = self._extract_generated_choice_letter(output.text, len(record.choices))
            payload = self._build_direct_payload(
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                key=key,
                prompt=prompt,
                pred_letter=pred_letter,
            )
            if on_record is not None:
                on_record(payload)
            payloads.append(payload)
        return payloads

    def _extract_generated_choice_letter(self, text: str, num_choices: int) -> str:
        valid_letters = ALPHABET[:num_choices]
        normalized = (text or "").strip().upper()
        boundary_match = re.search(rf"\b([{re.escape(valid_letters)}])\b", normalized)
        if boundary_match is not None:
            return boundary_match.group(1)
        for char in normalized:
            if char in valid_letters:
                return char
        return ""


def _multiple_choice_answer_sampling() -> SamplingConfig:
    return SamplingConfig(
        max_generate_tokens=8,
        # Keep top_k=1 deterministic while avoiding vLLM rapid-sampler greedy crashes.
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        alpha_presence=0.0,
        alpha_frequency=0.0,
        alpha_decay=1.0,
        stop_tokens=(),
        no_penalty_token_ids=(),
    )


__all__ = ["MultipleChoicePipeline", "MultipleChoicePipelineResult"]
