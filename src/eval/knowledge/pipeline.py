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
    LOGPROBS_PLACEHOLDER,
    build_multiple_choice_expected_context,
    concat_choices,
    normalize_subject,
    prompt_for_cot,
    prompt_for_marker,
)
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.infer.backend import InferenceBackend
from src.infer.sampling import GenerationOutput, SamplingConfig

TARGET_TOKEN_FORMAT = " <LETTER>"


@dataclass(slots=True)
class MultipleChoicePipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict]


class MultipleChoicePipeline:
    """Wrap direct and CoT multiple-choice execution into canonical payloads."""

    def __init__(self, backend: InferenceBackend, target_token_format: str = TARGET_TOKEN_FORMAT) -> None:
        self.backend = backend
        self.target_token_format = target_token_format

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
            if prompt_template is not None:
                prompt = self._format_prompt(record, prompt_template)
            else:
                expected_context = self._build_expected_context(record, cot_mode)
                prompt = self._prompt_for_final_answer(expected_context, None)
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
        batch_size = max(1, int(batch_size))

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
            expected_contexts = [self._build_expected_context(record, CoTMode.COT) for _key, record in chunk]
            prompts = [
                self._format_prompt(record, cot_prompt_template)
                if cot_prompt_template is not None
                else self._prompt_for_cot(expected_context)
                for expected_context, (_key, record) in zip(expected_contexts, chunk, strict=True)
            ]

            def _on_cot_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                key, record = chunk[local_idx]
                cot_prompt = prompts[local_idx]
                expected_context = expected_contexts[local_idx]
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
                if final_answer_template is not None:
                    final_prompt = (
                        final_answer_template.replace("<Q>", cot_prompt).replace("<COT>", output.text)
                    )
                else:
                    final_prompt = self._prompt_for_final_answer(expected_context, output.text)
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

            _ = self.backend.generate(
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

    def _build_expected_context(self, record: MultipleChoiceRecord, cot_mode: CoTMode) -> str:
        return build_multiple_choice_expected_context(
            subject=normalize_subject(record.subject, "unknown"),
            question=record.question,
            choices=record.choices,
            cot_mode=cot_mode,
        )

    def _prompt_for_cot(self, expected_context: str) -> str:
        return prompt_for_cot(expected_context)

    def _prompt_for_final_answer(self, expected_context: str, completions_of_cot: str | None) -> str:
        return prompt_for_marker(
            expected_context,
            LOGPROBS_PLACEHOLDER,
            completions_of_cot=completions_of_cot,
        )

    def _format_prompt(self, record: MultipleChoiceRecord, template: str) -> str:
        return (
            template.replace("<SUBJECT>", normalize_subject(record.subject, "unknown"))
            .replace("<Q>", record.question)
            .replace("<CHOICES>", concat_choices(record.choices))
        )

    def _choice_tokens(self, num_choices: int) -> list[int]:
        return [
            self.target_token_format.replace("<LETTER>", letter)
            for letter in ALPHABET[:num_choices]
        ]

    def _score_prompt(self, record: MultipleChoiceRecord, prompt: str) -> tuple[dict[str, float], str]:
        choice_texts = self._choice_tokens(len(record.choices))
        try:
            score_map, best_text = self.backend.score_choice_tokens(
                prompt=prompt,
                choice_token_texts=choice_texts,
            )
        except NotImplementedError:
            return self._score_prompt_via_generation(record, prompt)
        logits_map = {
            ALPHABET[index]: float(score_map.get(choice_texts[index], float("-inf")))
            for index in range(len(choice_texts))
        }
        try:
            pred_idx = choice_texts.index(best_text)
        except ValueError as exc:
            raise RuntimeError(f"backend returned unexpected choice token text: {best_text!r}") from exc
        return logits_map, ALPHABET[pred_idx]

    def _score_prompt_via_generation(
        self,
        record: MultipleChoiceRecord,
        prompt: str,
    ) -> tuple[dict[str, float], str]:
        outputs = self.backend.generate(
            [prompt],
            sampling=SamplingConfig(
                max_generate_tokens=8,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                alpha_presence=0.0,
                alpha_frequency=0.0,
                alpha_decay=1.0,
                stop_tokens=(),
                no_penalty_token_ids=(),
            ),
            batch_size=1,
            progress_desc="Generating MC answer",
            show_progress=False,
        )
        if not outputs:
            raise RuntimeError("backend returned no output for multiple-choice fallback generation")
        pred_letter = self._extract_generated_choice_letter(outputs[0].text, len(record.choices))
        score_map = {
            letter: (0.0 if letter == pred_letter else float("-inf"))
            for letter in ALPHABET[: len(record.choices)]
        }
        return score_map, pred_letter

    def _extract_generated_choice_letter(self, text: str, num_choices: int) -> str:
        valid_letters = ALPHABET[:num_choices]
        normalized = (text or "").strip().upper()
        boundary_match = re.search(rf"\b([{re.escape(valid_letters)}])\b", normalized)
        if boundary_match is not None:
            return boundary_match.group(1)
        for char in normalized:
            if char in valid_letters:
                return char
        raise RuntimeError(f"could not extract a valid choice letter from generated text: {text!r}")


__all__ = ["MultipleChoicePipeline", "MultipleChoicePipelineResult"]
