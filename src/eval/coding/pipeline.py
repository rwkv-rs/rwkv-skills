from __future__ import annotations

"""Field-oriented coding pipeline aligned with rwkv-rs coding datasets."""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.execution_plan import AttemptKey
from src.eval.prompt_builders import (
    CODE_COMPLETION_PLACEHOLDER,
    build_human_eval_expected_context,
    build_livecodebench_expected_context,
    build_mbpp_expected_context,
    extract_function_signature,
    prompt_for_cot,
    prompt_for_marker,
)
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.backend import InferenceBackend
from src.infer.sampling import GenerationOutput, SamplingConfig
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed

# Coding 默认只计算 pass@1；如需更高 k，请通过 CLI 传入
DEFAULT_PASS_K = (1,)


@dataclass(slots=True)
class CodingPipelineResult:
    dataset: str
    sample_count: int
    problem_count: int
    payloads: list[dict]


def _expand_attempt_entries(
    records: Sequence[tuple[int, CodeGenerationRecord]],
    *,
    repeats: int,
    attempt_keys: Sequence[AttemptKey] | None,
    skip_keys: set[tuple[int, int, int]] | None,
) -> list[tuple[AttemptKey, CodeGenerationRecord]]:
    normalized_skip = skip_keys or set()
    record_map = {int(idx): record for idx, record in records}
    expanded: list[tuple[AttemptKey, CodeGenerationRecord]] = []
    if attempt_keys is not None:
        for key in attempt_keys:
            record = record_map.get(int(key.sample_index))
            if record is None or key.as_tuple() in normalized_skip:
                continue
            expanded.append((key, record))
        return expanded

    for idx, record in records:
        for sample_idx in range(repeats):
            key = AttemptKey(sample_index=int(idx), repeat_index=int(sample_idx), pass_index=0)
            if key.as_tuple() in normalized_skip:
                continue
            expanded.append((key, record))
    return expanded


def _build_human_eval_prompt(prompt: str, *, echo_prompt: bool) -> str:
    expected_context = build_human_eval_expected_context(
        prompt,
        assistant_code_prefix=prompt if echo_prompt else None,
        cot_mode=CoTMode.NO_COT,
    )
    return prompt_for_marker(expected_context, CODE_COMPLETION_PLACEHOLDER)


def _build_mbpp_context(prompt: str, signature: str | None, cot_mode: CoTMode) -> str:
    return build_mbpp_expected_context(prompt, signature=signature, cot_mode=cot_mode)


def _build_livecodebench_context(prompt: str, starter_code: str | None) -> str:
    return build_livecodebench_expected_context(
        prompt,
        starter_code=starter_code,
        cot_mode=CoTMode.COT,
    )


class CodingPipeline:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def run_human_eval(
        self,
        dataset_path: str,
        *,
        sampling: SamplingConfig,
        batch_size: int = 64,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else int(samples_per_task or max(1, max(pass_k) if pass_k else 1))
        records, dataset_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, 0, [])

        is_human_eval_fix = "human_eval_fix" in dataset_path.lower()

        if probe_only:
            prompts = []
            probe_seeds: list[int] = []
            for idx in range(batch_size):
                _record_idx, record = records[idx % len(records)]
                prompt_text = _build_human_eval_prompt(record.prompt, echo_prompt=not is_human_eval_fix)
                prompts.append(prompt_text)
                probe_seeds.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            _ = self.backend.generate(
                prompts,
                sampling=sampling,
                batch_size=batch_size,
                progress_desc="Probing code",
                probe_only=True,
                prompt_seeds=probe_seeds,
            )
            return CodingPipelineResult(dataset_name, len(prompts), len(records), [])

        expanded = _expand_attempt_entries(
            records,
            repeats=samples_per_task,
            attempt_keys=attempt_keys,
            skip_keys=skip_keys,
        )
        total_expected = len(expanded)
        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return CodingPipelineResult(dataset_name, len(expanded), len(records), [])
            expanded = expanded[resume_start_index:]
            print(
                f"⏩ HumanEval 恢复运行：已完成 {resume_start_index}/{len(records) * samples_per_task}，剩余 {len(expanded)}"
            )
        entries = [
            (
                _build_human_eval_prompt(record.prompt, echo_prompt=not is_human_eval_fix),
                record,
                key,
            )
            for key, record in expanded
        ]
        skipped = total_expected - len(entries)
        if skipped > 0:
            print(f"⏩ HumanEval 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
        payloads: list[dict] = []
        if entries:
            chunk_size = max(1, int(batch_size))
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                prompts = [entry[0] for entry in chunk]
                def _on_complete(output: GenerationOutput) -> None:
                    local_idx = output.prompt_index
                    if local_idx < 0 or local_idx >= len(chunk):
                        return
                    prompt_text, _record, key = chunk[local_idx]
                    raw_output = output.text or ""
                    stage = StageRecord(
                        prompt=prompt_text,
                        completion=raw_output,
                        stop_reason=output.finish_reason,
                    )
                    payload = SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling_config=sampling_config,
                        stages=[stage],
                    ).as_payload()
                    if on_record is not None:
                        on_record(payload)
                    payloads.append(payload)

                _ = self.backend.generate(
                    prompts,
                    sampling=sampling,
                    batch_size=max(1, min(batch_size, len(prompts))),
                    progress_desc="Generating code",
                    on_complete=_on_complete,
                    prompt_seeds=[
                        sample_repeat_seed(
                            key.sample_index,
                            key.repeat_index,
                            pass_index=key.pass_index,
                            stage=1,
                        )
                        for _prompt_text, _record, key in chunk
                    ],
                )

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            problem_count=len(records),
            payloads=payloads,
        )

    def run_mbpp(
        self,
        dataset_path: str,
        *,
        sampling: SamplingConfig,
        cot_mode: CoTMode = CoTMode.NO_COT,
        batch_size: int = 64,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else int(samples_per_task or max(1, max(pass_k) if pass_k else 1))
        records, dataset_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, 0, [])

        if probe_only:
            expected_contexts = []
            prompts = []
            probe_seeds: list[int] = []
            for idx in range(batch_size):
                _record_idx, record = records[idx % len(records)]
                raw_code = record.metadata.get("code") if record.metadata else None
                signature = extract_function_signature(raw_code)
                expected_context = _build_mbpp_context(record.prompt, signature, cot_mode)
                prompt_text = (
                    prompt_for_cot(expected_context)
                    if cot_mode is CoTMode.COT
                    else prompt_for_marker(expected_context, CODE_COMPLETION_PLACEHOLDER)
                )
                expected_contexts.append(expected_context)
                prompts.append(prompt_text)
                probe_seeds.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            if cot_mode is CoTMode.COT:
                cot_outputs = self.backend.generate(
                    prompts,
                    sampling=sampling,
                    batch_size=batch_size,
                    progress_desc="Probing CoT",
                    probe_only=True,
                    prompt_seeds=probe_seeds,
                )
                cot_by_idx = {item.prompt_index: item for item in cot_outputs}
                final_prompts = []
                for local_idx, expected_context in enumerate(expected_contexts):
                    cot_seq = cot_by_idx.get(local_idx)
                    cot_text = cot_seq.text if cot_seq is not None else ""
                    final_prompts.append(
                        prompt_for_marker(
                            expected_context,
                            CODE_COMPLETION_PLACEHOLDER,
                            completions_of_cot=cot_text,
                        )
                    )
                if final_prompts:
                    final_prompt_seeds = [
                        sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=2)
                        for idx in range(len(final_prompts))
                    ]
                    _ = self.backend.generate(
                        final_prompts,
                        sampling=sampling,
                        batch_size=batch_size,
                        progress_desc="Probing final code",
                        probe_only=True,
                        prompt_seeds=final_prompt_seeds,
                    )
            else:
                _ = self.backend.generate(
                    prompts,
                    sampling=sampling,
                    batch_size=batch_size,
                    progress_desc="Probing code",
                    probe_only=True,
                    prompt_seeds=probe_seeds,
                )
            return CodingPipelineResult(dataset_name, len(prompts), len(records), [])

        expanded = _expand_attempt_entries(
            records,
            repeats=samples_per_task,
            attempt_keys=attempt_keys,
            skip_keys=skip_keys,
        )
        total_expected = len(expanded)
        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return CodingPipelineResult(dataset_name, len(expanded), len(records), [])
            expanded = expanded[resume_start_index:]
            print(
                f"⏩ MBPP 恢复运行：已完成 {resume_start_index}/{len(records) * samples_per_task}，剩余 {len(expanded)}"
            )
        entries = []
        for key, record in expanded:
            raw_code = record.metadata.get("code") if record.metadata else None
            signature = extract_function_signature(raw_code)
            expected_context = _build_mbpp_context(record.prompt, signature, cot_mode)
            prompt_text = (
                prompt_for_cot(expected_context)
                if cot_mode is CoTMode.COT
                else prompt_for_marker(expected_context, CODE_COMPLETION_PLACEHOLDER)
            )
            entries.append((expected_context, prompt_text, record, key))
        skipped = total_expected - len(entries)
        if skipped > 0:
            print(f"⏩ MBPP 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        sampling_config = normalize_sampling_config_by_stage(
            [(1, sampling)] if cot_mode is not CoTMode.COT else [(1, sampling), (2, sampling)]
        )
        payloads: list[dict] = []
        if entries:
            chunk_size = max(1, int(batch_size))
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                prompts = [entry[1] for entry in chunk]
                if cot_mode is CoTMode.COT:
                    def _on_cot_complete(output: GenerationOutput) -> None:
                        local_idx = output.prompt_index
                        if local_idx < 0 or local_idx >= len(chunk):
                            return
                        _expected_context, prompt_text, _record, key = chunk[local_idx]
                        cot_stage = StageRecord(
                            prompt=prompt_text,
                            completion=output.text,
                            stop_reason=output.finish_reason,
                        )
                        payload = SampleRecord(
                            benchmark_name=benchmark_name,
                            dataset_split=dataset_split,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            sampling_config=sampling_config,
                            stages=[cot_stage],
                        ).as_payload()
                        payload["_stage"] = "cot"
                        if on_record is not None:
                            on_record(payload)

                    cot_outputs = self.backend.generate(
                        prompts,
                        sampling=sampling,
                        batch_size=max(1, min(batch_size, len(prompts))),
                        progress_desc="Generating CoT",
                        on_complete=_on_cot_complete,
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=1,
                            )
                            for _expected_context, _prompt_text, _record, key in chunk
                        ],
                    )
                    cot_by_idx = {item.prompt_index: item for item in cot_outputs}
                    final_prompts: list[str] = []
                    final_prompt_indices: list[int] = []
                    for local_idx, (expected_context, _prompt_text, _record, _key) in enumerate(chunk):
                        cot_seq = cot_by_idx.get(local_idx)
                        if cot_seq is None:
                            continue
                        final_prompts.append(
                            prompt_for_marker(
                                expected_context,
                                CODE_COMPLETION_PLACEHOLDER,
                                completions_of_cot=cot_seq.text,
                            )
                        )
                        final_prompt_indices.append(local_idx)

                    def _on_final_complete(output: GenerationOutput) -> None:
                        local_idx = final_prompt_indices[output.prompt_index]
                        expected_context, prompt_text, _record, key = chunk[local_idx]
                        cot_seq = cot_by_idx.get(local_idx)
                        if cot_seq is None:
                            return
                        prior_context = f"{prompt_text}{cot_seq.text}"
                        final_prompt = prompt_for_marker(
                            expected_context,
                            CODE_COMPLETION_PLACEHOLDER,
                            completions_of_cot=cot_seq.text,
                        )
                        delta_prompt = prompt_delta(final_prompt, prior_context)
                        cot_stage = StageRecord(
                            prompt=prompt_text,
                            completion=cot_seq.text,
                            stop_reason=cot_seq.finish_reason,
                        )
                        final_stage = StageRecord(
                            prompt=delta_prompt,
                            completion=output.text,
                            stop_reason=output.finish_reason,
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

                    if final_prompts:
                        _ = self.backend.generate(
                            final_prompts,
                            sampling=sampling,
                            batch_size=max(1, min(batch_size, len(final_prompts))),
                            progress_desc="Generating code",
                            on_complete=_on_final_complete,
                            prompt_seeds=[
                                sample_repeat_seed(
                                    chunk[local_idx][3].sample_index,
                                    chunk[local_idx][3].repeat_index,
                                    pass_index=chunk[local_idx][3].pass_index,
                                    stage=2,
                                )
                                for local_idx in final_prompt_indices
                            ],
                        )
                else:
                    def _on_complete(output: GenerationOutput) -> None:
                        local_idx = output.prompt_index
                        if local_idx < 0 or local_idx >= len(chunk):
                            return
                        _expected_context, prompt_text, _record, key = chunk[local_idx]
                        raw_output = output.text or ""
                        stage = StageRecord(
                            prompt=prompt_text,
                            completion=raw_output,
                            stop_reason=output.finish_reason,
                        )
                        payload = SampleRecord(
                            benchmark_name=benchmark_name,
                            dataset_split=dataset_split,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            sampling_config=sampling_config,
                            stages=[stage],
                        ).as_payload()
                        if on_record is not None:
                            on_record(payload)
                        payloads.append(payload)

                    _ = self.backend.generate(
                        prompts,
                        sampling=sampling,
                        batch_size=max(1, min(batch_size, len(prompts))),
                        progress_desc="Generating code",
                        on_complete=_on_complete,
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=1,
                            )
                            for _expected_context, _prompt_text, _record, key in chunk
                        ],
                    )

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            problem_count=len(records),
            payloads=payloads,
        )

    def run_livecodebench(
        self,
        dataset_path: str,
        *,
        cot_sampling: SamplingConfig,
        final_sampling: SamplingConfig,
        batch_size: int = 64,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else int(samples_per_task or max(1, max(pass_k) if pass_k else 1))
        records, dataset_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, 0, [])

        if probe_only:
            expected_contexts = []
            prompts = []
            probe_seeds_stage1: list[int] = []
            for idx in range(batch_size):
                _record_idx, record = records[idx % len(records)]
                expected_context = _build_livecodebench_context(record.prompt, record.starter_code)
                prompt_text = prompt_for_cot(expected_context)
                expected_contexts.append(expected_context)
                prompts.append(prompt_text)
                probe_seeds_stage1.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            cot_outputs = self.backend.generate(
                prompts,
                sampling=cot_sampling,
                batch_size=batch_size,
                progress_desc="Probing CoT",
                probe_only=True,
                prompt_seeds=probe_seeds_stage1,
            )
            final_prompts: list[str] = []
            cot_by_idx = {item.prompt_index: item for item in cot_outputs}
            for local_idx, expected_context in enumerate(expected_contexts):
                cot_seq = cot_by_idx.get(local_idx)
                cot_text = cot_seq.text if cot_seq is not None else ""
                final_prompts.append(
                    prompt_for_marker(
                        expected_context,
                        CODE_COMPLETION_PLACEHOLDER,
                        completions_of_cot=cot_text,
                    )
                )
            if final_prompts:
                probe_seeds_stage2 = [
                        sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=2)
                        for idx in range(len(final_prompts))
                    ]
                _ = self.backend.generate(
                    final_prompts,
                    sampling=final_sampling,
                    batch_size=batch_size,
                    progress_desc="Probing final code",
                    probe_only=True,
                    prompt_seeds=probe_seeds_stage2,
                )
            return CodingPipelineResult(dataset_name, len(prompts), len(records), [])

        expanded = _expand_attempt_entries(
            records,
            repeats=samples_per_task,
            attempt_keys=attempt_keys,
            skip_keys=skip_keys,
        )
        total_expected = len(expanded)
        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return CodingPipelineResult(dataset_name, len(expanded), len(records), [])
            expanded = expanded[resume_start_index:]
            print(
                f"⏩ LiveCodeBench 恢复运行：已完成 {resume_start_index}/{len(records) * samples_per_task}，剩余 {len(expanded)}"
            )
        entries = []
        for key, record in expanded:
            expected_context = _build_livecodebench_context(record.prompt, record.starter_code)
            entries.append((expected_context, prompt_for_cot(expected_context), record, key))
        skipped = total_expected - len(entries)
        if skipped > 0:
            print(f"⏩ LiveCodeBench 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        sampling_config = normalize_sampling_config_by_stage(
            [(1, cot_sampling), (2, final_sampling)]
        )
        payloads: list[dict] = []
        if entries:
            chunk_size = max(1, int(batch_size))
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                prompts = [entry[1] for entry in chunk]

                def _on_cot_complete(output: GenerationOutput) -> None:
                    local_idx = output.prompt_index
                    if local_idx < 0 or local_idx >= len(chunk):
                        return
                    _expected_context, prompt_text, _record, key = chunk[local_idx]
                    cot_stage = StageRecord(
                        prompt=prompt_text,
                        completion=output.text,
                        stop_reason=output.finish_reason,
                    )
                    payload = SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling_config=sampling_config,
                        stages=[cot_stage],
                    ).as_payload()
                    payload["_stage"] = "cot"
                    if on_record is not None:
                        on_record(payload)

                cot_outputs = self.backend.generate(
                    prompts,
                    sampling=cot_sampling,
                    batch_size=max(1, min(batch_size, len(prompts))),
                    progress_desc="Generating CoT",
                    on_complete=_on_cot_complete,
                    prompt_seeds=[
                        sample_repeat_seed(
                            key.sample_index,
                            key.repeat_index,
                            pass_index=key.pass_index,
                            stage=1,
                        )
                        for _expected_context, _prompt_text, _record, key in chunk
                    ],
                )
                cot_by_idx = {item.prompt_index: item for item in cot_outputs}

                final_prompts: list[str] = []
                final_prompt_indices: list[int] = []
                for local_idx, (expected_context, _prompt_text, _record, _key) in enumerate(chunk):
                    cot_seq = cot_by_idx.get(local_idx)
                    if cot_seq is None:
                        continue
                    final_prompts.append(
                        prompt_for_marker(
                            expected_context,
                            CODE_COMPLETION_PLACEHOLDER,
                            completions_of_cot=cot_seq.text,
                        )
                    )
                    final_prompt_indices.append(local_idx)

                def _on_final_complete(output: GenerationOutput) -> None:
                    local_idx = final_prompt_indices[output.prompt_index]
                    expected_context, prompt_text, _record, key = chunk[local_idx]
                    cot_seq = cot_by_idx.get(local_idx)
                    if cot_seq is None:
                        return
                    prior_context = f"{prompt_text}{cot_seq.text}"
                    final_prompt = prompt_for_marker(
                        expected_context,
                        CODE_COMPLETION_PLACEHOLDER,
                        completions_of_cot=cot_seq.text,
                    )
                    delta_prompt = prompt_delta(final_prompt, prior_context)
                    cot_stage = StageRecord(
                        prompt=prompt_text,
                        completion=cot_seq.text,
                        stop_reason=cot_seq.finish_reason,
                    )
                    final_stage = StageRecord(
                        prompt=delta_prompt,
                        completion=output.text,
                        stop_reason=output.finish_reason,
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

                if final_prompts:
                    final_prompt_seeds = [
                        sample_repeat_seed(
                            chunk[local_idx][3].sample_index,
                            chunk[local_idx][3].repeat_index,
                            pass_index=chunk[local_idx][3].pass_index,
                            stage=2,
                        )
                        for local_idx in final_prompt_indices
                    ]
                    _ = self.backend.generate(
                        final_prompts,
                        sampling=final_sampling,
                        batch_size=max(1, min(batch_size, len(final_prompts))),
                        progress_desc="Generating final code",
                        on_complete=_on_final_complete,
                        prompt_seeds=final_prompt_seeds,
                    )

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            problem_count=len(records),
            payloads=payloads,
        )

    def _load_records(
        self,
        dataset_path: str,
        sample_limit: int | None,
        *,
        record_indices: Sequence[int] | None = None,
    ) -> tuple[list[tuple[int, CodeGenerationRecord]], str]:
        loader = JsonlCodeGenerationLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if record_indices is not None:
            indexed_records = [(int(index), records[int(index)]) for index in record_indices]
        else:
            indexed_records = list(enumerate(records))
            if sample_limit is not None and sample_limit > 0:
                indexed_records = indexed_records[: min(sample_limit, len(indexed_records))]
        return indexed_records, infer_dataset_slug_from_path(dataset_path)


__all__ = [
    "CodingPipeline",
    "CodingPipelineResult",
]
