from __future__ import annotations

"""Maths benchmark pipeline for free-response datasets."""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.execution_plan import AttemptKey
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think>"""

DEFAULT_FINAL_PROMPT = """<Q><COT></think>
Therefore, the answer is \\(\\boxed{"""


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    problem_count: int
    payloads: list[dict]


class FreeResponsePipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        *,
        cot_prompt_template: str = DEFAULT_COT_PROMPT,
        final_answer_template: str = DEFAULT_FINAL_PROMPT,
        cot_sampling: SamplingConfig,
        final_sampling: SamplingConfig,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        record_indices: Sequence[int] | None = None,
        pad_to_batch: bool = False,
        pass_k: Iterable[int] | None = None,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        attempt_keys: Sequence[AttemptKey] | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> FreeResponsePipelineResult:
        samples_per_task = (samples_per_task or max(1, max(pass_k) if pass_k else 1)) if not probe_only else 1
        raw_records, resolved_name = self._load_records(
            dataset_path,
            sample_limit,
            record_indices=record_indices,
        )
        problem_count = len(raw_records)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        record_map = {int(idx): record for idx, record in raw_records}
        expanded: list[tuple[AttemptKey, FreeAnswerRecord]] = []
        repeats = max(1, samples_per_task)
        skip_keys = skip_keys or set()
        if attempt_keys is not None:
            total_expected = len(attempt_keys)
            for key in attempt_keys:
                record = record_map.get(int(key.sample_index))
                if record is None or key.as_tuple() in skip_keys:
                    continue
                expanded.append((key, record))
        else:
            total_expected = len(raw_records) * repeats
            for idx, record in raw_records:
                for sample_id in range(repeats):
                    key = AttemptKey(sample_index=int(idx), repeat_index=int(sample_id), pass_index=0)
                    if key.as_tuple() in skip_keys:
                        continue
                    expanded.append((key, record))
        if pad_to_batch and expanded and len(expanded) < batch_size:
            original_len = len(expanded)
            repeat = (batch_size + original_len - 1) // original_len
            expanded = (expanded * repeat)[:batch_size]
        if not expanded:
            return FreeResponsePipelineResult(dataset_name, 0, problem_count, [])

        skipped = total_expected - len(expanded)
        if skipped > 0:
            print(f"⏩ 自由问答恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])
            if resume_start_index % repeats != 0:
                print(
                    f"⚠️  Resume index {resume_start_index} 未按 repeats={repeats} 对齐，可能存在漏样本。"
                )
            remaining_entries = expanded[resume_start_index:]
            print(
                f"⏩ 自由问答恢复运行：已完成 {resume_start_index}/{len(expanded)}，剩余 {len(remaining_entries)}"
            )
        else:
            remaining_entries = expanded
        if probe_only:
            cot_prompts = [
                cot_prompt_template.replace("<Q>", record.question) for _key, record in remaining_entries
            ]
            probe_seeds = [
                sample_repeat_seed(
                    key.sample_index,
                    key.repeat_index,
                    pass_index=key.pass_index,
                    stage=1,
                )
                for key, _record in remaining_entries
            ]
            _ = self.engine.generate(
                cot_prompts,
                sampling=final_sampling,
                batch_size=batch_size,
                progress_desc="Generating answers",
                probe_only=probe_only,
                prompt_seeds=probe_seeds,
            )
            return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])

        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling), (2, final_sampling)])
        payloads: list[dict] = []
        chunk_size = max(1, int(batch_size))
        for start in range(0, len(remaining_entries), chunk_size):
            chunk = remaining_entries[start : start + chunk_size]
            cot_prompts = [
                cot_prompt_template.replace("<Q>", record.question) for _key, record in chunk
            ]

            def _on_cot_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                key, _record = chunk[local_idx]
                cot_stage = StageRecord(
                    prompt=cot_prompts[local_idx],
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

            cot_outputs = self.engine.generate(
                cot_prompts,
                sampling=cot_sampling,
                batch_size=min(batch_size, len(cot_prompts)),
                progress_desc="Generating CoT",
                on_complete=_on_cot_complete,
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
            cot_by_idx = {item.prompt_index: item for item in cot_outputs}

            final_prompts: list[str] = []
            for local_idx in range(len(chunk)):
                cot_seq = cot_by_idx.get(local_idx)
                cot_text = cot_seq.text if cot_seq else ""
                prompt = final_answer_template.replace("<Q>", cot_prompts[local_idx]).replace("<COT>", cot_text)
                final_prompts.append(prompt)

            def _on_final_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                cot_seq = cot_by_idx.get(local_idx)
                if cot_seq is None:
                    return
                key, _record = chunk[local_idx]
                prior_context = f"{cot_prompts[local_idx]}{cot_seq.text}"
                delta_prompt2 = prompt_delta(final_prompts[local_idx], prior_context)
                stages = [
                    StageRecord(
                        prompt=cot_prompts[local_idx],
                        completion=cot_seq.text,
                        stop_reason=cot_seq.finish_reason,
                    ),
                    StageRecord(
                        prompt=delta_prompt2,
                        completion=output.text,
                        stop_reason=output.finish_reason,
                    ),
                ]
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=key.sample_index,
                    repeat_index=key.repeat_index,
                    pass_index=key.pass_index,
                    sampling_config=sampling_config,
                    stages=stages,
                ).as_payload()
                payload["_stage"] = "answer"
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.engine.generate(
                final_prompts,
                sampling=final_sampling,
                batch_size=min(batch_size, len(final_prompts)),
                progress_desc="Generating answers",
                on_complete=_on_final_complete,
                prompt_seeds=[
                    sample_repeat_seed(
                        key.sample_index,
                        key.repeat_index,
                        pass_index=key.pass_index,
                        stage=2,
                    )
                    for key, _record in chunk
                ],
            )
        return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, payloads)

    def _load_records(
        self,
        dataset_path: str,
        sample_limit: int | None,
        *,
        record_indices: Sequence[int] | None = None,
    ) -> tuple[list[tuple[int, FreeAnswerRecord]], str]:
        loader = JsonlFreeAnswerLoader(dataset_path)
        records = list(loader)
        if record_indices is not None:
            indexed_records = [(int(index), records[int(index)]) for index in record_indices]
        else:
            indexed_records = list(enumerate(records))
            if sample_limit is not None and sample_limit > 0:
                indexed_records = indexed_records[: min(sample_limit, len(indexed_records))]
        return indexed_records, infer_dataset_slug_from_path(dataset_path)


__all__ = ["FreeResponsePipeline", "FreeResponsePipelineResult"]
