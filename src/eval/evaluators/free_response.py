from __future__ import annotations

"""Free-form QA 评估流水线：读数据 -> 两阶段生成 -> JSONL 导出。"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import SampleRecord, StageRecord

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think"""

DEFAULT_FINAL_PROMPT = """<Q><COT>
Therefore, the answer is \\(\\boxed{"""


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    problem_count: int
    payloads: list[dict]


_PREFERRED_ANSWER_KEYS = (
    "expected_answer",
    "reference_answer",
    "target",
    "final_answer",
)


def _normalize_answer_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized = str(value)
        return normalized.strip() or None
    normalized = str(value).strip()
    return normalized or None


def _resolve_reference_answer(record: FreeAnswerRecord) -> str:
    metadata = record.metadata or {}
    for key in _PREFERRED_ANSWER_KEYS:
        normalized = _normalize_answer_value(metadata.get(key))
        if normalized:
            return normalized
    raw_record = metadata.get("raw_record")
    if isinstance(raw_record, dict):
        for key in _PREFERRED_ANSWER_KEYS:
            normalized = _normalize_answer_value(raw_record.get(key))
            if normalized:
                return normalized
    return record.answer


class FreeResponsePipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

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
        pad_to_batch: bool = False,
        pass_k: Iterable[int] | None = None,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> FreeResponsePipelineResult:
        samples_per_task = (samples_per_task or max(1, max(pass_k) if pass_k else 1)) if not probe_only else 1
        raw_records, resolved_name = self._load_records(dataset_path, sample_limit)
        problem_count = len(raw_records)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        expanded: list[tuple[int, FreeAnswerRecord, int]] = []
        repeats = max(1, samples_per_task)
        skip_keys = skip_keys or set()
        total_expected = len(raw_records) * repeats
        for idx, record in enumerate(raw_records):
            for sample_id in range(repeats):
                if (idx, sample_id) in skip_keys:
                    continue
                expanded.append((idx, record, sample_id))
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
        cot_prompts = [cot_prompt_template.replace("<Q>", record.question) for _, record, _ in remaining_entries]

        if probe_only:
            _ = self.engine.generate(
                cot_prompts,
                sampling=final_sampling,
                batch_size=batch_size,
                progress_desc="Generating answers",
                probe_only=probe_only,
            )
            return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])

        cot_outputs = self.engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=batch_size,
            progress_desc="Generating CoT",
        )
        cot_by_idx = {item.prompt_index: item for item in cot_outputs}

        final_prompts: list[str] = []
        for local_idx, _ in enumerate(remaining_entries):
            cot_seq = cot_by_idx.get(local_idx)
            cot_text = cot_seq.text if cot_seq else ""
            prompt = final_answer_template.replace("<Q>", cot_prompts[local_idx]).replace("<COT>", cot_text)
            final_prompts.append(prompt)

        final_outputs = self.engine.generate(
            final_prompts,
            sampling=final_sampling,
            batch_size=batch_size,
            progress_desc="Generating answers",
        )

        final_by_idx = {item.prompt_index: item for item in final_outputs}
        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling), (2, final_sampling)])
        payloads: list[dict] = []

        for local_idx, (problem_idx, record, sample_id) in enumerate(remaining_entries):
            cot_seq = cot_by_idx.get(local_idx)
            ans_seq = final_by_idx.get(local_idx)
            if cot_seq is None or ans_seq is None:
                continue
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
                    completion=ans_seq.text,
                    stop_reason=ans_seq.finish_reason,
                ),
            ]
            payload = SampleRecord(
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                sample_index=problem_idx,
                repeat_index=sample_id,
                sampling_config=sampling_config,
                stages=stages,
            ).as_payload()
            if on_record is not None:
                on_record(payload)
            payloads.append(payload)
        return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, payloads)

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[FreeAnswerRecord], str]:
        loader = JsonlFreeAnswerLoader(dataset_path)
        records: list[FreeAnswerRecord] = []
        limited = sample_limit is not None and sample_limit > 0
        for record in loader:
            records.append(record)
            if limited and len(records) >= sample_limit:
                break
        return records, infer_dataset_slug_from_path(dataset_path)


__all__ = ["FreeResponsePipeline", "FreeResponsePipelineResult"]
