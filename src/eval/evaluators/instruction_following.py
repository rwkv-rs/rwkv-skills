from __future__ import annotations

"""Instruction-following 评估流水线：生成响应 + 导出 JSONL。"""

from dataclasses import dataclass, replace
from typing import Callable

from src.eval.datasets.data_loader.instruction_following import (
    JsonlInstructionFollowingLoader,
)
from src.eval.datasets.data_struct.instruction_following import (
    InstructionFollowingRecord,
)
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from .common import SampleRecord, StageRecord

DEFAULT_BAN_TOKEN = 295

@dataclass(slots=True)
class InstructionFollowingPipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict]


class InstructionFollowingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        *,
        sampling: SamplingConfig,
        batch_size: int = 128,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        enable_think: bool = False,
        stop_tokens: tuple[int, ...],
        ban_tokens: tuple[int, ...] | None = None,
        samples_per_prompt: int | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int]] | None = None,
        on_record: Callable[[dict], None] | None = None,
    ) -> InstructionFollowingPipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        repeats = max(1, samples_per_prompt or 1)
        expanded: list[tuple[int, InstructionFollowingRecord, int]] = []
        for idx, record in enumerate(records):
            for sample_id in range(repeats):
                expanded.append((idx, record, sample_id))
        if not expanded:
            return InstructionFollowingPipelineResult(dataset_name, 0, [])

        skip_keys = skip_keys or set()
        total_expected = len(expanded)
        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(expanded):
                return InstructionFollowingPipelineResult(dataset_name, len(expanded), [])
            remaining_records = [
                item
                for item in expanded[resume_start_index:]
                if (item[0], item[2]) not in skip_keys
            ]
            print(
                f"⏩ Instruction-following 恢复运行：已完成 {resume_start_index}/{len(expanded)}，剩余 {len(remaining_records)}"
            )
        else:
            remaining_records = [item for item in expanded if (item[0], item[2]) not in skip_keys]
        if not remaining_records:
            return InstructionFollowingPipelineResult(dataset_name, 0, [])

        skipped = total_expected - len(remaining_records)
        if skipped > 0:
            print(f"⏩ Instruction-following 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        effective_ban = ban_tokens
        if effective_ban is None:
            effective_ban = () if enable_think else (DEFAULT_BAN_TOKEN,)

        sampling_cfg = replace(sampling, stop_tokens=stop_tokens, ban_tokens=effective_ban)
        sampling_config = normalize_sampling_config_by_stage([(1, sampling_cfg)])

        payloads: list[dict] = []
        chunk_size = max(1, int(batch_size))
        for start in range(0, len(remaining_records), chunk_size):
            chunk = remaining_records[start : start + chunk_size]
            prompts = [self._make_prompt(record.prompt, enable_think) for _, record, _ in chunk]
            def _on_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                problem_idx, _record, sample_id = chunk[local_idx]
                stage = StageRecord(
                    prompt=prompts[local_idx],
                    completion=output.text,
                    stop_reason=output.finish_reason,
                )
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=problem_idx,
                    repeat_index=sample_id,
                    sampling_config=sampling_config,
                    stages=[stage],
                ).as_payload()
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.engine.generate(
                prompts,
                sampling=sampling_cfg,
                batch_size=min(batch_size, len(prompts)),
                progress_desc="Generating instruction-following responses",
                on_complete=_on_complete,
            )
        return InstructionFollowingPipelineResult(dataset_name, len(expanded), payloads)

    def _make_prompt(self, prompt: str, enable_think: bool) -> str:
        suffix = " <think" if enable_think else ""
        return f"User: {prompt}\n\nAssistant:{suffix}"

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[InstructionFollowingRecord], str]:
        loader = JsonlInstructionFollowingLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, infer_dataset_slug_from_path(dataset_path)


__all__ = ["InstructionFollowingPipeline", "InstructionFollowingPipelineResult"]
