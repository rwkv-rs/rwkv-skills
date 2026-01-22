from __future__ import annotations

"""Instruction-following 评估流水线：生成响应 + 导出 JSONL。"""

from dataclasses import dataclass, replace
from pathlib import Path

from src.eval.datasets.data_loader.instruction_following import (
    JsonlInstructionFollowingLoader,
)
from src.eval.datasets.data_struct.instruction_following import (
    InstructionFollowingRecord,
)
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state

DEFAULT_BAN_TOKEN = 295

@dataclass(slots=True)
class InstructionFollowingPipelineResult:
    dataset: str
    sample_count: int
    output_path: Path


class InstructionFollowingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

    def run(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig,
        batch_size: int = 128,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        enable_think: bool = False,
        stop_tokens: tuple[int, ...],
        ban_tokens: tuple[int, ...] | None = None,
        samples_per_prompt: int | None = None,
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
            return InstructionFollowingPipelineResult(dataset_name, 0, Path(output_path))

        resume = detect_resume_state(output_path, repeats=repeats)
        start_index = min(resume.next_index, len(expanded))
        if start_index and len(expanded):
            remaining = max(len(expanded) - start_index, 0)
            print(f"⏩ Instruction-following 恢复运行：已完成 {start_index}/{len(expanded)}，剩余 {remaining}")
        remaining_records = expanded[start_index:]
        if not remaining_records:
            return InstructionFollowingPipelineResult(dataset_name, len(expanded), Path(output_path))

        effective_ban = ban_tokens
        if effective_ban is None:
            effective_ban = () if enable_think else (DEFAULT_BAN_TOKEN,)

        sampling_cfg = replace(sampling, stop_tokens=stop_tokens, ban_tokens=effective_ban)
        sampling_config = normalize_sampling_config_by_stage([(1, sampling_cfg)])

        writer = JsonlStageWriter(output_path, resume=resume.has_progress)
        prompts = [self._make_prompt(record.prompt, enable_think) for _, record, _ in remaining_records]
        outputs = self.engine.generate(
            prompts,
            sampling=sampling_cfg,
            batch_size=batch_size,
            progress_desc="Generating instruction-following responses",
        )
        output_by_idx = {item.prompt_index: item for item in outputs}
        for local_idx, (problem_idx, record, sample_id) in enumerate(remaining_records):
            seq = output_by_idx.get(local_idx)
            if seq is None:
                continue
            stage = StageRecord(
                prompt=prompts[local_idx],
                completion=seq.text,
                stop_reason=seq.finish_reason,
            )
            writer.write(
                SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=problem_idx,
                    repeat_index=sample_id,
                    sampling_config=sampling_config,
                    stages=[stage],
                )
            )
        writer.close()
        return InstructionFollowingPipelineResult(dataset_name, len(expanded), Path(output_path))

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
