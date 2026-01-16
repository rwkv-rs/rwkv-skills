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
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.infra.database import DatabaseManager
from src.infra.eval_db_service import EvalDbService
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state

DEFAULT_STOP_TOKENS = (0, 261, 24281)
DEFAULT_BAN_TOKEN = 295
DEFAULT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.3,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.99,
    stop_tokens=DEFAULT_STOP_TOKENS,
)


@dataclass(slots=True)
class InstructionFollowingPipelineResult:
    dataset: str
    sample_count: int
    output_path: Path


class InstructionFollowingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = DEFAULT_SAMPLING,
        batch_size: int = 128,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        enable_think: bool = False,
        stop_tokens: tuple[int, ...] = DEFAULT_STOP_TOKENS,
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

        db_service: EvalDbService | None = None
        run_ctx = None
        if DEFAULT_DB_CONFIG.enabled:
            db = DatabaseManager.instance()
            db.initialize(DEFAULT_DB_CONFIG)
            db_service = EvalDbService(db)
            run_ctx = db_service.prepare_run(
                dataset_slug=benchmark_name,
                split_name=dataset_split,
                model_path=str(self.model_path),
                is_cot=False,
                run_tag=Path(output_path).stem,
                sampling_config=sampling_config,
                runtime_config={
                    "batch_size": batch_size,
                    "samples_per_prompt": samples_per_prompt,
                    "enable_think": enable_think,
                },
                code_version=None,
            )

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
            if db_service and run_ctx:
                meta = {
                    "instruction_ids": record.instruction_ids,
                    "kwargs_list": record.kwargs_list,
                    "key": record.key,
                    **(record.metadata or {}),
                }
                sample_db_id = db_service.upsert_sample(
                    dataset_id=run_ctx.dataset_id,
                    split_id=run_ctx.split_id,
                    sample_index=problem_idx,
                    question=record.prompt,
                    reference_answer=None,
                    meta=meta,
                )
                run_sample_id = db_service.upsert_run_sample(
                    run_id=run_ctx.run_id,
                    sample_id=sample_db_id,
                    repeat_index=sample_id,
                    status="pending",
                    current_stage=None,
                )
                if db_service.fetch_latest_stage(run_sample_id=run_sample_id, stage="final"):
                    db_service.mark_run_sample_status(
                        run_sample_id=run_sample_id,
                        status="succeeded",
                        current_stage="final",
                        finished=True,
                    )
                else:
                    attempt_id, attempt_index = db_service.start_attempt(
                        run_sample_id=run_sample_id,
                        current_stage="final",
                    )
                    db_service.write_stage_output(
                        attempt_id=attempt_id,
                        stage="final",
                        seq=0,
                        prompt=prompts[local_idx],
                        completion=seq.text,
                        finish_reason=seq.finish_reason,
                        is_final=True,
                    )
                    db_service.mark_attempt_status(
                        attempt_id=attempt_id,
                        status="succeeded",
                        finished=True,
                    )
                    db_service.mark_run_sample_status(
                        run_sample_id=run_sample_id,
                        status="succeeded",
                        current_stage="final",
                        latest_attempt_index=attempt_index,
                        finished=True,
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
        if db_service and run_ctx:
            db_service.mark_run_status(run_id=run_ctx.run_id, status="succeeded")
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
