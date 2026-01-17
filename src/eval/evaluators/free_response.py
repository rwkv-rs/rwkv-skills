from __future__ import annotations

"""Free-form QA 评估流水线：读数据 -> 两阶段生成 -> JSONL 导出。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, safe_slug
from src.infra.database import DatabaseManager
from src.infra.eval_db_service import EvalDbService
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state, ensure_resume_samples_compatible

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think"""

DEFAULT_FINAL_PROMPT = """<Q><COT>
Therefore, the answer is \\(\\boxed{"""

DEFAULT_COT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.3,
    top_k=500,
    top_p=0.4,
    alpha_presence=0.5,
    alpha_frequency=0.1,
    alpha_decay=0.99,
    stop_tokens=(0, 261, 24281),
)

DEFAULT_FINAL_SAMPLING = SamplingConfig(
    max_generate_tokens=64,
    temperature=1.0,
    top_k=1,
    top_p=0.3,
    alpha_presence=0.0,
    alpha_frequency=0.0,
    alpha_decay=0.99,
    stop_tokens=(0, 2402, 4910),
)


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    output_path: Path
    problem_count: int


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
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        output_path: str,
        *,
        cot_prompt_template: str = DEFAULT_COT_PROMPT,
        final_answer_template: str = DEFAULT_FINAL_PROMPT,
        cot_sampling: SamplingConfig = DEFAULT_COT_SAMPLING,
        final_sampling: SamplingConfig = DEFAULT_FINAL_SAMPLING,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        pad_to_batch: bool = False,
        pass_k: Iterable[int] | None = None,
        write_output: bool = True,
        samples_per_task: int | None = None,
        probe_only: bool = False,
        overwrite_db_task: bool = False,
    ) -> FreeResponsePipelineResult:
        if DEFAULT_DB_CONFIG.enabled:
            return self._run_with_db(
                dataset_path,
                output_path,
                cot_prompt_template=cot_prompt_template,
                final_answer_template=final_answer_template,
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
                batch_size=batch_size,
                dataset_name=dataset_name,
                sample_limit=sample_limit,
                pass_k=pass_k,
                write_output=write_output,
                samples_per_task=samples_per_task,
                probe_only=probe_only,
                overwrite_db_task=overwrite_db_task,
            )

        samples_per_task = (samples_per_task or max(1, max(pass_k) if pass_k else 1)) if not probe_only else 1
        raw_records, resolved_name = self._load_records(dataset_path, sample_limit)
        problem_count = len(raw_records)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        target_path = Path(output_path)
        expanded: list[tuple[int, FreeAnswerRecord, int]] = []
        repeats = max(1, samples_per_task)
        for idx, record in enumerate(raw_records):
            for sample_id in range(repeats):
                expanded.append((idx, record, sample_id))
        if pad_to_batch and expanded and len(expanded) < batch_size:
            original_len = len(expanded)
            repeat = (batch_size + original_len - 1) // original_len
            expanded = (expanded * repeat)[:batch_size]
        if not expanded:
            return FreeResponsePipelineResult(dataset_name, 0, target_path, problem_count)

        if write_output and not probe_only:
            resume = detect_resume_state(target_path, repeats=repeats)
            if resume.has_progress:
                ensure_resume_samples_compatible(target_path, samples_per_task)
            start_index = min(resume.next_index, len(expanded))
            if start_index and len(expanded):
                remaining = max(len(expanded) - start_index, 0)
                print(f"⏩ 自由问答恢复运行：已完成 {start_index}/{len(expanded)}，剩余 {remaining}")
        else:
            start_index = 0
            resume = None
        remaining_entries = expanded[start_index:]
        if not remaining_entries:
            return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

        writer = JsonlStageWriter(target_path, resume=resume.has_progress) if write_output and not probe_only else None
        cot_prompts = [cot_prompt_template.replace("<Q>", record.question) for _, record, _ in remaining_entries]

        if probe_only:
            _ = self.engine.generate(
                cot_prompts,
                sampling=final_sampling,
                batch_size=batch_size,
                progress_desc="Generating answers",
                probe_only=probe_only,
            )
            return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

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

        if not write_output:
            return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

        final_by_idx = {item.prompt_index: item for item in final_outputs}
        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling), (2, final_sampling)])

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
            writer.write(
                SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=problem_idx,
                    repeat_index=sample_id,
                    sampling_config=sampling_config,
                    stages=stages,
                )
            )
        writer.close()
        return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

    def _run_with_db(
        self,
        dataset_path: str,
        output_path: str,
        *,
        cot_prompt_template: str,
        final_answer_template: str,
        cot_sampling: SamplingConfig,
        final_sampling: SamplingConfig,
        batch_size: int,
        dataset_name: str | None,
        sample_limit: int | None,
        pass_k: Iterable[int] | None,
        write_output: bool,
        samples_per_task: int | None,
        probe_only: bool,
        overwrite_db_task: bool,
    ) -> FreeResponsePipelineResult:
        db = DatabaseManager.instance()
        db.initialize(DEFAULT_DB_CONFIG)
        service = EvalDbService(db)

        samples_per_task = (samples_per_task or max(1, max(pass_k) if pass_k else 1)) if not probe_only else 1
        raw_records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        target_path = Path(output_path)
        problem_count = len(raw_records)
        run_tag = target_path.stem

        sampling_config = normalize_sampling_config_by_stage([(1, cot_sampling), (2, final_sampling)])
        runtime_config = {
            "batch_size": batch_size,
            "samples_per_task": samples_per_task,
            "probe_only": probe_only,
        }

        run_ctx = service.prepare_run(
            dataset_slug=benchmark_name,
            split_name=dataset_split,
            model_path=self.model_path,
            is_cot=True,
            run_tag=run_tag,
            sampling_config=sampling_config,
            runtime_config=runtime_config,
            code_version=None,
            overwrite_run=overwrite_db_task,
        )

        expanded: list[tuple[int, FreeAnswerRecord, int]] = []
        repeats = max(1, samples_per_task)
        for idx, record in enumerate(raw_records):
            for sample_id in range(repeats):
                expanded.append((idx, record, sample_id))

        if not expanded:
            return FreeResponsePipelineResult(dataset_name, 0, target_path, problem_count)

        try:
            total_items = len(expanded)
            for start_idx in range(0, total_items, batch_size):
                end_idx = min(start_idx + batch_size, total_items)
                batch = expanded[start_idx:end_idx]

                items_to_process: list[dict[str, Any]] = []
                for p_idx, rec, r_idx in batch:
                    ref_answer = _resolve_reference_answer(rec)
                    meta = {"subject": rec.subject, **(rec.metadata or {})}
                    sample_id = service.upsert_sample(
                        benchmark_name=run_ctx.benchmark_name,
                        dataset_split=run_ctx.dataset_split,
                        sample_index=p_idx,
                        question=rec.question,
                        reference_answer=ref_answer,
                        meta=meta,
                    )
                    run_sample_id = service.upsert_run_sample(
                        run_id=run_ctx.run_id,
                        sample_id=sample_id,
                        repeat_index=r_idx,
                        status="pending",
                        current_stage=None,
                    )
                    final_stage = service.fetch_latest_stage(run_sample_id=run_sample_id, stage="final")
                    if final_stage:
                        service.mark_run_sample_status(
                            run_sample_id=run_sample_id,
                            status="succeeded",
                            current_stage="final",
                            finished=True,
                        )
                        continue
                    cot_stage = service.fetch_latest_stage(run_sample_id=run_sample_id, stage="cot")
                    items_to_process.append(
                        {
                            "p_idx": p_idx,
                            "rec": rec,
                            "r_idx": r_idx,
                            "run_sample_id": run_sample_id,
                            "cot_prompt": cot_stage.prompt if cot_stage else None,
                            "cot": cot_stage.completion if cot_stage else None,
                        }
                    )

                if not items_to_process:
                    continue

                for item in items_to_process:
                    stage = "cot" if not item["cot"] else "final"
                    attempt_id, attempt_index = service.start_attempt(
                        run_sample_id=item["run_sample_id"],
                        current_stage=stage,
                    )
                    item["attempt_id"] = attempt_id
                    item["attempt_index"] = attempt_index

                cot_inputs = []
                cot_indices = []
                for i, item in enumerate(items_to_process):
                    if item["cot"]:
                        continue
                    prompt = cot_prompt_template.replace("<Q>", item["rec"].question)
                    item["cot_prompt"] = prompt
                    cot_inputs.append(prompt)
                    cot_indices.append(i)

                if cot_inputs:
                    if probe_only:
                        outputs = self.engine.generate(
                            cot_inputs,
                            sampling=final_sampling,
                            batch_size=len(cot_inputs),
                            probe_only=True,
                        )
                    else:
                        outputs = self.engine.generate(
                            cot_inputs,
                            sampling=cot_sampling,
                            batch_size=len(cot_inputs),
                            progress_desc=f"DB: CoT {start_idx}-{end_idx}",
                        )

                    for local_idx, output in enumerate(outputs):
                        orig_idx = cot_indices[local_idx]
                        item = items_to_process[orig_idx]
                        service.write_stage_output(
                            attempt_id=item["attempt_id"],
                            stage="cot",
                            seq=0,
                            prompt=output.prompt,
                            completion=output.text,
                            finish_reason=output.finish_reason,
                            is_final=True,
                        )
                        item["cot"] = output.text
                        item["cot_prompt"] = output.prompt

                if probe_only:
                    continue

                final_inputs = []
                final_indices = []
                for i, item in enumerate(items_to_process):
                    if not item["cot"]:
                        continue
                    cot_prompt_str = item.get("cot_prompt") or cot_prompt_template.replace("<Q>", item["rec"].question)
                    prompt = (
                        final_answer_template.replace("<Q>", cot_prompt_str)
                        .replace("<COT>", item["cot"])
                    )
                    final_inputs.append(prompt)
                    final_indices.append(i)

                if final_inputs:
                    outputs = self.engine.generate(
                        final_inputs,
                        sampling=final_sampling,
                        batch_size=len(final_inputs),
                        progress_desc=f"DB: Final {start_idx}-{end_idx}",
                    )
                    for local_idx, output in enumerate(outputs):
                        orig_idx = final_indices[local_idx]
                        item = items_to_process[orig_idx]
                        prior_context = f"{item['cot_prompt']}{item['cot']}"
                        delta_prompt2 = prompt_delta(output.prompt, prior_context)
                        service.write_stage_output(
                            attempt_id=item["attempt_id"],
                            stage="final",
                            seq=0,
                            prompt=delta_prompt2,
                            completion=output.text,
                            finish_reason=output.finish_reason,
                            is_final=True,
                        )
                        service.mark_attempt_status(
                            attempt_id=item["attempt_id"],
                            status="succeeded",
                            finished=True,
                        )
                        service.mark_run_sample_status(
                            run_sample_id=item["run_sample_id"],
                            status="succeeded",
                            current_stage="final",
                            latest_attempt_index=item["attempt_index"],
                            finished=True,
                        )
        except Exception as exc:
            service.mark_run_status(run_id=run_ctx.run_id, status="failed", error_msg=str(exc))
            raise
        service.mark_run_status(run_id=run_ctx.run_id, status="succeeded")

        if not write_output or probe_only:
            return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

        writer = JsonlStageWriter(target_path, resume=False)
        with db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    s.sample_index,
                    rs.repeat_index,
                    cot.prompt AS cot_prompt,
                    cot.completion AS cot_completion,
                    cot.finish_reason AS cot_finish_reason,
                    fin.prompt AS final_prompt,
                    fin.completion AS final_completion,
                    fin.finish_reason AS final_finish_reason
                FROM eval_run_sample rs
                JOIN eval_sample s ON s.id = rs.sample_id
                LEFT JOIN LATERAL (
                    SELECT so.prompt, so.completion, so.finish_reason
                    FROM eval_stage_output so
                    JOIN eval_attempt a ON a.id = so.attempt_id
                    WHERE a.run_sample_id = rs.id AND so.stage = 'cot' AND so.is_final = TRUE
                    ORDER BY so.created_at DESC
                    LIMIT 1
                ) cot ON TRUE
                LEFT JOIN LATERAL (
                    SELECT so.prompt, so.completion, so.finish_reason
                    FROM eval_stage_output so
                    JOIN eval_attempt a ON a.id = so.attempt_id
                    WHERE a.run_sample_id = rs.id AND so.stage = 'final' AND so.is_final = TRUE
                    ORDER BY so.created_at DESC
                    LIMIT 1
                ) fin ON TRUE
                WHERE rs.run_id = %s AND rs.status = 'succeeded'
                ORDER BY s.sample_index, rs.repeat_index
                """,
                (run_ctx.run_id,),
            )
            for row in cursor:
                cot_prompt = row["cot_prompt"]
                cot_text = row["cot_completion"]
                final_prompt = row["final_prompt"]
                final_text = row["final_completion"]
                if not cot_prompt or not cot_text or not final_text:
                    continue
                stages = [
                    StageRecord(
                        prompt=cot_prompt,
                        completion=cot_text,
                        stop_reason=row["cot_finish_reason"],
                    ),
                    StageRecord(
                        prompt=final_prompt,
                        completion=final_text,
                        stop_reason=row["final_finish_reason"],
                    ),
                ]
                writer.write(
                    SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=row["sample_index"],
                        repeat_index=row["repeat_index"],
                        sampling_config=sampling_config,
                        stages=stages,
                    )
                )
        writer.close()
        return FreeResponsePipelineResult(dataset_name, len(expanded), target_path, problem_count)

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
