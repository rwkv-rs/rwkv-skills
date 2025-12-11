from __future__ import annotations

"""Code generation / HumanEval evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.metrics.code_generation.human_eval import evaluate_functional_correctness
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state, ensure_resume_samples_compatible

DEFAULT_CODE_SAMPLING = SamplingConfig(
    max_generate_tokens=1024,
    temperature=0.6,
    top_k=50,
    top_p=0.6,
    alpha_presence=0.25,
    alpha_frequency=0.25,
    alpha_decay=0.996,
    stop_tokens=(0, 261, 24281),
    pad_zero=True,
)

# Coding 默认只计算 pass@1；如需更高 k，请通过 CLI 传入
DEFAULT_PASS_K = (1,)


def _compress_newlines(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _format_prompt(prompt: str) -> str:
    """Match the rwkv_mmlu HumanEval prompt: duplicate code after Assistant."""

    clean = _compress_newlines(prompt).strip()
    return (
        "User:You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{clean}\n\nAssistant:{clean}"
    )


@dataclass(slots=True)
class CodingPipelineResult:
    dataset: str
    sample_count: int
    output_path: Path
    problem_count: int
    eval_results: dict[str, float] | None = None
    eval_details_path: Path | None = None


class CodingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

    def run_human_eval(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = DEFAULT_CODE_SAMPLING,
        batch_size: int = 64,
        sample_limit: int | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> CodingPipelineResult:
        samples_per_task = max(1, max(pass_k) if pass_k else 1)
        write_output = write_output and (not probe_only)
        records, dataset_name = self._load_records(dataset_path, sample_limit)
        if not records:
            return CodingPipelineResult(dataset_name, 0, Path(output_path), 0)

        entries: list[tuple[str, CodeGenerationRecord, int, int]] = []
        for rec_idx, record in enumerate(records):
            for sample_idx in range(samples_per_task):
                entries.append((_format_prompt(record.prompt), record, rec_idx, sample_idx))

        target_path = Path(output_path)
        resume = detect_resume_state(target_path)
        if resume.has_progress:
            ensure_resume_samples_compatible(target_path, samples_per_task)
        start_index = min(resume.next_index, len(entries))
        if start_index and len(entries):
            remaining = max(len(entries) - start_index, 0)
            print(f"⏩ HumanEval 恢复运行：已完成 {start_index}/{len(entries)}，剩余 {remaining}")
        pending_entries = entries[start_index:]
        if probe_only and pending_entries:
            pending_entries = pending_entries[: min(batch_size, len(pending_entries))]
        if pending_entries:
            prompts = [entry[0] for entry in pending_entries]
            outputs = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=max(1, min(batch_size, len(prompts))),
                progress_desc="Generating code",
            )
            output_by_idx = {item.prompt_index: item for item in outputs}

            if not write_output:
                return CodingPipelineResult(dataset_name, len(outputs), target_path, len(records))

            writer = JsonlStageWriter(target_path, resume=resume.has_progress)
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                seq = output_by_idx.get(local_idx)
                if seq is None:
                    continue
                raw_output = seq.text or ""
                completion = raw_output.rstrip()
                stage = StageRecord(
                    prompt=prompt_text,
                    output=raw_output,
                    finish_reason=seq.finish_reason,
                )
                metadata = {
                    "task_id": getattr(record, "task_id", f"{dataset_name}_{rec_idx}"),
                    "sample_id": sample_idx,
                    "prompt_raw": record.prompt,
                    "entry_point": record.entry_point,
                    "canonical_solution": record.canonical_solution,
                    "test": record.metadata.get("test") if record.metadata else None,
                    "completion": completion,
                }
                writer.write(
                    SampleRecord(
                        index=start_index + local_idx,
                        dataset=dataset_name,
                        stages=[stage],
                        metadata=metadata,
                    )
                )
            writer.close()

        eval_results = None
        eval_details_path = None
        try:
            eval_results, eval_details_path = evaluate_functional_correctness(
                sample_file=str(output_path),
                k=tuple(pass_k),
                n_workers=eval_workers,
                timeout=eval_timeout,
                problem_file=dataset_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ HumanEval 评测失败：{exc}")

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            output_path=target_path,
            problem_count=len(records),
            eval_results=eval_results,
            eval_details_path=Path(eval_details_path) if eval_details_path else None,
        )

    def run_mbpp(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = DEFAULT_CODE_SAMPLING,
        batch_size: int = 64,
        sample_limit: int | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> CodingPipelineResult:
        samples_per_task = max(1, max(pass_k) if pass_k else 1)
        write_output = write_output and (not probe_only)
        records, dataset_name = self._load_records(dataset_path, sample_limit)
        if not records:
            return CodingPipelineResult(dataset_name, 0, Path(output_path), 0)

        entries: list[tuple[str, CodeGenerationRecord, int, int]] = []
        for rec_idx, record in enumerate(records):
            for sample_idx in range(samples_per_task):
                entries.append((_format_prompt(record.prompt), record, rec_idx, sample_idx))

        target_path = Path(output_path)
        resume = detect_resume_state(target_path)
        if resume.has_progress:
            ensure_resume_samples_compatible(target_path, samples_per_task)
        start_index = min(resume.next_index, len(entries))
        if start_index and len(entries):
            remaining = max(len(entries) - start_index, 0)
            print(f"⏩ MBPP 恢复运行：已完成 {start_index}/{len(entries)}，剩余 {remaining}")
        pending_entries = entries[start_index:]
        if probe_only and pending_entries:
            pending_entries = pending_entries[: min(batch_size, len(pending_entries))]
        if pending_entries:
            prompts = [entry[0] for entry in pending_entries]
            outputs = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=max(1, min(batch_size, len(prompts))),
                progress_desc="Generating code",
            )
            output_by_idx = {item.prompt_index: item for item in outputs}

            if not write_output:
                return CodingPipelineResult(dataset_name, len(outputs), target_path, len(records))

            writer = JsonlStageWriter(target_path, resume=resume.has_progress)
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                seq = output_by_idx.get(local_idx)
                if seq is None:
                    continue
                raw_output = seq.text or ""
                completion = raw_output.rstrip()
                stage = StageRecord(
                    prompt=prompt_text,
                    output=raw_output,
                    finish_reason=seq.finish_reason,
                )
                metadata = {
                    "task_id": getattr(record, "task_id", f"{dataset_name}_{rec_idx}"),
                    "sample_id": sample_idx,
                    "prompt_raw": record.prompt,
                    "entry_point": record.entry_point,
                    "canonical_solution": record.canonical_solution,
                    "completion": completion,
                }
                if record.test_cases is not None:
                    metadata["test_cases"] = record.test_cases
                writer.write(
                    SampleRecord(
                        index=start_index + local_idx,
                        dataset=dataset_name,
                        stages=[stage],
                        metadata=metadata,
                    )
                )
            writer.close()

        eval_results = None
        eval_details_path = None
        try:
            from src.eval.metrics.code_generation.mbpp import evaluate_mbpp

            eval_results, eval_details_path = evaluate_mbpp(
                sample_file=str(output_path),
                k=tuple(pass_k),
                n_workers=eval_workers,
                timeout=eval_timeout,
                problem_file=dataset_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ MBPP 评测失败：{exc}")

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            output_path=target_path,
            problem_count=len(records),
            eval_results=eval_results,
            eval_details_path=Path(eval_details_path) if eval_details_path else None,
        )

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[CodeGenerationRecord], str]:
        loader = JsonlCodeGenerationLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, Path(dataset_path).stem


__all__ = ["CodingPipeline", "CodingPipelineResult", "DEFAULT_CODE_SAMPLING"]
