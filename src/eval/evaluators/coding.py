from __future__ import annotations

"""Code generation / HumanEval evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state, ensure_resume_samples_compatible

HUMAN_EVAL_CODE_SAMPLING = SamplingConfig(
    max_generate_tokens=1024,
    temperature=0.6,
    top_k=50,
    top_p=0.6,
    alpha_presence=0.25,
    alpha_frequency=0.25,
    alpha_decay=0.996,
    stop_tokens=(0, 261, 6884, 21214, 24281),
    pad_zero=True,
)

MBPP_EVAL_CODE_SAMPLING = SamplingConfig(
    max_generate_tokens=1024,
    temperature=0.6,
    top_k=50,
    top_p=0.6,
    alpha_presence=0.25,
    alpha_frequency=0.25,
    alpha_decay=0.996,
    stop_tokens=(0, 261, 6884, 21214, 24281),
    pad_zero=True,
)

LCB_COT_SAMPLING = SamplingConfig(
    max_generate_tokens=8192,
    temperature=0.6,
    top_k=50,
    top_p=0.6,
    alpha_presence=0.25,
    alpha_frequency=0.25,
    alpha_decay=0.996,
    stop_tokens=(0,),
    pad_zero=True,
)

LCB_FINAL_SAMPLING = SamplingConfig(
    max_generate_tokens=8192,
    temperature=0.6,
    top_k=50,
    top_p=0.6,
    alpha_presence=0.25,
    alpha_frequency=0.25,
    alpha_decay=0.996,
    stop_tokens=(6884, 21214),
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


def _format_prompt_no_echo(prompt: str) -> str:
    """Variant without echoing prompt after Assistant (used for bug-fix style prompts)."""

    clean = _compress_newlines(prompt).strip()
    return (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{clean}\n\nAssistant: <think></think>\n```python"
    )


_LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
_LCB_FORMAT_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
_LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _format_lcb_body(question: str, starter_code: str | None) -> str:
    clean = (question or "").strip()
    body = f"### Question:\n{clean}\n\n"
    if starter_code and starter_code.strip():
        body += f"### Format: {_LCB_FORMAT_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Format: {_LCB_FORMAT_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return body


_LCB_FINAL_ANSWER_PREFIX = "\nTherefore, the correct code is ```python\n"


def _format_lcb_cot_prompt(question: str, starter_code: str | None) -> str:
    body = _format_lcb_body(question, starter_code)
    return f"User: {_LCB_SYSTEM_MESSAGE}\n{body}Assistant: <think"


def _format_lcb_final_prompt(cot_prompt: str, cot_completion: str) -> str:
    return f"{cot_prompt}{cot_completion}{_LCB_FINAL_ANSWER_PREFIX}"


@dataclass(slots=True)
class CodingPipelineResult:
    dataset: str
    sample_count: int
    output_path: Path
    problem_count: int


class CodingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

    def run_human_eval(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = HUMAN_EVAL_CODE_SAMPLING,
        batch_size: int = 64,
        sample_limit: int | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else max(1, max(pass_k) if pass_k else 1)
        write_output = write_output and (not probe_only)
        records, dataset_name = self._load_records(dataset_path, sample_limit)
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, Path(output_path), 0)

        is_human_eval_fix = "human_eval_fix" in dataset_path.lower()

        if probe_only:
            prompts = []
            for idx in range(batch_size):
                record = records[idx % len(records)]
                prompt_text = _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt)
                prompts.append(prompt_text)
            _ = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=batch_size,
                progress_desc="Probing code",
                probe_only=True,
            )
            return CodingPipelineResult(dataset_name, len(prompts), Path(output_path), len(records))

        entries: list[tuple[str, CodeGenerationRecord, int, int]] = []
        for rec_idx, record in enumerate(records):
            for sample_idx in range(samples_per_task):
                prompt_text = (
                    _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt)
                )
                entries.append((prompt_text, record, rec_idx, sample_idx))

        target_path = Path(output_path)
        resume = detect_resume_state(target_path, repeats=samples_per_task)
        if resume.has_progress:
            ensure_resume_samples_compatible(target_path, samples_per_task)
        start_index = min(resume.next_index, len(entries))
        if start_index and len(entries):
            remaining = max(len(entries) - start_index, 0)
            print(f"⏩ HumanEval 恢复运行：已完成 {start_index}/{len(entries)}，剩余 {remaining}")
        pending_entries = entries[start_index:]
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
            sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                seq = output_by_idx.get(local_idx)
                if seq is None:
                    continue
                raw_output = seq.text or ""
                stage = StageRecord(
                    prompt=prompt_text,
                    completion=raw_output,
                    stop_reason=seq.finish_reason,
                )
                writer.write(
                    SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=rec_idx,
                        repeat_index=sample_idx,
                        sampling_config=sampling_config,
                        stages=[stage],
                    )
                )
            writer.close()

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            output_path=target_path,
            problem_count=len(records),
        )

    def run_mbpp(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = MBPP_EVAL_CODE_SAMPLING,
        batch_size: int = 64,
        sample_limit: int | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else max(1, max(pass_k) if pass_k else 1)
        write_output = write_output and (not probe_only)
        records, dataset_name = self._load_records(dataset_path, sample_limit)
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, Path(output_path), 0)

        is_human_eval_fix = "human_eval_fix" in dataset_path.lower()

        if probe_only:
            prompts = []
            for idx in range(batch_size):
                record = records[idx % len(records)]
                prompt_text = _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt)
                prompts.append(prompt_text)
            _ = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=batch_size,
                progress_desc="Probing code",
                probe_only=True,
            )
            return CodingPipelineResult(dataset_name, len(prompts), Path(output_path), len(records))

        entries: list[tuple[str, CodeGenerationRecord, int, int]] = []
        for rec_idx, record in enumerate(records):
            for sample_idx in range(samples_per_task):
                prompt_text = (
                    _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt)
                )
                entries.append((prompt_text, record, rec_idx, sample_idx))

        target_path = Path(output_path)
        resume = detect_resume_state(target_path, repeats=samples_per_task)
        if resume.has_progress:
            ensure_resume_samples_compatible(target_path, samples_per_task)
        start_index = min(resume.next_index, len(entries))
        if start_index and len(entries):
            remaining = max(len(entries) - start_index, 0)
            print(f"⏩ MBPP 恢复运行：已完成 {start_index}/{len(entries)}，剩余 {remaining}")
        pending_entries = entries[start_index:]
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
            sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                seq = output_by_idx.get(local_idx)
                if seq is None:
                    continue
                raw_output = seq.text or ""
                stage = StageRecord(
                    prompt=prompt_text,
                    completion=raw_output,
                    stop_reason=seq.finish_reason,
                )
                writer.write(
                    SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=rec_idx,
                        repeat_index=sample_idx,
                        sampling_config=sampling_config,
                        stages=[stage],
                    )
                )
            writer.close()

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            output_path=target_path,
            problem_count=len(records),
        )

    def run_livecodebench(
        self,
        dataset_path: str,
        output_path: str,
        *,
        cot_sampling: SamplingConfig = LCB_COT_SAMPLING,
        final_sampling: SamplingConfig = LCB_FINAL_SAMPLING,
        batch_size: int = 64,
        sample_limit: int | None = None,
        eval_timeout: float = 3.0,
        eval_workers: int = 4,
        pass_k: Iterable[int] = DEFAULT_PASS_K,
        probe_only: bool = False,
        write_output: bool = True,
    ) -> CodingPipelineResult:
        batch_size = max(1, int(batch_size))
        if probe_only and (sample_limit is None or sample_limit <= 0 or sample_limit > batch_size):
            sample_limit = batch_size
        samples_per_task = 1 if probe_only else max(1, max(pass_k) if pass_k else 1)
        write_output = write_output and (not probe_only)
        records, dataset_name = self._load_records(dataset_path, sample_limit)
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        if not records:
            return CodingPipelineResult(dataset_name, 0, Path(output_path), 0)

        if probe_only:
            prompts = []
            for idx in range(batch_size):
                record = records[idx % len(records)]
                prompt_text = _format_lcb_cot_prompt(record.prompt, record.starter_code)
                prompts.append(prompt_text)
            cot_outputs = self.engine.generate(
                prompts,
                sampling=cot_sampling,
                batch_size=batch_size,
                progress_desc="Probing CoT",
                probe_only=True,
            )
            final_prompts: list[str] = []
            cot_by_idx = {item.prompt_index: item for item in cot_outputs}
            for local_idx, prompt_text in enumerate(prompts):
                cot_seq = cot_by_idx.get(local_idx)
                cot_text = cot_seq.text if cot_seq is not None else ""
                final_prompts.append(_format_lcb_final_prompt(prompt_text, cot_text))
            if final_prompts:
                _ = self.engine.generate(
                    final_prompts,
                    sampling=final_sampling,
                    batch_size=batch_size,
                    progress_desc="Probing final code",
                    probe_only=True,
                )
            return CodingPipelineResult(dataset_name, len(prompts), Path(output_path), len(records))

        entries: list[tuple[str, CodeGenerationRecord, int, int]] = []
        for rec_idx, record in enumerate(records):
            for sample_idx in range(samples_per_task):
                prompt_text = _format_lcb_cot_prompt(record.prompt, record.starter_code)
                entries.append((prompt_text, record, rec_idx, sample_idx))

        target_path = Path(output_path)
        resume = detect_resume_state(target_path, repeats=samples_per_task)
        if resume.has_progress:
            ensure_resume_samples_compatible(target_path, samples_per_task)
        start_index = min(resume.next_index, len(entries))
        if start_index and len(entries):
            remaining = max(len(entries) - start_index, 0)
            print(f"⏩ LiveCodeBench 恢复运行：已完成 {start_index}/{len(entries)}，剩余 {remaining}")
        pending_entries = entries[start_index:]
        if pending_entries:
            prompts = [entry[0] for entry in pending_entries]
            cot_outputs = self.engine.generate(
                prompts,
                sampling=cot_sampling,
                batch_size=max(1, min(batch_size, len(prompts))),
                progress_desc="Generating CoT",
            )
            cot_by_idx = {item.prompt_index: item for item in cot_outputs}

            final_prompts: list[str] = []
            final_prompt_indices: list[int] = []
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                cot_seq = cot_by_idx.get(local_idx)
                if cot_seq is None:
                    continue
                final_prompts.append(_format_lcb_final_prompt(prompt_text, cot_seq.text))
                final_prompt_indices.append(local_idx)

            final_outputs = []
            if final_prompts:
                final_outputs = self.engine.generate(
                    final_prompts,
                    sampling=final_sampling,
                    batch_size=max(1, min(batch_size, len(final_prompts))),
                    progress_desc="Generating final code",
                )
            final_by_idx = {
                final_prompt_indices[item.prompt_index]: item for item in final_outputs
            }

            if not write_output:
                return CodingPipelineResult(dataset_name, len(cot_outputs), target_path, len(records))

            writer = JsonlStageWriter(target_path, resume=resume.has_progress)
            sampling_config = normalize_sampling_config_by_stage(
                [(1, cot_sampling), (2, final_sampling)]
            )
            for local_idx, (prompt_text, record, rec_idx, sample_idx) in enumerate(pending_entries):
                cot_seq = cot_by_idx.get(local_idx)
                final_seq = final_by_idx.get(local_idx)
                if cot_seq is None or final_seq is None:
                    continue
                prior_context = f"{prompt_text}{cot_seq.text}"
                final_prompt = _format_lcb_final_prompt(prompt_text, cot_seq.text)
                delta_prompt = prompt_delta(final_prompt, prior_context)
                cot_stage = StageRecord(
                    prompt=prompt_text,
                    completion=cot_seq.text,
                    stop_reason=cot_seq.finish_reason,
                )
                final_stage = StageRecord(
                    prompt=delta_prompt,
                    completion=final_seq.text,
                    stop_reason=final_seq.finish_reason,
                )
                writer.write(
                    SampleRecord(
                        benchmark_name=benchmark_name,
                        dataset_split=dataset_split,
                        sample_index=rec_idx,
                        repeat_index=sample_idx,
                        sampling_config=sampling_config,
                        stages=[cot_stage, final_stage],
                    )
                )
            writer.close()

        return CodingPipelineResult(
            dataset=dataset_name,
            sample_count=len(entries),
            output_path=target_path,
            problem_count=len(records),
        )

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[CodeGenerationRecord], str]:
        loader = JsonlCodeGenerationLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, infer_dataset_slug_from_path(dataset_path)


__all__ = [
    "CodingPipeline",
    "CodingPipelineResult",
    "HUMAN_EVAL_CODE_SAMPLING",
    "MBPP_EVAL_CODE_SAMPLING",
    "LCB_COT_SAMPLING",
    "LCB_FINAL_SAMPLING",
]
