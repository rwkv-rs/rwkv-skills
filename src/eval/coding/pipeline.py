from __future__ import annotations

"""Field-oriented coding pipeline aligned with rwkv-rs coding datasets."""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.execution_plan import AttemptKey
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed

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


def _extract_function_signature(code: str | None) -> str | None:
    if not code:
        return None
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            return stripped
    return None


def _format_signature_prompt(prompt: str, signature: str ) -> str:
    prompt = f"{prompt}\nFunction signature: {signature}\nWrite the full function definition."
    return _format_prompt_no_echo(prompt)


def _format_mbpp_user_prompt(prompt: str, signature: str | None) -> str:
    body = (
        f"{prompt}\nFunction signature: {signature}\nWrite the full function definition."
        if signature
        else prompt
    )
    body = _compress_newlines(body).strip()
    return (
        "User: You are a top-level code master.\n"
        f"{body}\n"
        "Output only the full Python function definition without any additional text or explanation.\n\n"
        "Assistant: "
    )


def _format_mbpp_prompt(prompt: str, signature: str | None, cot_mode: CoTMode) -> str:
    user_prompt = _format_mbpp_user_prompt(prompt, signature)
    if cot_mode is CoTMode.NO_COT:
        return f"{user_prompt}```python\n"
    if cot_mode is CoTMode.FAKE_COT:
        return f"{user_prompt}<think>\n</think>\n```python\n"
    return f"{user_prompt}<think>"


def _format_mbpp_final_prompt(cot_prompt: str, cot_completion: str) -> str:
    return f"{cot_prompt}{cot_completion}</think>\n```python\n"


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


class CodingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

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
                prompt_text = _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt)
                prompts.append(prompt_text)
                probe_seeds.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            _ = self.engine.generate(
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
                _format_prompt_no_echo(record.prompt) if is_human_eval_fix else _format_prompt(record.prompt),
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

                _ = self.engine.generate(
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
            prompts = []
            probe_seeds: list[int] = []
            for idx in range(batch_size):
                _record_idx, record = records[idx % len(records)]
                raw_code = record.metadata.get("code") if record.metadata else None
                signature = _extract_function_signature(raw_code)
                prompt_text = _format_mbpp_prompt(record.prompt, signature, cot_mode)
                prompts.append(prompt_text)
                probe_seeds.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            if cot_mode is CoTMode.COT:
                cot_outputs = self.engine.generate(
                    prompts,
                    sampling=sampling,
                    batch_size=batch_size,
                    progress_desc="Probing CoT",
                    probe_only=True,
                    prompt_seeds=probe_seeds,
                )
                cot_by_idx = {item.prompt_index: item for item in cot_outputs}
                final_prompts = []
                for local_idx, prompt_text in enumerate(prompts):
                    cot_seq = cot_by_idx.get(local_idx)
                    cot_text = cot_seq.text if cot_seq is not None else ""
                    final_prompts.append(_format_mbpp_final_prompt(prompt_text, cot_text))
                if final_prompts:
                    _ = self.engine.generate(
                        final_prompts,
                        sampling=sampling,
                        batch_size=batch_size,
                        progress_desc="Probing final code",
                        probe_only=True,
                        prompt_seeds=probe_seeds,
                    )
            else:
                _ = self.engine.generate(
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
            signature = _extract_function_signature(raw_code)
            prompt_text = _format_mbpp_prompt(record.prompt, signature, cot_mode)
            entries.append((prompt_text, record, key))
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
                prompts = [entry[0] for entry in chunk]
                if cot_mode is CoTMode.COT:
                    def _on_cot_complete(output: GenerationOutput) -> None:
                        local_idx = output.prompt_index
                        if local_idx < 0 or local_idx >= len(chunk):
                            return
                        prompt_text, _record, key = chunk[local_idx]
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

                    cot_outputs = self.engine.generate(
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
                            for _prompt_text, _record, key in chunk
                        ],
                    )
                    cot_by_idx = {item.prompt_index: item for item in cot_outputs}
                    final_prompts: list[str] = []
                    final_prompt_indices: list[int] = []
                    for local_idx, (prompt_text, _record, _key) in enumerate(chunk):
                        cot_seq = cot_by_idx.get(local_idx)
                        if cot_seq is None:
                            continue
                        final_prompts.append(_format_mbpp_final_prompt(prompt_text, cot_seq.text))
                        final_prompt_indices.append(local_idx)

                    def _on_final_complete(output: GenerationOutput) -> None:
                        local_idx = final_prompt_indices[output.prompt_index]
                        prompt_text, _record, key = chunk[local_idx]
                        cot_seq = cot_by_idx.get(local_idx)
                        if cot_seq is None:
                            return
                        prior_context = f"{prompt_text}{cot_seq.text}"
                        final_prompt = _format_mbpp_final_prompt(prompt_text, cot_seq.text)
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
                        _ = self.engine.generate(
                            final_prompts,
                            sampling=sampling,
                            batch_size=max(1, min(batch_size, len(final_prompts))),
                            progress_desc="Generating code",
                            on_complete=_on_final_complete,
                            prompt_seeds=[
                                sample_repeat_seed(
                                    chunk[local_idx][2].sample_index,
                                    chunk[local_idx][2].repeat_index,
                                    pass_index=chunk[local_idx][2].pass_index,
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

                    _ = self.engine.generate(
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
            prompts = []
            probe_seeds_stage1: list[int] = []
            for idx in range(batch_size):
                _record_idx, record = records[idx % len(records)]
                prompt_text = _format_lcb_cot_prompt(record.prompt, record.starter_code)
                prompts.append(prompt_text)
                probe_seeds_stage1.append(sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=1))
            cot_outputs = self.engine.generate(
                prompts,
                sampling=cot_sampling,
                batch_size=batch_size,
                progress_desc="Probing CoT",
                probe_only=True,
                prompt_seeds=probe_seeds_stage1,
            )
            final_prompts: list[str] = []
            cot_by_idx = {item.prompt_index: item for item in cot_outputs}
            for local_idx, prompt_text in enumerate(prompts):
                cot_seq = cot_by_idx.get(local_idx)
                cot_text = cot_seq.text if cot_seq is not None else ""
                final_prompts.append(_format_lcb_final_prompt(prompt_text, cot_text))
            if final_prompts:
                probe_seeds_stage2 = [
                        sample_repeat_seed(records[idx % len(records)][0], idx // len(records), stage=2)
                        for idx in range(len(final_prompts))
                    ]
                _ = self.engine.generate(
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
        entries = [
            (_format_lcb_cot_prompt(record.prompt, record.starter_code), record, key)
            for key, record in expanded
        ]
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
                prompts = [entry[0] for entry in chunk]

                def _on_cot_complete(output: GenerationOutput) -> None:
                    local_idx = output.prompt_index
                    if local_idx < 0 or local_idx >= len(chunk):
                        return
                    prompt_text, _record, key = chunk[local_idx]
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

                cot_outputs = self.engine.generate(
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
                        for _prompt_text, _record, key in chunk
                    ],
                )
                cot_by_idx = {item.prompt_index: item for item in cot_outputs}

                final_prompts: list[str] = []
                final_prompt_indices: list[int] = []
                for local_idx, (prompt_text, _record, _key) in enumerate(chunk):
                    cot_seq = cot_by_idx.get(local_idx)
                    if cot_seq is None:
                        continue
                    final_prompts.append(_format_lcb_final_prompt(prompt_text, cot_seq.text))
                    final_prompt_indices.append(local_idx)

                def _on_final_complete(output: GenerationOutput) -> None:
                    local_idx = final_prompt_indices[output.prompt_index]
                    prompt_text, _record, key = chunk[local_idx]
                    cot_seq = cot_by_idx.get(local_idx)
                    if cot_seq is None:
                        return
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
                            chunk[local_idx][2].sample_index,
                            chunk[local_idx][2].repeat_index,
                            pass_index=chunk[local_idx][2].pass_index,
                            stage=2,
                        )
                        for local_idx in final_prompt_indices
                    ]
                    _ = self.engine.generate(
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
