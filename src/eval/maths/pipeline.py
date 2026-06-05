from __future__ import annotations

"""Maths benchmark pipeline for full free-response generation."""

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.execution_plan import AttemptKey
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.infer.backend import InferenceBackend
from src.infer.sampling import GenerationOutput, SamplingConfig

USER_SENTINEL = "\nUser:"
FREE_RESPONSE_STOP_TOKENS = (0,)

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think"""

DEFAULT_DIRECT_PROMPT = """User: <Q>

Assistant:"""

DEFAULT_FINAL_PROMPT = """<Q><COT>
Therefore, the answer is \\(\\boxed{"""


def _render_prompt(template: str, question: str) -> str:
    return template.replace("<Q>", question.lstrip()).rstrip(" ")


def _render_final_prompt(template: str, cot_prompt: str, cot_completion: str) -> str:
    return template.replace("<Q>", cot_prompt).replace("<COT>", cot_completion).rstrip(" ")


def _clip_user_sentinel(text: str) -> str:
    return text.split(USER_SENTINEL, 1)[0] if USER_SENTINEL in text else text


def _is_truncated(output: GenerationOutput) -> bool:
    return output.finish_reason in {"max_tokens", "max_length"}


def _output_stats(output: GenerationOutput) -> dict[str, object]:
    return {
        "truncated": _is_truncated(output),
        "stop_detail": output.finish_reason or "",
        "generated_token_count": len(output.token_ids),
    }


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    problem_count: int
    payloads: list[dict]


class FreeResponsePipeline:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def run(
        self,
        dataset_path: str,
        *,
        prompt_template: str = DEFAULT_COT_PROMPT,
        generation_sampling: SamplingConfig | None = None,
        cot_prompt_template: str | None = None,
        cot_sampling: SamplingConfig | None = None,
        final_answer_template: str | None = None,
        final_sampling: SamplingConfig | None = None,
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
        if cot_prompt_template:
            prompt_template = cot_prompt_template
        if generation_sampling is None:
            generation_sampling = cot_sampling
        if generation_sampling is None:
            raise ValueError("generation_sampling is required")
        use_final_stage = final_answer_template is not None or final_sampling is not None
        if use_final_stage:
            if final_answer_template is None:
                raise ValueError("final_answer_template is required when final_sampling is provided")
            if final_sampling is None:
                raise ValueError("final_sampling is required when final_answer_template is provided")

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

        generation_sampling = replace(generation_sampling, stop_tokens=FREE_RESPONSE_STOP_TOKENS)
        sampling_items = [(1, generation_sampling)]
        if use_final_stage:
            assert final_sampling is not None
            sampling_items.append((2, final_sampling))
        sampling_config = normalize_sampling_config_by_stage(sampling_items)

        if probe_only:
            prompts = [_render_prompt(prompt_template, record.question) for _key, record in remaining_entries]
            probe_seeds = [
                sample_repeat_seed(
                    key.sample_index,
                    key.repeat_index,
                    pass_index=key.pass_index,
                    stage=1,
                )
                for key, _record in remaining_entries
            ]
            cot_outputs = self.backend.generate(
                prompts,
                sampling=generation_sampling,
                batch_size=batch_size,
                progress_desc="Probing CoT" if use_final_stage else "Generating full responses",
                probe_only=probe_only,
                prompt_stop_suffixes=[(USER_SENTINEL,) for _ in prompts],
                prompt_seeds=probe_seeds,
            )
            if use_final_stage:
                assert final_answer_template is not None
                assert final_sampling is not None
                final_prompts: list[str] = []
                final_source_indices: list[int] = []
                for output in cot_outputs:
                    if 0 <= output.prompt_index < len(prompts):
                        final_prompts.append(
                            _render_final_prompt(
                                final_answer_template,
                                prompts[output.prompt_index],
                                _clip_user_sentinel(output.text),
                            )
                        )
                        final_source_indices.append(int(output.prompt_index))
                if final_prompts:
                    _ = self.backend.generate(
                        final_prompts,
                        sampling=final_sampling,
                        batch_size=min(batch_size, len(final_prompts)),
                        progress_desc="Probing final answers",
                        probe_only=probe_only,
                        prompt_stop_suffixes=[(USER_SENTINEL,) for _ in final_prompts],
                        prompt_seeds=[
                            sample_repeat_seed(
                                remaining_entries[idx][0].sample_index,
                                remaining_entries[idx][0].repeat_index,
                                pass_index=remaining_entries[idx][0].pass_index,
                                stage=2,
                            )
                            for idx in final_source_indices
                        ],
                    )
            return FreeResponsePipelineResult(dataset_name, len(expanded), problem_count, [])

        payloads: list[dict] = []
        chunk_size = max(1, int(batch_size))
        for start in range(0, len(remaining_entries), chunk_size):
            chunk = remaining_entries[start : start + chunk_size]
            prompts = [_render_prompt(prompt_template, record.question) for _key, record in chunk]

            if use_final_stage:
                assert final_answer_template is not None
                assert final_sampling is not None
                cot_outputs = self.backend.generate(
                    prompts,
                    sampling=generation_sampling,
                    batch_size=min(batch_size, len(prompts)),
                    progress_desc="Generating CoT",
                    prompt_stop_suffixes=[(USER_SENTINEL,) for _ in prompts],
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
                cot_by_idx: dict[int, GenerationOutput] = {
                    int(output.prompt_index): output
                    for output in cot_outputs
                    if 0 <= int(output.prompt_index) < len(chunk)
                }
                final_prompts: list[str] = []
                final_prompt_indices: list[int] = []
                final_prompt_deltas: dict[int, str] = {}
                cot_texts: dict[int, str] = {}
                for local_idx, cot_output in sorted(cot_by_idx.items()):
                    cot_text = _clip_user_sentinel(cot_output.text)
                    cot_texts[local_idx] = cot_text
                    full_final_prompt = _render_final_prompt(
                        final_answer_template,
                        prompts[local_idx],
                        cot_text,
                    )
                    prior_context = f"{prompts[local_idx]}{cot_text}"
                    try:
                        delta_prompt = prompt_delta(full_final_prompt, prior_context)
                    except ValueError as exc:
                        raise ValueError(
                            "math final_answer_template must preserve the <Q><COT> prefix "
                            "so stage payloads can be reconstructed"
                        ) from exc
                    final_prompt_deltas[local_idx] = delta_prompt
                    final_prompt_indices.append(local_idx)
                    final_prompts.append(full_final_prompt)

                if not final_prompts:
                    continue

                def _on_final_complete(output: GenerationOutput) -> None:
                    prompt_idx = output.prompt_index
                    if prompt_idx < 0 or prompt_idx >= len(final_prompt_indices):
                        return
                    local_idx = final_prompt_indices[prompt_idx]
                    cot_output = cot_by_idx.get(local_idx)
                    if cot_output is None:
                        return
                    key, _record = chunk[local_idx]
                    cot_text = cot_texts[local_idx]
                    final_text = _clip_user_sentinel(output.text)
                    cot_stage = StageRecord(
                        prompt=prompts[local_idx],
                        completion=cot_text,
                        stop_reason=cot_output.finish_reason,
                    )
                    final_stage = StageRecord(
                        prompt=final_prompt_deltas[local_idx],
                        completion=final_text,
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
                    cot_stats = _output_stats(cot_output)
                    final_stats = _output_stats(output)
                    payload["stats"] = {
                        "truncated": bool(final_stats["truncated"]),
                        "stop_detail": final_stats["stop_detail"],
                        "generated_token_count": int(cot_stats["generated_token_count"])
                        + int(final_stats["generated_token_count"]),
                        "stage1": cot_stats,
                        "stage2": final_stats,
                    }
                    payload["_stage"] = "answer"
                    if on_record is not None:
                        on_record(payload)
                    payloads.append(payload)

                _ = self.backend.generate(
                    final_prompts,
                    sampling=final_sampling,
                    batch_size=min(batch_size, len(final_prompts)),
                    progress_desc="Generating final answers",
                    on_complete=_on_final_complete,
                    prompt_stop_suffixes=[(USER_SENTINEL,) for _ in final_prompts],
                    prompt_seeds=[
                        sample_repeat_seed(
                            chunk[local_idx][0].sample_index,
                            chunk[local_idx][0].repeat_index,
                            pass_index=chunk[local_idx][0].pass_index,
                            stage=2,
                        )
                        for local_idx in final_prompt_indices
                    ],
                )
                continue

            def _on_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                key, _record = chunk[local_idx]
                clipped_text = _clip_user_sentinel(output.text)
                stats = _output_stats(output)
                stage = StageRecord(
                    prompt=prompts[local_idx],
                    completion=clipped_text,
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
                payload["stats"] = stats
                payload["_stage"] = "answer"
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.backend.generate(
                prompts,
                sampling=generation_sampling,
                batch_size=min(batch_size, len(prompts)),
                progress_desc="Generating full responses",
                on_complete=_on_complete,
                prompt_stop_suffixes=[(USER_SENTINEL,) for _ in prompts],
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


__all__ = [
    "DEFAULT_COT_PROMPT",
    "DEFAULT_DIRECT_PROMPT",
    "DEFAULT_FINAL_PROMPT",
    "FREE_RESPONSE_STOP_TOKENS",
    "FreeResponsePipeline",
    "FreeResponsePipelineResult",
    "USER_SENTINEL",
]
