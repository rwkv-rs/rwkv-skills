from __future__ import annotations

"""Simple tool-call benchmark evaluation pipeline."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.eval.benchmark_config import BenchmarkModelConfig
from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.common.events import FunctionCallEvent
from src.eval.function_calling.common.payload import (
    FunctionCallRunStats,
    build_one_step_completion_payload,
)
from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.sampling import SamplingConfig

from src.eval.evaluators.common import sample_repeat_seed

if TYPE_CHECKING:
    from src.infer.model import ModelLoadConfig

SUPPORTED_FUNCTION_CALL_ENVS = ("simple_tool_call",)


@dataclass(slots=True)
class FunctionCallPipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict[str, Any]]


class FunctionCallPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        from src.infer.model import load_rwkv_model

        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        *,
        sampling: SamplingConfig,
        batch_size: int = 8,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        samples_per_task: int | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int]] | None = None,
        config: BenchmarkModelConfig | None = None,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> FunctionCallPipelineResult:
        _ = config
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        repeats = max(1, int(samples_per_task or 1))
        skip_keys = skip_keys or set()

        entries: list[tuple[int, FunctionCallTaskRecord, int]] = []
        for idx, record in enumerate(records):
            for sample_id in range(repeats):
                if (idx, sample_id) in skip_keys:
                    continue
                entries.append((idx, record, sample_id))
        total_expected = len(records) * repeats
        if not entries:
            return FunctionCallPipelineResult(dataset_name, 0, [])

        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(entries):
                return FunctionCallPipelineResult(dataset_name, len(entries), [])
            entries = entries[resume_start_index:]
            print(
                f"⏩ Function-call 恢复运行：已完成 {resume_start_index}/{total_expected}，剩余 {len(entries)}"
            )

        skipped = total_expected - len(entries)
        if skipped > 0:
            print(f"⏩ Function-call 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        _ = batch_size
        chunk_size = 1
        sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
        payloads: list[dict[str, Any]] = []

        for start in range(0, len(entries), chunk_size):
            chunk = entries[start : start + chunk_size]
            prompts = [self._make_prompt(record) for _idx, record, _sample_id in chunk]
            env_types = [str(record.env.get("type") or "simple_tool_call") for _idx, record, _sample_id in chunk]
            for env_type in env_types:
                if env_type not in SUPPORTED_FUNCTION_CALL_ENVS:
                    raise NotImplementedError(f"不支持的 function_call env.type: {env_type}")

            def _on_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                record_idx, record, sample_id = chunk[local_idx]
                prompt = prompts[local_idx]
                completion = _trim_stop_suffixes(output.text or "", JSON_CALL_STOP_SUFFIXES)
                function_call_text = _complete_bfcl_exec_forced_prefix(record, completion).strip()
                events = self._build_events(
                    record=record,
                    completion=completion,
                    function_call_text=function_call_text,
                    finish_reason=output.finish_reason,
                )
                stats = FunctionCallRunStats(
                    steps=1,
                    tool_calls=0,
                    summaries=0,
                    truncated_observations=0,
                    prompt_chars=len(prompt),
                    completion_chars=len(completion),
                )
                payload = build_one_step_completion_payload(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=record_idx,
                    repeat_index=sample_id,
                    sampling_config=sampling_config,
                    prompt=prompt,
                    completion=completion,
                    stop_reason=output.finish_reason,
                    final_answer=function_call_text,
                    events=events,
                    stats=stats,
                    env_type=str(record.env.get("type") or "simple_tool_call"),
                    scorer_type=str(record.scorer.get("type") or "simple_tool_call"),
                )
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=min(chunk_size, len(prompts)),
                progress_desc="Generating function-call responses",
                on_complete=_on_complete,
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
                prompt_seeds=[
                    sample_repeat_seed(record_idx, sample_id, stage=1)
                    for record_idx, _record, sample_id in chunk
                ],
                preserve_prompt_whitespace=True,
            )
        return FunctionCallPipelineResult(dataset_name, len(entries), payloads)

    def _build_events(
        self,
        *,
        record: FunctionCallTaskRecord,
        completion: str,
        function_call_text: str,
        finish_reason: str,
    ) -> list[FunctionCallEvent]:
        events: list[FunctionCallEvent] = []
        for message in record.messages:
            role = str(message.get("role") or "user").strip().lower()
            events.append(
                FunctionCallEvent(
                    type=role,
                    role="user" if role in {"tool", "function"} else role,
                    content=self._message_content(message),
                    name=self._message_name(message),
                )
            )
        events.append(
            FunctionCallEvent(
                type="assistant",
                role="assistant",
                content=completion,
                metadata={"stop_reason": finish_reason},
            )
        )
        events.append(
            FunctionCallEvent(type="function_call", role="assistant", content=function_call_text)
        )
        return events

    def _make_prompt(self, record: FunctionCallTaskRecord) -> str:
        from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS
        from src.eval.function_calling.one_step.simple_tool_call import (
            build_simple_tool_call_prompt,
            normalize_simple_tool_call_manifest_row,
        )

        simple_record = normalize_simple_tool_call_manifest_row(
            {
                "task_id": record.task_id,
                "instruction": record.instruction or self._format_history(record),
                "tools": record.tools,
                "expected_tool_calls": record.expected_tool_calls,
                "metadata": record.metadata,
            },
            index=0,
        )
        prompt = build_simple_tool_call_prompt(
            simple_record,
            history_max_chars=DEFAULT_HISTORY_MAX_CHARS,
        )
        if _force_bfcl_exec_array_prefix(record):
            prompt += "[\n"
        return prompt

    def _format_history(self, record: FunctionCallTaskRecord) -> str:
        messages = record.messages or [{"role": "user", "content": record.instruction}]
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            content = self._message_content(message).lstrip().rstrip(" ")
            if role == "assistant":
                rendered.append(f"Assistant: {content}" if content else "Assistant:")
            elif role in {"tool", "function"}:
                rendered.append(f"User: Function output:\n{content}" if content else "User: Function output:")
            elif role == "system":
                rendered.append(f"System: {content}" if content else "System:")
            else:
                rendered.append(f"User: {content}" if content else "User:")
        return "\n\n".join(item for item in rendered if item.strip())

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        for key in ("content", "text", "observation", "result", "output"):
            value = message.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for key in ("tool_call", "function_call", "call"):
            value = message.get(key)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return ""

    @staticmethod
    def _message_name(message: dict[str, Any]) -> str | None:
        name = message.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    def _load_records(
        self,
        dataset_path: str,
        sample_limit: int | None,
    ) -> tuple[list[FunctionCallTaskRecord], str]:
        dataset = JsonlFunctionCallTaskLoader(str(dataset_path)).load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, infer_dataset_slug_from_path(dataset_path)



def _bfcl_expected_executable_calls(record: FunctionCallTaskRecord) -> list[str]:
    raw = (
        record.scorer.get("ground_truth")
        or record.metadata.get("expected_executable_calls")
        or record.metadata.get("bfcl_ground_truth")
    )
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list | tuple):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _force_bfcl_exec_array_prefix(record: FunctionCallTaskRecord) -> bool:
    category = str(record.metadata.get("category") or record.task_id or "").strip().lower()
    return "parallel" in category and len(_bfcl_expected_executable_calls(record)) > 1


def _complete_bfcl_exec_forced_prefix(record: FunctionCallTaskRecord, completion: str) -> str:
    if not _force_bfcl_exec_array_prefix(record):
        return completion
    stripped = completion.lstrip()
    if stripped.startswith("["):
        return completion
    if stripped.startswith("{") and not completion.rstrip().endswith("]"):
        return "[\n" + completion.rstrip() + "\n]"
    return "[\n" + completion

def _trim_stop_suffixes(text: str, stop_suffixes: tuple[str, ...]) -> str:
    earliest: int | None = None
    for suffix in stop_suffixes:
        index = text.find(suffix)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]
