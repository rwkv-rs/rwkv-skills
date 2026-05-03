from __future__ import annotations

"""Function-call benchmark evaluation pipeline."""

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.eval.benchmark_config import BenchmarkModelConfig
from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import GenerationOutput, InferenceEngine
from src.infer.sampling import SamplingConfig

from .common import sample_repeat_seed

if TYPE_CHECKING:
    from src.infer.model import ModelLoadConfig

DEFAULT_FUNCTION_CALL_SYSTEM_TEMPLATE = (
    "Tools:\n<TOOLS>\n"
    "Return only a JSON function call."
)
DEFAULT_FUNCTION_CALL_USER_TEMPLATE = "<HISTORY>"
SUPPORTED_FUNCTION_CALL_ENVS = (
    "json_function_call",
    "function_call",
    "single_turn_function_call",
    "multi_turn_function_call",
)


@dataclass(slots=True)
class FunctionCallEvent:
    type: str
    role: str | None = None
    content: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FunctionCallRunStats:
    steps: int = 0
    tool_calls: int = 0
    summaries: int = 0
    truncated_observations: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0


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

        chunk_size = max(1, int(batch_size))
        sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
        payloads: list[dict[str, Any]] = []

        for start in range(0, len(entries), chunk_size):
            chunk = entries[start : start + chunk_size]
            prompts = [self._make_prompt(record, config=config) for _idx, record, _sample_id in chunk]
            env_types = [str(record.env.get("type") or "json_function_call") for _idx, record, _sample_id in chunk]
            for env_type in env_types:
                if env_type not in SUPPORTED_FUNCTION_CALL_ENVS:
                    raise NotImplementedError(f"暂不支持的 function_call env.type: {env_type}")

            def _on_complete(output: GenerationOutput) -> None:
                local_idx = output.prompt_index
                if local_idx < 0 or local_idx >= len(chunk):
                    return
                record_idx, record, sample_id = chunk[local_idx]
                prompt = prompts[local_idx]
                completion = output.text or ""
                function_call_text = completion.strip()
                system_prompt = self._render_system_prompt(record, config=config)
                user_prompt = self._render_user_prompt(record, config=config)
                events = self._build_events(
                    record=record,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
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
                payload = {
                    "benchmark_name": benchmark_name,
                    "dataset_split": dataset_split,
                    "sample_index": record_idx,
                    "repeat_index": sample_id,
                    "sampling_config": sampling_config,
                    "prompt1": prompt,
                    "completion1": completion,
                    "stop_reason1": output.finish_reason,
                    "final_answer": function_call_text,
                    "events": [asdict(event) for event in events],
                    "stats": asdict(stats),
                    "function_call_env_type": str(record.env.get("type") or "json_function_call"),
                    "function_call_scorer_type": str(record.scorer.get("type") or "json_function_call_exact"),
                }
                if on_record is not None:
                    on_record(payload)
                payloads.append(payload)

            _ = self.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=min(chunk_size, len(prompts)),
                progress_desc="Generating function-call responses",
                on_complete=_on_complete,
                prompt_seeds=[
                    sample_repeat_seed(record_idx, sample_id, stage=1)
                    for record_idx, _record, sample_id in chunk
                ],
            )
        return FunctionCallPipelineResult(dataset_name, len(entries), payloads)

    def _build_events(
        self,
        *,
        record: FunctionCallTaskRecord,
        system_prompt: str,
        user_prompt: str,
        completion: str,
        function_call_text: str,
        finish_reason: str,
    ) -> list[FunctionCallEvent]:
        events: list[FunctionCallEvent] = []
        if system_prompt.strip():
            events.append(FunctionCallEvent(type="system", role="system", content=system_prompt))
        if record.messages:
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
        elif user_prompt.strip():
            events.append(FunctionCallEvent(type="user", role="user", content=user_prompt))
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

    def _make_prompt(self, record: FunctionCallTaskRecord, *, config: BenchmarkModelConfig | None) -> str:
        system_prompt = self._render_system_prompt(record, config=config)
        user_prompt = self._render_user_prompt(record, config=config)
        parts: list[str] = []
        if system_prompt.strip():
            parts.append(f"System: {system_prompt.strip()}")
        if user_prompt.strip():
            parts.append(user_prompt.strip())
        prompt = "\n\n".join(parts).rstrip()
        if not prompt.endswith("Assistant:"):
            prompt = f"{prompt}\n\nAssistant:"
        return prompt

    def _render_system_prompt(
        self,
        record: FunctionCallTaskRecord,
        *,
        config: BenchmarkModelConfig | None,
    ) -> str:
        template = (
            (config.function_call_system_template if config is not None else None)
            or (config.agent_system_template if config is not None else None)
            or DEFAULT_FUNCTION_CALL_SYSTEM_TEMPLATE
        )
        return self._render_template(template, record)

    def _render_user_prompt(
        self,
        record: FunctionCallTaskRecord,
        *,
        config: BenchmarkModelConfig | None,
    ) -> str:
        template = (
            (config.function_call_user_template if config is not None else None)
            or (config.agent_user_template if config is not None else None)
            or DEFAULT_FUNCTION_CALL_USER_TEMPLATE
        )
        return self._render_template(template, record)

    def _render_template(self, template: str, record: FunctionCallTaskRecord) -> str:
        attachments = self._format_named_items(record.attachments)
        tools = self._format_tools(record.tools)
        history = self._format_history(record)
        return (
            template.replace("<INSTRUCTION>", record.instruction)
            .replace("<ATTACHMENTS>", attachments)
            .replace("<TOOLS>", tools)
            .replace("<HISTORY>", history)
            .replace("<MESSAGES>", history)
            .replace("<MAX_STEPS>", str(record.max_steps or 1))
        )

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]]) -> str:
        return json.dumps(tools, ensure_ascii=False, indent=2)

    def _format_history(self, record: FunctionCallTaskRecord) -> str:
        messages = record.messages or [{"role": "user", "content": record.instruction}]
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            content = self._message_content(message)
            if role == "assistant":
                rendered.append(f"Assistant: {content}")
            elif role in {"tool", "function"}:
                rendered.append(f"User: Function output:\n{content}")
            elif role == "system":
                rendered.append(f"System: {content}")
            else:
                rendered.append(f"User: {content}")
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

    @staticmethod
    def _format_named_items(items: list[dict[str, Any]]) -> str:
        if not items:
            return "None"
        lines: list[str] = []
        for item in items:
            name = str(item.get("name") or item.get("path") or item.get("type") or "item").strip()
            desc = str(item.get("description") or item.get("summary") or item.get("value") or "").strip()
            if desc:
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}")
        return "\n".join(lines)

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


AgentEvent = FunctionCallEvent
AgentRunStats = FunctionCallRunStats
AgentPipelineResult = FunctionCallPipelineResult
AgentPipeline = FunctionCallPipeline


__all__ = [
    "FunctionCallEvent",
    "FunctionCallRunStats",
    "FunctionCallPipeline",
    "FunctionCallPipelineResult",
    "AgentEvent",
    "AgentRunStats",
    "AgentPipeline",
    "AgentPipelineResult",
]
