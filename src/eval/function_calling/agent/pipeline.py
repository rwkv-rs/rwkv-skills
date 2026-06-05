from __future__ import annotations

"""Multi-turn function-calling agent evaluation pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.eval.benchmark_config import BenchmarkModelConfig
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.evaluators.common import sample_repeat_seed
from src.eval.function_calling.agent.adapters.registry import (
    FunctionCallAgentAdapter,
    create_agent_env,
    prompt_adapter_for_env,
    render_agent_trajectory,
)
from src.eval.function_calling.agent.runner import AgentRunConfig, run_function_calling_agent
from src.eval.function_calling.common.payload import (
    FunctionCallRunStats,
    build_agent_completion_payload,
)
from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
    tool_routing_config_from_benchmark_config,
)
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.engine import InferenceEngine
from src.infer.sampling import SamplingConfig

if TYPE_CHECKING:
    from src.infer.model import ModelLoadConfig


@dataclass(slots=True)
class FunctionCallAgentPipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict[str, Any]]


class FunctionCallAgentPipeline:
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
        batch_size: int = 1,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        samples_per_task: int | None = None,
        resume_start_index: int = 0,
        skip_keys: set[tuple[int, int]] | None = None,
        config: BenchmarkModelConfig | None = None,
        tool_routing_config: ToolRoutingConfig | None = None,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> FunctionCallAgentPipelineResult:
        _ = batch_size
        records, resolved_name = load_agent_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        repeats = max(1, int(samples_per_task or 1))
        skip_keys = skip_keys or set()
        resolved_tool_routing_config = tool_routing_config or tool_routing_config_from_benchmark_config(config)

        entries: list[tuple[int, FunctionCallTaskRecord, int]] = []
        for idx, record in enumerate(records):
            for sample_id in range(repeats):
                if (idx, sample_id) not in skip_keys:
                    entries.append((idx, record, sample_id))
        total_expected = len(records) * repeats
        if not entries:
            return FunctionCallAgentPipelineResult(dataset_name, 0, [])
        if resume_start_index < 0:
            resume_start_index = 0
        if resume_start_index:
            if resume_start_index >= len(entries):
                return FunctionCallAgentPipelineResult(dataset_name, len(entries), [])
            entries = entries[resume_start_index:]
            print(f"⏩ Function-agent 恢复运行：已完成 {resume_start_index}/{total_expected}，剩余 {len(entries)}")
        skipped = total_expected - len(entries)
        if skipped > 0:
            print(f"⏩ Function-agent 恢复运行：已跳过 {skipped}/{total_expected} 个样本")

        sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
        payloads: list[dict[str, Any]] = []
        for record_idx, record, sample_id in entries:
            adapter = prompt_adapter_for_env(str(record.env.get("type") or ""))
            env = create_agent_env(record)
            prompts: list[str] = []
            completions: list[str] = []
            tool_routes: list[dict[str, Any]] = []

            def _generate_action(events, observation, step):
                prompt = self._make_prompt(
                    record,
                    events,
                    observation,
                    step,
                    tool_routing_config=resolved_tool_routing_config,
                    tool_route_sink=tool_routes,
                    adapter=adapter,
                )
                prompts.append(prompt)
                outputs = self.engine.generate(
                    [prompt],
                    sampling=sampling,
                    batch_size=1,
                    progress_desc="Generating function-agent action",
                    prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                    prompt_seeds=[sample_repeat_seed(record_idx, sample_id, stage=step + 1)],
                    preserve_prompt_whitespace=True,
                )
                completion = _trim_stop_suffixes(outputs[0].text or "", JSON_CALL_STOP_SUFFIXES).strip()
                completions.append(completion)
                return completion

            result = run_function_calling_agent(
                env,
                _generate_action,
                config=AgentRunConfig(max_steps=record.max_steps or 8, timeout_s=record.time_limit_s),
            )
            stats = FunctionCallRunStats(
                steps=int(result.details.get("steps") or 0),
                tool_calls=sum(1 for event in result.events if event.get("type") == "action"),
                prompt_chars=sum(len(prompt) for prompt in prompts),
                completion_chars=sum(len(completion) for completion in completions),
            )
            details = dict(result.details)
            if tool_routes:
                details["tool_routes"] = tool_routes
            payload = build_agent_completion_payload(
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                sample_index=record_idx,
                repeat_index=sample_id,
                sampling_config=sampling_config,
                prompts=prompts,
                completions=completions,
                events=result.events,
                stats=stats,
                env_type=str(record.env.get("type") or ""),
                scorer_type=str(record.scorer.get("type") or ""),
                success=result.success,
                official_score=result.score,
                details=details,
            )
            if adapter.augment_payload is not None:
                adapter.augment_payload(payload, result.details)
            if on_record is not None:
                on_record(payload)
            payloads.append(payload)
        return FunctionCallAgentPipelineResult(dataset_name, len(entries), payloads)

    def _make_prompt(
        self,
        record: FunctionCallTaskRecord,
        events: list[Mapping[str, Any]],
        observation: Any,
        step: int,
        *,
        tool_routing_config: ToolRoutingConfig | None = None,
        tool_route_sink: list[dict[str, Any]] | None = None,
        adapter: FunctionCallAgentAdapter | None = None,
    ) -> str:
        tools = list(record.tools or [])
        env_type = str(record.env.get("type") or "")
        resolved_adapter = adapter or prompt_adapter_for_env(env_type)
        route = route_tools_for_prompt(
            tools,
            _tool_route_context_messages(record, events, observation, step),
            config=tool_routing_config,
            control_tool_names=resolved_adapter.control_tool_names,
        )
        selected_tools = route.selected_tools
        if route.mode != "off" and tool_route_sink is not None:
            tool_route_sink.append({"step": int(step), **route.trace_payload()})
        return resolved_adapter.render_prompt(record, events, observation, step, selected_tools)


def load_agent_records(
    dataset_path: str,
    sample_limit: int | None = None,
) -> tuple[list[FunctionCallTaskRecord], str]:
    path = Path(dataset_path)
    records: list[FunctionCallTaskRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: function agent row must be a JSON object")
            records.append(_agent_record_from_payload(payload, index=index))
            if sample_limit is not None and sample_limit > 0 and len(records) >= sample_limit:
                break
    return records, infer_dataset_slug_from_path(str(path))


def _agent_record_from_payload(payload: Mapping[str, Any], *, index: int) -> FunctionCallTaskRecord:
    task_id = payload.get("task_id") or payload.get("id") or f"function_agent_{index}"
    metadata = _dict_value(payload.get("metadata"))
    for key in ("domain", "index", "task", "benchmark_version", "tau_policy", "source_path"):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    return FunctionCallTaskRecord(
        task_id=str(task_id),
        instruction=str(payload.get("instruction") or ""),
        messages=_list_of_dicts(payload.get("messages")),
        expected_tool_calls=_list_of_dicts(payload.get("expected_tool_calls")),
        env=_dict_value(payload.get("env") or payload.get("environment") or payload.get("env_spec")),
        scorer=_dict_value(payload.get("scorer") or payload.get("scorer_spec") or payload.get("evaluation")),
        tools=_list_of_dicts(payload.get("tools") or payload.get("tools_spec")),
        attachments=_list_of_dicts(payload.get("attachments") or payload.get("files")),
        max_steps=_positive_int(payload.get("max_steps")),
        time_limit_s=_positive_float(payload.get("time_limit_s")),
        metadata=metadata,
    )


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    return None


def _tool_route_context_messages(
    record: FunctionCallTaskRecord,
    events: Sequence[Mapping[str, Any]],
    observation: Any,
    step: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if record.instruction:
        messages.append({"role": "user", "content": record.instruction})
    for message in record.messages[-4:]:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    trajectory = render_agent_trajectory(list(events))
    if trajectory:
        messages.append({"role": "assistant", "content": trajectory})
    current = str(getattr(observation, "content", "") or "")
    if current:
        messages.append({"role": "user", "content": current})
    messages.append({"role": "user", "content": f"step={int(step)}"})
    return messages

def _trim_stop_suffixes(text: str, stop_suffixes: tuple[str, ...]) -> str:
    earliest: int | None = None
    for suffix in stop_suffixes:
        index = text.find(suffix)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]


__all__ = [
    "FunctionCallAgentPipeline",
    "FunctionCallAgentPipelineResult",
    "create_agent_env",
    "load_agent_records",
]
