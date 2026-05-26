from __future__ import annotations

"""Multi-turn function-calling agent evaluation pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from src.eval.benchmark_config import BenchmarkModelConfig
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.evaluators.common import sample_repeat_seed
from src.eval.function_calling.agent.adapters.apibank import create_apibank_level2_env
from src.eval.function_calling.agent.adapters.browsecomp_plus import (
    browsecomp_plus_run_from_agent_details,
    create_browsecomp_plus_env,
)
from src.eval.function_calling.agent.runner import AgentRunConfig, run_function_calling_agent
from src.eval.function_calling.common.payload import (
    FunctionCallRunStats,
    build_agent_completion_payload,
)
from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
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
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> FunctionCallAgentPipelineResult:
        _ = config
        _ = batch_size
        records, resolved_name = load_agent_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        repeats = max(1, int(samples_per_task or 1))
        skip_keys = skip_keys or set()

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
            env = create_agent_env(record)
            prompts: list[str] = []
            completions: list[str] = []

            def _generate_action(events, observation, step):
                prompt = self._make_prompt(record, events, observation, step)
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
                details=result.details,
            )
            browsecomp_plus_run = browsecomp_plus_run_from_agent_details(result.details)
            if browsecomp_plus_run is not None:
                payload["browsecomp_plus_run"] = browsecomp_plus_run
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
    ) -> str:
        tools_json = json.dumps(record.tools or [], ensure_ascii=False, indent=2)
        trajectory = _render_agent_trajectory(events)
        current = str(getattr(observation, "content", "") or "")
        return (
            "You are controlling tools in a function-calling environment.\n"
            "Respond with exactly one JSON tool call and no extra text.\n"
            'Use this shape: {"name":"ToolName","arguments":{"arg":"value"}}\n'
            "Available tools:\n"
            f"{tools_json}\n\n"
            f"Trajectory:\n{trajectory}\n\n"
            f"Current observation:\n{current}\n\n"
            "Assistant: <think>\n</think>\n```json\n"
        )


def create_agent_env(record: FunctionCallTaskRecord):
    env_type = str(record.env.get("type") or "")
    if env_type == "apibank_level2":
        return create_apibank_level2_env(record)
    if env_type == "browsecomp_plus":
        return create_browsecomp_plus_env(record)
    raise NotImplementedError(f"Unsupported function-calling agent env.type: {env_type}")


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
        metadata=_dict_value(payload.get("metadata")),
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


def _render_agent_trajectory(events: list[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        content = str(event.get("content") or "")
        if event_type == "observation":
            rendered.append(f"Environment: {content}")
        elif event_type == "model_output":
            rendered.append(f"Assistant action: {content}")
        elif event_type == "action":
            rendered.append(
                "Tool call: "
                + json.dumps(
                    {"name": event.get("name"), "arguments": event.get("arguments") or {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        elif event_type == "env_result":
            rendered.append(f"Environment: {content}")
        elif event_type == "error":
            rendered.append(f"Error: {content}")
    return "\n".join(part for part in rendered if part.strip()).strip()


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
