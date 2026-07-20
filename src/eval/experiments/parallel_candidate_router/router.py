"""Experimental parallel candidate router for TAU-style tool decisions.

This module intentionally does not touch task persistence or DB services. It
turns a full tool table into parallel tool shards, asks the same model for one
candidate tool call per shard, then asks the model to aggregate the candidate
set into one candidate-layer JSON object:

{"name":"...","arguments":{...},"confidence":0.0,"evidence":"..."}
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.eval.tasks.agent_bench.tau_official import (
    RESPOND_TOOL_NAME,
    _compact_tool_schema,
    _minimal_tool_schema,
    _tau_airline_reservation_criteria_from_user_text,
    _tau_airline_reservation_match_score,
)
from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text, truncate_text
from src.eval.tasks.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    build_rwkv_json_call_prompt,
)
from src.eval.tasks.function_calling.tool_call_contract import (
    allowed_arguments_by_tool_name,
    load_tool_call_payload,
    normalize_argument_aliases,
    normalize_tool_schema,
    parse_tool_call_text,
    required_arguments_by_tool_name,
    tool_name,
)

CANDIDATE_LAYER_KEYS = ("name", "arguments", "confidence", "evidence")
NO_CANDIDATE_TOOL_NAME = "__no_candidate__"
PARALLEL_CANDIDATE_ASSISTANT_PREFIX = assistant_json_prefix(enable_think=False, prefill_object=False)
_RESERVATION_ID_RE = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b", re.IGNORECASE)
_USER_ID_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z][a-z0-9]*_\d+\b", re.IGNORECASE)
_FLIGHT_NUMBER_RE = re.compile(r"\b[A-Z]{2,4}\d{2,4}\b", re.IGNORECASE)
_PAYMENT_ID_RE = re.compile(
    r"\b(?:(?:credit_card|gift_card|certificate|paypal)_\d+|payment_\d+|\d{6,})\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AIRLINE_CANCEL_REQUEST_RE = re.compile(
    r"\b(?:would like|want|wants|need|needs|proceed|help|assist|please|could you|can you|i'd like|i would like)"
    r"\b.{0,80}\b(?:cancel|canceling|cancelling|cancellation)\b"
    r"|\b(?:cancel|canceling|cancelling|cancellation)\b.{0,60}\b(?:reservation|booking|trip|flight)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ParallelCandidateRouterConfig:
    chunk_tools: int = 2
    batch_size: int = 16
    context_chars: int = 6000
    prompt_max_chars: int = 12288
    candidate_max_tokens: int = 192
    aggregate_max_tokens: int = 192
    max_candidates: int = 12
    tool_schema_mode: str = "compact"
    include_respond: bool = True
    fallback_to_highest_confidence: bool = True
    evidence_chars: int = 220
    policy_chars: int = 1200
    ground_identifier_arguments: bool = True


@dataclass(frozen=True, slots=True)
class CandidateToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: str = ""

    def layer_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "confidence": float(self.confidence),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class CandidateChunkTrace:
    chunk_index: int
    tool_names: tuple[str, ...]
    tool_schemas: tuple[dict[str, Any], ...]
    prompt: str
    completion: str
    finish_reason: str
    candidate: CandidateToolCall | None = None
    error: str | None = None

    def trace_payload(self, *, include_prompts: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_index": int(self.chunk_index),
            "tool_names": list(self.tool_names),
            "completion": self.completion,
            "finish_reason": self.finish_reason,
        }
        if self.candidate is not None:
            payload["candidate"] = self.candidate.layer_payload()
        if self.error:
            payload["error"] = self.error
        if include_prompts:
            payload["prompt"] = self.prompt
            payload["tools"] = [dict(schema) for schema in self.tool_schemas]
        return payload


@dataclass(frozen=True, slots=True)
class ParallelCandidateRouteResult:
    selected: CandidateToolCall | None
    chunks: tuple[CandidateChunkTrace, ...]
    aggregate_prompt: str
    aggregate_completion: str
    aggregate_finish_reason: str
    config: ParallelCandidateRouterConfig | None = None
    aggregate_error: str | None = None
    fallback_used: bool = False

    @property
    def candidates(self) -> tuple[CandidateToolCall, ...]:
        return tuple(chunk.candidate for chunk in self.chunks if chunk.candidate is not None)

    def trace_payload(self, *, include_prompts: bool = False) -> dict[str, Any]:
        selected_names = [self.selected.name] if self.selected is not None else []
        payload: dict[str, Any] = {
            "mode": "parallel_candidate",
            "routed": True,
            "reason": "parallel_candidate_fallback" if self.fallback_used else "parallel_candidate",
            "selected_names": selected_names,
            "candidate_count": len(self.candidates),
            "parallel_chunk_count": len(self.chunks),
            "candidates": [candidate.layer_payload() for candidate in self.candidates],
            "selected_candidate": self.selected.layer_payload() if self.selected is not None else None,
            "aggregate_completion": self.aggregate_completion,
            "aggregate_finish_reason": self.aggregate_finish_reason,
            "fallback_used": bool(self.fallback_used),
        }
        if self.config is not None:
            payload["config"] = _candidate_router_config_payload(self.config)
        if self.aggregate_error:
            payload["aggregate_error"] = self.aggregate_error
        payload["chunks"] = [chunk.trace_payload(include_prompts=include_prompts) for chunk in self.chunks]
        if include_prompts:
            payload["aggregate_prompt"] = self.aggregate_prompt
        return payload


def route_parallel_candidate_tool_call(
    *,
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, object]],
    messages_by_chunk: Sequence[Sequence[Mapping[str, object]]] | None = None,
    domain_policy: str,
    domain: str | None,
    facts_text: str | None,
    engine: Any,
    sampling: Any,
    config: ParallelCandidateRouterConfig | None = None,
    progress_desc: str = "ParallelCandidateRouter",
    prompt_seed: int | None = None,
) -> ParallelCandidateRouteResult:
    cfg = config or ParallelCandidateRouterConfig()
    tool_chunks = _chunk_tools(tools, chunk_size=max(1, int(cfg.chunk_tools)))
    if messages_by_chunk is not None and len(messages_by_chunk) != len(tool_chunks):
        raise ValueError(
            f"messages_by_chunk count={len(messages_by_chunk)} does not match tool chunk count={len(tool_chunks)}"
        )
    chunk_messages = (
        [list(chunk) for chunk in messages_by_chunk]
        if messages_by_chunk is not None
        else [list(messages) for _ in tool_chunks]
    )
    prompts = [
        build_candidate_prompt(
            chunk,
            messages=chunk_messages[chunk_index],
            domain_policy=domain_policy,
            domain=domain,
            facts_text=facts_text,
            config=cfg,
        )
        for chunk_index, chunk in enumerate(tool_chunks)
    ]
    required_args_by_name = _required_args_by_tool_name(tools, include_respond=cfg.include_respond)
    allowed_args_by_name = _allowed_args_by_tool_name(tools, include_respond=cfg.include_respond)
    chunks_by_index: dict[int, CandidateChunkTrace] = {}
    allowed_rows: list[tuple[int, list[Any], str]] = []
    for chunk_index, (chunk, prompt) in enumerate(zip(tool_chunks, prompts, strict=True)):
        valid_names = _chunk_valid_names(chunk, include_respond=cfg.include_respond)
        if len(prompt) > max(1, int(cfg.prompt_max_chars)):
            chunks_by_index[chunk_index] = CandidateChunkTrace(
                chunk_index=chunk_index,
                tool_names=tuple(name for name in valid_names if name != RESPOND_TOOL_NAME),
                tool_schemas=tuple(normalize_tool_schema(tool) for tool in chunk if tool_name(tool)),
                prompt=prompt,
                completion="",
                finish_reason="prompt_over_budget",
                candidate=None,
                error=f"candidate shard prompt_chars={len(prompt)} exceeds budget={int(cfg.prompt_max_chars)}",
            )
            continue
        allowed_rows.append((chunk_index, chunk, prompt))

    outputs = []
    if allowed_rows:
        allowed_prompts = [prompt for _chunk_index, _chunk, prompt in allowed_rows]
        outputs = engine.generate(
            allowed_prompts,
            sampling=_clamp_sampling(sampling, cfg.candidate_max_tokens),
            batch_size=min(len(allowed_prompts), max(1, int(cfg.batch_size))),
            progress_desc=progress_desc,
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in allowed_prompts],
            prompt_seeds=(
                None
                if prompt_seed is None
                else [int(prompt_seed) + chunk_index for chunk_index, _chunk, _prompt in allowed_rows]
            ),
            show_progress=False,
        )
    for output_index, (chunk_index, chunk, prompt) in enumerate(allowed_rows):
        output = outputs[output_index] if output_index < len(outputs) else None
        completion = _generation_text(output)
        finish_reason = _generation_finish_reason(output) if output is not None else "missing_output"
        valid_names = _chunk_valid_names(chunk, include_respond=cfg.include_respond)
        candidate_valid_names = set(valid_names)
        candidate_messages = chunk_messages[chunk_index]
        context_hint = _last_context_hint(candidate_messages)
        if not cfg.include_respond:
            candidate_valid_names.add(NO_CANDIDATE_TOOL_NAME)
        try:
            candidate = parse_candidate_tool_call(completion)
            candidate = _normalize_candidate_name_alias(candidate, valid_names=candidate_valid_names)
            _validate_candidate_name(candidate, valid_names=candidate_valid_names)
            if candidate.name == NO_CANDIDATE_TOOL_NAME:
                candidate = None
            else:
                candidate = _normalize_candidate_argument_aliases(candidate)
                candidate = _prune_candidate_arguments(candidate, allowed_args_by_name=allowed_args_by_name)
                _validate_candidate_arguments(candidate, required_args_by_name=required_args_by_name)
                if cfg.ground_identifier_arguments:
                    _validate_candidate_grounded_identifiers(candidate, messages=candidate_messages)
                _validate_candidate_domain_intent(candidate, messages=candidate_messages, domain=domain)
                candidate = _complete_candidate_layer(
                    candidate,
                    fallback_confidence=0.5,
                    fallback_evidence=f"candidate shard {chunk_index} selected {candidate.name} for: {context_hint}",
                    evidence_chars=cfg.evidence_chars,
                )
            error = None
        except Exception as exc:  # noqa: BLE001 - one bad shard should not discard other shards.
            candidate = None
            error = str(exc)
        chunks_by_index[chunk_index] = CandidateChunkTrace(
            chunk_index=chunk_index,
            tool_names=tuple(name for name in valid_names if name != RESPOND_TOOL_NAME),
            tool_schemas=tuple(normalize_tool_schema(tool) for tool in chunk if tool_name(tool)),
            prompt=prompt,
            completion=completion,
            finish_reason=finish_reason,
            candidate=candidate,
            error=error,
        )
    chunks = [chunks_by_index[index] for index in range(len(tool_chunks))]

    valid_candidates = [chunk.candidate for chunk in chunks if chunk.candidate is not None]
    aggregate_prompt = build_candidate_aggregate_prompt(
        valid_candidates,
        messages=messages,
        domain_policy=domain_policy,
        domain=domain,
        valid_tool_names=sorted(_all_valid_names(tools, include_respond=cfg.include_respond)),
        config=cfg,
    )
    aggregate_completion = ""
    aggregate_finish_reason = "missing_output"
    aggregate_error: str | None = None
    selected: CandidateToolCall | None = None
    fallback_used = False
    if not valid_candidates:
        aggregate_error = "no valid candidate tool calls"
    elif len(aggregate_prompt) > max(1, int(cfg.prompt_max_chars)):
        aggregate_error = f"candidate aggregate prompt_chars={len(aggregate_prompt)} exceeds budget={int(cfg.prompt_max_chars)}"
        aggregate_finish_reason = "prompt_over_budget"
    else:
        try:
            aggregate_outputs = engine.generate(
                [aggregate_prompt],
                sampling=_clamp_sampling(sampling, cfg.aggregate_max_tokens),
                batch_size=1,
                progress_desc=f"{progress_desc}-Aggregate",
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                prompt_seeds=None if prompt_seed is None else [int(prompt_seed) + 100_000],
                show_progress=False,
            )
            aggregate_output = aggregate_outputs[0] if aggregate_outputs else None
            aggregate_completion = _generation_text(aggregate_output)
            aggregate_finish_reason = _generation_finish_reason(aggregate_output)
            selected = parse_candidate_tool_call(aggregate_completion)
            aggregate_valid_names = _all_valid_names(tools, include_respond=cfg.include_respond)
            selected = _normalize_candidate_name_alias(selected, valid_names=aggregate_valid_names)
            _validate_candidate_name(selected, valid_names=aggregate_valid_names)
            selected = _normalize_candidate_argument_aliases(selected)
            selected = _prune_candidate_arguments(selected, allowed_args_by_name=allowed_args_by_name)
            _validate_candidate_arguments(selected, required_args_by_name=required_args_by_name)
            if cfg.ground_identifier_arguments:
                _validate_candidate_grounded_identifiers(selected, messages=messages)
            _validate_candidate_domain_intent(selected, messages=messages, domain=domain)
            matched = _matching_candidate(selected, valid_candidates)
            selected = _complete_candidate_layer(
                selected,
                fallback_confidence=(matched.confidence if matched is not None else 0.5),
                fallback_evidence=(
                    matched.evidence
                    if matched is not None and matched.evidence
                    else f"aggregator selected {selected.name} from parallel candidates for: {context_hint}"
                ),
                evidence_chars=cfg.evidence_chars,
            )
        except Exception as exc:  # noqa: BLE001 - experiment should continue and record fallback evidence.
            aggregate_error = str(exc)
            selected = None

    if selected is None and cfg.fallback_to_highest_confidence and valid_candidates:
        selected = _select_fallback_candidate(valid_candidates, messages=messages, domain=domain)
        fallback_used = True
    if selected is None and cfg.include_respond and not valid_candidates:
        selected = CandidateToolCall(
            name=RESPOND_TOOL_NAME,
            arguments={"content": _grounding_guard_question(domain=domain, messages=messages)},
            confidence=0.4,
            evidence="deterministic validation filtered every candidate tool call",
        )
        fallback_used = True
    return ParallelCandidateRouteResult(
        selected=selected,
        chunks=tuple(chunks),
        aggregate_prompt=aggregate_prompt,
        aggregate_completion=aggregate_completion,
        aggregate_finish_reason=aggregate_finish_reason,
        config=cfg,
        aggregate_error=aggregate_error,
        fallback_used=fallback_used,
    )


def build_candidate_prompt(
    tools: Sequence[Any],
    *,
    messages: Sequence[Mapping[str, object]],
    domain_policy: str,
    domain: str | None,
    facts_text: str | None,
    config: ParallelCandidateRouterConfig | None = None,
) -> str:
    cfg = config or ParallelCandidateRouterConfig()
    system_prompt = build_candidate_system_prompt(
        tools,
        domain_policy=domain_policy,
        domain=domain,
        facts_text=facts_text,
        config=cfg,
    )
    prompt = build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=max(1, int(cfg.context_chars)),
        assistant_prefix=PARALLEL_CANDIDATE_ASSISTANT_PREFIX,
        single_user_turn=False,
    )
    if len(prompt) <= cfg.prompt_max_chars:
        return prompt
    overflow = len(prompt) - cfg.prompt_max_chars
    trimmed_context = max(0, int(cfg.context_chars) - overflow - 512)
    return build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=trimmed_context,
        assistant_prefix=PARALLEL_CANDIDATE_ASSISTANT_PREFIX,
        single_user_turn=False,
    )


def build_candidate_system_prompt(
    tools: Sequence[Any],
    *,
    domain_policy: str,
    domain: str | None,
    facts_text: str | None,
    config: ParallelCandidateRouterConfig | None = None,
) -> str:
    cfg = config or ParallelCandidateRouterConfig()
    schemas = [_schema_for_prompt(tool, mode=cfg.tool_schema_mode) for tool in tools if tool_name(tool)]
    if cfg.include_respond:
        schemas.append(_respond_schema())
    else:
        schemas.append(_no_candidate_schema())
    lines = [
        "You are one worker in a parallel candidate tool-call router.",
        "Given this tool shard and the conversation, propose the single best next action.",
        "Return exactly one JSON object with only these fields:",
        '{"name":"tool_name","arguments":{},"confidence":0.0,"evidence":"short reason"}',
        "confidence must be a number from 0 to 1.",
        "evidence must cite the user request, prior tool output, policy, or tool schema that supports the candidate.",
        "Use exactly one name from this shard's Tools array. Do not invent tool names.",
        "Never invent ids, emails, phones, order ids, reservation ids, customer ids, line ids, item ids, or payment ids.",
        "Do not include id, type, tool_calls, function, requestor, role, rationale, analysis, markdown, or extra fields.",
    ]
    if cfg.include_respond:
        lines.extend(
            [
                "Use respond only when the assistant should send a user-facing message instead of calling a real tool.",
                "For reservation_id, user_id, payment_id/payment_method_id, flight_number, and required travel dates, use only exact values from the user text or successful tool output; otherwise choose respond and ask.",
            ]
        )
    else:
        lines.extend(
            [
                f"Use {NO_CANDIDATE_TOOL_NAME} when none of this shard's real tools is the right next action.",
                "Use final_answer only when the task is complete and the conversation already contains the answer or verification evidence.",
                "If required evidence or identifiers are missing, prefer a real tool that can inspect, search, execute, or verify; otherwise use __no_candidate__ for this shard.",
            ]
        )
    domain_name = str(domain or "").strip().lower()
    if domain_name == "telecom":
        if cfg.include_respond:
            lines.append("Telecom device actions in policy prose are not JSON tool names; use respond for user instructions.")
        else:
            lines.append("Telecom device actions in policy prose are not JSON tool names.")
    elif domain_name == "retail":
        lines.append("Retail IDs must come from the user request or prior tool outputs; preserve leading # on order IDs.")
    elif domain_name == "airline":
        lines.append("Airline reservation/user IDs must come from the user request or prior tool outputs.")
    if facts_text:
        lines.extend(["Known facts:", truncate_text(normalize_rwkv_text(facts_text), 1200)])
    lines.extend(
        [
            "Tools:",
            json.dumps(schemas, ensure_ascii=False, sort_keys=False, separators=(",", ":")),
            "Policy:",
            truncate_text(normalize_rwkv_text(domain_policy), max(200, int(cfg.policy_chars))),
        ]
    )
    return normalize_rwkv_text("\n".join(lines))


def build_candidate_aggregate_prompt(
    candidates: Sequence[CandidateToolCall],
    *,
    messages: Sequence[Mapping[str, object]],
    domain_policy: str,
    domain: str | None,
    valid_tool_names: Sequence[str],
    config: ParallelCandidateRouterConfig | None = None,
) -> str:
    cfg = config or ParallelCandidateRouterConfig()
    ranked_candidates = sorted(candidates, key=lambda item: float(item.confidence), reverse=True)[
        : max(1, int(cfg.max_candidates))
    ]
    if cfg.include_respond:
        missing_evidence_rule = (
            "Do not invent tool names or identifiers. If a required reservation/user/payment/flight ID or travel date is not in user text or successful tool output, choose respond and ask."
        )
    else:
        missing_evidence_rule = (
            "Do not invent tool names or identifiers. If required evidence is missing, choose a real candidate tool that can inspect, search, execute, or verify; do not use final_answer for missing evidence."
        )
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "You are the aggregator for a parallel candidate tool-call router.",
                "Choose the best next action from the candidate set for the official sandbox.",
                "Return exactly one JSON object with only these fields:",
                '{"name":"tool_name","arguments":{},"confidence":0.0,"evidence":"short reason"}',
                "Use only a valid tool name listed below and prefer a candidate's name/arguments unless the transcript evidence clearly fixes an ID or empty argument.",
                missing_evidence_rule,
                "Use final_answer only when the task is complete and the transcript contains the answer or verification evidence.",
                "Do not include id, type, tool_calls, function, requestor, role, analysis, or markdown.",
                f"Domain: {str(domain or '').strip() or 'unknown'}",
                "Valid tool names:",
                json.dumps(list(valid_tool_names), ensure_ascii=False, separators=(",", ":")),
                "Candidates:",
                json.dumps([item.layer_payload() for item in ranked_candidates], ensure_ascii=False, separators=(",", ":")),
                "Policy excerpt:",
                truncate_text(normalize_rwkv_text(domain_policy), min(900, max(200, int(cfg.policy_chars)))),
            ]
        )
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=max(1, int(cfg.context_chars)),
        assistant_prefix=PARALLEL_CANDIDATE_ASSISTANT_PREFIX,
        single_user_turn=False,
    )


def parse_candidate_tool_call(text: str) -> CandidateToolCall:
    try:
        parsed = parse_tool_call_text(
            text,
            context_label="candidate",
            allowed_metadata_keys=(
                "id",
                "confidence",
                "score",
                "evidence",
                "explanation",
                "reason",
                "rationale",
                "annotations",
                "citations",
            ),
        )
    except ValueError:
        direct_final = _parse_direct_final_answer_candidate(text)
        if direct_final is not None:
            return direct_final
        raise
    raw = parsed.raw_payload
    arguments = dict(parsed.arguments)
    misplaced_explanation = raw.get("explanation")
    if parsed.name == "final_answer":
        # G1h occasionally closes a stringified arguments object after `answer`
        # and emits the remaining final-answer fields in the candidate envelope.
        # They are tool arguments here, not router metadata.
        if misplaced_explanation is None:
            misplaced_explanation = _recover_escaped_candidate_string_field(text, "explanation")
        if "explanation" not in arguments and misplaced_explanation:
            arguments["explanation"] = misplaced_explanation
        if "confidence" not in arguments and raw.get("confidence") is not None:
            arguments["confidence"] = raw["confidence"]
    return CandidateToolCall(
        name=parsed.name,
        arguments=arguments,
        confidence=_coerce_confidence(raw.get("confidence", raw.get("score", 0.0))),
        evidence=normalize_rwkv_text(
            str(raw.get("evidence") or misplaced_explanation or raw.get("reason") or raw.get("rationale") or "")
        ),
    )


def _recover_escaped_candidate_string_field(text: str, field: str) -> str:
    marker = f'"{field}\\":\\"'
    source = str(text or "")
    start = source.find(marker)
    if start < 0:
        return ""
    escaped = source[start + len(marker) :]
    for match in re.finditer(r'\\"', escaped):
        suffix = escaped[match.end() :]
        if not (
            re.match(r'\s*,\s*"confidence"\s*:', suffix)
            or re.fullmatch(r"\s*\}\s*\}\s*", suffix)
        ):
            continue
        encoded_value = escaped[: match.start()]
        try:
            value = json.loads(f'"{encoded_value}"')
        except json.JSONDecodeError:
            return ""
        return normalize_rwkv_text(str(value))
    return ""


def _parse_direct_final_answer_candidate(text: str) -> CandidateToolCall | None:
    try:
        value = load_tool_call_payload(text, context_label="candidate")
    except ValueError:
        return None
    if not isinstance(value, Mapping):
        return None
    name = normalize_rwkv_text(str(value.get("name") or ""))
    arguments: dict[str, Any] = {}
    raw_arguments = value.get("arguments")
    if isinstance(raw_arguments, Mapping):
        arguments.update(raw_arguments)
    elif isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            decoded_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            decoded_arguments = None
        if isinstance(decoded_arguments, Mapping):
            arguments.update(decoded_arguments)
    explanation_value = (
        arguments.get("explanation")
        or value.get("explanation")
        or value.get("evidence")
        or _recover_escaped_candidate_string_field(text, "explanation")
        or ""
    )
    explanation = (
        normalize_rwkv_text(json.dumps(explanation_value, ensure_ascii=False, separators=(",", ":")))
        if isinstance(explanation_value, (Mapping, list))
        else normalize_rwkv_text(str(explanation_value))
    )
    confidence_value = arguments.get("confidence", value.get("confidence", value.get("score")))
    explicit_final = name == "final_answer" or "answer" in value or "final_answer" in value
    if not explicit_final:
        if name and not arguments and explanation and confidence_value is not None:
            return CandidateToolCall(
                name=name,
                confidence=_coerce_confidence(confidence_value),
                evidence=explanation,
            )
        return None
    answer = normalize_rwkv_text(
        str(arguments.get("answer") or value.get("answer") or value.get("final_answer") or "")
    )
    if not answer:
        return None
    arguments = {"answer": answer}
    if explanation:
        arguments["explanation"] = explanation
    if confidence_value is not None:
        arguments["confidence"] = confidence_value
    return CandidateToolCall(
        name="final_answer",
        arguments=arguments,
        confidence=_coerce_confidence(confidence_value),
        evidence=explanation,
    )


def _schema_for_prompt(tool: Any, *, mode: str) -> dict[str, Any]:
    schema = normalize_tool_schema(tool)
    if mode == "minimal":
        return _minimal_tool_schema(schema)
    if mode == "compact":
        return _compact_tool_schema(schema)
    if mode == "full":
        return schema
    raise ValueError(f"unsupported candidate router tool schema mode: {mode}")


def _respond_schema() -> dict[str, Any]:
    return {
        "name": RESPOND_TOOL_NAME,
        "description": "Send a natural-language message to the user. Include ###STOP### when the task is complete.",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    }


def _no_candidate_schema() -> dict[str, Any]:
    return {
        "name": NO_CANDIDATE_TOOL_NAME,
        "description": "Candidate-router-only abstention. Use when this shard has no appropriate real tool.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def _chunk_tools(tools: Sequence[Any], *, chunk_size: int) -> list[list[Any]]:
    routeable = [tool for tool in tools if tool_name(tool)]
    size = max(1, int(chunk_size))
    return [routeable[index : index + size] for index in range(0, len(routeable), size)] or [[]]


def _chunk_valid_names(tools: Sequence[Any], *, include_respond: bool) -> set[str]:
    names = {tool_name(tool) for tool in tools if tool_name(tool)}
    if include_respond:
        names.add(RESPOND_TOOL_NAME)
    return names


def _all_valid_names(tools: Sequence[Any], *, include_respond: bool) -> set[str]:
    return _chunk_valid_names(tools, include_respond=include_respond)


def _required_args_by_tool_name(tools: Sequence[Any], *, include_respond: bool) -> dict[str, set[str]]:
    extra_tools = (_respond_schema(),) if include_respond else ()
    return required_arguments_by_tool_name(tools, extra_tools=extra_tools)


def _allowed_args_by_tool_name(tools: Sequence[Any], *, include_respond: bool) -> dict[str, set[str]]:
    extra_tools = (_respond_schema(),) if include_respond else ()
    return allowed_arguments_by_tool_name(tools, extra_tools=extra_tools)


def _validate_candidate_name(candidate: CandidateToolCall, *, valid_names: set[str]) -> None:
    if candidate.name not in valid_names:
        raise ValueError(f"candidate name {candidate.name!r} not in valid tool names")


def _normalize_candidate_name_alias(
    candidate: CandidateToolCall,
    *,
    valid_names: set[str],
) -> CandidateToolCall:
    real_valid_names = valid_names.difference({NO_CANDIDATE_TOOL_NAME})
    if (
        real_valid_names == {"final_answer"}
        and candidate.name not in valid_names
        and not candidate.arguments
        and candidate.name
        and candidate.evidence
    ):
        return CandidateToolCall(
            name="final_answer",
            arguments={
                "answer": candidate.name,
                "explanation": candidate.evidence,
                "confidence": candidate.confidence,
            },
            confidence=candidate.confidence,
            evidence=candidate.evidence,
        )
    normalized_name = _candidate_name_alias(candidate, valid_names=valid_names)
    if normalized_name == candidate.name:
        return candidate
    return CandidateToolCall(
        name=normalized_name,
        arguments=dict(candidate.arguments),
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def _candidate_name_alias(candidate: CandidateToolCall, *, valid_names: set[str]) -> str:
    if candidate.name in valid_names:
        return candidate.name
    key = _canonical_candidate_name(candidate.name)
    aliases = {
        "get_reservation": "get_reservation_details",
        "get_reservation_detail": "get_reservation_details",
        "reservation_details": "get_reservation_details",
        "search_reservation": "get_reservation_details",
        "get_flight": "get_flight_status",
        "get_flight_detail": "get_flight_status",
        "get_flight_details": "get_flight_status",
        "get_flight_status": "get_flight_status",
        "flight_status": "get_flight_status",
        "cancel": "cancel_reservation",
        "cancel_flight": "cancel_reservation",
        "cancel_reservation": "cancel_reservation",
        "cancel_booking": "cancel_reservation",
        "search_direct": "search_direct_flight",
        "direct_flight": "search_direct_flight",
        "search_direct_flight": "search_direct_flight",
        "search_onestop": "search_onestop_flight",
        "search_one_stop": "search_onestop_flight",
        "search_one_stop_flight": "search_onestop_flight",
        "search_onestop_flight": "search_onestop_flight",
        "send_text": RESPOND_TOOL_NAME,
        "send_message": RESPOND_TOOL_NAME,
        "message_user": RESPOND_TOOL_NAME,
        "transfer_to_human": "transfer_to_human_agents",
        "transfer_to_human_agent": "transfer_to_human_agents",
        "transfer_to_human_travel": "transfer_to_human_agents",
    }
    alias = aliases.get(key)
    if alias in valid_names:
        return alias
    if key in {"update_reservation", "modify_reservation", "change_reservation"}:
        update_alias = _update_reservation_alias(candidate, valid_names=valid_names)
        if update_alias:
            return update_alias
    return candidate.name


def _update_reservation_alias(candidate: CandidateToolCall, *, valid_names: set[str]) -> str | None:
    argument_text = _canonical_candidate_name(json.dumps(candidate.arguments, ensure_ascii=False, sort_keys=True))
    intent_targets = [
        (
            "update_reservation_baggages",
            ("baggage", "baggages", "bag", "bags", "suitcase", "suitcases", "nonfree_baggages", "total_baggages"),
        ),
        (
            "update_reservation_passengers",
            ("passenger", "passengers", "first_name", "last_name", "dob", "date_of_birth"),
        ),
        (
            "update_reservation_flights",
            ("flight", "flights", "flight_number", "cabin", "upgrade", "class"),
        ),
    ]
    for target, markers in intent_targets:
        if target in valid_names and any(marker in argument_text for marker in markers):
            return target
    update_names = sorted(name for name in valid_names if name.startswith("update_reservation_"))
    if len(update_names) == 1:
        return update_names[0]
    return None


def _canonical_candidate_name(value: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or "").strip())
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def _prune_candidate_arguments(
    candidate: CandidateToolCall,
    *,
    allowed_args_by_name: Mapping[str, set[str]],
) -> CandidateToolCall:
    allowed = allowed_args_by_name.get(candidate.name)
    if allowed is None:
        return candidate
    pruned = {key: value for key, value in candidate.arguments.items() if key in allowed}
    if pruned == candidate.arguments:
        return candidate
    return CandidateToolCall(
        name=candidate.name,
        arguments=pruned,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def _normalize_candidate_argument_aliases(candidate: CandidateToolCall) -> CandidateToolCall:
    normalized = _normalize_argument_aliases(candidate.arguments)
    if normalized == candidate.arguments:
        return candidate
    return CandidateToolCall(
        name=candidate.name,
        arguments=normalized,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def _normalize_argument_aliases(value: Any) -> Any:
    return normalize_argument_aliases(value, aliases={"date_of_birth": "dob"})


def _select_fallback_candidate(
    candidates: Sequence[CandidateToolCall],
    *,
    messages: Sequence[Mapping[str, object]],
    domain: str | None,
) -> CandidateToolCall:
    indexed = list(enumerate(candidates))
    selected_index, selected_candidate = max(
        indexed,
        key=lambda item: (
            _fallback_candidate_score(item[1], messages=messages, domain=domain, candidate_count=len(candidates)),
            -item[0],
        ),
    )
    del selected_index
    return selected_candidate


def _fallback_candidate_score(
    candidate: CandidateToolCall,
    *,
    messages: Sequence[Mapping[str, object]],
    domain: str | None,
    candidate_count: int,
) -> float:
    score = float(candidate.confidence)
    if str(domain or "").strip().lower() != "airline":
        return score
    user_text = _actual_user_request_text(messages).lower()
    if _airline_user_requests_reservation_update(user_text):
        if candidate.name.startswith("update_reservation_"):
            score += 0.7
        if candidate.name == "update_reservation_baggages" and _airline_user_mentions_baggage(user_text):
            score += 0.3
        if candidate.name == "update_reservation_passengers" and re.search(
            r"\b(?:passenger|name|dob|date of birth|remove)\b",
            user_text,
        ):
            score += 0.3
        if candidate.name == "update_reservation_flights" and re.search(
            r"\b(?:flight|cabin|class|upgrade|reschedule|change|move)\b",
            user_text,
        ):
            score += 0.3
        if candidate.name in {"get_reservation_details", "get_user_details", "list_all_airports"}:
            score -= 0.15
    if _prior_candidate_tool_call_matches(candidate, messages):
        score -= 0.35
    if candidate.name == RESPOND_TOOL_NAME and candidate_count > 1:
        score -= 0.1
    return score


def _airline_user_requests_reservation_update(user_text: str) -> bool:
    text = str(user_text or "").lower()
    if not re.search(r"\b(?:change|update|modify|add|remove|upgrade|reschedule|move)\b", text):
        return False
    return bool(
        re.search(
            r"\b(?:reservation|flight|passenger|name|baggage|bags?|suitcases?|luggage|cabin|class|ticket)\b",
            text,
        )
    )


def _airline_user_mentions_baggage(user_text: str) -> bool:
    return bool(re.search(r"\b(?:baggage|bags?|suitcases?|luggage)\b", str(user_text or ""), re.IGNORECASE))


def _prior_candidate_tool_call_matches(
    candidate: CandidateToolCall,
    messages: Sequence[Mapping[str, object]],
) -> bool:
    target_args = _canonical_json(candidate.arguments)
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role not in {"assistant", "agent"}:
            continue
        parsed = _parse_prior_candidate_call(str(message.get("content") or ""))
        if parsed is None:
            continue
        name, arguments = parsed
        if name == candidate.name and _canonical_json(arguments) == target_args:
            return True
    return False


def _parse_prior_candidate_call(content: str) -> tuple[str, dict[str, Any]] | None:
    text = str(content or "").strip()
    if not text:
        return None
    if "<tool_call>" in text:
        start = text.find("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>", start)
        if end > start:
            text = text[start:end].strip()
    try:
        candidate = parse_candidate_tool_call(text)
    except Exception:
        return None
    return candidate.name, candidate.arguments


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_candidate_arguments(
    candidate: CandidateToolCall,
    *,
    required_args_by_name: Mapping[str, set[str]],
) -> None:
    required = required_args_by_name.get(candidate.name) or set()
    missing = sorted(key for key in required if key not in candidate.arguments or candidate.arguments.get(key) is None)
    if missing:
        raise ValueError(f"candidate {candidate.name!r} missing required arguments: {missing}")


def _validate_candidate_grounded_identifiers(
    candidate: CandidateToolCall,
    *,
    messages: Sequence[Mapping[str, object]],
) -> None:
    if candidate.name == RESPOND_TOOL_NAME:
        return
    grounded = _grounded_identifier_values(messages)
    violations: list[str] = []
    for kind, path, raw_value in _iter_candidate_identifier_arguments(
        candidate.arguments,
        tool_name=candidate.name,
    ):
        normalized = _normalize_identifier(kind, raw_value)
        if not normalized:
            continue
        if normalized not in grounded.get(kind, set()):
            violations.append(f"{'.'.join(path)}={str(raw_value)!r} ({kind})")
    if violations:
        joined = "; ".join(violations[:8])
        if len(violations) > 8:
            joined += f"; +{len(violations) - 8} more"
        raise ValueError(
            f"candidate {candidate.name!r} has ungrounded identifier arguments: {joined}"
        )


def _validate_candidate_domain_intent(
    candidate: CandidateToolCall,
    *,
    messages: Sequence[Mapping[str, object]],
    domain: str | None,
) -> None:
    if str(domain or "").strip().lower() != "airline":
        return
    if candidate.name == "cancel_reservation" and not _airline_user_requests_cancellation(messages):
        raise ValueError("candidate 'cancel_reservation' lacks explicit user cancellation intent")
    if candidate.name == "cancel_reservation":
        reservation_id = _normalize_identifier("reservation", candidate.arguments.get("reservation_id"))
        if (
            reservation_id
            and reservation_id not in _user_provided_reservation_ids(messages)
            and not _cancel_reservation_id_matches_user_request(reservation_id, messages)
        ):
            raise ValueError(
                "candidate 'cancel_reservation' uses a tool-output reservation_id that does not match the user request"
            )


def _airline_user_requests_cancellation(messages: Sequence[Mapping[str, object]]) -> bool:
    user_text = _actual_user_request_text(messages)
    if not user_text:
        return False
    return bool(_AIRLINE_CANCEL_REQUEST_RE.search(user_text))


def _actual_user_request_text(messages: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = str(message.get("content") or "")
        if not content:
            continue
        if _parse_tool_response_payload(content) is not None or _is_known_facts_message(content):
            continue
        parts.append(content)
    return normalize_rwkv_text("\n".join(parts))


def _user_provided_reservation_ids(messages: Sequence[Mapping[str, object]]) -> set[str]:
    ids: set[str] = set()
    for match in _RESERVATION_ID_RE.finditer(_actual_user_request_text(messages)):
        candidate = match.group(0)
        if not _looks_like_flight_number(candidate):
            ids.add(_normalize_identifier("reservation", candidate))
    return ids


def _cancel_reservation_id_matches_user_request(
    reservation_id: str,
    messages: Sequence[Mapping[str, object]],
) -> bool:
    requested = str(reservation_id or "").strip().upper()
    if not requested:
        return False
    criteria = _tau_airline_reservation_criteria_from_user_text(_actual_user_request_text(messages))
    for message in messages:
        content = str(message.get("content") or "")
        payload = _parse_tool_response_payload(content)
        if payload is None or not bool(payload.get("ok", True)):
            continue
        output = _parse_tool_output(payload.get("output"))
        if not isinstance(output, Mapping):
            continue
        observed_id = str(output.get("reservation_id") or "").strip().upper()
        if observed_id == requested and _tau_airline_reservation_match_score(output, criteria) > 0:
            return True
    return False


def _grounded_identifier_values(messages: Sequence[Mapping[str, object]]) -> dict[str, set[str]]:
    grounded: dict[str, set[str]] = {
        "reservation": set(),
        "user": set(),
        "payment": set(),
        "flight": set(),
        "date": set(),
    }
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if not content:
            continue
        tool_payload = _parse_tool_response_payload(content)
        if tool_payload is not None:
            if bool(tool_payload.get("ok", True)):
                _add_successful_tool_output_identifiers(grounded, tool_payload.get("output"))
            continue
        if _is_known_facts_message(content):
            _add_known_fact_identifiers(grounded, content)
            continue
        if role == "user":
            _add_user_text_identifiers(grounded, content)
    return grounded


def _iter_candidate_identifier_arguments(
    value: Any,
    *,
    tool_name: str,
    path: tuple[str, ...] = (),
) -> list[tuple[str, tuple[str, ...], Any]]:
    found: list[tuple[str, tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            child_path = (*path, key)
            if key_lower in {"reservation_id", "reservation_ids"}:
                found.extend(_iter_scalar_identifier_values("reservation", raw_item, path=child_path))
            elif key_lower == "reservations":
                found.extend(_iter_scalar_identifier_values("reservation", raw_item, path=child_path))
            elif key_lower == "user_id":
                found.extend(_iter_scalar_identifier_values("user", raw_item, path=child_path))
            elif key_lower in {"payment_id", "payment_method_id"}:
                found.extend(_iter_scalar_identifier_values("payment", raw_item, path=child_path))
            elif key_lower in {"payment_methods", "payment_method_ids"}:
                found.extend(_iter_payment_identifier_values(raw_item, path=child_path))
            elif key_lower in {"flight_id", "flight_number", "flight_numbers"}:
                found.extend(_iter_scalar_identifier_values("flight", raw_item, path=child_path))
            elif key_lower == "flights":
                found.extend(_iter_flight_identifier_values(raw_item, path=child_path))
            elif key_lower == "date" and _requires_grounded_date(tool_name, path=path):
                found.extend(_iter_scalar_identifier_values("date", raw_item, path=child_path))
            else:
                found.extend(_iter_candidate_identifier_arguments(raw_item, tool_name=tool_name, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_iter_candidate_identifier_arguments(item, tool_name=tool_name, path=(*path, str(index))))
    return found


def _iter_scalar_identifier_values(
    kind: str,
    value: Any,
    *,
    path: tuple[str, ...],
) -> list[tuple[str, tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        found: list[tuple[str, tuple[str, ...], Any]] = []
        for key, item in value.items():
            found.extend(_iter_scalar_identifier_values(kind, item, path=(*path, str(key))))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        found = []
        for index, item in enumerate(value):
            found.extend(_iter_scalar_identifier_values(kind, item, path=(*path, str(index))))
        return found
    return [(kind, path, value)] if str(value or "").strip() else []


def _iter_payment_identifier_values(value: Any, *, path: tuple[str, ...]) -> list[tuple[str, tuple[str, ...], Any]]:
    found: list[tuple[str, tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            if key_lower in {"payment_id", "payment_method_id"}:
                found.extend(_iter_scalar_identifier_values("payment", item, path=(*path, key)))
            elif _looks_like_payment_id(key):
                found.append(("payment", (*path, key), key))
            else:
                found.extend(_iter_payment_identifier_values(item, path=(*path, key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_iter_payment_identifier_values(item, path=(*path, str(index))))
    elif _looks_like_payment_id(str(value or "")):
        found.append(("payment", path, value))
    return found


def _iter_flight_identifier_values(value: Any, *, path: tuple[str, ...]) -> list[tuple[str, tuple[str, ...], Any]]:
    found: list[tuple[str, tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in {"flight_id", "flight_number", "flight_numbers"}:
                found.extend(_iter_scalar_identifier_values("flight", item, path=(*path, key)))
            else:
                found.extend(_iter_flight_identifier_values(item, path=(*path, key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_iter_flight_identifier_values(item, path=(*path, str(index))))
    elif _looks_like_flight_number(str(value or "")):
        found.append(("flight", path, value))
    return found


def _parse_tool_response_payload(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if text.startswith("Function output:"):
        text = text[len("Function output:") :].strip()
    elif "<tool_response>" in text:
        start = text.find("<tool_response>") + len("<tool_response>")
        end = text.find("</tool_response>", start)
        if end < 0:
            return None
        text = text[start:end].strip()
    else:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _add_successful_tool_output_identifiers(grounded: dict[str, set[str]], output: Any) -> None:
    parsed = _parse_tool_output(output)
    _add_output_identifier_values(grounded, parsed)
    if isinstance(parsed, str):
        _add_user_text_identifiers(grounded, parsed)


def _parse_tool_output(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return value


def _add_output_identifier_values(
    grounded: dict[str, set[str]],
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            if key_lower in {"reservation_id", "reservation_ids", "reservations"}:
                _add_identifier_values(grounded, "reservation", item)
            elif key_lower == "user_id":
                _add_identifier_values(grounded, "user", item)
            elif key_lower in {"payment_id", "payment_method_id", "payment_methods"}:
                _add_identifier_values(grounded, "payment", item)
            elif key_lower in {"flight_id", "flight_number", "flight_numbers"}:
                _add_identifier_values(grounded, "flight", item)
            elif key_lower == "date":
                _add_identifier_values(grounded, "date", item)
            else:
                _add_output_identifier_values(grounded, item, path=(*path, key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _add_output_identifier_values(grounded, item, path=(*path, str(index)))


def _add_known_fact_identifiers(grounded: dict[str, set[str]], content: str) -> None:
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        key, raw_value = line[1:].split(":", 1)
        key_lower = key.strip().lower()
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
        if key_lower in {"reservation_id", "reservations"}:
            _add_identifier_values(grounded, "reservation", values)
        elif key_lower == "user_id":
            _add_identifier_values(grounded, "user", values)
        elif key_lower in {"payment_id", "payment_method_id", "payment_methods"}:
            _add_identifier_values(grounded, "payment", values)
        elif key_lower in {"flight_id", "flight_number", "flight_numbers"}:
            _add_identifier_values(grounded, "flight", values)
        elif key_lower == "date":
            _add_identifier_values(grounded, "date", values)


def _add_user_text_identifiers(grounded: dict[str, set[str]], text: str) -> None:
    normalized = str(text or "")
    for match in _USER_ID_RE.finditer(normalized):
        _add_identifier_values(grounded, "user", match.group(0))
    for match in _PAYMENT_ID_RE.finditer(normalized):
        _add_identifier_values(grounded, "payment", match.group(0))
    for match in _FLIGHT_NUMBER_RE.finditer(normalized):
        _add_identifier_values(grounded, "flight", match.group(0))
    for match in _ISO_DATE_RE.finditer(normalized):
        _add_identifier_values(grounded, "date", match.group(0))
    for match in _RESERVATION_ID_RE.finditer(normalized):
        candidate = match.group(0)
        if not _looks_like_flight_number(candidate):
            _add_identifier_values(grounded, "reservation", candidate)


def _add_identifier_values(grounded: dict[str, set[str]], kind: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if kind == "payment" and _looks_like_payment_id(str(key)):
                normalized_key = _normalize_identifier(kind, key)
                if normalized_key:
                    grounded[kind].add(normalized_key)
            _add_identifier_values(grounded, kind, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _add_identifier_values(grounded, kind, item)
        return
    normalized = _normalize_identifier(kind, value)
    if normalized:
        grounded[kind].add(normalized)


def _normalize_identifier(kind: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if kind in {"reservation", "flight"}:
        return text.upper()
    return text.casefold()


def _is_known_facts_message(content: str) -> bool:
    return str(content or "").lstrip().startswith("Known facts from previous tool outputs.")


def _looks_like_payment_id(value: str) -> bool:
    return bool(_PAYMENT_ID_RE.fullmatch(str(value or "").strip()))


def _looks_like_flight_number(value: str) -> bool:
    return bool(_FLIGHT_NUMBER_RE.fullmatch(str(value or "").strip().upper()))


def _requires_grounded_date(tool_name: str, *, path: tuple[str, ...]) -> bool:
    normalized_name = str(tool_name or "").strip()
    if normalized_name in {
        "search_direct_flight",
        "search_onestop_flight",
        "book_reservation",
        "update_reservation_flights",
    }:
        return True
    return any(part.lower() == "flights" for part in path)


def _grounding_guard_question(*, domain: str | None, messages: Sequence[Mapping[str, object]]) -> str:
    del messages
    if str(domain or "").strip().lower() == "airline":
        return (
            "I need the relevant reservation ID, user ID, payment method, flight number, or travel date "
            "before I can safely call a tool. Could you provide the missing detail?"
        )
    return "I need the relevant ID or required detail before I can safely call a tool. Could you provide it?"


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _complete_candidate_layer(
    candidate: CandidateToolCall,
    *,
    fallback_confidence: float,
    fallback_evidence: str,
    evidence_chars: int,
) -> CandidateToolCall:
    confidence = float(candidate.confidence)
    evidence = candidate.evidence.strip()
    if confidence <= 0.0:
        confidence = _coerce_confidence(fallback_confidence)
    if not evidence:
        evidence = fallback_evidence
    return CandidateToolCall(
        name=candidate.name,
        arguments=dict(candidate.arguments),
        confidence=confidence,
        evidence=truncate_text(evidence, max(1, int(evidence_chars))),
    )


def _matching_candidate(
    selected: CandidateToolCall,
    candidates: Sequence[CandidateToolCall],
) -> CandidateToolCall | None:
    for candidate in candidates:
        if candidate.name == selected.name and candidate.arguments == selected.arguments:
            return candidate
    for candidate in candidates:
        if candidate.name == selected.name:
            return candidate
    return None


def _last_context_hint(messages: Sequence[Mapping[str, object]]) -> str:
    for message in reversed(messages):
        role = str(message.get("role") or "").strip().lower()
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if role == "user" and content:
            return truncate_text(content, 140)
    for message in reversed(messages):
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if content:
            return truncate_text(content, 140)
    return "current conversation"


def _candidate_router_config_payload(config: ParallelCandidateRouterConfig) -> dict[str, Any]:
    return {
        "chunk_tools": int(config.chunk_tools),
        "batch_size": int(config.batch_size),
        "context_chars": int(config.context_chars),
        "prompt_max_chars": int(config.prompt_max_chars),
        "candidate_max_tokens": int(config.candidate_max_tokens),
        "aggregate_max_tokens": int(config.aggregate_max_tokens),
        "max_candidates": int(config.max_candidates),
        "tool_schema_mode": str(config.tool_schema_mode),
        "include_respond": bool(config.include_respond),
        "fallback_to_highest_confidence": bool(config.fallback_to_highest_confidence),
        "evidence_chars": int(config.evidence_chars),
        "policy_chars": int(config.policy_chars),
        "ground_identifier_arguments": bool(config.ground_identifier_arguments),
    }


def _clamp_sampling(sampling: Any, max_tokens: int) -> Any:
    clamp = getattr(sampling, "clamp", None)
    if callable(clamp):
        return clamp(max_tokens)
    return sampling


def _generation_text(output: Any) -> str:
    return normalize_rwkv_text(str(getattr(output, "text", "") or ""))


def _generation_finish_reason(output: Any) -> str:
    return str(getattr(output, "finish_reason", "stop") or "stop")


__all__ = [
    "CANDIDATE_LAYER_KEYS",
    "CandidateToolCall",
    "ParallelCandidateRouteResult",
    "ParallelCandidateRouterConfig",
    "build_candidate_aggregate_prompt",
    "build_candidate_prompt",
    "build_candidate_system_prompt",
    "parse_candidate_tool_call",
    "route_parallel_candidate_tool_call",
]
