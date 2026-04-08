from __future__ import annotations

"""OpenAI chat request preparation and response adaptation for the infer service."""

from dataclasses import dataclass
import json
from typing import Literal

from .api import (
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    ChatCompletionChunkToolCall,
    ChatCompletionChunkToolCallFunction,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionToolCall,
    ChatCompletionToolCallFunction,
    ChatNamedToolChoice,
    ChatResponseFormat,
    ChatTool,
    CompletionChoice,
    CompletionChunkResponse,
    CompletionRequest,
    CompletionResponse,
    completion_finish_reason_to_chat,
    completion_logprobs_to_chat_logprobs,
)


StructuredResponseMode = Literal["plain_text", "json_text", "tool_call"]


@dataclass(frozen=True, slots=True)
class PreparedChatCompletionRequest:
    completion_request: CompletionRequest
    response_mode: StructuredResponseMode


@dataclass(frozen=True, slots=True)
class _StructuredPlan:
    prompt_preamble: str | None
    response_mode: StructuredResponseMode


@dataclass(frozen=True, slots=True)
class _ValidatedToolDefinition:
    name: str
    description: str | None
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ValidatedToolChoice:
    mode: Literal["none", "auto", "required", "named"]
    name: str | None = None


def prepare_chat_completion_request(request: ChatCompletionRequest) -> PreparedChatCompletionRequest:
    if not request.messages:
        raise ValueError("messages cannot be empty")
    if request.n not in (None, 1):
        raise ValueError("only n=1 is supported")
    if request.top_logprobs is not None and request.logprobs is not True:
        raise ValueError("top_logprobs requires logprobs=true")

    structured_plan = _validate_chat_structured_output(request)
    prompt = _build_chat_prompt(
        request.messages,
        prompt_preamble=structured_plan.prompt_preamble,
        response_mode=structured_plan.response_mode,
    )
    max_tokens = request.max_completion_tokens
    if max_tokens is None:
        max_tokens = request.max_tokens

    completion_request = CompletionRequest(
        model=request.model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty,
        penalty_decay=request.penalty_decay,
        stop=request.stop,
        stream=request.stream,
        logprobs=request.top_logprobs if request.logprobs else None,
        candidate_token_texts=request.candidate_token_texts,
        seed=request.seed,
    )
    return PreparedChatCompletionRequest(
        completion_request=completion_request,
        response_mode=structured_plan.response_mode,
    )


def build_chat_completion_response(
    request: ChatCompletionRequest,
    prepared: PreparedChatCompletionRequest,
    response: CompletionResponse,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=response.id.replace("cmpl-", "chatcmpl-", 1),
        created=response.created,
        model=response.model,
        choices=[
            _build_chat_choice(
                request=request,
                response_mode=prepared.response_mode,
                choice=choice,
            )
            for choice in response.choices
        ],
    )


def build_completion_stream_responses(response: CompletionResponse) -> list[CompletionChunkResponse]:
    chunks: list[CompletionChunkResponse] = []
    for choice in response.choices:
        chunks.append(
            CompletionChunkResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    CompletionChoice(
                        text=choice.text,
                        index=choice.index,
                        finish_reason=None,
                        logprobs=choice.logprobs,
                    )
                ],
            )
        )
        chunks.append(
            CompletionChunkResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    CompletionChoice(
                        text="",
                        index=choice.index,
                        finish_reason=choice.finish_reason,
                        logprobs=None,
                    )
                ],
            )
        )
    return chunks


def build_chat_completion_stream_responses(
    request: ChatCompletionRequest,
    prepared: PreparedChatCompletionRequest,
    response: CompletionResponse,
) -> list[ChatCompletionChunkResponse]:
    chat_response = build_chat_completion_response(request, prepared, response)
    chunks: list[ChatCompletionChunkResponse] = []
    for choice in chat_response.choices:
        chunks.append(
            ChatCompletionChunkResponse(
                id=chat_response.id,
                created=chat_response.created,
                model=chat_response.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=ChatCompletionDelta(role="assistant"),
                    )
                ],
            )
        )

        delta = _build_stream_delta(choice)
        if delta.content is not None or delta.tool_calls is not None or choice.logprobs is not None:
            chunks.append(
                ChatCompletionChunkResponse(
                    id=chat_response.id,
                    created=chat_response.created,
                    model=chat_response.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=choice.index,
                            delta=delta,
                            finish_reason=None,
                            logprobs=choice.logprobs,
                        )
                    ],
                )
            )

        chunks.append(
            ChatCompletionChunkResponse(
                id=chat_response.id,
                created=chat_response.created,
                model=chat_response.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=ChatCompletionDelta(),
                        finish_reason=choice.finish_reason,
                    )
                ],
            )
        )
    return chunks


def _build_chat_choice(
    *,
    request: ChatCompletionRequest,
    response_mode: StructuredResponseMode,
    choice,
) -> ChatCompletionChoice:
    message = ChatCompletionMessage(role="assistant", content=choice.text)
    finish_reason = completion_finish_reason_to_chat(choice.finish_reason)
    if response_mode == "tool_call":
        try:
            parsed = _parse_tool_model_output(choice.text)
        except ValueError:
            pass
        else:
            if parsed["type"] == "message":
                message = ChatCompletionMessage(role="assistant", content=str(parsed["content"]))
            else:
                message = ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionToolCall(
                            id=f"call_{index}",
                            function=ChatCompletionToolCallFunction(
                                name=item["name"],
                                arguments=item["arguments"],
                            ),
                        )
                        for index, item in enumerate(parsed["tool_calls"])
                    ],
                )
                finish_reason = "tool_calls"

    return ChatCompletionChoice(
        index=choice.index,
        message=message,
        finish_reason=finish_reason,
        logprobs=completion_logprobs_to_chat_logprobs(choice.logprobs) if request.logprobs else None,
    )


def _build_stream_delta(choice: ChatCompletionChoice) -> ChatCompletionDelta:
    tool_calls = choice.message.tool_calls or []
    if tool_calls:
        return ChatCompletionDelta(
            tool_calls=[
                ChatCompletionChunkToolCall(
                    index=index,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=ChatCompletionChunkToolCallFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for index, tool_call in enumerate(tool_calls)
            ]
        )
    return ChatCompletionDelta(content=choice.message.text_content() or None)


def _validate_chat_structured_output(request: ChatCompletionRequest) -> _StructuredPlan:
    tools = request.tools or []
    has_tools = bool(tools)
    if request.response_format is not None and has_tools:
        raise ValueError("response_format and tools cannot be used together yet")

    if request.response_format is not None:
        if request.stop is not None:
            raise ValueError("stop is not supported together with response_format")
        return _validate_response_format(request.response_format)

    if not has_tools:
        if request.tool_choice is not None:
            raise ValueError("tool_choice requires tools to be provided")
        if request.parallel_tool_calls is not None:
            raise ValueError("parallel_tool_calls requires tools to be provided")
        return _StructuredPlan(prompt_preamble=None, response_mode="plain_text")

    if request.stop is not None:
        raise ValueError("stop is not supported together with tools")
    if request.logprobs:
        raise ValueError("logprobs are not supported together with tools yet")

    validated_tools = _validate_tools(tools)
    tool_choice = _validate_tool_choice(request.tool_choice, validated_tools)
    if tool_choice.mode == "none":
        return _StructuredPlan(prompt_preamble=None, response_mode="plain_text")
    return _StructuredPlan(
        prompt_preamble=_build_tool_prompt_preamble(
            validated_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=bool(request.parallel_tool_calls),
        ),
        response_mode="tool_call",
    )


def _validate_response_format(response_format: ChatResponseFormat) -> _StructuredPlan:
    if response_format.type == "text":
        return _StructuredPlan(prompt_preamble=None, response_mode="plain_text")
    if response_format.type == "json_object":
        return _StructuredPlan(
            prompt_preamble="Return only a valid JSON object.",
            response_mode="json_text",
        )
    json_schema = response_format.json_schema
    if json_schema is None or json_schema.schema_ is None:
        raise ValueError("response_format.json_schema.schema is required")
    prompt_lines = [f"Return only JSON that matches the schema `{json_schema.name}`."]
    if json_schema.description:
        prompt_lines.append(f"Schema description: {json_schema.description}")
    prompt_lines.append("JSON schema:")
    prompt_lines.append(json.dumps(json_schema.schema_, ensure_ascii=False, indent=2, sort_keys=True))
    return _StructuredPlan(
        prompt_preamble="\n".join(prompt_lines),
        response_mode="json_text",
    )


def _validate_tools(tools: list[ChatTool]) -> list[_ValidatedToolDefinition]:
    validated: list[_ValidatedToolDefinition] = []
    seen_names: set[str] = set()
    for tool in tools:
        if tool.type.lower() != "function":
            raise ValueError(f"unsupported tool type: {tool.type}")
        name = tool.function.name.strip()
        if not name:
            raise ValueError("tool.function.name cannot be empty")
        if name in seen_names:
            raise ValueError(f"duplicate tool name: {name}")
        seen_names.add(name)
        parameters = tool.function.parameters or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        validated.append(
            _ValidatedToolDefinition(
                name=name,
                description=tool.function.description,
                parameters=parameters,
            )
        )
    return validated


def _validate_tool_choice(
    tool_choice: str | ChatNamedToolChoice | None,
    tools: list[_ValidatedToolDefinition],
) -> _ValidatedToolChoice:
    if tool_choice is None:
        return _ValidatedToolChoice(mode="auto")
    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in {"none", "auto", "required"}:
            return _ValidatedToolChoice(mode=mode)
        raise ValueError(f"unsupported tool_choice mode: {tool_choice}")
    if tool_choice.type.lower() != "function":
        raise ValueError(f"unsupported tool_choice type: {tool_choice.type}")
    name = tool_choice.function.name.strip()
    if any(tool.name == name for tool in tools):
        return _ValidatedToolChoice(mode="named", name=name)
    raise ValueError(f"tool_choice references unknown function: {name}")


def _build_tool_prompt_preamble(
    tools: list[_ValidatedToolDefinition],
    *,
    tool_choice: _ValidatedToolChoice,
    parallel_tool_calls: bool,
) -> str:
    lines = [
        "You are using the OpenAI tool-calling interface.",
        "Return only JSON.",
    ]
    if tool_choice.mode == "auto":
        lines.append('For a direct answer, emit {"type":"message","content":"..."}')
    elif tool_choice.mode == "required":
        lines.append(
            "You must return one or more tool calls."
            if parallel_tool_calls
            else "You must return exactly one tool call."
        )
    elif tool_choice.mode == "named":
        assert tool_choice.name is not None
        lines.append(
            f"You must call the tool `{tool_choice.name}` one or more times."
            if parallel_tool_calls
            else f"You must call the tool `{tool_choice.name}` exactly once."
        )

    lines.append(
        'For tool calls, emit {"type":"tool_calls","tool_calls":[{"name":"tool_name","arguments":{...}}]}'
    )
    lines.append("Available tools:")
    for tool in tools:
        title = f"- {tool.name}"
        if tool.description:
            title = f"{title}: {tool.description}"
        lines.append(title)
        lines.append(json.dumps(tool.parameters, ensure_ascii=False, indent=2, sort_keys=True))
    return "\n".join(lines)


def _build_chat_prompt(
    messages: list[ChatCompletionMessage],
    *,
    prompt_preamble: str | None,
    response_mode: StructuredResponseMode,
) -> str:
    parts: list[str] = []
    if prompt_preamble:
        parts.append(f"System: {prompt_preamble}")
    for message in messages:
        parts.append(_render_prompt_message(message))

    last_role = _normalize_chat_role(messages[-1].role)
    if response_mode != "plain_text" or last_role != "assistant":
        parts.append("Assistant:")
    return "\n\n".join(parts)


def _render_prompt_message(message: ChatCompletionMessage) -> str:
    role = _normalize_chat_role(message.role)
    has_tool_calls = bool(message.tool_calls)
    if has_tool_calls and role != "assistant":
        raise ValueError("tool_calls are only valid on assistant messages")
    if message.tool_call_id and role != "tool":
        raise ValueError("tool_call_id is only valid on tool messages")

    if role == "assistant" and has_tool_calls:
        if message.text_content().strip():
            raise ValueError("assistant tool_calls messages cannot include content")
        tool_history = _render_tool_call_history_json(message.tool_calls or [])
        return f"Assistant: ```json\n{tool_history}\n```"

    content = message.text_content()
    if role == "tool":
        prefix = "Tool"
        if message.tool_call_id:
            prefix = f"Tool[{message.tool_call_id}]"
        return f"{prefix}: {content}"

    labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    return f"{labels[role]}: {content}"


def _normalize_chat_role(role: str) -> Literal["system", "user", "assistant", "tool"]:
    value = role.strip().lower()
    if value in {"system", "developer"}:
        return "system"
    if value in {"user", "assistant", "tool"}:
        return value
    raise ValueError(f"unknown chat role: {role}")


def _render_tool_call_history_json(tool_calls: list[ChatCompletionToolCall]) -> str:
    payload = {
        "type": "tool_calls",
        "tool_calls": [
            {
                "name": tool_call.function.name,
                "arguments": _parse_json_like(tool_call.function.arguments),
            }
            for tool_call in tool_calls
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_tool_model_output(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("tool output must be valid JSON") from exc

    if isinstance(parsed, dict):
        parsed_type = parsed.get("type")
        if parsed_type == "message" and isinstance(parsed.get("content"), str):
            return {"type": "message", "content": parsed["content"]}
        if parsed_type == "tool_calls" and isinstance(parsed.get("tool_calls"), list):
            return {"type": "tool_calls", "tool_calls": _parse_tool_call_items(parsed["tool_calls"])}
        if isinstance(parsed.get("tool_calls"), list):
            return {"type": "tool_calls", "tool_calls": _parse_tool_call_items(parsed["tool_calls"])}
        if "name" in parsed and "arguments" in parsed:
            return {"type": "tool_calls", "tool_calls": _parse_tool_call_items([parsed])}
        if isinstance(parsed.get("content"), str):
            return {"type": "message", "content": parsed["content"]}
    if isinstance(parsed, list):
        return {"type": "tool_calls", "tool_calls": _parse_tool_call_items(parsed)}
    raise ValueError("tool output JSON shape is unsupported")


def _parse_tool_call_items(items: list[object]) -> list[dict[str, str]]:
    tool_calls: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("tool call entries must be objects")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool call name is required")
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                arguments_text = json.dumps(arguments, ensure_ascii=False)
            else:
                arguments_text = arguments
        else:
            arguments_text = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        tool_calls.append({"name": name.strip(), "arguments": arguments_text})
    return tool_calls


def _parse_json_like(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("tool call arguments must be valid JSON") from exc


__all__ = [
    "PreparedChatCompletionRequest",
    "build_chat_completion_stream_responses",
    "build_chat_completion_response",
    "build_completion_stream_responses",
    "prepare_chat_completion_request",
]
