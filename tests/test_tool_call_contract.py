from __future__ import annotations

import pytest

from src.eval.tasks.function_calling.tool_call_contract import (
    allowed_arguments_by_tool_name,
    normalize_tool_call_arguments,
    parse_tool_call_text,
    prune_tool_call_arguments,
    required_arguments_by_tool_name,
    validate_tool_call_name,
    validate_tool_call_required_arguments,
)


def _tool(name: str, *required: str) -> dict:
    return {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in required},
            "required": list(required),
        },
    }


def test_parse_tool_call_text_keeps_normal_and_native_contracts_separate() -> None:
    normal = parse_tool_call_text(
        '{"name":"assistant.lookup","arguments":{"id":"A1"},"confidence":0.9}',
        context_label="candidate",
        allowed_metadata_keys=("confidence",),
    )
    openai = parse_tool_call_text(
        '{"tool_calls":[{"function":{"name":"inspect","arguments":"{\\"id\\":\\"B2\\"}"}}]}'
    )

    assert normal.name == "lookup"
    assert normal.arguments == {"id": "A1"}
    assert normal.raw_payload["confidence"] == 0.9
    assert openai.name == "inspect"
    assert openai.arguments == {"id": "B2"}


def test_parse_tool_call_text_rejects_legacy_aliases_and_unapproved_metadata() -> None:
    with pytest.raises(ValueError, match="unsupported legacy fields"):
        parse_tool_call_text('{"tool_name":"lookup","parameters":{"id":"A1"}}')
    with pytest.raises(ValueError, match="unsupported extra fields"):
        parse_tool_call_text('{"name":"lookup","arguments":{"id":"A1"},"unexpected":true}')


def test_tool_call_schema_contract_prunes_and_validates_arguments() -> None:
    tools = [_tool("lookup", "id")]
    required = required_arguments_by_tool_name(tools)
    allowed = allowed_arguments_by_tool_name(tools)
    call = parse_tool_call_text('{"name":"lookup","arguments":{"id":"A1","extra":"drop"}}')

    pruned = prune_tool_call_arguments(call, allowed_args_by_name=allowed)
    validate_tool_call_name(pruned, valid_names={"lookup"})
    validate_tool_call_required_arguments(pruned, required_args_by_name=required)

    assert pruned.arguments == {"id": "A1"}


def test_tool_call_schema_contract_reports_missing_required_arguments() -> None:
    call = parse_tool_call_text('{"name":"lookup","arguments":{}}')

    with pytest.raises(ValueError, match="missing required arguments"):
        validate_tool_call_required_arguments(call, required_args_by_name={"lookup": {"id"}})


def test_normalize_tool_call_arguments_applies_recursive_aliases() -> None:
    call = parse_tool_call_text(
        '{"name":"update","arguments":{"passengers":[{"date_of_birth":"1992-11-12"}]}}'
    )

    normalized = normalize_tool_call_arguments(call, aliases={"date_of_birth": "dob"})

    assert normalized.arguments == {"passengers": [{"dob": "1992-11-12"}]}
