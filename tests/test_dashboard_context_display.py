from __future__ import annotations

from src.dashboard.web.context_display import clean_context_for_display, clean_legacy_role_message_labels


def test_clean_legacy_role_message_labels_restores_display_headers() -> None:
    assert (
        clean_legacy_role_message_labels("User message: hi\nAssistant message: hello\nSystem message: keep order")
        == "User: hi\nAssistant: hello\nSystem: keep order"
    )


def test_clean_legacy_role_message_labels_formats_g1h_flower_template() -> None:
    assert (
        clean_legacy_role_message_labels("User✿question✿Bot✿<think>")
        == "User: question\n\nAssistant: <think>"
    )


def test_clean_context_for_display_only_changes_stage_prompts() -> None:
    context = {
        "stages": [
            {
                "prompt": "User message: old prompt\nAssistant message: old answer",
                "completion": "Assistant message: real model output",
                "stop_reason": "stop",
            }
        ],
        "agent_result": "User message: parser error",
        "strategy_a": {"prompt": "User✿alternate question✿Bot✿", "completion": "raw"},
    }

    display = clean_context_for_display(context)

    assert display["stages"][0]["prompt"] == "User: old prompt\nAssistant: old answer"
    assert display["stages"][0]["completion"] == "Assistant message: real model output"
    assert display["agent_result"] == "User message: parser error"
    assert display["strategy_a"]["prompt"] == "User: alternate question\n\nAssistant: "
    assert display["strategy_a"]["completion"] == "raw"
    assert context["stages"][0]["prompt"] == "User message: old prompt\nAssistant message: old answer"
