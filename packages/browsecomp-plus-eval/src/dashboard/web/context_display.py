from __future__ import annotations

import copy
from typing import Any

_ROLE_MESSAGE_LABELS = {
    "User message:": "User:",
    "Assistant message:": "Assistant:",
    "System message:": "System:",
    "Tool message:": "Tool:",
}

_FLOWER_ROLE_LABELS = {
    "User✿": "User: ",
    "Bot✿": "\nAssistant: ",
    "Assistant✿": "\nAssistant: ",
    "System✿": "\nSystem: ",
    "Tool✿": "\nTool: ",
}


def clean_legacy_role_message_labels(text: str) -> str:
    cleaned = text
    for old, new in _ROLE_MESSAGE_LABELS.items():
        cleaned = cleaned.replace(old, new)
    # G1h prompts use a flower delimiter as a chat-template sentinel. It is
    # meaningful to the model, but showing it verbatim makes the dashboard
    # look like the prompt/completion was concatenated or corrupted.
    for old, new in _FLOWER_ROLE_LABELS.items():
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("✿", "\n")
    return cleaned


def clean_context_for_display(context: dict[str, Any]) -> dict[str, Any]:
    display_context = copy.deepcopy(context)
    stages = display_context.get("stages")
    prompt_groups: list[Any] = [stages, display_context.get("strategy_a")]
    for group in prompt_groups:
        if isinstance(group, list):
            candidates = group
        elif isinstance(group, dict):
            candidates = [group]
        else:
            continue
        for stage in candidates:
            if not isinstance(stage, dict):
                continue
            prompt = stage.get("prompt")
            if isinstance(prompt, str):
                stage["prompt"] = clean_legacy_role_message_labels(prompt)
    return display_context


__all__ = ["clean_context_for_display", "clean_legacy_role_message_labels"]
