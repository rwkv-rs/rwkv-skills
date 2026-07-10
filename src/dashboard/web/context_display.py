from __future__ import annotations

import copy
from typing import Any

_ROLE_MESSAGE_LABELS = {
    "User message:": "User:",
    "Assistant message:": "Assistant:",
    "System message:": "System:",
    "Tool message:": "Tool:",
}


def clean_legacy_role_message_labels(text: str) -> str:
    cleaned = text
    for old, new in _ROLE_MESSAGE_LABELS.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def clean_context_for_display(context: dict[str, Any]) -> dict[str, Any]:
    display_context = copy.deepcopy(context)
    stages = display_context.get("stages")
    if not isinstance(stages, list):
        return display_context
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        prompt = stage.get("prompt")
        if isinstance(prompt, str):
            stage["prompt"] = clean_legacy_role_message_labels(prompt)
    return display_context


__all__ = ["clean_context_for_display", "clean_legacy_role_message_labels"]
