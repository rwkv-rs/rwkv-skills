from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .context_budget import normalize_rwkv_text

_TRANSCRIPT_PREFIX = "Conversation transcript JSON:"
_LEGACY_ROLE_RE = re.compile(r"^(User|Assistant|API):(?:\s?(.*))$")


def render_api_bank_history(history: Sequence[Mapping[str, Any]]) -> str:
    transcript: list[dict[str, Any]] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        if role == "User":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            transcript.append({"role": "user", "content": text})
        elif role == "AI":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            transcript.append({"role": "assistant", "content": text})
        elif role == "API":
            param_dict = item.get("param_dict")
            if not isinstance(param_dict, Mapping):
                param_dict = {}
            transcript.append(
                {
                    "role": "api",
                    "name": str(item.get("api_name") or "").strip(),
                    "arguments": dict(param_dict),
                    "response": item.get("result"),
                }
            )
    return render_api_bank_transcript(transcript)


def normalize_api_bank_instruction_for_prompt(instruction: str) -> str:
    normalized = normalize_rwkv_text(instruction)
    if normalized.startswith(_TRANSCRIPT_PREFIX):
        return normalized
    transcript = _parse_legacy_api_bank_transcript(normalized)
    if transcript is None:
        return normalized
    return render_api_bank_transcript(transcript)


def render_api_bank_transcript(transcript: Sequence[Mapping[str, Any]]) -> str:
    rows = [dict(item) for item in transcript if isinstance(item, Mapping)]
    if not rows:
        return ""
    return _TRANSCRIPT_PREFIX + "\n" + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _parse_legacy_api_bank_transcript(text: str) -> list[dict[str, str]] | None:
    transcript: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    preamble: list[str] = []
    saw_role = False
    for line in text.split("\n"):
        match = _LEGACY_ROLE_RE.match(line)
        if match is not None:
            saw_role = True
            if preamble:
                transcript.append({"role": "instruction", "content": "\n".join(preamble).strip()})
                preamble = []
            label, content = match.groups()
            role = {"User": "user", "Assistant": "assistant", "API": "api"}[label]
            current = {"role": role, "content": content or ""}
            transcript.append(current)
            continue
        if current is None:
            if line.strip():
                preamble.append(line)
            continue
        current["content"] = current["content"] + "\n" + line
    return transcript if saw_role and transcript else None


__all__ = [
    "normalize_api_bank_instruction_for_prompt",
    "render_api_bank_history",
    "render_api_bank_transcript",
]
