from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STOP_SIGNAL = "###STOP###"
TRANSFER_SIGNAL = "###TRANSFER###"
OUT_OF_SCOPE_SIGNAL = "###OUT-OF-SCOPE###"


def contains_stop_signal(text: str | None) -> bool:
    if not text:
        return False
    return (
        STOP_SIGNAL in text
        or TRANSFER_SIGNAL in text
        or OUT_OF_SCOPE_SIGNAL in text
    )


@dataclass(slots=True)
class HumanUser:
    """Simple terminal-backed user for manual runs."""

    total_cost: float = 0.0

    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return self.total_cost


@dataclass(slots=True)
class LLMUserSimulator:
    """LiteLLM-backed user simulator for tau v1-style interactions."""

    model_name: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    num_retries: int = 2
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0

    def build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = f"\n\nInstruction:\n{instruction}\n" if instruction else ""
        return (
            "You are a user interacting with a support agent."
            f"{instruction_display}\n"
            "Rules:\n"
            "- Reply with one natural user message per turn.\n"
            "- Reveal task details progressively instead of all at once.\n"
            "- Do not fabricate unknown details.\n"
            f"- If goal is satisfied, reply exactly {STOP_SIGNAL}.\n"
            "- Keep responses concise and realistic."
        )

    def reset(self, instruction: str | None = None) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self._generate_next_message()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._generate_next_message()

    def _generate_next_message(self) -> str:
        completion = _import_litellm_completion()
        response = completion(
            model=self.model_name,
            messages=self.messages,
            api_key=self.api_key,
            api_base=self.base_url,
            temperature=self.temperature,
            num_retries=self.num_retries,
        )
        message = response.choices[0].message
        payload = message.model_dump() if hasattr(message, "model_dump") else dict(message)
        self.messages.append(payload)
        hidden = getattr(response, "_hidden_params", None) or {}
        try:
            self.total_cost += float(hidden.get("response_cost") or 0.0)
        except Exception:
            pass
        content = payload.get("content")
        return "" if content is None else str(content)

    def get_total_cost(self) -> float:
        return self.total_cost


def _import_litellm_completion():
    try:
        from litellm import completion
    except Exception as exc:  # pragma: no cover - dependency is optional during unit tests
        raise RuntimeError(
            "litellm is required for LLM user simulation. Please install litellm first."
        ) from exc
    return completion


__all__ = [
    "STOP_SIGNAL",
    "TRANSFER_SIGNAL",
    "OUT_OF_SCOPE_SIGNAL",
    "contains_stop_signal",
    "HumanUser",
    "LLMUserSimulator",
]
