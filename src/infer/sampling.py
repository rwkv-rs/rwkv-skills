from __future__ import annotations

"""Sampling primitives shared across所有推理/评估流水线。"""

from dataclasses import dataclass, replace

DEFAULT_NO_PENALTY_TOKEN_IDS = (33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58)


@dataclass(slots=True)
class SamplingConfig:
    """描述一次生成过程的 sampling 策略。"""

    max_generate_tokens: int = 4096
    temperature: float = 0.3
    top_k: int = 50
    top_p: float = 0.3
    alpha_presence: float = 0.5
    alpha_frequency: float = 0.5
    alpha_decay: float = 0.99
    stop_tokens: tuple[int, ...] = (0, 261, 24281)
    ban_tokens: tuple[int, ...] | None = None
    pad_zero: bool = True
    no_penalty_token_ids: tuple[int, ...] = DEFAULT_NO_PENALTY_TOKEN_IDS

    def clamp(self, max_tokens: int | None) -> "SamplingConfig":
        if not max_tokens or max_tokens <= 0:
            return self
        return replace(self, max_generate_tokens=max(1, min(self.max_generate_tokens, max_tokens)))


@dataclass(slots=True)
class GenerationOutput:
    """一次批量生成的单条结果。"""

    prompt_index: int
    prompt: str
    token_ids: list[int]
    text: str
    finish_reason: str


__all__ = ["SamplingConfig", "GenerationOutput"]
