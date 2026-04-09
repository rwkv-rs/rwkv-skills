from __future__ import annotations

"""Sampling primitives shared across所有推理/评估流水线。"""

from dataclasses import dataclass, field, replace

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

    @property
    def max_new_tokens(self) -> int:
        return int(self.max_generate_tokens)

    @property
    def presence_penalty(self) -> float:
        return float(self.alpha_presence)

    @property
    def repetition_penalty(self) -> float:
        return float(self.alpha_frequency)

    @property
    def penalty_decay(self) -> float:
        return float(self.alpha_decay)

    def penalties_enabled(self) -> bool:
        return self.presence_penalty != 0.0 or self.repetition_penalty != 0.0

    def checked(self, vocab_size: int) -> "SamplingConfig":
        top_k = int(self.top_k)
        top_p = float(self.top_p)
        temperature = float(self.temperature)
        if temperature <= 0.0:
            temperature = 0.001
        else:
            temperature = min(temperature, 1000.0)

        if not (0 <= top_k <= int(vocab_size)):
            top_k = int(vocab_size)
        if not (0.0 <= top_p <= 1.0):
            top_p = 1.0
        if top_p == 0.0:
            top_k = 1
            top_p = 1.0

        return replace(
            self,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_generate_tokens=max(1, int(self.max_generate_tokens)),
        )


@dataclass(slots=True)
class GenerationOutput:
    """一次批量生成的单条结果。"""

    prompt_index: int
    prompt: str
    token_ids: list[int]
    text: str
    finish_reason: str
    tokens: list["GeneratedToken"] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedTokenCandidate:
    token_id: int | None
    text: str
    logprob: float
    bytes: bytes = b""


@dataclass(slots=True)
class GeneratedToken:
    token_id: int | None
    text: str
    bytes: bytes = b""
    logprob: float | None = None
    top_logprobs: list[GeneratedTokenCandidate] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedTextDelta:
    text: str
    tokens: list[GeneratedToken] = field(default_factory=list)

__all__ = [
    "GeneratedTextDelta",
    "GeneratedToken",
    "GeneratedTokenCandidate",
    "GenerationOutput",
    "SamplingConfig",
]
