from __future__ import annotations

"""Sampling primitives shared across all inference / evaluation pipelines."""

import json
from dataclasses import dataclass, field, replace
from typing import Any

# RWKV world 词表（rwkv_vocab_v20230424）中前 256 个 token 为单字节，token id == byte + 1。
# 这里豁免重复惩罚的是：空格(33→0x20)、制表符(10→0x09)、数字 '0'-'9'(49-58→0x30-0x39)，
# 避免对空白与数字的正常重复施加 presence/frequency penalty。
DEFAULT_NO_PENALTY_TOKEN_IDS = (33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58)
MIN_NONZERO_SAMPLING_FLOAT = 1e-5


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
    # 0 = EOS/pad；261、24281 为 RWKV world 词表中的对话分隔 token（双换行 / 角色边界类），
    # 命中即停止生成，对应 RWKV 对话模板的回合结束标记。
    stop_tokens: tuple[int, ...] = (0, 261, 24281)
    ban_tokens: tuple[int, ...] | None = None
    # Optional hard vocabulary constraint supported by vLLM-compatible
    # backends.  This is intentionally token-id based so callers can resolve
    # model-specific literals (for example multiple-choice answer tokens)
    # through the serving tokenizer instead of hard-coding a vocabulary.
    allowed_token_ids: tuple[int, ...] | None = None
    # Optional string sequences to suppress in the generated text.  G1h CoT
    # uses this together with min_think_tokens so an immediately generated
    # </think> cannot collapse the reasoning block to an empty think.
    bad_words: tuple[str, ...] = ()
    min_think_tokens: int = 0
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
            temperature = MIN_NONZERO_SAMPLING_FLOAT
        else:
            temperature = min(temperature, 1000.0)

        if not (0 <= top_k <= int(vocab_size)):
            top_k = int(vocab_size)
        if not (0.0 <= top_p <= 1.0):
            top_p = 1.0
        if top_p == 0.0:
            top_k = 1
            top_p = MIN_NONZERO_SAMPLING_FLOAT

        return replace(
            self,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_generate_tokens=max(1, int(self.max_generate_tokens)),
            min_think_tokens=max(0, min(int(self.min_think_tokens), int(self.max_generate_tokens))),
            allowed_token_ids=(
                None
                if self.allowed_token_ids is None
                else tuple(
                    dict.fromkeys(
                        int(token_id)
                        for token_id in self.allowed_token_ids
                        if 0 <= int(token_id) < int(vocab_size)
                    )
                )
            ),
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
class ChatToolCall:
    """OpenAI-compatible tool call returned by chat-native inference."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def as_openai_tool_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }


@dataclass(slots=True)
class ToolCallGenerationOutput:
    """A chat completion result that preserves native tool-call structure."""

    prompt_index: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    content: str
    tool_calls: list[ChatToolCall]
    finish_reason: str
    raw_message: dict[str, Any] = field(default_factory=dict)
    response_source: str = "content"


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
    "ChatToolCall",
    "GeneratedTextDelta",
    "GeneratedToken",
    "GeneratedTokenCandidate",
    "GenerationOutput",
    "SamplingConfig",
    "ToolCallGenerationOutput",
]
