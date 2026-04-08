from __future__ import annotations

"""OpenAI-compatible request/response models for the RWKV infer service."""

from dataclasses import dataclass
from time import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .engine import DEFAULT_PREFILL_CHUNK_SIZE
from .sampling import SamplingConfig


def current_unix_seconds() -> int:
    return int(time())


def next_completion_id() -> str:
    return f"cmpl-{uuid4().hex}"


def next_chat_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


class CompletionLogprobs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tokens: list[str] | None = None
    token_logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    index: int = 0
    finish_reason: str | None = None
    logprobs: CompletionLogprobs | None = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=next_completion_id)
    object: str = "text_completion"
    created: int = Field(default_factory=current_unix_seconds)
    model: str
    choices: list[CompletionChoice]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_decay: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    logprobs: int | None = Field(default=None, ge=0)
    candidate_token_texts: list[str] | None = None
    seed: int | None = None
    stop_tokens: list[int] | None = None
    ban_tokens: list[int] | None = None
    pad_zero: bool | None = None
    no_penalty_token_ids: list[int] | None = None
    prefill_chunk_size: int | None = Field(default=None, ge=1)

    def to_sampling_config(self) -> SamplingConfig:
        defaults = SamplingConfig()
        return SamplingConfig(
            max_generate_tokens=(
                int(self.max_tokens)
                if self.max_tokens is not None
                else defaults.max_generate_tokens
            ),
            temperature=(
                float(self.temperature)
                if self.temperature is not None
                else defaults.temperature
            ),
            top_k=int(self.top_k) if self.top_k is not None else defaults.top_k,
            top_p=float(self.top_p) if self.top_p is not None else defaults.top_p,
            alpha_presence=(
                float(self.presence_penalty)
                if self.presence_penalty is not None
                else defaults.alpha_presence
            ),
            alpha_frequency=(
                float(self.frequency_penalty)
                if self.frequency_penalty is not None
                else defaults.alpha_frequency
            ),
            alpha_decay=(
                float(self.penalty_decay)
                if self.penalty_decay is not None
                else defaults.alpha_decay
            ),
            stop_tokens=(
                tuple(int(token_id) for token_id in self.stop_tokens)
                if self.stop_tokens is not None
                else defaults.stop_tokens
            ),
            ban_tokens=(
                tuple(int(token_id) for token_id in self.ban_tokens)
                if self.ban_tokens is not None
                else defaults.ban_tokens
            ),
            pad_zero=defaults.pad_zero if self.pad_zero is None else bool(self.pad_zero),
            no_penalty_token_ids=(
                tuple(int(token_id) for token_id in self.no_penalty_token_ids)
                if self.no_penalty_token_ids is not None
                else defaults.no_penalty_token_ids
            ),
        )

    def is_choice_scoring_request(self) -> bool:
        return bool(self.candidate_token_texts)

    def effective_prefill_chunk_size(self) -> int:
        if self.prefill_chunk_size is None:
            return DEFAULT_PREFILL_CHUNK_SIZE
        return max(1, int(self.prefill_chunk_size))

    def generation_batch_key(self) -> tuple[object, ...]:
        sampling = self.to_sampling_config()
        return (
            sampling.max_generate_tokens,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.alpha_presence,
            sampling.alpha_frequency,
            sampling.alpha_decay,
            sampling.stop_tokens,
            sampling.ban_tokens,
            sampling.pad_zero,
            sampling.no_penalty_token_ids,
            self.effective_prefill_chunk_size(),
        )


class ChatMessageTextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ChatMessageTextPart]
    name: str | None = None

    def text_content(self) -> str:
        return chat_message_content_to_text(self.content)


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=next_chat_completion_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=current_unix_seconds)
    model: str
    choices: list[ChatCompletionChoice]

    @classmethod
    def from_completion_response(cls, response: CompletionResponse) -> "ChatCompletionResponse":
        return cls(
            id=response.id.replace("cmpl-", "chatcmpl-", 1),
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatCompletionMessage(role="assistant", content=choice.text),
                    finish_reason=completion_finish_reason_to_chat(choice.finish_reason),
                )
                for choice in response.choices
            ],
        )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    seed: int | None = None
    n: int | None = None

    def to_completion_request(self) -> CompletionRequest:
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if self.n not in (None, 1):
            raise ValueError("only n=1 is supported")
        return CompletionRequest(
            model=self.model,
            prompt=render_chat_messages_as_prompt(self.messages),
            max_tokens=(
                self.max_completion_tokens
                if self.max_completion_tokens is not None
                else self.max_tokens
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop=self.stop,
            stream=self.stream,
            seed=self.seed,
        )


def chat_message_content_to_text(content: str | list[ChatMessageTextPart]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def render_chat_messages_as_prompt(messages: list[ChatCompletionMessage]) -> str:
    if not messages:
        raise ValueError("messages cannot be empty")
    role_labels = {
        "system": "System",
        "developer": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }
    rendered: list[str] = []
    for message in messages:
        label = role_labels[message.role]
        text = message.text_content()
        if text:
            rendered.append(f"{label}: {text}")
        else:
            rendered.append(f"{label}:")
    prompt = "\n\n".join(rendered)
    if messages[-1].role != "assistant":
        prompt = f"{prompt}\n\nAssistant:"
    return prompt


def completion_finish_reason_to_chat(finish_reason: str | None) -> str | None:
    if finish_reason == "stop_token":
        return "stop"
    if finish_reason == "max_length":
        return "length"
    return finish_reason


@dataclass(frozen=True, slots=True)
class ChoiceScore:
    text: str
    logprob: float


__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessageTextPart",
    "ChoiceScore",
    "CompletionChoice",
    "CompletionLogprobs",
    "CompletionRequest",
    "CompletionResponse",
    "chat_message_content_to_text",
    "completion_finish_reason_to_chat",
    "current_unix_seconds",
    "next_chat_completion_id",
    "next_completion_id",
    "render_chat_messages_as_prompt",
]
