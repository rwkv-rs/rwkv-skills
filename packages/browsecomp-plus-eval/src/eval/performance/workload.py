from __future__ import annotations

from typing import Sequence

from .tokenizers import BenchmarkTokenizer


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"期望正整数，得到: {value}")
        values.append(value)
    if not values:
        raise ValueError("至少需要一个正整数配置")
    return tuple(values)


def build_prompt_for_target_tokens(tokenizer: BenchmarkTokenizer, target_tokens: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens 必须大于 0")

    seed_texts = (
        " benchmark",
        " performance",
        " latency",
        " throughput",
        " token",
        " request",
    )
    seed_token_ids: list[int] = []
    for text in seed_texts:
        encoded = tokenizer.encode(text)
        if encoded:
            seed_token_ids = encoded
            break
    if not seed_token_ids:
        raise ValueError("无法为 tokenizer 构造性能测试 prompt")

    repeats = (target_tokens + len(seed_token_ids) - 1) // len(seed_token_ids)
    prompt_token_ids = (seed_token_ids * repeats)[:target_tokens]
    prompt_text = tokenizer.decode(prompt_token_ids)
    verified = tokenizer.encode(prompt_text)
    if len(verified) != target_tokens:
        raise ValueError(
            f"无法稳定构造目标 token 长度 prompt: target={target_tokens}, actual={len(verified)}"
        )
    return prompt_text, len(verified)


def repeat_prompts(prompt_text: str, count: int) -> list[str]:
    return [prompt_text for _ in range(max(1, int(count)))]


__all__ = [
    "build_prompt_for_target_tokens",
    "parse_int_csv",
    "repeat_prompts",
]
