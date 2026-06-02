from __future__ import annotations

import torch

from src.infer.engine import InferenceEngine
from src.infer.sampling import SamplingConfig


def test_engine_records_token_zero_stop(monkeypatch) -> None:
    _patch_sampler(monkeypatch, [0])
    engine = InferenceEngine(_FakeModel(), _FakeTokenizer({}))

    output = engine.generate(
        ["p"],
        sampling=SamplingConfig(max_generate_tokens=4, stop_tokens=(0,), pad_zero=False),
        batch_size=1,
        progress_desc="test",
    )[0]

    assert output.finish_reason == "stop_condition"
    assert output.finish_detail == "token_0"
    assert output.truncated is False
    assert output.token_ids == []


def test_engine_records_user_sentinel_stop(monkeypatch) -> None:
    _patch_sampler(monkeypatch, [1])
    engine = InferenceEngine(_FakeModel(), _FakeTokenizer({1: "\nUser:"}))

    output = engine.generate(
        ["p"],
        sampling=SamplingConfig(max_generate_tokens=4, stop_tokens=(0,), pad_zero=False),
        batch_size=1,
        progress_desc="test",
        prompt_stop_suffixes=[("\nUser:",)],
    )[0]

    assert output.finish_reason == "stop_condition"
    assert output.finish_detail == "user_sentinel"
    assert output.truncated is False
    assert output.text == "\nUser:"


def test_engine_records_max_tokens_truncation(monkeypatch) -> None:
    _patch_sampler(monkeypatch, [2])
    engine = InferenceEngine(_FakeModel(), _FakeTokenizer({2: "x"}))

    output = engine.generate(
        ["p"],
        sampling=SamplingConfig(max_generate_tokens=1, stop_tokens=(0,), pad_zero=False),
        batch_size=1,
        progress_desc="test",
    )[0]

    assert output.finish_reason == "max_tokens"
    assert output.finish_detail == "max_tokens"
    assert output.truncated is True
    assert output.token_ids == [2]
    assert output.text == "x"


def _patch_sampler(monkeypatch, tokens: list[int]) -> None:
    class FakeSampler:
        def __init__(self) -> None:
            self._tokens = list(tokens)

        def setup_rand(self, _seed: int, batch_size: int) -> torch.Tensor:
            return torch.zeros(batch_size, dtype=torch.uint8)

        def batch_sampling_repetition_temperature_topk_topp(self, *_args) -> torch.Tensor:
            value = self._tokens.pop(0) if self._tokens else 0
            return torch.tensor([value], dtype=torch.int64)

    monkeypatch.setattr("src.infer.engine.get_rapid_sampling_module", lambda: FakeSampler())


class _FakeTokenizer:
    def __init__(self, decoded: dict[int, str]) -> None:
        self._decoded = decoded

    def encode(self, text: str) -> list[int]:
        return [99] if text else []

    def decode(self, token_ids) -> str:
        return "".join(self._decoded.get(int(token), "") for token in token_ids)


class _FakeModel:
    vocab_size = 256

    def generate_zero_state(self, batch_size: int):
        return [
            torch.zeros((1, 1, batch_size, 1)),
            torch.zeros((1, batch_size, 1, 1, 1)),
            torch.zeros((batch_size, 1)),
        ]

    def forward_batch(self, tokens, state, full_output: bool = False):
        _ = state, full_output
        return torch.zeros((len(tokens), self.vocab_size), dtype=torch.float32)
