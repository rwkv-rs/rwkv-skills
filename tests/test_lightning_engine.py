from __future__ import annotations

from types import SimpleNamespace

import torch

from src.infer.lightning_engine import LightningEngineConfig, LightningInferenceEngineAdapter
from src.infer.sampling import SamplingConfig


class _FakeRapidSampler:
    def setup_rand(self, _seed: int, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, dtype=torch.int64)

    def batch_sampling_repetition_temperature_topk_topp(
        self,
        logits: torch.Tensor,
        _penalties: torch.Tensor,
        _sampler_states: torch.Tensor,
        *_args,
    ) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    def batch_sampling_temperature_topk_topp(
        self,
        logits: torch.Tensor,
        _sampler_states: torch.Tensor,
        *_args,
    ) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._table = {
            "abcd": [1, 2, 3, 4],
            "abcdef": [1, 2, 3, 4, 5, 6],
        }

    def encode(self, text: str) -> list[int]:
        return list(self._table[text])

    def decode(self, token_ids) -> str:
        return "".join(str(token) for token in token_ids)


class _FakeModel:
    def __init__(self) -> None:
        self.args = SimpleNamespace(vocab_size=8)
        self.z = {"head.weight": torch.zeros((1,), dtype=torch.float32)}
        self.forward_calls: list[tuple[int, ...]] = []

    def generate_zero_state(self, batch_size: int):
        return [
            torch.zeros((1, 2, batch_size, 1), dtype=torch.float32),
            torch.zeros((1, batch_size, 1, 1, 1), dtype=torch.float32),
            torch.zeros((batch_size,), dtype=torch.int32),
        ]

    def forward_batch(self, tokens, state, full_output: bool = False):
        del full_output
        logits = torch.full((len(tokens), self.args.vocab_size), -1000.0, dtype=torch.float32)
        for idx, token_list in enumerate(tokens):
            as_tuple = tuple(int(token) for token in token_list)
            self.forward_calls.append(as_tuple)
            state[2][idx] += len(token_list)
            state[0][0, 0, idx, 0] = float(sum(token_list))
            logits[idx, 0] = 10.0
        return logits


def _sampling() -> SamplingConfig:
    return SamplingConfig(
        max_generate_tokens=4,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        alpha_presence=0.0,
        alpha_frequency=0.0,
        alpha_decay=0.99,
        stop_tokens=(0,),
        pad_zero=False,
        no_penalty_token_ids=(),
    )


def test_lightning_engine_uses_full_prefix_cache_without_forward(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    engine = LightningInferenceEngineAdapter(
        model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "full-cache.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        first = engine.generate(
            ["abcd"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            show_progress=False,
        )
        assert [output.finish_reason for output in first] == ["stop_token"]
        assert model.forward_calls == [(1, 2), (3, 4)]

        engine.state_cache.clear_prefix_memory()
        model.forward_calls.clear()

        second = engine.generate(
            ["abcd"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            show_progress=False,
        )
        assert [output.finish_reason for output in second] == ["stop_token"]
        assert model.forward_calls == []
    finally:
        engine.shutdown()


def test_lightning_engine_only_forwards_prompt_suffix_after_prefix_hit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    engine = LightningInferenceEngineAdapter(
        model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "partial-cache.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        first = engine.generate(
            ["abcdef"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            show_progress=False,
        )
        assert [output.finish_reason for output in first] == ["stop_token"]
        assert model.forward_calls == [(1, 2), (3, 4), (5, 6)]

        engine.state_cache.clear_prefix_memory()
        model.forward_calls.clear()

        second = engine.generate(
            ["abcdef"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            show_progress=False,
        )
        assert [output.finish_reason for output in second] == ["stop_token"]
        assert model.forward_calls == [(5, 6)]
    finally:
        engine.shutdown()
