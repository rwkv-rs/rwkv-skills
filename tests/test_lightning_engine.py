from __future__ import annotations

from types import SimpleNamespace

import torch

from src.infer.constraints import LiteralChoiceConstraint
from src.infer.engine import InferenceEngine
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


class _StopTokenizer:
    def __init__(self) -> None:
        self._encode_table = {"prompt": [1]}
        self._decode_table = {
            1: b"prompt",
            2: b"hello",
            3: b"User",
            4: b":",
            5: b"ignored",
            0: b"",
        }

    def encode(self, text: str) -> list[int]:
        return list(self._encode_table[text])

    def decodeBytes(self, token_ids) -> bytes:
        return b"".join(self._decode_table[int(token_id)] for token_id in token_ids)

    def decode(self, token_ids) -> str:
        return self.decodeBytes(token_ids).decode("utf-8")


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


class _StreamingFakeModel(_FakeModel):
    def forward_batch(self, tokens, state, full_output: bool = False):
        del full_output
        logits = torch.full((len(tokens), self.args.vocab_size), -1000.0, dtype=torch.float32)
        for idx, token_list in enumerate(tokens):
            as_tuple = tuple(int(token) for token in token_list)
            self.forward_calls.append(as_tuple)
            state[2][idx] += len(token_list)
            if as_tuple == (7,):
                logits[idx, 0] = 10.0
            else:
                logits[idx, 7] = 10.0
        return logits


class _StopSuffixFakeModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.args = SimpleNamespace(vocab_size=16)

    def forward_batch(self, tokens, state, full_output: bool = False):
        del full_output
        logits = torch.full((len(tokens), self.args.vocab_size), -1000.0, dtype=torch.float32)
        for idx, token_list in enumerate(tokens):
            as_tuple = tuple(int(token) for token in token_list)
            self.forward_calls.append(as_tuple)
            state[2][idx] += len(token_list)
            if as_tuple == (1,):
                logits[idx, 2] = 10.0
            elif as_tuple == (2,):
                logits[idx, 3] = 10.0
            elif as_tuple == (3,):
                logits[idx, 4] = 10.0
            else:
                logits[idx, 5] = 10.0
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


def test_local_engines_emit_token_callbacks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())

    tokenizer = _FakeTokenizer()

    classic_model = _StreamingFakeModel()
    classic_engine = InferenceEngine(classic_model, tokenizer)
    classic_tokens = []
    classic_outputs = classic_engine.generate(
        ["abcd"],
        sampling=_sampling(),
        batch_size=1,
        prefill_chunk_size=8,
        on_token=lambda _prompt_index, delta: classic_tokens.append((delta.text, [token.token_id for token in delta.tokens])),
        top_logprobs=1,
        show_progress=False,
    )

    lightning_model = _StreamingFakeModel()
    lightning_engine = LightningInferenceEngineAdapter(
        lightning_model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "stream-cache.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        lightning_tokens = []
        lightning_outputs = lightning_engine.generate(
            ["abcd"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            on_token=lambda _prompt_index, delta: lightning_tokens.append((delta.text, [token.token_id for token in delta.tokens])),
            top_logprobs=1,
            show_progress=False,
        )
    finally:
        lightning_engine.shutdown()

    assert classic_tokens == [("7", [7])]
    assert classic_outputs[0].token_ids == [7]
    assert classic_outputs[0].tokens[0].logprob is not None
    assert lightning_tokens == [("7", [7])]
    assert lightning_outputs[0].token_ids == [7]
    assert lightning_outputs[0].tokens[0].logprob is not None


def test_local_engines_stop_on_text_suffix_without_extra_forward(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())

    tokenizer = _StopTokenizer()

    classic_model = _StopSuffixFakeModel()
    classic_engine = InferenceEngine(classic_model, tokenizer)
    classic_deltas = []
    classic_outputs = classic_engine.generate(
        ["prompt"],
        sampling=_sampling(),
        batch_size=1,
        prefill_chunk_size=8,
        prompt_stop_suffixes=[["User:"]],
        on_token=lambda _prompt_index, delta: classic_deltas.append((delta.text, [token.token_id for token in delta.tokens])),
        show_progress=False,
    )

    lightning_model = _StopSuffixFakeModel()
    lightning_engine = LightningInferenceEngineAdapter(
        lightning_model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "stop-suffix.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        lightning_deltas = []
        lightning_outputs = lightning_engine.generate(
            ["prompt"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            prompt_stop_suffixes=[["User:"]],
            on_token=lambda _prompt_index, delta: lightning_deltas.append((delta.text, [token.token_id for token in delta.tokens])),
            show_progress=False,
        )
    finally:
        lightning_engine.shutdown()

    assert classic_deltas == [("hello", [2])]
    assert classic_outputs[0].text == "hello"
    assert classic_outputs[0].token_ids == [2]
    assert classic_outputs[0].finish_reason == "stop_token"
    assert classic_model.forward_calls == [(1,), (2,), (3,)]

    assert lightning_deltas == [("hello", [2])]
    assert lightning_outputs[0].text == "hello"
    assert lightning_outputs[0].token_ids == [2]
    assert lightning_outputs[0].finish_reason == "stop_token"
    assert lightning_model.forward_calls == [(1,), (2,), (3,)]


def test_local_engines_support_literal_choice_constraints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())

    tokenizer = _FakeTokenizer()

    classic_model = _StreamingFakeModel()
    classic_engine = InferenceEngine(classic_model, tokenizer)
    classic_outputs = classic_engine.generate(
        ["abcd"],
        sampling=_sampling(),
        batch_size=1,
        prefill_chunk_size=8,
        prompt_constraints=[LiteralChoiceConstraint(("7",))],
        show_progress=False,
    )

    lightning_model = _StreamingFakeModel()
    lightning_engine = LightningInferenceEngineAdapter(
        lightning_model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "constraint-cache.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        lightning_outputs = lightning_engine.generate(
            ["abcd"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            prompt_constraints=[LiteralChoiceConstraint(("7",))],
            show_progress=False,
        )
    finally:
        lightning_engine.shutdown()

    assert classic_outputs[0].finish_reason == "constraint_stop"
    assert classic_outputs[0].token_ids == [7]
    assert classic_outputs[0].text == "7"
    assert lightning_outputs[0].finish_reason == "constraint_stop"
    assert lightning_outputs[0].token_ids == [7]
    assert lightning_outputs[0].text == "7"


def test_local_engines_report_constraint_dead_end(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.infer.engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())
    monkeypatch.setattr("src.infer.lightning_engine.get_rapid_sampling_module", lambda: _FakeRapidSampler())

    tokenizer = _FakeTokenizer()

    classic_model = _StreamingFakeModel()
    classic_engine = InferenceEngine(classic_model, tokenizer)
    classic_outputs = classic_engine.generate(
        ["abcd"],
        sampling=_sampling(),
        batch_size=1,
        prefill_chunk_size=8,
        prompt_constraints=[LiteralChoiceConstraint(("9",))],
        show_progress=False,
    )

    lightning_model = _StreamingFakeModel()
    lightning_engine = LightningInferenceEngineAdapter(
        lightning_model,
        tokenizer,
        config=LightningEngineConfig(
            state_db_path=str(tmp_path / "constraint-dead-end.db"),
            prefix_cache_buckets=(2, 4),
            prefix_bucket_capacity=8,
        ),
    )
    try:
        lightning_outputs = lightning_engine.generate(
            ["abcd"],
            sampling=_sampling(),
            batch_size=1,
            prefill_chunk_size=8,
            prompt_constraints=[LiteralChoiceConstraint(("9",))],
            show_progress=False,
        )
    finally:
        lightning_engine.shutdown()

    assert classic_outputs[0].finish_reason == "constraint_dead_end"
    assert classic_outputs[0].token_ids == []
    assert classic_outputs[0].text == ""
    assert lightning_outputs[0].finish_reason == "constraint_dead_end"
    assert lightning_outputs[0].token_ids == []
    assert lightning_outputs[0].text == ""
