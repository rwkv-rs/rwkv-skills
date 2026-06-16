from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.infer.nano_vllm_backend import NanoVLLMBackendConfig, NanoVLLMInferenceBackend
from src.infer.sampling import SamplingConfig


def test_nano_vllm_backend_loads_engine_scheduler_and_model_runner(monkeypatch, tmp_path: Path) -> None:
    root, fake = _install_fake_nanovllm(monkeypatch, tmp_path)

    backend = NanoVLLMInferenceBackend.from_config(
        NanoVLLMBackendConfig(
            model_path="/models/rwkv-demo.pth",
            model_name="demo",
            nano_vllm_path=root,
            max_num_seqs=7,
            max_num_batched_tokens=99,
            rwkv_prefill_token_budget=33,
            tensor_parallel_size=1,
        )
    )

    assert backend.model_name == "demo"
    assert isinstance(backend.engine, fake.LLMEngine)
    assert isinstance(backend.scheduler, fake.Scheduler)
    assert isinstance(backend.model_runner, fake.ModelRunner)
    assert fake.calls["model"] == "/models/rwkv-demo.pth"
    assert fake.calls["engine_kwargs"]["max_num_seqs"] == 7
    assert fake.calls["engine_kwargs"]["max_num_batched_tokens"] == 99
    assert fake.calls["engine_kwargs"]["rwkv_prefill_token_budget"] == 33


def test_nano_vllm_backend_generate_maps_sampling_and_stop_suffixes(monkeypatch, tmp_path: Path) -> None:
    root, fake = _install_fake_nanovllm(monkeypatch, tmp_path)
    backend = NanoVLLMInferenceBackend.from_config(
        NanoVLLMBackendConfig(model_path="/models/rwkv-demo.pth", nano_vllm_path=root)
    )
    completed = []
    token_deltas = []

    outputs = backend.generate(
        ["alpha", "beta"],
        sampling=SamplingConfig(
            max_generate_tokens=128,
            temperature=0.7,
            top_k=0,
            top_p=0.8,
            alpha_presence=0.1,
            alpha_frequency=0.2,
            alpha_decay=0.97,
        ),
        batch_size=2,
        prompt_stop_suffixes=[(" STOP",), None],
        on_complete=completed.append,
        on_token=lambda prompt_index, delta: token_deltas.append((prompt_index, delta.text)),
        show_progress=False,
    )

    assert [output.prompt_index for output in outputs] == [0, 1]
    assert outputs[0].text == "alpha-done"
    assert outputs[0].finish_reason == "stop_token"
    assert outputs[1].text == "beta-done STOP tail"
    assert completed == outputs
    assert token_deltas == [(0, "alpha-done"), (1, "beta-done STOP tail")]

    params = fake.calls["sampling_params"][0]
    assert params.temperature == 0.7
    assert params.top_k == -1
    assert params.top_p == 0.8
    assert params.presence_penalty == 0.1
    assert params.repetition_penalty == 0.2
    assert params.penalty_decay == 0.97
    assert params.max_tokens == 128
    assert params.ignore_eos is False

    alpha_seq = fake.calls["seqs"][0]
    assert (0,) in alpha_seq.stop_token_seqs
    assert tuple(ord(char) for char in " STOP") in alpha_seq.stop_token_seqs


def test_nano_vllm_backend_rejects_prompt_seeds(monkeypatch, tmp_path: Path) -> None:
    root, _fake = _install_fake_nanovllm(monkeypatch, tmp_path)
    backend = NanoVLLMInferenceBackend.from_config(
        NanoVLLMBackendConfig(model_path="/models/rwkv-demo.pth", nano_vllm_path=root)
    )

    with pytest.raises(NotImplementedError, match="per-prompt seeds"):
        backend.generate(
            ["prompt"],
            sampling=SamplingConfig(max_generate_tokens=1),
            batch_size=1,
            prompt_seeds=[123],
            show_progress=False,
        )
    backend.shutdown()


def test_nano_vllm_backend_submit_rejects_prompt_seed(monkeypatch, tmp_path: Path) -> None:
    root, _fake = _install_fake_nanovllm(monkeypatch, tmp_path)
    backend = NanoVLLMInferenceBackend.from_config(
        NanoVLLMBackendConfig(model_path="/models/rwkv-demo.pth", nano_vllm_path=root)
    )

    try:
        with pytest.raises(NotImplementedError, match="per-prompt seeds"):
            backend.submit(
                "prompt",
                sampling=SamplingConfig(max_generate_tokens=1),
                prompt_seed=123,
            )
    finally:
        backend.shutdown()


def _install_fake_nanovllm(monkeypatch, tmp_path: Path):
    root = tmp_path / "nano-vllm-rwkv-315cf53"
    (root / "nanovllm" / "engine").mkdir(parents=True)
    calls: dict[str, object] = {"sampling_params": [], "seqs": []}

    class Scheduler:
        pass

    class ModelRunner:
        pass

    class SamplingParams:
        def __init__(
            self,
            *,
            temperature: float,
            top_k: int,
            top_p: float,
            presence_penalty: float,
            repetition_penalty: float,
            penalty_decay: float,
            max_tokens: int,
            ignore_eos: bool,
        ) -> None:
            self.temperature = temperature
            self.top_k = top_k
            self.top_p = top_p
            self.presence_penalty = presence_penalty
            self.repetition_penalty = repetition_penalty
            self.penalty_decay = penalty_decay
            self.max_tokens = max_tokens
            self.ignore_eos = ignore_eos
            calls["sampling_params"].append(self)

    class _Tokenizer:
        eos_token_id = 0

        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

    class LLMEngine:
        def __init__(self, model: str, **kwargs) -> None:
            calls["model"] = model
            calls["engine_kwargs"] = kwargs
            self.scheduler = Scheduler()
            self.model_runner = ModelRunner()
            self.tokenizer = _Tokenizer()
            self._pending = []
            self._next_seq_id = 0
            self.exit_calls = 0

        def add_request(self, prompt: str, sampling_params: SamplingParams):
            seq = SimpleNamespace(
                seq_id=self._next_seq_id,
                max_tokens=sampling_params.max_tokens,
                num_raw_completion_tokens=0,
            )
            self._next_seq_id += 1
            self._pending.append((seq, prompt))
            calls["seqs"].append(seq)
            return seq

        def is_finished(self) -> bool:
            return not self._pending

        def step(self):
            outputs = []
            for seq, prompt in self._pending:
                token_ids = [ord(char) for char in f"{prompt}-done STOP tail"]
                seq.num_raw_completion_tokens = len(token_ids)
                outputs.append((seq.seq_id, token_ids))
            self._pending = []
            return outputs, -len(outputs)

        def exit(self) -> None:
            self.exit_calls += 1

    fake = SimpleNamespace(
        LLMEngine=LLMEngine,
        Scheduler=Scheduler,
        ModelRunner=ModelRunner,
        SamplingParams=SamplingParams,
        calls=calls,
    )
    modules = {
        "nanovllm": ModuleType("nanovllm"),
        "nanovllm.engine": ModuleType("nanovllm.engine"),
        "nanovllm.engine.llm_engine": ModuleType("nanovllm.engine.llm_engine"),
        "nanovllm.engine.scheduler": ModuleType("nanovllm.engine.scheduler"),
        "nanovllm.engine.model_runner": ModuleType("nanovllm.engine.model_runner"),
        "nanovllm.sampling_params": ModuleType("nanovllm.sampling_params"),
    }
    modules["nanovllm"].__path__ = []  # type: ignore[attr-defined]
    modules["nanovllm.engine"].__path__ = []  # type: ignore[attr-defined]
    modules["nanovllm.engine.llm_engine"].LLMEngine = LLMEngine
    modules["nanovllm.engine.scheduler"].Scheduler = Scheduler
    modules["nanovllm.engine.model_runner"].ModelRunner = ModelRunner
    modules["nanovllm.sampling_params"].SamplingParams = SamplingParams
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return root, fake
