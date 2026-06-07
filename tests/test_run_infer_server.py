from __future__ import annotations

from pathlib import Path

from src.bin import run_infer_server
from src.infer.nano_vllm_backend import NanoVLLMBackendConfig


def test_run_infer_server_no_longer_imports_local_backend() -> None:
    source = Path(run_infer_server.__file__).read_text(encoding="utf-8")

    assert "LocalInferenceBackend" not in source
    assert "ModelLoadConfig" not in source


def test_build_backend_uses_nano_vllm_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    fake_backend = object()

    def _fake_from_config(config: NanoVLLMBackendConfig):
        captured["config"] = config
        return fake_backend

    monkeypatch.setattr(
        run_infer_server.NanoVLLMInferenceBackend,
        "from_config",
        staticmethod(_fake_from_config),
    )
    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--model-name",
            "demo",
            "--nano-vllm-path",
            str(tmp_path / "nano"),
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "2048",
            "--max-model-len",
            "8192",
            "--rwkv-prefill-token-budget",
            "512",
            "--rwkv-prefill-max-batch-size",
            "16",
            "--rwkv-prefill-chunk-size",
            "32",
            "--rwkv-state-cache-enable",
            "--max-state-slots",
            "128",
            "--rwkv-state-cache-safety-reserve-slots",
            "4",
            "--sampling-bucket-temperature-resolution",
            "0.05",
            "--sampling-bucket-top-p-resolution",
            "0.1",
            "--rwkv-quant-int8",
            "--rwkv-int8-fp16-lm-head",
            "--gpu-memory-utilization",
            "0.75",
            "--tensor-parallel-size",
            "2",
            "--enforce-eager",
        ]
    )

    backend = run_infer_server.build_backend(args)

    assert backend is fake_backend
    config = captured["config"]
    assert isinstance(config, NanoVLLMBackendConfig)
    assert config.model_path == "/models/rwkv-demo.pth"
    assert config.model_name == "demo"
    assert config.nano_vllm_path == str(tmp_path / "nano")
    assert config.max_num_seqs == 64
    assert config.max_num_batched_tokens == 2048
    assert config.max_model_len == 8192
    assert config.rwkv_prefill_token_budget == 512
    assert config.rwkv_prefill_max_batch_size == 16
    assert config.rwkv_prefill_chunk_size == 32
    assert config.rwkv_state_cache_enable is True
    assert config.max_state_slots == 128
    assert config.rwkv_state_cache_safety_reserve_slots == 4
    assert config.sampling_bucket_temperature_resolution == 0.05
    assert config.sampling_bucket_top_p_resolution == 0.1
    assert config.rwkv_quant_int8 is True
    assert config.rwkv_int8_fp16_lm_head is True
    assert config.gpu_memory_utilization == 0.75
    assert config.tensor_parallel_size == 2
    assert config.enforce_eager is True


def test_main_serves_nano_vllm_backend(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeBackend:
        model_name = "demo"

    class _FakeService:
        def __init__(self, backend, *, max_batch_size: int, batch_collect_ms: int) -> None:
            captured["service_backend"] = backend
            captured["max_batch_size"] = max_batch_size
            captured["batch_collect_ms"] = batch_collect_ms

    fake_backend = _FakeBackend()

    monkeypatch.setattr(run_infer_server, "build_backend", lambda args: fake_backend)
    monkeypatch.setattr(run_infer_server, "InferenceService", _FakeService)
    monkeypatch.setattr(
        run_infer_server,
        "create_app",
        lambda service, *, api_key: {"service": service, "api_key": api_key},
    )
    monkeypatch.setattr(
        run_infer_server.uvicorn,
        "run",
        lambda app, **kwargs: captured.update({"app": app, "uvicorn_kwargs": kwargs}),
    )

    result = run_infer_server.main(
        [
            "--model-path",
            str(tmp_path / "rwkv-demo.pth"),
            "--model-name",
            "demo",
            "--host",
            "0.0.0.0",
            "--port",
            "18081",
            "--api-key",
            "secret",
            "--max-batch-size",
            "8",
            "--batch-collect-ms",
            "3",
            "--log-level",
            "warning",
        ]
    )

    assert result == 0
    assert captured["service_backend"] is fake_backend
    assert captured["max_batch_size"] == 8
    assert captured["batch_collect_ms"] == 3
    assert captured["app"]["api_key"] == "secret"
    assert captured["uvicorn_kwargs"] == {
        "host": "0.0.0.0",
        "port": 18081,
        "log_level": "warning",
        "access_log": False,
    }
