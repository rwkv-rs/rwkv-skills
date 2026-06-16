from __future__ import annotations

from pathlib import Path

from src.bin import run_infer_server
from src.infer.auto_config import GpuProfile
from src.infer.nano_vllm_backend import DEFAULT_NANO_VLLM_PATH, NanoVLLMBackendConfig


def test_run_infer_server_no_longer_imports_local_backend() -> None:
    source = Path(run_infer_server.__file__).read_text(encoding="utf-8")

    assert "LocalInferenceBackend" not in source
    assert "ModelLoadConfig" not in source


def test_default_nano_vllm_path_points_to_repo_vendor() -> None:
    assert DEFAULT_NANO_VLLM_PATH.name == "nano-vllm-rwkv"
    assert DEFAULT_NANO_VLLM_PATH.parent.name == "vendor"
    assert (DEFAULT_NANO_VLLM_PATH / "nanovllm" / "engine" / "llm_engine.py").is_file()


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


def test_parse_args_applies_throughput_auto_config_for_large_visible_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        run_infer_server,
        "detect_visible_gpu_profile",
        lambda: GpuProfile(name="generic-cuda-device", total_memory_mb=97887),
    )

    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv7-g1g-7.2b-20260523-ctx8192.pth",
            "--model-name",
            "rwkv7-g1g-7.2b-20260523-ctx8192",
        ]
    )

    assert args.infer_auto_config == "throughput"
    assert args.infer_auto_config_applied is True
    assert "gpu_memory_mb=97887" in args.infer_auto_config_reason
    assert "model_params_b=7.2" in args.infer_auto_config_reason
    assert "generic-cuda-device" not in args.infer_auto_config_reason
    assert args.max_batch_size == 1024
    assert args.batch_collect_ms == 50
    assert args.max_num_seqs == -1
    assert args.max_num_batched_tokens == 32768
    assert args.rwkv_prefill_token_budget == 8192
    assert args.rwkv_prefill_max_batch_size == 1024
    assert args.max_state_slots == -1
    assert args.gpu_memory_utilization == 0.98


def test_parse_args_keeps_explicit_values_when_auto_config_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        run_infer_server,
        "detect_visible_gpu_profile",
        lambda: GpuProfile(name="large-gpu", total_memory_mb=97887),
    )

    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv7-g1f-2.9b-20260420-ctx8192.pth",
            "--max-batch-size",
            "77",
            "--gpu-memory-utilization",
            "0.91",
        ]
    )

    assert args.max_batch_size == 77
    assert args.gpu_memory_utilization == 0.91
    assert args.batch_collect_ms == 50
    assert args.max_num_batched_tokens == 32768
    assert args.rwkv_prefill_max_batch_size == 1024


def test_parse_args_auto_config_off_uses_legacy_defaults(monkeypatch) -> None:
    def _fail_gpu_probe() -> None:
        raise AssertionError("GPU detection should not run when auto config is off")

    monkeypatch.setattr(run_infer_server, "detect_visible_gpu_profile", _fail_gpu_probe)

    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--infer-auto-config",
            "off",
        ]
    )

    assert args.infer_auto_config_applied is False
    assert args.max_batch_size == 32
    assert args.batch_collect_ms == 5
    assert args.max_num_seqs == 512
    assert args.max_num_batched_tokens == 16384
    assert args.rwkv_prefill_token_budget == 2048
    assert args.rwkv_prefill_max_batch_size == 128
    assert args.max_state_slots == -1
    assert args.gpu_memory_utilization == 0.9


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
