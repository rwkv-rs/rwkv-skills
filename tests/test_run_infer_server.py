from __future__ import annotations

from pathlib import Path

from src.bin import run_infer_server
from src.infer.auto_config import GpuProfile
from src.infer.backend import LocalInferenceBackend
from src.infer.model import ModelLoadConfig
from src.infer.rwkv_lightning_server import DEFAULT_RWKV_LIGHTNING_PATH, RWKVLightningServerConfig


def test_run_infer_server_defaults_to_rwkv_lightning_backend() -> None:
    source = Path(run_infer_server.__file__).read_text(encoding="utf-8")

    assert '_ENGINE_MODE_CHOICES = ("rwkv-lightning", "lightning", "classic")' in source
    assert "default=_default_engine_mode()" in source
    assert "build_rwkv_lightning_app" in source
    assert "LocalInferenceBackend" in source


def test_default_rwkv_lightning_path_points_to_repo_vendor() -> None:
    assert DEFAULT_RWKV_LIGHTNING_PATH.name == "rwkv-lightning"
    assert DEFAULT_RWKV_LIGHTNING_PATH.parent.name == "vendor"
    assert (DEFAULT_RWKV_LIGHTNING_PATH / "API_servers" / "fastapi_service.py").is_file()
    assert (DEFAULT_RWKV_LIGHTNING_PATH / "infer" / "inference.py").is_file()


def test_build_backend_uses_local_lightning_when_requested(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    fake_backend = object()

    def _fake_from_model_config(config, *, engine_mode: str, state_db_path: str | None):  # noqa: ANN001
        captured["config"] = config
        captured["engine_mode"] = engine_mode
        captured["state_db_path"] = state_db_path
        return fake_backend

    monkeypatch.setattr(
        LocalInferenceBackend,
        "from_model_config",
        staticmethod(_fake_from_model_config),
    )
    args = run_infer_server.parse_args(
        [
            "--model-path",
            str(tmp_path / "rwkv-demo.pth"),
            "--model-name",
            "demo",
            "--engine-mode",
            "lightning",
            "--device",
            "cuda:1",
            "--state-db-path",
            str(tmp_path / "states.sqlite"),
        ]
    )

    backend = run_infer_server.build_backend(args)

    assert backend is fake_backend
    config = captured["config"]
    assert isinstance(config, ModelLoadConfig)
    assert config.weights_path == str(tmp_path / "rwkv-demo.pth")
    assert config.device == "cuda:1"
    assert config.tokenizer_path is None
    assert captured["engine_mode"] == "lightning"
    assert captured["state_db_path"] == str(tmp_path / "states.sqlite")


def test_parse_args_ignores_removed_nano_engine_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RWKV_INFER_ENGINE_MODE", "nano-vllm")

    args = run_infer_server.parse_args(
        [
            "--model-path",
            str(tmp_path / "rwkv-demo.pth"),
            "--infer-auto-config",
            "off",
        ]
    )

    assert args.engine_mode == "rwkv-lightning"


def test_main_serves_rwkv_lightning_server_by_default(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeServer:
        app = {"kind": "rwkv-lightning"}

        @staticmethod
        def cleanup() -> None:
            captured["cleanup"] = True

    def _fake_build(config: RWKVLightningServerConfig):
        captured["config"] = config
        return _FakeServer()

    monkeypatch.setattr(run_infer_server, "build_rwkv_lightning_app", _fake_build)
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
            "--log-level",
            "warning",
        ]
    )

    assert result == 0
    config = captured["config"]
    assert isinstance(config, RWKVLightningServerConfig)
    assert config.model_path == str(tmp_path / "rwkv-demo.pth")
    assert config.model_name == "demo"
    assert config.password == "secret"
    assert config.rwkv_lightning_path == str(DEFAULT_RWKV_LIGHTNING_PATH)
    assert captured["app"] == {"kind": "rwkv-lightning"}
    assert captured["cleanup"] is True
    assert captured["uvicorn_kwargs"] == {
        "host": "0.0.0.0",
        "port": 18081,
        "log_level": "warning",
        "access_log": False,
    }


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


def test_main_serves_local_backend(monkeypatch, tmp_path: Path) -> None:
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
            "--engine-mode",
            "classic",
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
