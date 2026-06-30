from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.bin import run_infer_server
from src.infer.auto_config import GpuProfile


def test_parse_args_uses_vllm_rwkv_defaults(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_INFER_ENGINE_MODE", "rwkv-lightning")

    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--infer-auto-config",
            "off",
        ]
    )

    assert not hasattr(args, "engine_mode")
    assert args.vllm_rwkv_path == str(run_infer_server.DEFAULT_VLLM_RWKV_PATH)
    assert args.max_num_seqs == 512
    assert args.max_num_batched_tokens == 16384
    assert args.gpu_memory_utilization == 0.9


def test_build_vllm_command_maps_server_flags() -> None:
    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--model-name",
            "demo",
            "--infer-auto-config",
            "off",
            "--vllm-python",
            "/opt/vllm/bin/python",
            "--host",
            "0.0.0.0",
            "--port",
            "19082",
            "--api-key",
            "secret",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "-1",
            "--max-num-batched-tokens",
            "32768",
            "--gpu-memory-utilization",
            "0.97",
            "--tensor-parallel-size",
            "2",
            "--enforce-eager",
            "--extra-vllm-arg=--trust-remote-code",
        ]
    )

    command = run_infer_server.build_vllm_command(args)

    assert command[:4] == ["/opt/vllm/bin/python", "-m", "vllm.entrypoints.cli.main", "serve"]
    assert command[4] == "/models/rwkv-demo.pth"
    assert command[command.index("--served-model-name") + 1] == "demo"
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--port") + 1] == "19082"
    assert command[command.index("--api-key") + 1] == "secret"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert "--max-num-seqs" not in command
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.97"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--enforce-eager" in command
    assert "--trust-remote-code" in command


def test_build_vllm_env_prepends_reference_checkout(tmp_path: Path) -> None:
    args = run_infer_server.parse_args(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--infer-auto-config",
            "off",
            "--vllm-rwkv-path",
            str(tmp_path / "vllm-rwkv"),
            "--cuda-visible-devices",
            "3",
            "--rwkv-wkv-mode",
            "fla",
            "--rwkv-emb-device",
            "cpu",
        ]
    )

    env = run_infer_server.build_vllm_env(args, base_env={"PYTHONPATH": "/existing"})

    assert env["PYTHONPATH"].split(":")[0] == str((tmp_path / "vllm-rwkv").resolve())
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["VLLM_RWKV7_WKV_MODE"] == "fla"
    assert env["VLLM_RWKV7_EMB_DEVICE"] == "cpu"
    assert env["VLLM_RWKV7_RKV_MODE"] == "off"


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
    assert not hasattr(args, "max_batch_size")
    assert not hasattr(args, "batch_collect_ms")
    assert args.max_num_seqs == -1
    assert args.max_num_batched_tokens == 32768
    assert args.gpu_memory_utilization == 0.98


def test_main_runs_vllm_subprocess(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(command, *, env, check):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_infer_server.subprocess, "run", _fake_run)

    result = run_infer_server.main(
        [
            "--model-path",
            "/models/rwkv-demo.pth",
            "--model-name",
            "demo",
            "--infer-auto-config",
            "off",
            "--cuda-visible-devices",
            "2",
        ]
    )

    assert result == 7
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:4] == [
        run_infer_server.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
    ]
    assert command[4] == "/models/rwkv-demo.pth"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert captured["check"] is False
