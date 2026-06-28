from __future__ import annotations

import json
from pathlib import Path

from src.bin.run_infer_fleet import (
    InferServiceSpec,
    RunningInferService,
    build_command,
    parse_args,
    plan_deployments,
    resolve_max_batch_sizes,
    resolve_model_names,
    write_manifest,
)


def test_resolve_model_names_defaults_to_weight_stems() -> None:
    paths = (Path("/models/rwkv7-g1e-7.2b.pth"), Path("/models/rwkv7-g1f-13.3b.pth"))

    assert resolve_model_names(paths, None) == ("rwkv7-g1e-7.2b", "rwkv7-g1f-13.3b")


def test_resolve_max_batch_sizes_supports_per_model_values() -> None:
    paths = (Path("/models/rwkv7-g1e-13.3b.pth"), Path("/models/rwkv7-g1e-7.2b.pth"))

    assert resolve_max_batch_sizes(paths, max_batch_size=8, max_batch_sizes=None) == (8, 8)
    assert resolve_max_batch_sizes(paths, max_batch_size=8, max_batch_sizes=(64, 128)) == (64, 128)


def test_plan_deployments_skips_assigned_gpus_and_increments_ports(tmp_path: Path) -> None:
    paths = (
        Path("/models/a.pth"),
        Path("/models/b.pth"),
        Path("/models/c.pth"),
    )
    names = ("model-a", "model-b", "model-c")

    specs = plan_deployments(
        model_paths=paths,
        model_names=names,
        max_batch_sizes=(64, 128, 64),
        idle_gpus=("0", "1", "2"),
        assigned_gpus={"0"},
        base_port=18081,
        log_dir=tmp_path / "logs",
        state_db_dir=tmp_path / "state",
        launched_count=1,
    )

    assert [(spec.model_name, spec.gpu, spec.port) for spec in specs] == [
        ("model-a", "1", 18082),
        ("model-b", "2", 18083),
    ]
    assert [spec.max_batch_size for spec in specs] == [64, 128]
    assert specs[0].log_path == tmp_path / "logs" / "model-a.port18082.log"


def test_plan_deployments_spreads_replicas_across_distinct_gpus(tmp_path: Path) -> None:
    specs = plan_deployments(
        model_paths=(Path("/models/a.pth"), Path("/models/b.pth")),
        model_names=("model-a", "model-b"),
        max_batch_sizes=(64, 128),
        idle_gpus=("0", "1", "2"),
        assigned_gpus=set(),
        base_port=18081,
        log_dir=tmp_path / "logs",
        state_db_dir=tmp_path / "state",
        launched_count=0,
        replicas_per_model=2,
    )

    assert [(spec.model_name, spec.gpu, spec.port, spec.replica_index) for spec in specs] == [
        ("model-a", "0", 18081, 0),
        ("model-a", "1", 18082, 1),
    ]


def test_build_command_targets_visible_cuda_zero(tmp_path: Path) -> None:
    spec = InferServiceSpec(
        model_path=tmp_path / "rwkv.pth",
        model_name="demo",
        gpu="3",
        port=18081,
        max_batch_size=4,
        log_path=tmp_path / "demo.log",
    )

    command = build_command(
        spec,
        host="127.0.0.1",
        api_key="secret",
        engine_mode="vllm-rwkv",
        vllm_rwkv_path="/opt/vllm-rwkv",
        vllm_python="/opt/vllm/bin/python",
        infer_auto_config="off",
        batch_collect_ms=10,
        log_level="warning",
        max_model_len=8192,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.97,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    assert command[:3][-2:] == ["-m", "src.bin.run_infer_server"]
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--cuda-visible-devices") + 1] == "3"
    assert command[command.index("--engine-mode") + 1] == "vllm-rwkv"
    assert command[command.index("--vllm-rwkv-path") + 1] == "/opt/vllm-rwkv"
    assert command[command.index("--vllm-python") + 1] == "/opt/vllm/bin/python"
    assert command[command.index("--port") + 1] == "18081"
    assert command[command.index("--max-batch-size") + 1] == "4"
    assert command[command.index("--infer-auto-config") + 1] == "off"
    assert command[command.index("--api-key") + 1] == "secret"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--max-num-seqs") + 1] == "256"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.97"
    assert "--enforce-eager" in command


def test_parse_args_ignores_removed_nano_engine_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RWKV_INFER_ENGINE_MODE", "nano-vllm")

    args = parse_args(
        [
            "--model-paths",
            str(tmp_path / "rwkv-demo.pth"),
        ]
    )

    assert args.engine_mode == "vllm-rwkv"


def test_write_manifest_serializes_service_urls(tmp_path: Path) -> None:
    spec = InferServiceSpec(
        model_path=tmp_path / "rwkv.pth",
        model_name="demo",
        gpu="0",
        port=18081,
        max_batch_size=64,
        log_path=tmp_path / "demo.log",
    )
    manifest = tmp_path / "fleet.json"

    write_manifest(
        manifest,
        services=(RunningInferService(spec=spec, pid=1234),),
        host="127.0.0.1",
        api_key_set=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["api_key_set"] is True
    assert payload["services"][0]["base_url"] == "http://127.0.0.1:18081"
    assert payload["services"][0]["health_url"] == "http://127.0.0.1:18081/healthz"
    assert payload["services"][0]["max_batch_size"] == 64
    assert payload["services"][0]["pid"] == 1234
    assert "state_db_path" not in payload["services"][0]
