from __future__ import annotations

import json
from pathlib import Path

from src.bin.run_infer_fleet import (
    InferServiceSpec,
    RunningInferService,
    build_command,
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
    assert specs[0].state_db_path == tmp_path / "state" / "model-a.sqlite3"


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
        engine_mode="classic",
        batch_collect_ms=10,
        log_level="warning",
    )

    assert command[:3][-2:] == ["-m", "src.bin.run_infer_server"]
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--port") + 1] == "18081"
    assert command[command.index("--max-batch-size") + 1] == "4"
    assert command[command.index("--api-key") + 1] == "secret"


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
