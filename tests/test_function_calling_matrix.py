from __future__ import annotations

import tomllib
from pathlib import Path

from src.bin.run_function_calling_matrix import _build_run_config, parse_args, parse_model_spec


def test_parse_model_spec() -> None:
    assert parse_model_spec("18082:rwkv7-g1e-7.2b-20260301-ctx8192:128") == (
        18082,
        "rwkv7-g1e-7.2b-20260301-ctx8192",
        128,
    )


def test_build_run_config_overrides_remote_model_and_batch(tmp_path: Path) -> None:
    source = tmp_path / "bfcl_v3.toml"
    source.write_text(
        """
[run]
max_samples = 50

[dataset]
name = "bfcl_v3"

[model]
infer_base_url = "http://127.0.0.1:18081"
infer_model = "old"
infer_max_workers = 32

[runner]
benchmark_kind = "bfcl_v3"
avg_ks = [1]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output = _build_run_config(
        source,
        output_path=tmp_path / "out.toml",
        port=18083,
        model_name="rwkv7-g1f-13.3b-20260415-ctx8192",
        batch_size=64,
    )

    payload = tomllib.loads(output.read_text(encoding="utf-8"))
    assert payload["model"]["infer_base_url"] == "http://127.0.0.1:18083"
    assert payload["model"]["infer_model"] == "rwkv7-g1f-13.3b-20260415-ctx8192"
    assert payload["model"]["infer_max_workers"] == 64
    assert payload["run"]["batch_size"] == 64


def test_build_run_config_applies_agent_long_context_ablation(tmp_path: Path) -> None:
    source = tmp_path / "tau3_bench_mock_long_context.toml"
    source.write_text(
        """
[run]
max_samples = 2

[dataset]
name = "tau3_bench_mock_long_context"

[model]
infer_base_url = "http://127.0.0.1:19081"
infer_model = "old"

[runner]
benchmark_kind = "tau3_bench"
long_doc_mode = "lexical"
tool_router_mode = "off"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output = _build_run_config(
        source,
        output_path=tmp_path / "out.toml",
        port=19081,
        model_name="rwkv7-g1f-7.2b-20260414-ctx8192",
        batch_size=1,
        ablation_variant="tool_router_lexical",
    )

    payload = tomllib.loads(output.read_text(encoding="utf-8"))
    assert payload["run"]["id"] == "ablation_tool_router_lexical"
    assert payload["runner"]["long_doc_mode"] == "off"
    assert payload["runner"]["tool_router_mode"] == "lexical"


def test_parse_args_accepts_ablation_variant() -> None:
    args = parse_args(["--ablation", "agent-long-context", "--ablation-variant", "baseline"])

    assert args.ablation == "agent-long-context"
    assert args.ablation_variants == ["baseline"]
