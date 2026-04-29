from __future__ import annotations

import tomllib
from pathlib import Path

from src.bin.run_function_calling_matrix import _build_run_config, parse_model_spec


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
