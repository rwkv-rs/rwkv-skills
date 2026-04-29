from __future__ import annotations

"""Run the function-calling benchmark matrix against remote infer services."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Mapping, Sequence


NO_JUDGE_BENCHMARKS: tuple[str, ...] = (
    "bfcl_v3",
    "tau_bench_airline",
    "tau_bench_retail",
    "tau_bench_telecom",
    "tau2_bench_airline",
    "tau2_bench_retail",
    "tau2_bench_telecom",
)
JUDGE_BENCHMARKS: tuple[str, ...] = (
    "browsecomp",
    "browsecomp_zh",
    "mcp_bench",
)
ALL_BENCHMARKS: tuple[str, ...] = NO_JUDGE_BENCHMARKS + JUDGE_BENCHMARKS

DEFAULT_MODEL_SPECS: tuple[str, ...] = (
    "18081:rwkv7-g1e-13.3b-20260309-ctx8192:64",
    "18082:rwkv7-g1e-7.2b-20260301-ctx8192:128",
    "18083:rwkv7-g1f-13.3b-20260415-ctx8192:64",
    "18084:rwkv7-g1f-7.2b-20260414-ctx8192:128",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all function-calling configs for multiple remote RWKV services")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model service spec: PORT:MODEL_NAME:BATCH_SIZE. Repeat for multiple models.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        choices=ALL_BENCHMARKS,
        help="Benchmark to run. Repeat to run a subset. Defaults to all function-calling benchmarks.",
    )
    parser.add_argument("--config-dir", default="configs/run", help="Directory containing run configs")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run src.main")
    parser.add_argument("--keep-going", action="store_true", help="Continue with the next run after a failure")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--enable-checker",
        action="store_true",
        help="Do not set RWKV_SKILLS_DISABLE_CHECKER=1 for BFCL/tau/tau2 runs",
    )
    return parser.parse_args(argv)


def parse_model_spec(raw: str) -> tuple[int, str, int]:
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"model spec must be PORT:MODEL_NAME:BATCH_SIZE, got {raw!r}")
    port_raw, model_name, batch_raw = parts
    port = int(port_raw)
    batch_size = max(1, int(batch_raw))
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("model name cannot be empty")
    return port, model_name, batch_size


def run_matrix(args: argparse.Namespace) -> int:
    model_specs = tuple(parse_model_spec(raw) for raw in (args.models or DEFAULT_MODEL_SPECS))
    benchmarks = tuple(args.benchmarks or ALL_BENCHMARKS)
    config_dir = Path(args.config_dir).expanduser()
    with tempfile.TemporaryDirectory(prefix="rwkv-fc-matrix-") as tmp:
        tmp_root = Path(tmp)
        for port, model_name, batch_size in model_specs:
            for benchmark in benchmarks:
                source_config = config_dir / f"{benchmark}.toml"
                run_config = _build_run_config(
                    source_config,
                    output_path=tmp_root / f"{benchmark}.{model_name}.toml",
                    port=port,
                    model_name=model_name,
                    batch_size=batch_size,
                )
                command = [str(args.python), "-m", "src.main", "--config", str(run_config)]
                env = dict(os.environ)
                if benchmark in NO_JUDGE_BENCHMARKS and not args.enable_checker:
                    env["RWKV_SKILLS_DISABLE_CHECKER"] = "1"
                print(f"$ {' '.join(command)}", flush=True)
                if args.dry_run:
                    continue
                result = subprocess.run(command, env=env, check=False)
                if result.returncode != 0 and not args.keep_going:
                    return int(result.returncode)
        return 0


def _build_run_config(
    source_path: Path,
    *,
    output_path: Path,
    port: int,
    model_name: str,
    batch_size: int,
) -> Path:
    payload = _load_toml_mapping(source_path)
    updated: dict[str, Any] = {key: _copy_table(value) for key, value in payload.items()}

    model = _ensure_table(updated, "model")
    model["infer_base_url"] = f"http://127.0.0.1:{int(port)}"
    model["infer_model"] = model_name
    model["infer_max_workers"] = max(1, int(batch_size))

    run = _ensure_table(updated, "run")
    if source_path.stem != "mcp_bench":
        run["batch_size"] = max(1, int(batch_size))
    else:
        run.pop("batch_size", None)

    output_path.write_text(_render_toml(updated), encoding="utf-8")
    return output_path


def _load_toml_mapping(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = tomllib.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"config must be a TOML table: {path}")
    return payload


def _copy_table(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_table(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


def _ensure_table(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    table: dict[str, Any] = {}
    payload[key] = table
    return table


def _render_toml(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for section, values in payload.items():
        if not isinstance(values, Mapping):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, Mapping):
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return _toml_quote(str(value))


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main(argv: Sequence[str] | None = None) -> int:
    return run_matrix(parse_args(argv))


__all__ = [
    "ALL_BENCHMARKS",
    "DEFAULT_MODEL_SPECS",
    "JUDGE_BENCHMARKS",
    "NO_JUDGE_BENCHMARKS",
    "main",
    "parse_args",
    "parse_model_spec",
    "run_matrix",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
