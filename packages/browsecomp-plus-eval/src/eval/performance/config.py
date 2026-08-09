from __future__ import annotations

import shlex
import tomllib
from pathlib import Path
from typing import Any, Mapping

from src.eval.scheduler.dataset_utils import canonical_slug


REPO_ROOT = Path(__file__).resolve().parents[3]
PERF_CONFIG_ROOT = REPO_ROOT / "configs" / "perf"


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a table/object")


def _maybe_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    raise TypeError(f"{field_name} must be a string")


def _maybe_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _maybe_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    raise TypeError(f"{field_name} must be an integer")


def _maybe_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field_name} must be a number")


def _list_int(value: object, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    values: list[int] = []
    for item in value:
        parsed = _maybe_int(item, field_name=field_name)
        if parsed is None:
            continue
        values.append(parsed)
    return tuple(values)


def _list_str(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must contain only strings")
        text = item.strip()
        if text:
            values.append(text)
    return tuple(values)


def _csv(values: tuple[int, ...]) -> str | None:
    if not values:
        return None
    return ",".join(str(value) for value in values)


def _shell_join(values: tuple[str, ...]) -> str | None:
    if not values:
        return None
    return shlex.join(values)


def resolve_perf_config_path(config_path: str | Path) -> Path:
    raw = str(config_path).strip()
    if not raw:
        raise ValueError("config path must not be empty")

    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()

    candidate = PERF_CONFIG_ROOT / raw
    if candidate.is_file():
        return candidate.resolve()

    if direct.suffix:
        raise FileNotFoundError(f"perf config file not found: {direct.resolve()}")

    named = PERF_CONFIG_ROOT / f"{canonical_slug(raw)}.toml"
    if named.is_file():
        return named.resolve()

    raise FileNotFoundError(
        f"perf config file not found: {raw!r}; searched {direct.resolve()} and {named}"
    )


def load_perf_config_defaults(config_path: str | Path) -> dict[str, Any]:
    path = resolve_perf_config_path(config_path)
    with path.open("rb") as fh:
        payload = tomllib.load(fh)
    if not isinstance(payload, Mapping):
        raise TypeError("perf config must be a table/object")

    service = _as_mapping(payload.get("service"), field_name="service")
    tokenizer = _as_mapping(payload.get("tokenizer"), field_name="tokenizer")
    workload = _as_mapping(payload.get("workload"), field_name="workload")
    hardware = _as_mapping(payload.get("hardware"), field_name="hardware")
    report = _as_mapping(payload.get("report"), field_name="report")
    vllm = _as_mapping(payload.get("vllm"), field_name="vllm")

    defaults: dict[str, Any] = {
        "config": str(path),
    }

    pairs: tuple[tuple[str, Any], ...] = (
        ("base_url", _maybe_str(service.get("base_url"), field_name="service.base_url")),
        ("model", _maybe_str(service.get("model"), field_name="service.model")),
        ("api_key", _maybe_str(service.get("api_key"), field_name="service.api_key")),
        ("protocol", _maybe_str(service.get("protocol"), field_name="service.protocol")),
        ("stack_name", _maybe_str(service.get("stack_name"), field_name="service.stack_name")),
        ("engine_name", _maybe_str(service.get("engine_name"), field_name="service.engine_name")),
        ("precision", _maybe_str(service.get("precision"), field_name="service.precision")),
        ("tokenizer_type", _maybe_str(tokenizer.get("type"), field_name="tokenizer.type")),
        ("tokenizer_ref", _maybe_str(tokenizer.get("ref"), field_name="tokenizer.ref")),
        ("ctx_lens", _csv(_list_int(workload.get("ctx_lens"), field_name="workload.ctx_lens"))),
        (
            "concurrency_levels",
            _csv(_list_int(workload.get("concurrency_levels"), field_name="workload.concurrency_levels")),
        ),
        ("batch_sizes", _csv(_list_int(workload.get("batch_sizes"), field_name="workload.batch_sizes"))),
        ("output_tokens", _maybe_int(workload.get("output_tokens"), field_name="workload.output_tokens")),
        ("warmup_runs", _maybe_int(workload.get("warmup_runs"), field_name="workload.warmup_runs")),
        ("measure_runs", _maybe_int(workload.get("measure_runs"), field_name="workload.measure_runs")),
        ("temperature", _maybe_float(workload.get("temperature"), field_name="workload.temperature")),
        ("top_p", _maybe_float(workload.get("top_p"), field_name="workload.top_p")),
        ("timeout_s", _maybe_float(workload.get("timeout_s"), field_name="workload.timeout_s")),
        ("seed", _maybe_int(workload.get("seed"), field_name="workload.seed")),
        (
            "skip_concurrency_matrix",
            _maybe_bool(workload.get("skip_concurrency_matrix"), field_name="workload.skip_concurrency_matrix"),
        ),
        (
            "skip_batch_size_matrix",
            _maybe_bool(workload.get("skip_batch_size_matrix"), field_name="workload.skip_batch_size_matrix"),
        ),
        ("gpu_index", _maybe_int(hardware.get("gpu_index"), field_name="hardware.gpu_index")),
        ("hardware_label", _maybe_str(hardware.get("hardware_label"), field_name="hardware.hardware_label")),
        ("result_path", _maybe_str(report.get("result_path"), field_name="report.result_path")),
        ("launch_vllm", _maybe_bool(vllm.get("launch"), field_name="vllm.launch")),
        (
            "vllm_python",
            _maybe_str(vllm.get("python_executable"), field_name="vllm.python_executable"),
        ),
        ("vllm_host", _maybe_str(vllm.get("host"), field_name="vllm.host")),
        ("vllm_port", _maybe_int(vllm.get("port"), field_name="vllm.port")),
        ("vllm_dtype", _maybe_str(vllm.get("dtype"), field_name="vllm.dtype")),
        (
            "vllm_tensor_parallel_size",
            _maybe_int(vllm.get("tensor_parallel_size"), field_name="vllm.tensor_parallel_size"),
        ),
        (
            "vllm_gpu_memory_utilization",
            _maybe_float(vllm.get("gpu_memory_utilization"), field_name="vllm.gpu_memory_utilization"),
        ),
        ("vllm_max_model_len", _maybe_int(vllm.get("max_model_len"), field_name="vllm.max_model_len")),
        ("vllm_max_num_seqs", _maybe_int(vllm.get("max_num_seqs"), field_name="vllm.max_num_seqs")),
        (
            "vllm_max_num_batched_tokens",
            _maybe_int(vllm.get("max_num_batched_tokens"), field_name="vllm.max_num_batched_tokens"),
        ),
        (
            "vllm_trust_remote_code",
            _maybe_bool(vllm.get("trust_remote_code"), field_name="vllm.trust_remote_code"),
        ),
        (
            "vllm_command",
            _shell_join(_list_str(vllm.get("command"), field_name="vllm.command")),
        ),
        (
            "vllm_extra_args",
            _shell_join(_list_str(vllm.get("extra_args"), field_name="vllm.extra_args")),
        ),
        (
            "vllm_startup_timeout_s",
            _maybe_float(vllm.get("startup_timeout_s"), field_name="vllm.startup_timeout_s"),
        ),
        ("vllm_log_path", _maybe_str(vllm.get("log_path"), field_name="vllm.log_path")),
    )

    for key, value in pairs:
        if value is not None:
            defaults[key] = value
    return defaults


__all__ = [
    "PERF_CONFIG_ROOT",
    "load_perf_config_defaults",
    "resolve_perf_config_path",
]
