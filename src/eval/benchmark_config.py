from __future__ import annotations

"""Benchmark-level overrides loaded from configs/<benchmark>.toml."""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import (
    canonical_slug,
    safe_slug,
    split_benchmark_and_split,
)
from src.infer.sampling import SamplingConfig

CONFIG_ROOT = REPO_ROOT / "configs"
TEMPLATE_PATH = CONFIG_ROOT / "_templates.toml"

_INT_FIELDS = {"max_generate_tokens", "top_k"}
_FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "alpha_presence",
    "alpha_frequency",
    "alpha_decay",
    "noise",
}
_TUPLE_INT_FIELDS = {"stop_tokens", "ban_tokens", "no_penalty_token_ids"}
_BOOL_FIELDS = {"pad_zero"}
_STR_FIELDS = {"sample_mode"}

_CONFIG_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}


@dataclass(slots=True)
class BenchmarkModelConfig:
    sampling_overrides: dict[str, object]
    # Optional evaluation-level overrides (e.g. free-response pass@k / avg@k).
    pass_k: tuple[int, ...] | None = None
    avg_k: tuple[int, ...] | None = None
    report_pass_k: tuple[int, ...] | None = None
    report_avg_k: tuple[int, ...] | None = None

    def apply_sampling(self, base: SamplingConfig) -> SamplingConfig:
        if not self.sampling_overrides:
            return base
        return replace(base, **self.sampling_overrides)


def config_path_for_benchmark(benchmark_name: str) -> Path:
    slug = canonical_slug(benchmark_name)
    return CONFIG_ROOT / f"{slug}.toml"


def resolve_benchmark_model_config(
    dataset_slug: str,
    model_name: str,
    *,
    stage: str | None = None,
) -> BenchmarkModelConfig | None:
    benchmark, _ = split_benchmark_and_split(dataset_slug)
    tables = _load_benchmark_tables(benchmark)
    if not tables:
        return None
    default_table = _select_table(tables, "default")
    stage_table = _select_table(tables, stage) if stage else None
    stage_direct, stage_models = _split_stage_table(stage_table)
    model_table = _select_model_table(tables, model_name)
    stage_model_table = _select_model_table(stage_models, model_name) if stage_models else None
    merged: dict[str, Any] = {}
    if default_table:
        merged.update(default_table)
    if stage_direct:
        merged.update(stage_direct)
    if model_table:
        merged.update(model_table)
    if stage_model_table:
        merged.update(stage_model_table)
    if not merged:
        return None
    merged = _merge_templates(merged)
    return _parse_table(merged)


def _load_benchmark_tables(benchmark_name: str) -> dict[str, Mapping[str, Any]]:
    path = config_path_for_benchmark(benchmark_name)
    payload = _load_toml(path)
    tables: dict[str, Mapping[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            tables[str(key)] = value
    return tables


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cached = _CONFIG_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    if not isinstance(payload, dict):
        payload = {}
    _CONFIG_CACHE[path] = (mtime, payload)
    return payload


def _load_template_tables() -> dict[str, Mapping[str, Any]]:
    payload = _load_toml(TEMPLATE_PATH)
    tables: dict[str, Mapping[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            tables[str(key)] = value
    return tables


def _select_table(
    tables: Mapping[str, Mapping[str, Any]], key: str
) -> Mapping[str, Any] | None:
    if key in tables:
        return tables[key]
    lower_key = key.lower()
    for name, table in tables.items():
        if name.lower() == lower_key:
            return table
    return None


def _split_stage_table(
    table: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Mapping[str, Any]]]:
    if not table:
        return None, {}
    direct: dict[str, Any] = {}
    nested: dict[str, Mapping[str, Any]] = {}
    for key, value in table.items():
        if isinstance(value, Mapping):
            nested[str(key)] = value
        else:
            direct[key] = value
    return direct or None, nested


def _merge_templates(table: Mapping[str, Any]) -> dict[str, Any]:
    templates = _extract_template_names(table)
    if not templates:
        return dict(table)
    template_tables = _load_template_tables()
    merged: dict[str, Any] = {}
    for name in templates:
        template = _select_table(template_tables, name)
        if template:
            merged.update(template)
    for key, value in table.items():
        if key in {"template", "templates"}:
            continue
        merged[key] = value
    return merged


def _extract_template_names(table: Mapping[str, Any]) -> tuple[str, ...]:
    raw = table.get("template")
    if raw is None:
        raw = table.get("templates")
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        names = [item for item in raw if isinstance(item, str)]
        return tuple(names)
    return tuple()


def _select_model_table(
    tables: Mapping[str, Mapping[str, Any]], model_name: str
) -> Mapping[str, Any] | None:
    if model_name in tables:
        return tables[model_name]
    lower_name = model_name.lower()
    for name, table in tables.items():
        if name.lower() == lower_name:
            return table
    normalized_target = _normalize_model_key(model_name)
    for name, table in tables.items():
        if _normalize_model_key(name) == normalized_target:
            return table
    return None


def _normalize_model_key(value: str) -> str:
    return safe_slug(value).lower()


def _parse_table(table: Mapping[str, Any]) -> BenchmarkModelConfig:
    sampling_overrides: dict[str, object] = {}
    pass_k: tuple[int, ...] | None = None
    avg_k: tuple[int, ...] | None = None
    report_pass_k: tuple[int, ...] | None = None
    report_avg_k: tuple[int, ...] | None = None

    for key, raw in table.items():
        if key in _INT_FIELDS:
            value = _coerce_int(raw)
        elif key in _FLOAT_FIELDS:
            value = _coerce_float(raw)
        elif key in _TUPLE_INT_FIELDS:
            value = _coerce_int_tuple(raw)
        elif key in _BOOL_FIELDS:
            value = raw if isinstance(raw, bool) else None
        elif key in _STR_FIELDS:
            value = str(raw) if isinstance(raw, str) else None
        elif key == "pass_k":
            pass_k = _coerce_k_tuple(raw)
            continue
        elif key == "avg_k":
            avg_k = _coerce_k_tuple(raw)
            continue
        elif key == "report_pass_k":
            report_pass_k = _coerce_k_tuple(raw)
            continue
        elif key == "report_avg_k":
            report_avg_k = _coerce_k_tuple(raw)
            continue
        else:
            continue
        if value is not None:
            sampling_overrides[key] = value

    return BenchmarkModelConfig(
        sampling_overrides=sampling_overrides,
        pass_k=pass_k,
        avg_k=avg_k,
        report_pass_k=report_pass_k,
        report_avg_k=report_avg_k,
    )


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return None
        return (int(value),)
    if isinstance(value, (list, tuple)):
        items: list[int] = []
        for item in value:
            if isinstance(item, bool):
                return None
            if isinstance(item, (int, float)):
                if isinstance(item, float) and not item.is_integer():
                    return None
                items.append(int(item))
            else:
                return None
        return tuple(items)
    return None


def _coerce_k_tuple(value: Any) -> tuple[int, ...] | None:
    """Coerce pass@k / avg@k style configs.

    Accepts int or list/tuple of ints; filters out non-positive values and sorts/uniques.
    Returns an empty tuple when explicitly configured as an empty list.
    """

    raw = _coerce_int_tuple(value)
    if raw is None:
        return None
    filtered = sorted({int(item) for item in raw if int(item) > 0})
    return tuple(filtered)


def resolve_sampling_config(
    dataset_slug: str,
    model_name: str,
    *,
    stage: str | None = None,
    base: SamplingConfig | None = None,
    fallback_templates: str | Sequence[str] | None = None,
) -> SamplingConfig | None:
    config = resolve_benchmark_model_config(dataset_slug, model_name, stage=stage)
    if config is None and fallback_templates:
        merged = _merge_templates(
            {"templates": _normalize_template_names(fallback_templates)}
        )
        config = _parse_table(merged)
    if config is None:
        return None
    sampling_base = base or SamplingConfig()
    return config.apply_sampling(sampling_base)


def _normalize_template_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    names = [item for item in value if isinstance(item, str)]
    return tuple(names)


__all__ = [
    "BenchmarkModelConfig",
    "config_path_for_benchmark",
    "resolve_benchmark_model_config",
    "resolve_sampling_config",
]
