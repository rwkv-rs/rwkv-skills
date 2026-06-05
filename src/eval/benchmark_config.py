from __future__ import annotations

"""Benchmark-level overrides loaded from configs/<model_name>/<benchmark>.toml."""

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for existing server envs.
    import tomli as tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.k_values import NumericK
from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import (
    canonical_slug,
    safe_slug,
    split_benchmark_and_split,
)
from src.infer.sampling import SamplingConfig

CONFIG_ROOT = REPO_ROOT / "configs"
CONFIG_OVERRIDE_ROOT_ENV = "RWKV_BENCHMARK_CONFIG_ROOT"

_INT_FIELDS = {"max_generate_tokens", "top_k"}
_FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "alpha_presence",
    "alpha_frequency",
    "alpha_decay",
}
_TUPLE_INT_FIELDS = {"stop_tokens", "ban_tokens", "no_penalty_token_ids"}
_BOOL_FIELDS = {"pad_zero"}
_STRING_FIELDS = {
    "direct_prompt_template",
    "cot_prompt_template",
    "final_prompt_template",
    "judge_prompt_template",
    "user_model",
    "user_api_key",
    "user_base_url",
    "judge_model",
    "judge_api_key",
    "judge_base_url",
}
_CONFIG_KEY_ALIASES = {
    "long_doc_mode": "long_context_router_mode",
    "long_doc_min_chars": "long_context_min_chars",
    "long_doc_max_chars": "long_context_chunk_chars",
    "long_doc_overlap_lines": "long_context_overlap_lines",
    "long_doc_max_evidence_chunks": "long_context_max_evidence_chunks",
    "long_doc_max_evidence_chars": "long_context_max_evidence_chars",
    "long_doc_query_chars": "long_context_query_chars",
}

_CONFIG_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}


@dataclass(slots=True)
class BenchmarkModelConfig:
    sampling_overrides: dict[str, object]
    # Optional evaluation-level overrides (e.g. free-response pass@k / avg@k).
    pass_k: tuple[int, ...] | None = None
    avg_k: tuple[NumericK, ...] | None = None
    report_pass_k: tuple[int, ...] | None = None
    report_avg_k: tuple[NumericK, ...] | None = None
    max_samples: int | None = None
    direct_prompt_template: str | None = None
    cot_prompt_template: str | None = None
    final_prompt_template: str | None = None
    judge_prompt_template: str | None = None
    browsecomp_plus_judge: dict[str, Any] | None = None
    tool_router_mode: str | None = None
    tool_router_max_tools: int | None = None
    tool_router_trigger_tool_count: int | None = None
    tool_router_trigger_catalog_chars: int | None = None
    tool_router_context_chars: int | None = None
    tool_router_description_chars: int | None = None
    long_context_router_mode: str | None = None
    long_context_min_chars: int | None = None
    long_context_chunk_chars: int | None = None
    long_context_overlap_lines: int | None = None
    long_context_max_evidence_chunks: int | None = None
    long_context_max_evidence_chars: int | None = None
    long_context_query_chars: int | None = None
    history_max_chars: int | None = None
    prompt_max_chars: int | None = None
    max_steps: int | None = None
    max_tool_errors: int | None = None
    decision_max_tokens: int | None = None
    max_repeated_tool_calls: int | None = None
    user_model: str | None = None
    user_api_key: str | None = None
    user_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None

    def apply_sampling(self, base: SamplingConfig) -> SamplingConfig:
        if not self.sampling_overrides:
            return base
        return replace(base, **self.sampling_overrides)


def config_path_for_benchmark(benchmark_name: str, model_name: str | None = None) -> Path:
    roots = _config_roots(override_first=True)
    if model_name:
        for root in roots:
            path = _config_path_for_root(root, benchmark_name, model_name)
            if path.exists():
                return path
        for root in roots:
            path = _config_path_for_root(root, benchmark_name, None)
            if path.exists():
                return path
        return _config_path_for_root(CONFIG_ROOT, benchmark_name, model_name)

    for root in roots:
        path = _config_path_for_root(root, benchmark_name, None)
        if path.exists():
            return path
    return _config_path_for_root(CONFIG_ROOT, benchmark_name, None)


def _config_path_for_root(
    root: Path,
    benchmark_name: str,
    model_name: str | None = None,
) -> Path:
    raw_slug = safe_slug(canonical_slug(benchmark_name)).lower()
    if model_name:
        model_slug = safe_slug(model_name)
        direct = root / model_slug / f"{raw_slug}.toml"
        if direct.exists():
            return direct
        base, _ = split_benchmark_and_split(raw_slug)
        return root / model_slug / f"{safe_slug(base).lower()}.toml"

    direct = root / f"{raw_slug}.toml"
    if direct.exists():
        return direct
    base, _ = split_benchmark_and_split(raw_slug)
    return root / f"{safe_slug(base).lower()}.toml"

def _config_roots(*, override_first: bool = False) -> tuple[Path, ...]:
    override = _config_override_root()
    if override is None:
        return (CONFIG_ROOT,)
    if override_first:
        return (override, CONFIG_ROOT)
    return (CONFIG_ROOT, override)


def _config_override_root() -> Path | None:
    raw = os.environ.get(CONFIG_OVERRIDE_ROOT_ENV)
    if not raw:
        return None
    return Path(raw).expanduser()


def resolve_benchmark_model_config(
    dataset_slug: str,
    model_name: str,
    *,
    stage: str | None = None,
) -> BenchmarkModelConfig | None:
    benchmark, _ = split_benchmark_and_split(dataset_slug)
    tables = _load_benchmark_tables(benchmark, model_name)
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


def _load_benchmark_tables(benchmark_name: str, model_name: str) -> dict[str, Mapping[str, Any]]:
    tables: dict[str, Mapping[str, Any]] = {}
    for path in _benchmark_config_paths(benchmark_name, model_name):
        payload = _load_toml(path)
        for key, value in payload.items():
            if not isinstance(value, Mapping):
                continue
            key = str(key)
            previous = tables.get(key, {})
            tables[key] = _merge_mapping(previous, value)
    return tables


def _benchmark_config_paths(benchmark_name: str, model_name: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in _config_roots():
        for path in (
            _config_path_for_root(root, benchmark_name, None),
            _config_path_for_root(root, benchmark_name, model_name),
        ):
            if path.exists() and path not in paths:
                paths.append(path)
    return tuple(paths)


def _merge_mapping(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(existing, value)
        else:
            merged[key] = value
    return merged


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
    tables: dict[str, Mapping[str, Any]] = {}
    for root in _config_roots():
        payload = _load_toml(root / "_templates.toml")
        for key, value in payload.items():
            if not isinstance(value, Mapping):
                continue
            key = str(key)
            previous = tables.get(key, {})
            tables[key] = _merge_mapping(previous, value)
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
    avg_k: tuple[NumericK, ...] | None = None
    report_pass_k: tuple[int, ...] | None = None
    report_avg_k: tuple[NumericK, ...] | None = None
    max_samples: int | None = None
    direct_prompt_template: str | None = None
    cot_prompt_template: str | None = None
    final_prompt_template: str | None = None
    judge_prompt_template: str | None = None
    browsecomp_plus_judge: dict[str, Any] | None = None
    tool_router_mode: str | None = None
    tool_router_max_tools: int | None = None
    tool_router_trigger_tool_count: int | None = None
    tool_router_trigger_catalog_chars: int | None = None
    tool_router_context_chars: int | None = None
    tool_router_description_chars: int | None = None
    long_context_router_mode: str | None = None
    long_context_min_chars: int | None = None
    long_context_chunk_chars: int | None = None
    long_context_overlap_lines: int | None = None
    long_context_max_evidence_chunks: int | None = None
    long_context_max_evidence_chars: int | None = None
    long_context_query_chars: int | None = None
    history_max_chars: int | None = None
    prompt_max_chars: int | None = None
    max_steps: int | None = None
    max_tool_errors: int | None = None
    decision_max_tokens: int | None = None
    max_repeated_tool_calls: int | None = None
    user_model: str | None = None
    user_api_key: str | None = None
    user_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None

    for raw_key, raw in table.items():
        key = _CONFIG_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key in _INT_FIELDS:
            value = _coerce_int(raw)
        elif key in _FLOAT_FIELDS:
            value = _coerce_float(raw)
        elif key in _TUPLE_INT_FIELDS:
            value = _coerce_int_tuple(raw)
        elif key in _BOOL_FIELDS:
            value = raw if isinstance(raw, bool) else None
        elif key == "pass_k":
            pass_k = _coerce_k_tuple(raw)
            continue
        elif key == "avg_k":
            avg_k = _coerce_avg_k_tuple(raw)
            continue
        elif key == "report_pass_k":
            report_pass_k = _coerce_k_tuple(raw)
            continue
        elif key == "report_avg_k":
            report_avg_k = _coerce_avg_k_tuple(raw)
            continue
        elif key == "max_samples":
            max_samples = _coerce_int(raw)
            continue
        elif key == "direct_prompt_template":
            direct_prompt_template = _coerce_str(raw)
            continue
        elif key == "cot_prompt_template":
            cot_prompt_template = _coerce_str(raw)
            continue
        elif key == "final_prompt_template":
            final_prompt_template = _coerce_str(raw)
            continue
        elif key == "judge_prompt_template":
            judge_prompt_template = _coerce_str(raw)
            continue
        elif key == "browsecomp_plus_judge":
            browsecomp_plus_judge = _coerce_str_mapping(raw)
            continue
        elif key == "tool_router_mode":
            tool_router_mode = _coerce_str(raw)
            continue
        elif key == "tool_router_max_tools":
            tool_router_max_tools = _coerce_int(raw)
            continue
        elif key == "tool_router_trigger_tool_count":
            tool_router_trigger_tool_count = _coerce_int(raw)
            continue
        elif key == "tool_router_trigger_catalog_chars":
            tool_router_trigger_catalog_chars = _coerce_int(raw)
            continue
        elif key == "tool_router_context_chars":
            tool_router_context_chars = _coerce_int(raw)
            continue
        elif key == "tool_router_description_chars":
            tool_router_description_chars = _coerce_int(raw)
            continue
        elif key == "long_context_router_mode":
            long_context_router_mode = _coerce_str(raw)
            continue
        elif key == "long_context_min_chars":
            long_context_min_chars = _coerce_int(raw)
            continue
        elif key == "long_context_chunk_chars":
            long_context_chunk_chars = _coerce_int(raw)
            continue
        elif key == "long_context_overlap_lines":
            long_context_overlap_lines = _coerce_int(raw)
            continue
        elif key == "long_context_max_evidence_chunks":
            long_context_max_evidence_chunks = _coerce_int(raw)
            continue
        elif key == "long_context_max_evidence_chars":
            long_context_max_evidence_chars = _coerce_int(raw)
            continue
        elif key == "long_context_query_chars":
            long_context_query_chars = _coerce_int(raw)
            continue
        elif key == "history_max_chars":
            history_max_chars = _coerce_int(raw)
            continue
        elif key == "prompt_max_chars":
            prompt_max_chars = _coerce_int(raw)
            continue
        elif key == "max_steps":
            max_steps = _coerce_int(raw)
            continue
        elif key == "max_tool_errors":
            max_tool_errors = _coerce_int(raw)
            continue
        elif key == "decision_max_tokens":
            decision_max_tokens = _coerce_int(raw)
            continue
        elif key == "max_repeated_tool_calls":
            max_repeated_tool_calls = _coerce_int(raw)
            continue
        elif key == "user_model":
            user_model = _coerce_str(raw)
            continue
        elif key == "user_api_key":
            user_api_key = _coerce_str(raw)
            continue
        elif key == "user_base_url":
            user_base_url = _coerce_str(raw)
            continue
        elif key == "judge_model":
            judge_model = _coerce_str(raw)
            continue
        elif key == "judge_api_key":
            judge_api_key = _coerce_str(raw)
            continue
        elif key == "judge_base_url":
            judge_base_url = _coerce_str(raw)
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
        max_samples=max_samples,
        direct_prompt_template=direct_prompt_template,
        cot_prompt_template=cot_prompt_template,
        final_prompt_template=final_prompt_template,
        judge_prompt_template=judge_prompt_template,
        browsecomp_plus_judge=browsecomp_plus_judge,
        tool_router_mode=tool_router_mode,
        tool_router_max_tools=tool_router_max_tools,
        tool_router_trigger_tool_count=tool_router_trigger_tool_count,
        tool_router_trigger_catalog_chars=tool_router_trigger_catalog_chars,
        tool_router_context_chars=tool_router_context_chars,
        tool_router_description_chars=tool_router_description_chars,
        long_context_router_mode=long_context_router_mode,
        long_context_min_chars=long_context_min_chars,
        long_context_chunk_chars=long_context_chunk_chars,
        long_context_overlap_lines=long_context_overlap_lines,
        long_context_max_evidence_chunks=long_context_max_evidence_chunks,
        long_context_max_evidence_chars=long_context_max_evidence_chars,
        long_context_query_chars=long_context_query_chars,
        history_max_chars=history_max_chars,
        prompt_max_chars=prompt_max_chars,
        max_steps=max_steps,
        max_tool_errors=max_tool_errors,
        decision_max_tokens=decision_max_tokens,
        max_repeated_tool_calls=max_repeated_tool_calls,
        user_model=user_model,
        user_api_key=user_api_key,
        user_base_url=user_base_url,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
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


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_str_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


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


def _coerce_avg_k_tuple(value: Any) -> tuple[NumericK, ...] | None:
    """Coerce avg@k configs.

    Accepts:
    - positive integers (e.g. 16 -> avg@16)
    - positive ratios in (0, 1) (e.g. 0.2 -> avg@0.2)
    """

    if value is None or isinstance(value, bool):
        return None
    values: list[NumericK] = []
    raw_items = value if isinstance(value, (list, tuple)) else (value,)
    for item in raw_items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if number <= 0:
            continue
        if number >= 1:
            if not number.is_integer():
                return None
            values.append(int(number))
            continue
        values.append(number)
    normalized: list[NumericK] = []
    seen: set[str] = set()
    for item in sorted(values, key=float):
        key = str(int(item)) if isinstance(item, int) else f"{float(item):.12g}"
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return tuple(normalized)


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
