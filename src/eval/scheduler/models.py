from __future__ import annotations

"""Model discovery helpers."""

import glob
import os
import re
from pathlib import Path
from typing import Final, Sequence

from .config import REPO_ROOT


MODEL_SELECT_CHOICES: Final[tuple[str, ...]] = ("all", "param-extrema", "latest-data")
MODEL_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z]+[0-9]+)(?P<tag>[a-z]+)?(?P<number>[0-9]+)?$",
    re.IGNORECASE,
)

ARCH_ORDER: Final[tuple[str, ...]] = ("rwkv7", "rwkv7a", "rwkv7b")
DATA_VERSION_ORDER: Final[tuple[str, ...]] = (
    "g0",
    "g0a",
    "g0a2",
    "g0a3",
    "g0a4",
    "g0b",
    "g0c",
    "g1",
    "g1a",
    "g1a2",
    "g1a3",
    "g1a4",
    "g1b",
    "g1c",
)
NUM_PARAM_ORDER: Final[tuple[str, ...]] = ("0.1b", "0.4b", "1.5b", "2.9b", "7.2b", "13.3b")
NUM_PARAM_SKIP: Final[set[str]] = {"0.1b", "0.4b"}


def expand_model_paths(patterns: Sequence[str]) -> list[Path]:
    matched: set[Path] = set()
    for pattern in patterns:
        if not pattern:
            continue
        expanded = os.path.expanduser(os.path.expandvars(pattern))
        if not os.path.isabs(expanded):
            expanded = str(REPO_ROOT / expanded)
        for candidate in glob.glob(expanded):
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                matched.add(candidate_path.resolve())
    return sorted(matched)


def _normalize_model_identifier(raw: str) -> str:
    if not isinstance(raw, str):
        return raw
    sanitized = raw.replace("_", "-")

    def _decimal_replacer(match: re.Match[str]) -> str:
        whole = match.group("whole")
        frac = match.group("frac")
        return f"{whole}.{frac}b"

    sanitized = re.sub(r"(?P<whole>\d+)-(?P<frac>\d+)b", _decimal_replacer, sanitized)
    parts = sanitized.split("-")
    head_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"\d{8}", part):
            break
        if part.lower().startswith("ctx"):
            break
        head_parts.append(part)
    return "-".join(head_parts) if head_parts else sanitized


def _parse_model_tags(identifier: str) -> tuple[str | None, str | None, str | None]:
    """Extract (arch_version, data_version, num_params token) from normalized identifier.

    Example: rwkv7-g1a4-2.9b -> ("rwkv7", "g1a4", "2.9b")
    """

    if not identifier:
        return None, None, None
    parts = identifier.split("-")
    arch = parts[0].lower() if parts else None
    data_version = None
    num_params = None
    for token in parts[1:]:
        low = token.lower()
        if low in DATA_VERSION_ORDER:
            data_version = low
        if re.fullmatch(r"\d+(?:\.\d+)?b", low):
            num_params = low
    return arch, data_version, num_params


def _rank(seq: Sequence[str], value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return list(seq).index(value)
    except ValueError:
        return None


def _extract_param_count(model_short: str) -> float | None:
    if not isinstance(model_short, str):
        return None
    for token in model_short.split("-"):
        if token.lower().endswith("b"):
            number = token[:-1]
            try:
                return float(number)
            except ValueError:
                continue
    return None


def _model_version_sort_key(model: str) -> tuple[str, int, int, str]:
    if not isinstance(model, str) or not model:
        return "", 0, 0, ""
    prefix = model.split("-")[0]
    match = MODEL_PREFIX_PATTERN.match(prefix)
    if not match:
        return prefix, 0, 0, model
    family = match.group("family") or ""
    tag = match.group("tag") or ""
    number = match.group("number") or ""

    if not tag:
        tag_rank = 0
    elif tag == "a":
        tag_rank = 1
    else:
        tag_rank = 2

    try:
        number_rank = int(number) if number else 0
    except ValueError:
        number_rank = 0

    return (family, tag_rank, number_rank, model)


def filter_model_paths(
    model_paths: Sequence[Path],
    strategy: str,
    min_param_b: float | None,
    max_param_b: float | None,
) -> list[Path]:
    entries: list[tuple[float | None, str, Path, str | None, str | None]] = []
    for path in model_paths:
        normalized = _normalize_model_identifier(path.stem)
        arch, data_version, num_token = _parse_model_tags(normalized)
        params_val = _extract_param_count(normalized)
        if min_param_b is not None and params_val is not None and params_val < min_param_b:
            continue
        if max_param_b is not None and (params_val is None or params_val > max_param_b):
            continue
        entries.append((params_val, normalized, path, data_version, num_token))

    if strategy == "all":
        return [path for _, _, path, *_ in entries]

    if strategy == "param-extrema":
        grouped: dict[float | None, list[tuple[str, Path]]] = {}
        for params_val, normalized, path, *_ in entries:
            grouped.setdefault(params_val, []).append((normalized, path))

        selected: set[Path] = set()
        for params_val, models in grouped.items():
            model_map: dict[str, Path] = {}
            for normalized, path in models:
                model_map.setdefault(normalized, path)
            if params_val is None or len(model_map) <= 2:
                selected.update(model_map.values())
                continue
            ordered_models = sorted(model_map.keys(), key=_model_version_sort_key)
            keep = {ordered_models[0], ordered_models[-1]}
            for model_name in keep:
                selected.add(model_map[model_name])
        return [path for _, _, path, *_ in entries if path in selected]

    if strategy == "latest-data":
        selected: set[Path] = set()
        fallback: list[Path] = []
        groups: dict[tuple[str, str], list[tuple[str | None, str, Path]]] = {}
        for _, normalized, path, data_version, num_token in entries:
            if num_token in NUM_PARAM_SKIP:
                continue
            arch, _, _ = _parse_model_tags(normalized)
            if arch is None or data_version is None or num_token is None:
                fallback.append(path)
                continue
            groups.setdefault((arch, num_token), []).append((data_version, normalized, path))

        for (arch, num_token), items in groups.items():
            # pick the highest data_version according to DATA_VERSION_ORDER
            items_sorted = sorted(
                items,
                key=lambda x: (
                    _rank(ARCH_ORDER, arch) or 0,
                    _rank(NUM_PARAM_ORDER, num_token) or 0,
                    _rank(DATA_VERSION_ORDER, x[0]) or -1,
                    _model_version_sort_key(x[1]),
                ),
            )
            best = items_sorted[-1]
            selected.add(best[2])

        # include anything we couldn't classify
        selected.update(fallback)
        return [path for _, _, path, *_ in entries if path in selected]

    raise ValueError(f"未知的模型选择策略: {strategy!r}")


__all__ = [
    "expand_model_paths",
    "filter_model_paths",
    "MODEL_SELECT_CHOICES",
]
