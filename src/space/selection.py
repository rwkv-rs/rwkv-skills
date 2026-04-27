"""Model selection, snapshot resolution, sorting, and lineage helpers."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .data import (
    ARCH_VERSIONS,
    DATA_VERSIONS,
    NUM_PARAMS,
    ScoreEntry,
    latest_entries_for_model,
    list_models,
    load_scores,
    parse_model_signature,
    pick_latest_model,
)
from .constants import (
    AUTO_MODEL_LABEL,
    SelectionState,
    ParamLineage,
)
from .metrics import (
    _format_param,
    _dataset_base,
    _method_tag,
)


# ---------------------------------------------------------------------------
# Rank / sort helpers
# ---------------------------------------------------------------------------

def _rank_token(candidates: Sequence[str], value: str | None) -> int | None:
    if not value:
        return None
    low = value.lower()
    try:
        return candidates.index(low)
    except ValueError:
        return None


def _sort_entries(entries: Iterable[ScoreEntry]) -> list[ScoreEntry]:
    def sort_key(item: ScoreEntry) -> tuple[Any, ...]:
        arch_rank = _rank_token(ARCH_VERSIONS, item.arch_version)
        param_rank = _rank_token(NUM_PARAMS, item.num_params)
        data_rank = _rank_token(DATA_VERSIONS, item.data_version)
        return (
            arch_rank if arch_rank is not None else len(ARCH_VERSIONS),
            param_rank if param_rank is not None else len(NUM_PARAMS),
            -(data_rank if data_rank is not None else -1),
            item.domain,
            item.dataset,
            item.task or "",
            item.model,
        )

    return sorted(entries, key=sort_key)


def _series_sort_key(model: str) -> tuple[int, int, int, str]:
    sig = parse_model_signature(model)
    param_rank = sig.param_rank if sig.param_rank is not None else len(NUM_PARAMS)
    arch_rank = sig.arch_rank if sig.arch_rank is not None else len(ARCH_VERSIONS)
    data_rank = sig.data_rank if sig.data_rank is not None else len(DATA_VERSIONS)
    return (param_rank, arch_rank, data_rank, model)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_entry(entry: ScoreEntry) -> dict[str, Any]:
    arch_rank = _rank_token(ARCH_VERSIONS, entry.arch_version)
    data_rank = _rank_token(DATA_VERSIONS, entry.data_version)
    param_rank = _rank_token(NUM_PARAMS, entry.num_params)
    return {
        "model": entry.model,
        "arch": entry.arch_version,
        "data": entry.data_version,
        "params": entry.num_params,
        "arch_rank": arch_rank,
        "data_rank": data_rank,
        "param_rank": param_rank,
        "created": entry.created_at,
        "has_signature": bool(entry.arch_version and entry.num_params),
    }


def _model_snapshots(entries: Iterable[ScoreEntry]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for entry in entries:
        current = snapshots.get(entry.model)
        if current is None or entry.created_at > current["created"]:
            snapshots[entry.model] = _snapshot_entry(entry)
    return snapshots


def _select_signature_snapshots(entries: Iterable[ScoreEntry]) -> list[dict[str, Any]]:
    snapshots = _model_snapshots(entries)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fallback: list[dict[str, Any]] = []
    for snap in snapshots.values():
        if snap["has_signature"]:
            key = (snap["arch"], snap["params"])
            grouped.setdefault(key, []).append(snap)
        else:
            fallback.append(snap)

    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        best = max(
            items,
            key=lambda snap: (
                snap["data_rank"] if snap["data_rank"] is not None else -1,
                snap["created"].timestamp(),
                snap["model"],
            ),
        )
        selected.append(best)

    selected.sort(
        key=lambda snap: (
            snap["arch_rank"] if snap["arch_rank"] is not None else len(ARCH_VERSIONS),
            snap["param_rank"] if snap["param_rank"] is not None else len(NUM_PARAMS),
            -(snap["data_rank"] if snap["data_rank"] is not None else -1),
            snap["model"],
        )
    )
    fallback.sort(key=lambda snap: (-snap["created"].timestamp(), snap["model"]))
    return selected + fallback


def _latest_entries_for_signatures(entries: Iterable[ScoreEntry]) -> tuple[list[ScoreEntry], list[dict[str, Any]], list[str]]:
    snapshots = _select_signature_snapshots(entries)
    if not snapshots:
        return [], [], []
    ordered_models: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        model = snap["model"]
        if model in seen:
            continue
        seen.add(model)
        ordered_models.append(model)
    combined: list[ScoreEntry] = []
    entry_list = list(entries)
    for model in ordered_models:
        combined.extend(latest_entries_for_model(entry_list, model))
    return _sort_entries(combined), snapshots, ordered_models


# ---------------------------------------------------------------------------
# Display name helpers
# ---------------------------------------------------------------------------

def _model_display_name(model: str) -> str:
    sig = parse_model_signature(model)
    arch = (sig.arch or "").lower()
    data = (sig.data or "").lower()
    params = sig.params
    param_label = _format_param(params).lower() if params else ""
    if arch and data and param_label:
        return f"{arch}-{data}-{param_label}"
    parts = model.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return model


def _model_data_param_label(model: str, *, include_params: bool = True) -> str:
    sig = parse_model_signature(model)
    data = (sig.data or "").lower()
    params = _format_param(sig.params).lower() if sig.params else ""
    if include_params:
        if data and params:
            return f"{data}-{params}"
        if params:
            return params
        if data:
            return data
    else:
        if data:
            return data
        if params:
            return params
    return _model_display_name(model)


def _format_combo_label(snapshot: dict[str, Any]) -> str:
    if snapshot.get("has_signature"):
        arch = snapshot.get("arch") or "未知架构"
        params = _format_param(snapshot.get("params"))
        data = snapshot.get("data") or "?"
        return f"{arch} · {params} → {data}"
    return snapshot.get("model") or "未知模型"


def _summarise_snapshots(snapshots: Sequence[dict[str, Any]]) -> list[str]:
    combo_labels = [_format_combo_label(snap) for snap in snapshots if snap.get("has_signature")]
    extra_labels = [_format_combo_label(snap) for snap in snapshots if not snap.get("has_signature")]

    lines: list[str] = []
    if combo_labels:
        preview = " / ".join(combo_labels[:4])
        if len(combo_labels) > 4:
            preview += f" 等 {len(combo_labels)} 个组合"
        lines.append(f"- 覆盖组合：{preview}")
    if extra_labels:
        preview = " / ".join(extra_labels[:4])
        if len(extra_labels) > 4:
            preview += f" 等 {len(extra_labels)} 个模型"
        lines.append(f"- 其他未解析模型：{preview}")
    return lines


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _prepare_selection(entries: list[ScoreEntry], selection_value: str | None) -> SelectionState:
    if not entries:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    if selection_value == AUTO_MODEL_LABEL or selection_value is None:
        combined_entries, snapshots, ordered_models = _latest_entries_for_signatures(entries)
        allowed_models = ordered_models
        filtered_entries = [entry for entry in combined_entries if entry.model in allowed_models]
        return SelectionState(
            entries=_sort_entries(filtered_entries),
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label=AUTO_MODEL_LABEL,
            auto_selected=False,
            model_sequence=allowed_models,
            aggregated_models=snapshots,
            skipped_small_params=0,
        )

    models = set(list_models(entries))
    target_model = selection_value if selection_value in models else pick_latest_model(entries)
    auto_selected = target_model != selection_value
    if not target_model:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    latest = latest_entries_for_model(entries, target_model)
    return SelectionState(
        entries=_sort_entries(latest),
        dropdown_value=target_model,
        selected_label=target_model,
        auto_selected=auto_selected,
        model_sequence=[target_model],
        aggregated_models=None,
        skipped_small_params=0,
    )


def _compute_choices(entries: list[ScoreEntry]) -> list[str]:
    return [AUTO_MODEL_LABEL] + list_models(entries)


def _initial_payload() -> tuple[list[ScoreEntry], SelectionState, list[str]]:
    errors: list[str] = []
    entries = load_scores(errors=errors)
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL)
    return entries, selection, errors


# ---------------------------------------------------------------------------
# Model-entries cache & lineage resolution
# ---------------------------------------------------------------------------

def _build_model_entries_cache(entries: list[ScoreEntry]) -> dict[str, list[ScoreEntry]]:
    cache: dict[str, list[ScoreEntry]] = {}
    for model in list_models(entries):
        cache[model] = latest_entries_for_model(entries, model)
    return cache


def _resolve_param_lineages(entries: list[ScoreEntry], selection: SelectionState) -> list[ParamLineage]:
    snapshots = [snap for snap in _model_snapshots(entries).values() if snap.get("has_signature")]
    if not snapshots:
        return []

    by_arch_param: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for snap in snapshots:
        arch = snap.get("arch")
        param = snap.get("params")
        if not arch or not param:
            continue
        by_arch_param.setdefault((arch, param), []).append(snap)

    for items in by_arch_param.values():
        items.sort(
            key=lambda snap: (
                snap.get("data_rank") if snap.get("data_rank") is not None else -1,
                snap["created"].timestamp(),
                snap.get("model", ""),
            )
        )

    lineages: list[ParamLineage] = []

    if selection.dropdown_value != AUTO_MODEL_LABEL:
        target_model = selection.dropdown_value
        sig = parse_model_signature(target_model)
        if sig.arch and sig.params:
            items = by_arch_param.get((sig.arch, sig.params), [])
            prev_model: str | None = None
            for idx, snap in enumerate(items):
                if snap.get("model") != target_model:
                    continue
                if idx > 0:
                    prev_model = str(items[idx - 1]["model"])
                break
            lineages.append(
                ParamLineage(
                    param=sig.params,
                    latest_model=target_model,
                    latest_label=_model_display_name(target_model),
                    prev_model=prev_model,
                    prev_label=_model_display_name(prev_model) if prev_model else "—",
                )
            )
        else:
            lineages.append(
                ParamLineage(
                    param="custom",
                    latest_model=target_model,
                    latest_label=_model_display_name(target_model),
                    prev_model=None,
                    prev_label="—",
                )
            )
        return lineages

    by_param: dict[str, list[dict[str, Any]]] = {}
    for (_, param), items in by_arch_param.items():
        if not items:
            continue
        by_param.setdefault(param, []).append(items[-1])

    for param, candidates in by_param.items():
        latest = max(
            candidates,
            key=lambda snap: (
                snap.get("data_rank") if snap.get("data_rank") is not None else -1,
                snap["created"].timestamp(),
                snap.get("arch_rank") if snap.get("arch_rank") is not None else -1,
                snap.get("model", ""),
            ),
        )
        latest_model = str(latest["model"])
        arch = str(latest["arch"])
        chain = by_arch_param.get((arch, param), [])
        prev_model = None
        for idx, snap in enumerate(chain):
            if snap.get("model") != latest_model:
                continue
            if idx > 0:
                prev_model = str(chain[idx - 1]["model"])
            break
        lineages.append(
            ParamLineage(
                param=param,
                latest_model=latest_model,
                latest_label=_model_display_name(latest_model),
                prev_model=prev_model,
                prev_label=_model_display_name(prev_model) if prev_model else "—",
            )
        )

    lineages.sort(
        key=lambda item: (
            _rank_token(NUM_PARAMS, item.param) if _rank_token(NUM_PARAMS, item.param) is not None else len(NUM_PARAMS),
            item.param,
        )
    )
    return lineages


def _entries_for_model_in_domains(
    model_cache: dict[str, list[ScoreEntry]],
    model: str | None,
    domains: set[str],
) -> list[ScoreEntry]:
    if not model:
        return []
    return [entry for entry in model_cache.get(model, []) if entry.domain in domains]


def _build_detail_point_map(entries: Iterable[ScoreEntry]) -> dict[tuple[str, str, str], "DetailPoint"]:
    from .constants import DetailPoint
    from .metrics import _detail_rows_for_entry

    mapped: dict[tuple[str, str, str], DetailPoint] = {}
    for entry in entries:
        for benchmark, method, k_metric, score in _detail_rows_for_entry(entry):
            key = (benchmark, method, k_metric)
            previous = mapped.get(key)
            should_replace = False
            if previous is None:
                should_replace = True
            elif score is None and previous.score is None:
                should_replace = entry.created_at > previous.entry.created_at
            elif score is not None and previous.score is None:
                should_replace = True
            elif score is None and previous.score is not None:
                should_replace = entry.created_at > previous.entry.created_at
            elif previous.score is not None and score is not None:
                should_replace = score > previous.score or (
                    score == previous.score and entry.created_at > previous.entry.created_at
                )
            if should_replace:
                mapped[key] = DetailPoint(score=score, entry=entry)
    return mapped
