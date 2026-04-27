#!/usr/bin/env python3
"""Export latest wrong eval records using the same DB split as the space UI.

Fill ``EXPORT_TARGETS`` below, or pass one or more ``--target benchmark:model``
items from the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.db.eval_db_service import EvalDbService
from src.db.orm import init_orm
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import canonical_slug
from src.space.data import ScoreEntry, _score_entry_from_db
from src.space.metrics import _dataset_base, _format_param, _method_tag


# Edit this list on the server when you want a repeatable export.
EXPORT_TARGETS: list[dict[str, str]] = [
    {"benchmark": "aime24_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "aime25_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "amc23_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "asdiv_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "beyond_aime_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "college_math_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "comp_math_24_25_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "math_500_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "math_odyssey_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "olympiadbench_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "svamp_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "gpqa_main_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "mmlu_nocot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "human_eval_fix_nocot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "gsm8k_cot", "model": "rwkv7-g1f-13.3b"},
    {"benchmark": "algebra222_cot", "model": "rwkv7-g1f-13.3b"},
]


@dataclass(frozen=True)
class ExportTarget:
    benchmark: str
    model: str


@dataclass(frozen=True)
class BenchmarkKey:
    dataset: str
    method: str


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _strip_method_suffix(raw: str) -> tuple[str, str | None]:
    value = raw.strip().lower()
    for suffix, method in (("__cot", "cot"), ("_nocot", "nocot"), ("_cot", "cot")):
        if value.endswith(suffix):
            return value[: -len(suffix)], method
    return value, None


def _benchmark_key(raw: str) -> BenchmarkKey:
    dataset, method = _strip_method_suffix(raw)
    return BenchmarkKey(dataset=canonical_slug(dataset), method=method or "cot")


def _entry_benchmark_keys(entry: ScoreEntry) -> set[str]:
    dataset = canonical_slug(entry.dataset)
    base = canonical_slug(_dataset_base(dataset))
    return {dataset, base}


def _entry_model_keys(entry: ScoreEntry) -> set[str]:
    keys = {entry.model.strip().lower()}
    arch = (entry.arch_version or "").strip().lower()
    data = (entry.data_version or "").strip().lower()
    params = _format_param(entry.num_params).strip().lower() if entry.num_params else ""
    if arch and data and params:
        keys.add(f"{arch}-{data}-{params}")

    parts = entry.model.split("-")
    if len(parts) >= 3:
        keys.add("-".join(parts[:3]).strip().lower())
    return {key for key in keys if key}


def _matches_target(entry: ScoreEntry, target: ExportTarget) -> bool:
    key = _benchmark_key(target.benchmark)
    model = target.model.strip().lower()
    return (
        _method_tag(entry.cot) == key.method
        and key.dataset in _entry_benchmark_keys(entry)
        and model in _entry_model_keys(entry)
    )


def _resolve_latest_entry(entries: Iterable[ScoreEntry], target: ExportTarget) -> ScoreEntry | None:
    matches = [entry for entry in entries if _matches_target(entry, target) and entry.task_id is not None]
    if not matches:
        return None
    return max(matches, key=lambda entry: (entry.created_at, entry.task_id or -1, entry.model))


def _record_for_export(record: Mapping[str, Any], *, include_indexes: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "answer": str(record.get("answer") or ""),
        "ref-answer": str(record.get("ref_answer") or ""),
        "fail-reason": str(record.get("fail_reason") or ""),
        "context": record.get("context"),
    }
    if include_indexes:
        payload = {
            "sample_index": int(record.get("sample_index") or 0),
            "repeat_index": int(record.get("repeat_index") or 0),
            **payload,
        }
    return payload


def _load_frontend_latest_entries(*, include_param_search: bool) -> list[ScoreEntry]:
    init_orm(DEFAULT_DB_CONFIG)
    rows = EvalDbService().list_latest_scores_for_space(include_param_search=include_param_search)
    entries: list[ScoreEntry] = []
    errors: list[str] = []
    for row in rows:
        payload = dict(row) if isinstance(row, Mapping) else None
        if payload is None:
            continue
        entry = _score_entry_from_db(payload, errors)
        if entry is not None:
            entries.append(entry)
    if errors:
        print("[warn] space score normalization warnings:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
    return entries


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in value.strip())
    return cleaned.strip("._-") or "unknown"


def _target_output_path(output_root: Path, entry: ScoreEntry) -> Path:
    num_params = _safe_path_part((_format_param(entry.num_params).lower() if entry.num_params else "unknown"))
    data_version = _safe_path_part((entry.data_version or "unknown").lower())
    benchmark_name = _safe_path_part(f"{_dataset_base(entry.dataset)}_{_method_tag(entry.cot)}")
    return output_root / num_params / data_version / f"{benchmark_name}.json"


def export_wrong_records(
    targets: Sequence[ExportTarget],
    *,
    output_root: Path,
    include_param_search: bool = False,
    include_indexes: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    entries = _load_frontend_latest_entries(include_param_search=include_param_search)
    service = EvalDbService()
    exported_targets: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for target in targets:
        entry = _resolve_latest_entry(entries, target)
        if entry is None:
            missing.append(
                {
                    "benchmark": target.benchmark,
                    "model": target.model,
                    "reason": "no latest frontend score matched this benchmark/model",
                }
            )
            continue

        rows = service.list_eval_records_for_space(
            task_id=str(entry.task_id),
            only_wrong=True,
            include_context=True,
        )
        records = [_record_for_export(row, include_indexes=include_indexes) for row in rows]
        target_output_path = _target_output_path(output_root, entry)
        target_output = {
            "target": {
                "benchmark": target.benchmark,
                "model": target.model,
            },
            "resolved": {
                "benchmark": f"{_dataset_base(entry.dataset)}_{_method_tag(entry.cot)}",
                "dataset": entry.dataset,
                "model": entry.model,
                "task_id": entry.task_id,
                "score_created_at": _utc_iso(entry.created_at),
            },
            "wrong_count": len(records),
            "records": records,
        }
        target_output_path.parent.mkdir(parents=True, exist_ok=True)
        target_output_path.write_text(json.dumps(target_output, ensure_ascii=False, indent=2), encoding="utf-8")
        exported_targets.append(
            {
                "target": target_output["target"],
                "resolved": target_output["resolved"],
                "wrong_count": len(records),
                "path": str(target_output_path),
            }
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_param_search": include_param_search,
        "targets": exported_targets,
        "missing": missing,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _target_from_mapping(raw: Mapping[str, Any]) -> ExportTarget:
    benchmark = raw.get("benchmark") or raw.get("benchmark_name")
    model = raw.get("model") or raw.get("model_name")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError(f"target missing benchmark: {raw!r}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"target missing model: {raw!r}")
    return ExportTarget(benchmark=benchmark.strip(), model=model.strip())


def _load_targets_file(path: Path) -> list[ExportTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("targets")
    if not isinstance(payload, list):
        raise ValueError("targets file must be a JSON list or an object with a 'targets' list")
    targets: list[ExportTarget] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"target item must be an object: {item!r}")
        targets.append(_target_from_mapping(item))
    return targets


def _parse_cli_target(raw: str) -> ExportTarget:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("target must use benchmark:model format")
    benchmark, model = raw.split(":", 1)
    benchmark = benchmark.strip()
    model = model.strip()
    if not benchmark or not model:
        raise argparse.ArgumentTypeError("target must include both benchmark and model")
    return ExportTarget(benchmark=benchmark, model=model)


def _collect_targets(args: argparse.Namespace) -> list[ExportTarget]:
    targets = [_target_from_mapping(item) for item in EXPORT_TARGETS]
    if args.targets_file:
        targets.extend(_load_targets_file(args.targets_file))
    targets.extend(args.target or [])
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export wrong answer/ref-answer/fail-reason/context records for latest frontend scores.",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_cli_target,
        help="Benchmark/model pair in benchmark:model format. Example: aime24_cot:rwkv7-g1f-13.3b",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="JSON list of {'benchmark': ..., 'model': ...} targets, or an object with a 'targets' list.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exports/wrong_eval_records"),
        help="Output root. Files are written as num-params/data-version/benchmark-name.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON path listing exported files and missing targets.",
    )
    parser.add_argument(
        "--include-param-search",
        action="store_true",
        help="Include param-search scores. Default matches the frontend and excludes them.",
    )
    parser.add_argument(
        "--include-indexes",
        action="store_true",
        help="Include sample_index and repeat_index in each exported record.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    targets = _collect_targets(args)
    if not targets:
        parser.error("no targets provided; edit EXPORT_TARGETS or pass --target/--targets-file")

    output = export_wrong_records(
        targets,
        output_root=args.output_root,
        include_param_search=bool(args.include_param_search),
        include_indexes=bool(args.include_indexes),
        manifest_path=args.manifest,
    )
    print(f"exported {len(output['targets'])} target(s), missing {len(output['missing'])}: {args.output_root}")
    for item in output["targets"]:
        print(f"  - {item['path']}")
    if args.manifest:
        print(f"manifest: {args.manifest}")
    return 1 if output["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
