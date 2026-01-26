"""Migrate legacy flat results layout into the per-model directory structure."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import re
from pathlib import Path

from src.eval.results.layout import (
    COMPLETIONS_ROOT,
    CONSOLE_LOG_ROOT,
    EVAL_RESULTS_ROOT,
    PARAM_SEARCH_ROOT,
    SCORES_ROOT,
    eval_details_path,
    jsonl_path,
    scores_path,
)
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug


def _model_dataset_relpath(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    model_dir = safe_slug(model_name)
    slug = canonical_slug(dataset_slug)
    stem = f"{slug}__cot" if is_cot else slug
    return Path(model_dir) / stem


def _artifact_path(root: Path, dataset_slug: str, *, is_cot: bool, model_name: str, suffix: str) -> Path:
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return root / rel.parent / f"{rel.name}{suffix}"


def _param_search_directory(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return PARAM_SEARCH_ROOT / rel


def _param_search_trial(dataset_slug: str, *, is_cot: bool, model_name: str, trial: int) -> Path:
    return _param_search_directory(dataset_slug, is_cot=is_cot, model_name=model_name) / f"trial_{trial:02d}.jsonl"


def _param_search_records(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    return _param_search_directory(dataset_slug, is_cot=is_cot, model_name=model_name) / "records.jsonl"


def _parse_legacy_stem(stem: str) -> tuple[str, bool, str] | None:
    if "_cot_" in stem:
        dataset_part, _, rest = stem.partition("_cot_")
        return canonical_slug(dataset_part), True, rest
    if "_nocot_" in stem:
        dataset_part, _, rest = stem.partition("_nocot_")
        return canonical_slug(dataset_part), False, rest
    return None


def _should_use_canonical_path(stem: str) -> bool:
    return "__" not in stem


_VARIANT_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(__trial_\d+)$"),
)


def _match_variant_suffix(text: str | None) -> str | None:
    if not text:
        return None
    search_targets = [text]
    name = Path(text).name
    if name not in search_targets:
        search_targets.append(name)
    stem = Path(name).stem
    if stem not in search_targets:
        search_targets.append(stem)
    for target in search_targets:
        for pattern in _VARIANT_SUFFIX_PATTERNS:
            match = pattern.search(target)
            if match:
                return match.group(1)
    return None


def _split_model_variant(model_slug: str) -> tuple[str, str | None]:
    suffix = _match_variant_suffix(model_slug)
    if suffix:
        base = model_slug[: -len(suffix)]
        if base:
            return base, suffix
    return model_slug, None


def _append_variant_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _move_file(source: Path, target: Path, *, dry_run: bool, overwrite: bool) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists():
        if not overwrite:
            print(f"⚠️  Skip {source} -> {target} (exists)")
            return
        if dry_run:
            print(f"🧪 rm {target}")
        else:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"🧪 mv {source} -> {target}")
    else:
        source.replace(target)


def migrate_category(
    root: Path,
    *,
    suffix: str,
    resolver,
    derived_resolver,
    dry_run: bool,
    overwrite: bool,
    strip_results_suffix: bool = False,
) -> None:
    if not root.exists():
        return
    for path in root.iterdir():
        if not path.is_file() or not path.name.endswith(suffix):
            continue
        stem = path.stem
        if strip_results_suffix and stem.endswith("_results"):
            stem = stem[: -len("_results")]
        parsed = _parse_legacy_stem(stem)
        if not parsed:
            continue
        dataset_slug, is_cot, model_slug = parsed
        base_model_slug, variant_suffix = _split_model_variant(model_slug)
        use_canonical = _should_use_canonical_path(stem) or variant_suffix is not None
        if use_canonical:
            target = resolver(dataset_slug, is_cot=is_cot, model_name=base_model_slug or model_slug)
            target = _append_variant_suffix(target, variant_suffix)
        else:
            target = derived_resolver(model_slug, path.name)
        _move_file(path, target, dry_run=dry_run, overwrite=overwrite)


def _completion_resolver(dataset: str, *, is_cot: bool, model_name: str) -> Path:
    return jsonl_path(dataset, is_cot=is_cot, model_name=model_name)


def _eval_resolver(dataset: str, *, is_cot: bool, model_name: str) -> Path:
    return eval_details_path(dataset, is_cot=is_cot, model_name=model_name)


def _score_resolver(dataset: str, *, is_cot: bool, model_name: str) -> Path:
    return scores_path(dataset, is_cot=is_cot, model_name=model_name)


def _derived_path(root: Path, model_slug: str, name: str) -> Path:
    return root / model_slug / name


def migrate_logs(root: Path, *, dry_run: bool, overwrite: bool) -> None:
    if not root.exists():
        return
    for path in root.iterdir():
        if not path.is_file() or not path.suffix:
            continue
        stem_prefix = path.stem.split("--", 1)[0]
        parsed = _parse_legacy_stem(stem_prefix)
        if not parsed:
            continue
        dataset_slug, is_cot, model_slug = parsed
        base_model_slug, variant_suffix = _split_model_variant(model_slug)
        stem = canonical_slug(dataset_slug)
        if is_cot:
            stem = f"{stem}__cot"
        _, _, tail = path.name.partition(model_slug)
        if not tail:
            tail = path.suffix or ""
        suffix = variant_suffix or ""
        target_name = f"{stem}{suffix}{tail}"
        target = _derived_path(CONSOLE_LOG_ROOT, (base_model_slug or model_slug).split("--", 1)[0], target_name)
        _move_file(path, target, dry_run=dry_run, overwrite=overwrite)


def _resolve_existing_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    raw = Path(path_str).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
    name = raw.name
    if name:
        candidates.append(COMPLETIONS_ROOT / name)
        candidates.append(CONSOLE_LOG_ROOT / name)
        candidates.append(EVAL_RESULTS_ROOT / name)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    if name:
        escaped = glob.escape(name)
        for root in (COMPLETIONS_ROOT, CONSOLE_LOG_ROOT, EVAL_RESULTS_ROOT):
            try:
                if not root.exists():
                    continue
            except OSError:
                continue
            try:
                found = next(root.rglob(escaped))
            except StopIteration:
                continue
            except OSError:
                continue
            return found
    return None


def _guess_trial_index(path_str: str) -> int | None:
    stem = Path(path_str).stem
    if "__trial_" in stem:
        fragment = stem.rsplit("__trial_", 1)[1]
    elif "trial_" in stem:
        fragment = stem.rsplit("trial_", 1)[1]
    else:
        return None
    digits = "".join(ch for ch in fragment if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def _relocate_param_search_file(
    source_str: str,
    target: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    source = _resolve_existing_path(source_str)
    if source is None:
        print(f"⚠️  Param-search文件缺失，无法迁移：{source_str}")
        return False

    try:
        if source.resolve() == target.resolve():
            return False
    except OSError:
        pass

    if dry_run:
        print(f"🧪 mv {source} -> {target}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            print(f"⚠️  跳过 {target}：目标已存在（使用 --overwrite 覆盖）")
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    source.replace(target)
    return True


def _rewrite_param_search_paths(
    summary: dict,
    *,
    dataset_slug: str,
    is_cot: bool,
    model_name: str,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    changed = False
    relocated_trials: dict[int, Path] = {}
    if "records_path" in summary:
        new_records = str(_param_search_records(dataset_slug, is_cot=is_cot, model_name=model_name))
        if summary.get("records_path") != new_records:
            if _relocate_param_search_file(summary["records_path"], Path(new_records), dry_run=dry_run, overwrite=overwrite):
                summary["records_path"] = new_records
                changed = True

    def _determine_trial_index(raw_trial, fallback_path: str | None) -> int | None:
        if isinstance(raw_trial, int) and raw_trial > 0:
            return raw_trial
        if isinstance(fallback_path, str):
            guess = _guess_trial_index(fallback_path)
            if guess and guess > 0:
                return guess
        return None

    records = summary.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            source_path = record.get("log_path")
            trial_index = _determine_trial_index(record.get("trial"), source_path)
            if not trial_index:
                continue
            target_path = _param_search_trial(dataset_slug, is_cot=is_cot, model_name=model_name, trial=trial_index)
            if _relocate_param_search_file(source_path, target_path, dry_run=dry_run, overwrite=overwrite):
                changed = True
            desired = str(target_path)
            if record.get("log_path") != desired:
                record["log_path"] = desired
                changed = True
            relocated_trials[trial_index] = target_path

    best_trial = _determine_trial_index(summary.get("best_trial"), summary.get("best_log_path"))
    if best_trial:
        best_target = relocated_trials.get(best_trial) or _param_search_trial(
            dataset_slug, is_cot=is_cot, model_name=model_name, trial=best_trial
        )
        if _relocate_param_search_file(summary.get("best_log_path"), best_target, dry_run=dry_run, overwrite=overwrite):
            changed = True
        best_target_str = str(best_target)
        if summary.get("best_log_path") != best_target_str:
            summary["best_log_path"] = best_target_str
            changed = True

    return changed


def rewrite_score_paths(root: Path, *, dry_run: bool, overwrite: bool) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        dataset = payload.get("dataset")
        model = payload.get("model")
        if not isinstance(dataset, str) or not isinstance(model, str):
            continue
        slug = canonical_slug(dataset)
        is_cot = bool(payload.get("cot", False))
        changed = False

        new_log_path = str(_artifact_path(COMPLETIONS_ROOT, slug, is_cot=is_cot, model_name=model, suffix=".jsonl"))
        if payload.get("log_path") != new_log_path:
            payload["log_path"] = new_log_path
            changed = True

        task_details = payload.get("task_details")
        if isinstance(task_details, dict):
            eval_path = task_details.get("eval_details_path")
            if eval_path is not None:
                new_eval_path = str(
                    _artifact_path(EVAL_RESULTS_ROOT, slug, is_cot=is_cot, model_name=model, suffix="_results.jsonl")
                )
                if eval_path != new_eval_path:
                    task_details["eval_details_path"] = new_eval_path
                    changed = True
            param_search = task_details.get("param_search")
            if isinstance(param_search, dict):
                if _rewrite_param_search_paths(
                    param_search,
                    dataset_slug=slug,
                    is_cot=is_cot,
                    model_name=model,
                    dry_run=dry_run,
                    overwrite=overwrite,
                ):
                    changed = True

        if changed:
            if dry_run:
                print(f"🧪 rewrite score paths: {path}")
            else:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate flat results layout to per-model directories")
    parser.add_argument("--dry-run", action="store_true", help="Print moves without touching files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite conflicting targets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    migrate_category(
        COMPLETIONS_ROOT,
        suffix=".jsonl",
        resolver=_completion_resolver,
        derived_resolver=lambda model, name: _derived_path(COMPLETIONS_ROOT, model, name),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    migrate_category(
        EVAL_RESULTS_ROOT,
        suffix=".jsonl",
        resolver=_eval_resolver,
        derived_resolver=lambda model, name: _derived_path(EVAL_RESULTS_ROOT, model, name),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        strip_results_suffix=True,
    )
    migrate_category(
        SCORES_ROOT,
        suffix=".json",
        resolver=_score_resolver,
        derived_resolver=lambda model, name: _derived_path(SCORES_ROOT, model, name),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    migrate_logs(CONSOLE_LOG_ROOT, dry_run=args.dry_run, overwrite=args.overwrite)
    rewrite_score_paths(SCORES_ROOT, dry_run=args.dry_run, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
