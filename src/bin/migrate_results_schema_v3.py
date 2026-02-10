from __future__ import annotations

"""Migrate legacy `results/completions` + `results/eval` JSONL into the v3 schema.

v3 schema (completions):
  benchmark_name, dataset_split, sample_index, repeat_index, sampling_config,
  promptN, completionN, stop_reasonN...

v3 schema (eval):
  benchmark_name, dataset_split, sample_index, repeat_index, context,
  is_passed, fail_reason

Notes:
- This migration is best-effort for legacy logs. `sampling_config` cannot be
  reliably reconstructed from old artifacts and is left empty.
- For logits-only legacy stages (multi-choice), the token string is inferred
  heuristically from `logitsN` / `predicted`.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import orjson

from src.eval.results.schema import (
    build_context_from_completions,
    dataset_slug_parts,
    strip_artifact_suffix,
    IndexValidationError,
)


_LETTER_RE = re.compile(r"^[A-Z]$")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
    tmp.replace(path)


def _stage_indices(payload: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    for key in payload:
        for prefix in ("prompt", "output", "finish_reason", "logits", "completion", "stop_reason"):
            if key.startswith(prefix) and key.removeprefix(prefix).isdigit():
                indices.add(int(key.removeprefix(prefix)))
    return sorted(indices)


def _pick_logits_token(logits: Any) -> str | None:
    if not isinstance(logits, dict) or not logits:
        return None

    def score(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("-inf")

    token, value = max(logits.items(), key=lambda kv: score(kv[1]))
    if score(value) == float("-inf"):
        return None
    return str(token)


def _infer_repeats_from_legacy(payloads: Iterable[dict[str, Any]]) -> int:
    max_id: int | None = None
    for payload in payloads:
        sample_id = payload.get("sample_id")
        if isinstance(sample_id, int) and sample_id >= 0:
            max_id = sample_id if max_id is None else max(max_id, sample_id)
    return (max_id + 1) if max_id is not None else 1


def _infer_logits_token_text(prompt: str, token: str) -> str:
    """Heuristic: if token is a single A-Z letter, prefix a space unless prompt already ends with whitespace."""
    token = token.strip()
    if not _LETTER_RE.match(token):
        return token
    if prompt and prompt[-1].isspace():
        return token
    return f" {token}"


def _convert_completions_record(
    legacy: dict[str, Any],
    *,
    benchmark_name: str,
    dataset_split: str,
    repeats: int,
) -> dict[str, Any]:
    # Resolve indices
    sample_index = legacy.get("problem_index")
    repeat_index = legacy.get("sample_id")
    if not isinstance(sample_index, int):
        raw_idx = legacy.get("sample_index")
        if isinstance(raw_idx, int) and isinstance(repeat_index, int) and repeats > 0:
            sample_index = raw_idx // repeats
        elif isinstance(raw_idx, int):
            sample_index = raw_idx
        else:
            raise IndexValidationError(
                f"sample_index is required but could not be resolved from legacy record: {legacy.keys()}"
            )
    if not isinstance(repeat_index, int):
        raw_idx = legacy.get("sample_index")
        if isinstance(raw_idx, int) and repeats > 0:
            repeat_index = raw_idx % repeats
        else:
            raise IndexValidationError(
                f"repeat_index is required but could not be resolved from legacy record: {legacy.keys()}"
            )

    stages = _stage_indices(legacy)
    out: dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "dataset_split": dataset_split,
        "sample_index": int(sample_index),
        "repeat_index": int(repeat_index),
        "sampling_config": {},
    }

    context_so_far = ""
    for idx in stages:
        raw_prompt = legacy.get(f"prompt{idx}")
        if raw_prompt is None and idx == 1:
            # Legacy writers sometimes had `prompt` in metadata; only use it as a last resort.
            raw_prompt = legacy.get("prompt")
        if raw_prompt is None:
            continue
        raw_prompt = str(raw_prompt)

        # Completion
        raw_completion = legacy.get(f"output{idx}")
        if raw_completion is None:
            raw_completion = legacy.get(f"completion{idx}")
        if raw_completion is None:
            # logits-only
            token = legacy.get("predicted")
            if token is None:
                token = _pick_logits_token(legacy.get(f"logits{idx}"))
            token = "" if token is None else str(token)
            raw_completion = _infer_logits_token_text(raw_prompt, token)
        else:
            raw_completion = str(raw_completion)

        # Stop reason
        stop_reason = legacy.get(f"finish_reason{idx}")
        if stop_reason is None:
            stop_reason = legacy.get(f"stop_reason{idx}")
        if stop_reason is None:
            stop_reason = "logits_only" if legacy.get(f"logits{idx}") is not None else ""
        stop_reason = str(stop_reason)

        if idx == 1:
            prompt_seg = raw_prompt
        else:
            # Legacy promptN is usually the full prompt; store delta so concatenation matches final context.
            if raw_prompt.startswith(context_so_far):
                prompt_seg = raw_prompt[len(context_so_far) :]
            else:
                prompt_seg = raw_prompt

        out[f"prompt{idx}"] = prompt_seg
        out[f"completion{idx}"] = raw_completion
        out[f"stop_reason{idx}"] = stop_reason

        context_so_far = f"{context_so_far}{prompt_seg}{raw_completion}"

    return out


def migrate_completions_file(in_path: Path, out_path: Path) -> None:
    stem = strip_artifact_suffix(in_path.stem)
    benchmark_name, dataset_split = dataset_slug_parts(stem)
    legacy_payloads = list(_iter_jsonl(in_path))
    repeats = _infer_repeats_from_legacy(legacy_payloads)

    def rows():
        for payload in legacy_payloads:
            yield _convert_completions_record(
                payload,
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                repeats=repeats,
            )

    _write_jsonl_atomic(out_path, rows())


def _eval_key_from_legacy(
    legacy: dict[str, Any],
    *,
    repeats: int,
    key_map: dict[tuple[int, int], int] | None = None,
) -> tuple[int, int]:
    # Prefer explicit (problem_index, sample_id)
    sample_index = legacy.get("problem_index")
    repeat_index = legacy.get("sample_id")
    if isinstance(sample_index, int) and isinstance(repeat_index, int):
        return int(sample_index), int(repeat_index)

    # IFEval legacy uses (sample_key, sample_id)
    if key_map is not None:
        sample_key = legacy.get("sample_key")
        sample_id = legacy.get("sample_id")
        if isinstance(sample_key, int) and isinstance(sample_id, int):
            mapped = key_map.get((sample_key, sample_id))
            if mapped is not None:
                return int(mapped), int(sample_id)

    # Fall back to sample_index / repeat_index if present
    sample_index = legacy.get("sample_index")
    repeat_index = legacy.get("repeat_index")
    if isinstance(sample_index, int) and isinstance(repeat_index, int):
        return int(sample_index), int(repeat_index)

    # Or derive from global row index + repeats
    if isinstance(sample_index, int) and repeats > 0:
        return int(sample_index // repeats), int(sample_index % repeats)
    raise IndexValidationError(
        f"sample_index/repeat_index could not be resolved from legacy eval record: {legacy.keys()}"
    )


def _infer_pass_fail(legacy: dict[str, Any]) -> tuple[bool, str]:
    if "passed" in legacy:
        passed = bool(legacy.get("passed"))
        reason = str(legacy.get("result") or "")
        return passed, reason
    if "correct" in legacy:
        passed = bool(legacy.get("correct"))
        return passed, ""
    if "follow_all" in legacy:
        passed = bool(legacy.get("follow_all"))
        return passed, ""
    judge = legacy.get("judge_correct")
    if isinstance(judge, bool):
        return judge, ""
    exact = legacy.get("correct_exact")
    if isinstance(exact, bool):
        return exact, ""
    return False, ""


def migrate_eval_file(in_path: Path, out_path: Path, *, completions_path: Path, legacy_completions_path: Path) -> None:
    # Build context map from migrated completions.
    context_map: dict[tuple[int, int], tuple[str, str, str]] = {}
    for payload in _iter_jsonl(completions_path):
        sample_index = payload.get("sample_index")
        repeat_index = payload.get("repeat_index")
        if not isinstance(sample_index, int) or not isinstance(repeat_index, int):
            raise IndexValidationError(
                f"sample_index/repeat_index must be int in completions: got {sample_index!r}, {repeat_index!r}"
            )
        key = (sample_index, repeat_index)
        context_map[key] = (
            str(payload.get("benchmark_name", "")),
            str(payload.get("dataset_split", "")),
            build_context_from_completions(payload),
        )

    legacy_eval = list(_iter_jsonl(in_path))

    # Special mapping for legacy IFEval: (sample_key, sample_id) -> problem_index
    key_map: dict[tuple[int, int], int] = {}
    legacy_comp_payloads = list(_iter_jsonl(legacy_completions_path)) if legacy_completions_path.exists() else []
    for payload in legacy_comp_payloads:
        key = payload.get("key")
        sample_id = payload.get("sample_id")
        problem_index = payload.get("problem_index")
        if isinstance(key, int) and isinstance(sample_id, int) and isinstance(problem_index, int):
            key_map[(key, sample_id)] = problem_index
    repeats = _infer_repeats_from_legacy(legacy_comp_payloads) if legacy_comp_payloads else 1

    stem = strip_artifact_suffix(completions_path.stem)
    fallback_benchmark, fallback_split = dataset_slug_parts(stem)

    def rows():
        for payload in legacy_eval:
            sample_index, repeat_index = _eval_key_from_legacy(payload, repeats=repeats, key_map=key_map or None)
            passed, reason = _infer_pass_fail(payload)
            mapped = context_map.get((sample_index, repeat_index))
            if mapped is None:
                benchmark_name, dataset_split, context = fallback_benchmark, fallback_split, ""
            else:
                benchmark_name, dataset_split, context = mapped
            fail_reason = "" if passed else (reason or "incorrect")
            yield {
                "benchmark_name": benchmark_name,
                "dataset_split": dataset_split,
                "sample_index": int(sample_index),
                "repeat_index": int(repeat_index),
                "context": context,
                "is_passed": bool(passed),
                "fail_reason": fail_reason,
            }

    _write_jsonl_atomic(out_path, rows())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate results JSONL to v3 schema")
    parser.add_argument("--input-root", default="results", help="Root results directory (default: results)")
    parser.add_argument("--output-root", default=None, help="Output results directory (default: in-place)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--max-files", type=int, help="Limit number of files (debug)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else input_root

    in_comp_root = input_root / "completions"
    in_eval_root = input_root / "eval"
    out_comp_root = output_root / "completions"
    out_eval_root = output_root / "eval"

    comp_files = sorted(p for p in in_comp_root.glob("*/*.jsonl") if p.is_file())
    eval_files = sorted(p for p in in_eval_root.glob("*/*_results.jsonl") if p.is_file())
    if args.max_files is not None:
        comp_files = comp_files[: max(0, args.max_files)]
        eval_files = eval_files[: max(0, args.max_files)]

    for in_path in comp_files:
        rel = in_path.relative_to(in_comp_root)
        out_path = out_comp_root / rel
        if out_path.exists() and not args.overwrite and out_path.resolve() != in_path.resolve():
            continue
        if args.dry_run:
            print(f"[dry-run] completions: {in_path} -> {out_path}")
            continue
        migrate_completions_file(in_path, out_path)

    for in_path in eval_files:
        rel = in_path.relative_to(in_eval_root)
        # map xxx_results.jsonl -> xxx.jsonl in completions tree
        comp_name = in_path.name.replace("_results.jsonl", ".jsonl")
        comp_rel = rel.with_name(comp_name)
        out_comp_path = out_comp_root / comp_rel
        legacy_comp_path = in_comp_root / comp_rel

        out_path = out_eval_root / rel
        if out_path.exists() and not args.overwrite and out_path.resolve() != in_path.resolve():
            continue
        if args.dry_run:
            print(f"[dry-run] eval: {in_path} -> {out_path} (completions={out_comp_path})")
            continue
        if not out_comp_path.exists():
            # Cannot build context without migrated completions; skip.
            continue
        migrate_eval_file(in_path, out_path, completions_path=out_comp_path, legacy_completions_path=legacy_comp_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

