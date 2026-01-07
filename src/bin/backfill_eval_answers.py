from __future__ import annotations

"""Backfill `answer` + `ref_answer` into legacy `results/eval/*_results.jsonl`.

This script upgrades existing eval JSONL files that were written before the
`answer`/`ref_answer` fields were added to the eval schema.

How it works (per eval file):
- infer the matching completions JSONL path (results/completions)
- infer the dataset slug from the eval filename and locate the dataset JSONL
- extract `answer` from the completions payload (task-specific)
- extract `ref_answer` from the dataset record (task-specific)
- rewrite eval JSONL in-place (atomic .tmp replace)
"""

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable

import orjson

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_loader.instruction_following import JsonlInstructionFollowingLoader
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.metrics.free_response import resolve_reference_answer as resolve_free_response_reference_answer
from src.eval.metrics.instruction_following import instructions_registry
from src.eval.results.schema import strip_artifact_suffix
from src.eval.scheduler.config import DEFAULT_COMPLETION_DIR, DEFAULT_EVAL_RESULT_DIR
from src.eval.scheduler.datasets import find_dataset_file
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import detect_job_from_dataset


_LETTER_RE = re.compile(r"[A-Z]")
_RESULTS_SUFFIX = "_results.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill answer/ref_answer into results/eval JSONL")
    parser.add_argument(
        "targets",
        nargs="+",
        help="Eval JSONL file or directory (will scan for *_results.jsonl)",
    )
    parser.add_argument(
        "--prepare-missing-dataset",
        action="store_true",
        help="If dataset JSONL is missing locally, attempt to auto-prepare it (may download data).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing answer/ref_answer fields if already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes; only print what would be updated.",
    )
    return parser.parse_args(argv)


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


def _iter_eval_files(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in targets:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob(f"*{_RESULTS_SUFFIX}")))
        else:
            paths.append(path)
    return paths


def _infer_eval_stem(eval_path: Path) -> str:
    name = eval_path.name
    if name.endswith(_RESULTS_SUFFIX):
        return name[: -len(_RESULTS_SUFFIX)]
    if name.endswith(".jsonl"):
        return eval_path.stem
    raise ValueError(f"Unknown eval filename: {eval_path}")


def _infer_is_cot(eval_stem: str) -> bool:
    return canonical_slug(eval_stem).endswith("__cot")


def _infer_dataset_slug(eval_stem: str) -> str:
    return canonical_slug(strip_artifact_suffix(eval_stem))


def _infer_completions_path(eval_path: Path, eval_stem: str) -> Path:
    """Best-effort mapping: results/eval/.../<stem>_results.jsonl -> results/completions/.../<stem>.jsonl"""
    eval_root = DEFAULT_EVAL_RESULT_DIR
    completion_root = DEFAULT_COMPLETION_DIR
    completions_name = f"{eval_stem}.jsonl"

    if eval_path.is_relative_to(eval_root):
        rel = eval_path.relative_to(eval_root)
        return (completion_root / rel.parent / completions_name).resolve()

    # Fallback: replace a path segment named "eval" with "completions".
    parts = list(eval_path.parts)
    idx: int | None = None
    for i, part in enumerate(parts):
        if part == "eval":
            idx = i
            break
    if idx is not None:
        parts[idx] = "completions"
        candidate = Path(*parts)
        return candidate.with_name(completions_name).resolve()
    return (completion_root / eval_path.parent.name / completions_name).resolve()


def _max_stage_index(payload: dict[str, Any]) -> int:
    stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            stage = max(stage, int(key.removeprefix("completion")))
    return stage


def _extract_choice_letter(text: str) -> str | None:
    match = _LETTER_RE.search(text or "")
    return match.group(0) if match else None


def _instruction_response_from_completion(payload: dict[str, Any]) -> str:
    last_stage = _max_stage_index(payload)
    completion = str(payload.get(f"completion{last_stage}", "") or "")
    prompt1 = str(payload.get("prompt1", "") or "")
    if "<think" in prompt1:
        return completion.split("</think>")[-1].strip()
    return completion.strip()


def _resolve_code_ref_answer(record) -> str:
    canonical = (getattr(record, "canonical_solution", None) or "").strip()
    if canonical:
        return canonical
    tests = getattr(record, "test_cases", None)
    if tests is None:
        return ""
    if isinstance(tests, str):
        return tests
    try:
        return json.dumps(tests, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tests)


def _build_instruction_ref_answer(record) -> str:
    """Render a stable reference string from IFEval-style instruction ids/kwargs."""
    registry = instructions_registry.INSTRUCTION_DICT
    parts: list[str] = []
    for idx, instruction_id in enumerate(record.instruction_ids):
        instruction_cls = registry[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = record.kwargs_list[idx]
        description = instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            description = instruction.build_description(prompt=record.prompt)
        if description:
            parts.append(f"{instruction_id}: {description}")
        else:
            parts.append(str(instruction_id))
    return "\n".join(parts)


def _extract_answers_from_completions(job_name: str, completions_path: Path) -> dict[tuple[int, int], str]:
    """Return {(sample_index, repeat_index): answer}."""
    answers: dict[tuple[int, int], str] = {}
    for payload in _iter_jsonl(completions_path):
        sample_index = int(payload.get("sample_index", 0))
        repeat_index = int(payload.get("repeat_index", 0))
        last_stage = _max_stage_index(payload)
        completion = str(payload.get(f"completion{last_stage}", "") or "")

        if job_name.startswith("multi_choice"):
            answer = _extract_choice_letter(completion) or ""
        elif job_name.startswith("free_response"):
            answer = completion.strip()
        elif job_name == "instruction_following":
            answer = _instruction_response_from_completion(payload)
        elif job_name.startswith("code_"):
            answer = completion.rstrip()
        else:
            answer = completion.strip()

        answers.setdefault((sample_index, repeat_index), answer)
    return answers


def _resolve_ref_answers(job_name: str, dataset_path: Path) -> list[str]:
    if job_name.startswith("multi_choice"):
        dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [alphabet[record.answer_index] for record in dataset]
    if job_name.startswith("free_response"):
        dataset = list(JsonlFreeAnswerLoader(str(dataset_path)))
        return [resolve_free_response_reference_answer(record) for record in dataset]
    if job_name == "instruction_following":
        dataset = list(JsonlInstructionFollowingLoader(str(dataset_path)).load())
        return [_build_instruction_ref_answer(record) for record in dataset]
    if job_name.startswith("code_"):
        dataset = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
        return [_resolve_code_ref_answer(record) for record in dataset]
    return []


def _resolve_dataset_path(dataset_slug: str, *, prepare_missing: bool) -> Path:
    found = find_dataset_file(dataset_slug)
    if found:
        return found
    if prepare_missing:
        return resolve_or_prepare_dataset(dataset_slug)
    raise FileNotFoundError(
        f"Dataset JSONL not found for slug={dataset_slug!r}. "
        "Place it under ./data or re-run with --prepare-missing-dataset."
    )


def backfill_eval_file(eval_path: Path, *, prepare_missing: bool, overwrite: bool, dry_run: bool) -> None:
    eval_path = eval_path.expanduser().resolve()
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    eval_stem = _infer_eval_stem(eval_path)
    dataset_slug = _infer_dataset_slug(eval_stem)
    is_cot = _infer_is_cot(eval_stem)
    job_name = detect_job_from_dataset(dataset_slug, is_cot=is_cot)
    if not job_name:
        raise ValueError(f"Cannot infer job type from dataset={dataset_slug!r} (is_cot={is_cot})")

    dataset_path = _resolve_dataset_path(dataset_slug, prepare_missing=prepare_missing)
    completions_path = _infer_completions_path(eval_path, eval_stem)
    if not completions_path.exists():
        raise FileNotFoundError(f"Matching completions JSONL not found: {completions_path}")

    answer_map = _extract_answers_from_completions(job_name, completions_path)
    ref_list = _resolve_ref_answers(job_name, dataset_path)

    total = 0
    updated = 0
    missing_answer = 0
    missing_ref = 0

    def rows():
        nonlocal total, updated, missing_answer, missing_ref
        for row in _iter_jsonl(eval_path):
            total += 1
            has_fields = "answer" in row and "ref_answer" in row
            if has_fields and not overwrite:
                yield row
                continue

            sample_index = int(row.get("sample_index", 0))
            repeat_index = int(row.get("repeat_index", 0))
            answer = answer_map.get((sample_index, repeat_index), "")
            if not answer:
                missing_answer += 1
            ref_answer = ref_list[sample_index] if 0 <= sample_index < len(ref_list) else ""
            if not ref_answer:
                missing_ref += 1

            row = dict(row)
            row["answer"] = answer
            row["ref_answer"] = ref_answer
            updated += 1
            yield row

    if dry_run:
        # Materialize counts without writing.
        for _ in rows():
            pass
        print(
            f"🧪 dry-run: {eval_path} -> would update {updated}/{total} rows "
            f"(missing answer: {missing_answer}, missing ref: {missing_ref})"
        )
        return

    _write_jsonl_atomic(eval_path, rows())
    print(
        f"✅ updated: {eval_path} ({updated}/{total} rows) "
        f"(missing answer: {missing_answer}, missing ref: {missing_ref})"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_files = _iter_eval_files(args.targets)
    if not eval_files:
        print("⚠️  no eval files found")
        return 1
    failures = 0
    for path in eval_files:
        try:
            backfill_eval_file(
                path,
                prepare_missing=bool(args.prepare_missing_dataset),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
        except (FileNotFoundError, ValueError) as exc:
            failures += 1
            print(f"❌ failed: {path} -> {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
