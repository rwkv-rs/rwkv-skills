from __future__ import annotations

"""Run LLM wrong-answer checker over existing `results/eval/*_results.jsonl`.

Typical usage:
  python -m src.bin.run_llm_checker                # scan default results/eval
  python -m src.bin.run_llm_checker results/eval/<model_name>
  python -m src.bin.run_llm_checker results/eval/<model_name>/<bench>_results.jsonl
"""

import argparse
from pathlib import Path
from typing import Sequence

from src.eval.checkers.llm_checker import run_llm_checker
from src.eval.scheduler.config import DEFAULT_EVAL_RESULT_DIR


_RESULTS_SUFFIX = "_results.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llm_checker over legacy/new eval JSONL files")
    parser.add_argument(
        "targets",
        nargs="*",
        default=[str(DEFAULT_EVAL_RESULT_DIR)],
        help="Eval JSONL file or directory (will scan for *_results.jsonl). Default: results/eval",
    )
    parser.add_argument(
        "--eval-root",
        default=str(DEFAULT_EVAL_RESULT_DIR),
        help="Root dir used to infer model_name from path (default: results/eval).",
    )
    parser.add_argument(
        "--model-name",
        help="Force model_name (useful when passing eval files outside --eval-root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched eval files; do not call the checker.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of eval files processed (after sorting).",
    )
    return parser.parse_args(argv)


def _iter_eval_files(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in targets:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob(f"*{_RESULTS_SUFFIX}")))
        else:
            paths.append(path)
    return paths


def _infer_model_name(eval_path: Path, *, eval_root: Path, forced: str | None) -> str:
    if forced:
        return forced
    eval_path = eval_path.resolve()
    eval_root = eval_root.resolve()
    if eval_path.is_relative_to(eval_root):
        rel = eval_path.relative_to(eval_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    return eval_path.parent.name


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    eval_root = Path(args.eval_root).expanduser()
    eval_files = [p for p in _iter_eval_files(list(args.targets)) if p.name.endswith(_RESULTS_SUFFIX)]
    eval_files = sorted({p.expanduser() for p in eval_files})
    if args.limit is not None:
        eval_files = eval_files[: max(0, int(args.limit))]

    if not eval_files:
        print("⚠️  no eval files found")
        return 1

    if args.dry_run:
        for path in eval_files:
            model_name = _infer_model_name(path, eval_root=eval_root, forced=args.model_name)
            print(f"{path}  (model_name={model_name})")
        print(f"🧪 dry-run: {len(eval_files)} eval files")
        return 0

    for path in eval_files:
        model_name = _infer_model_name(path, eval_root=eval_root, forced=args.model_name)
        run_llm_checker(path, model_name=model_name)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
