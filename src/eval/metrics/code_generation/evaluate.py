from __future__ import annotations

"""Code-generation evaluation adapter for canonical `results/completions` JSONL.

The upstream HumanEval/MBPP evaluators expect sample JSONL lines containing
`task_id` and `completion`. Our canonical completions schema intentionally
does not store dataset labels/ids, so this module bridges by joining:
- canonical completions (sample_index/repeat_index + completion text)
- the original dataset file (to recover task_id)

It then writes canonical evaluator output (results/eval) with:
benchmark_name, dataset_split, sample_index, repeat_index, context,
answer, ref_answer, is_passed, fail_reason
"""

import json
from pathlib import Path
import tempfile
from typing import Iterable

import orjson

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.results.schema import build_context_from_completions
from src.eval.metrics.code_generation.human_eval import evaluate_functional_correctness
from src.eval.metrics.code_generation.mbpp import evaluate_mbpp


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _max_stage_index(payload: dict) -> int:
    stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            stage = max(stage, int(key.removeprefix("completion")))
    return stage


def _write_temp_samples(
    completions_path: str | Path,
    *,
    dataset_records: list,
    temp_path: Path,
) -> int:
    """Write evaluator-compatible samples JSONL into `temp_path`."""

    count = 0
    with temp_path.open("wb") as out_f:
        for payload in _iter_jsonl(completions_path):
            sample_index = int(payload.get("sample_index", 0))
            repeat_index = int(payload.get("repeat_index", 0))
            task_id = ""
            if 0 <= sample_index < len(dataset_records):
                task_id = str(getattr(dataset_records[sample_index], "task_id", ""))
            last_stage = _max_stage_index(payload)
            completion = str(payload.get(f"completion{last_stage}", "") or "")
            context = build_context_from_completions(payload)
            sample = {
                "benchmark_name": payload.get("benchmark_name", ""),
                "dataset_split": payload.get("dataset_split", ""),
                "sample_index": sample_index,
                "repeat_index": repeat_index,
                "context": context,
                "task_id": task_id,
                # Keep evaluator behaviour consistent with prior runs: strip trailing whitespace.
                "completion": completion.rstrip(),
            }
            out_f.write(orjson.dumps(sample, option=orjson.OPT_APPEND_NEWLINE))
            count += 1
    return count


def _resolve_reference_answer(record: CodeGenerationRecord | None) -> str:
    if record is None:
        return ""
    canonical = (record.canonical_solution or "").strip() if record.canonical_solution is not None else ""
    if canonical:
        return canonical
    tests = record.test_cases
    if tests is None:
        return ""
    if isinstance(tests, str):
        return tests
    try:
        return json.dumps(tests, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(tests)


def _write_canonical_eval_from_results(
    results_path: Path,
    *,
    eval_output_path: Path,
    dataset_records: list[CodeGenerationRecord],
) -> None:
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("r", encoding="utf-8") as in_f, eval_output_path.open("wb") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            passed = bool(payload.get("passed", False))
            result = str(payload.get("result", "") or "")
            sample_index = int(payload.get("sample_index", 0))
            record = dataset_records[sample_index] if 0 <= sample_index < len(dataset_records) else None
            out_row = {
                "benchmark_name": str(payload.get("benchmark_name", "")),
                "dataset_split": str(payload.get("dataset_split", "")),
                "sample_index": sample_index,
                "repeat_index": int(payload.get("repeat_index", 0)),
                "context": str(payload.get("context", "")),
                "answer": str(payload.get("completion", "") or ""),
                "ref_answer": _resolve_reference_answer(record),
                "is_passed": passed,
                "fail_reason": "" if passed else result,
            }
            out_f.write(orjson.dumps(out_row, option=orjson.OPT_APPEND_NEWLINE))


def evaluate_human_eval(
    completions_path: str | Path,
    *,
    dataset_path: str | Path,
    eval_output_path: str | Path,
    pass_k: tuple[int, ...] = (1,),
    n_workers: int = 4,
    timeout: float = 3.0,
) -> dict[str, float]:
    """Run HumanEval functional correctness eval and write canonical eval JSONL."""

    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    eval_output_path = Path(eval_output_path)

    with tempfile.TemporaryDirectory(prefix="rwkv_skills_humaneval_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        sample_file = tmp_dir_path / "samples.jsonl"
        _write_temp_samples(
            completions_path,
            dataset_records=dataset_records,
            temp_path=sample_file,
        )
        metrics, results_path_str = evaluate_functional_correctness(
            sample_file=str(sample_file),
            k=tuple(pass_k),
            n_workers=n_workers,
            timeout=timeout,
            problem_file=str(dataset_path),
        )
        _write_canonical_eval_from_results(
            Path(results_path_str),
            eval_output_path=eval_output_path,
            dataset_records=dataset_records,
        )
        return metrics or {}


def evaluate_mbpp_dataset(
    completions_path: str | Path,
    *,
    dataset_path: str | Path,
    eval_output_path: str | Path,
    pass_k: tuple[int, ...] = (1,),
    n_workers: int = 4,
    timeout: float = 3.0,
) -> dict[str, float]:
    """Run MBPP (or MBPP+) eval and write canonical eval JSONL."""

    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    eval_output_path = Path(eval_output_path)

    with tempfile.TemporaryDirectory(prefix="rwkv_skills_mbpp_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        sample_file = tmp_dir_path / "samples.jsonl"
        _write_temp_samples(
            completions_path,
            dataset_records=dataset_records,
            temp_path=sample_file,
        )
        metrics, results_path_str = evaluate_mbpp(
            sample_file=str(sample_file),
            k=tuple(pass_k),
            n_workers=n_workers,
            timeout=timeout,
            problem_file=str(dataset_path),
        )
        _write_canonical_eval_from_results(
            Path(results_path_str),
            eval_output_path=eval_output_path,
            dataset_records=dataset_records,
        )
        return metrics or {}


__all__ = ["evaluate_human_eval", "evaluate_mbpp_dataset"]
