from __future__ import annotations

"""SWE-bench prompt, prediction, and official harness helpers."""

import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_long_text
from src.eval.results.schema import build_context_from_completions, strict_nonneg_int

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class SweBenchHarnessResult:
    metrics: dict[str, float]
    instance_results: dict[str, dict[str, Any]]
    results_path: Path | None = None
    instance_results_path: Path | None = None


def build_swebench_prompt(
    record: CodeGenerationRecord,
    *,
    max_context_chars: int | None = None,
    long_doc_config: LongDocEvidenceConfig | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
    prompt_profile: str = "normal",
) -> str:
    prompt, _trace = build_swebench_prompt_with_trace(
        record,
        max_context_chars=max_context_chars,
        long_doc_config=long_doc_config,
        engine=engine,
        sampling=sampling,
        prompt_seed=prompt_seed,
        prompt_profile=prompt_profile,
    )
    return prompt


def build_swebench_prompt_with_trace(
    record: CodeGenerationRecord,
    *,
    max_context_chars: int | None = None,
    long_doc_config: LongDocEvidenceConfig | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
    prompt_profile: str = "normal",
) -> tuple[str, dict[str, Any]]:
    metadata = dict(record.metadata)
    if str(prompt_profile or "normal").strip().lower() == "naive":
        clean_prompt = str(record.prompt or "").strip()
        trace = _empty_long_doc_trace("", None)
        prompt = f"User: {clean_prompt}\n\nAssistant: <think>\n</think>\n```diff\n"
        trace["prompt_profile"] = "naive"
        trace["prompt_chars"] = len(prompt)
        return prompt, trace

    repo = str(metadata.get("repo") or "")
    base_commit = str(metadata.get("base_commit") or "")
    instance_id = str(metadata.get("instance_id") or record.task_id)
    hints_text = str(metadata.get("hints_text") or "").strip()
    raw_retrieved_context = str(metadata.get("retrieved_context") or "").strip()
    retrieved_context = raw_retrieved_context
    trace = _empty_long_doc_trace(raw_retrieved_context, long_doc_config)
    if retrieved_context and long_doc_config is not None:
        compaction = compact_long_text(
            retrieved_context,
            query=_swebench_context_query(record.prompt, hints_text),
            config=long_doc_config,
            label=f"swebench:{instance_id}",
            engine=engine,
            sampling=sampling,
            progress_desc="SWE-bench-LongDoc",
            prompt_seed=prompt_seed,
        )
        retrieved_context = compaction.text
        trace = {
            "mode": long_doc_config.mode if long_doc_config.enabled else "off",
            "enabled": bool(long_doc_config.enabled),
            "original_context_chars": int(compaction.original_chars),
            "rendered_context_chars": len(retrieved_context),
            "trimmed_context_chars": 0,
            "compacted": bool(compaction.compacted),
            "chunk_count": int(compaction.chunk_count),
            "selected_chunk_ids": list(compaction.selected_chunk_ids),
        }
    if max_context_chars is not None and max_context_chars > 0 and len(retrieved_context) > max_context_chars:
        before_trim = len(retrieved_context)
        retrieved_context = retrieved_context[:max_context_chars].rstrip()
        trace["rendered_context_chars"] = len(retrieved_context)
        trace["trimmed_context_chars"] = int(trace.get("trimmed_context_chars", 0)) + before_trim - len(retrieved_context)

    lines = [
        "User: You are resolving a real GitHub issue from SWE-bench.",
        "Return only a unified git diff patch. Do not include prose, commands, or markdown outside the patch.",
        "The patch must be applicable with git apply from the repository root.",
        "",
        f"Instance: {instance_id}",
    ]
    if repo:
        lines.append(f"Repository: {repo}")
    if base_commit:
        lines.append(f"Base commit: {base_commit}")
    lines.extend(["", "Issue:", record.prompt.strip()])
    if hints_text:
        lines.extend(["", "Hints:", hints_text])
    if retrieved_context:
        lines.extend(["", "Retrieved repository context:", retrieved_context])
    lines.extend(["", "Assistant: <think>\n</think>\n```diff\n"])
    trace["prompt_chars"] = len("\n".join(lines))
    return "\n".join(lines), trace


def extract_swebench_patch(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    fenced = _extract_last_fenced_block(cleaned)
    if fenced is not None:
        cleaned = fenced.strip()
    for marker in ("diff --git ", "--- a/"):
        index = cleaned.find(marker)
        if index >= 0:
            return cleaned[index:].strip()
    return ""


def _empty_long_doc_trace(
    retrieved_context: str,
    config: LongDocEvidenceConfig | None,
) -> dict[str, Any]:
    return {
        "mode": str(config.mode) if config is not None and config.enabled else "off",
        "enabled": bool(config is not None and config.enabled),
        "original_context_chars": len(str(retrieved_context or "")),
        "rendered_context_chars": len(str(retrieved_context or "")),
        "trimmed_context_chars": 0,
        "compacted": False,
        "chunk_count": 0,
        "selected_chunk_ids": [],
    }


def _swebench_context_query(problem_statement: str, hints_text: str) -> str:
    parts = [str(problem_statement or "").strip()]
    if hints_text:
        parts.append(str(hints_text).strip())
    return "\n\n".join(part for part in parts if part)


def make_swebench_prediction(
    completion_payload: Mapping[str, Any],
    *,
    record: CodeGenerationRecord,
    model_name: str,
) -> dict[str, str]:
    completion = str(_last_completion(completion_payload) or "")
    return {
        "instance_id": str(record.metadata.get("instance_id") or record.task_id),
        "model_name_or_path": model_name,
        "model_patch": extract_swebench_patch(completion),
    }


def write_swebench_predictions(
    completions: Iterable[Mapping[str, Any]],
    *,
    dataset_path: str | Path,
    model_name: str,
    output_path: str | Path,
) -> Path:
    records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for payload in completions:
            sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
            if sample_index >= len(records):
                continue
            prediction = make_swebench_prediction(payload, record=records[sample_index], model_name=model_name)
            fh.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    return path


def evaluate_swebench_predictions(
    completions: Iterable[Mapping[str, Any]],
    *,
    dataset_path: str | Path,
    model_name: str,
    predictions_path: str | Path,
    run_harness: bool,
    dataset_name: str | None = None,
    split: str = "test",
    run_id: str | None = None,
    max_workers: int = 1,
    cache_level: str | None = None,
    clean: bool = False,
    timeout_s: float | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], Path]:
    completion_rows = list(completions)
    predictions = write_swebench_predictions(
        completion_rows,
        dataset_path=dataset_path,
        model_name=model_name,
        output_path=predictions_path,
    )
    if not run_harness:
        return {"swebench_predictions": float(len(completion_rows)), "swebench_harness_ran": 0.0}, [], predictions

    resolved_dataset_name = dataset_name or infer_harness_dataset_name(dataset_path)
    harness = run_swebench_harness(
        predictions_path=predictions,
        dataset_name=resolved_dataset_name,
        split=split,
        run_id=run_id,
        max_workers=max_workers,
        cache_level=cache_level,
        clean=clean,
        timeout_s=timeout_s,
    )
    eval_payloads = build_swebench_eval_payloads(
        completion_rows,
        dataset_path=dataset_path,
        harness_result=harness,
    )
    metrics = dict(harness.metrics)
    metrics["swebench_harness_ran"] = 1.0
    metrics["swebench_predictions"] = float(len(completion_rows))
    return metrics, eval_payloads, predictions


def run_swebench_harness(
    *,
    predictions_path: str | Path,
    dataset_name: str,
    split: str = "test",
    run_id: str | None = None,
    max_workers: int = 1,
    cache_level: str | None = None,
    clean: bool = False,
    timeout_s: float | None = None,
) -> SweBenchHarnessResult:
    if importlib.util.find_spec("swebench") is None:
        raise ModuleNotFoundError("Install the official SWE-bench harness first: pip install swebench")
    resolved_run_id = run_id or f"rwkv-skills-{Path(predictions_path).stem}"
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max(1, int(max_workers))),
        "--run_id",
        resolved_run_id,
    ]
    if cache_level:
        cmd.extend(["--cache_level", cache_level])
    if clean:
        cmd.extend(["--clean", "True"])
    subprocess.run(cmd, check=True, timeout=timeout_s)
    return load_swebench_harness_result(run_id=resolved_run_id)


def load_swebench_harness_result(*, run_id: str, root: str | Path = "evaluation_results") -> SweBenchHarnessResult:
    root_path = Path(root)
    result_path = _latest_matching_path(root_path, run_id, "results.json")
    instance_path = _latest_matching_path(root_path, run_id, "instance_results.jsonl")
    result_payload: Mapping[str, Any] = {}
    if result_path is not None:
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    instance_results = _load_instance_results(instance_path)
    return SweBenchHarnessResult(
        metrics=_normalize_harness_metrics(result_payload, instance_results),
        instance_results=instance_results,
        results_path=result_path,
        instance_results_path=instance_path,
    )


def build_swebench_eval_payloads(
    completions: Sequence[Mapping[str, Any]],
    *,
    dataset_path: str | Path,
    harness_result: SweBenchHarnessResult,
) -> list[dict[str, Any]]:
    records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    payloads: list[dict[str, Any]] = []
    for completion in completions:
        sample_index = strict_nonneg_int(completion.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(completion.get("repeat_index"), "repeat_index")
        pass_index = strict_nonneg_int(completion.get("pass_index", 0), "pass_index")
        record = records[sample_index] if sample_index < len(records) else None
        instance_id = str(record.metadata.get("instance_id") or record.task_id) if record is not None else ""
        instance_result = harness_result.instance_results.get(instance_id, {})
        resolved = _is_instance_resolved(instance_result)
        answer = extract_swebench_patch(str(_last_completion(completion) or ""))
        fail_reason = "" if resolved else _instance_fail_reason(instance_result)
        payloads.append(
            {
                "benchmark_name": str(completion.get("benchmark_name", "")),
                "dataset_split": str(completion.get("dataset_split", "")),
                "sample_index": sample_index,
                "repeat_index": repeat_index,
                "pass_index": pass_index,
                "context": build_context_from_completions(dict(completion)),
                "answer": answer,
                "ref_answer": str(record.metadata.get("patch") or "") if record is not None else "",
                "is_passed": resolved,
                "fail_reason": fail_reason,
                "instance_id": instance_id,
            }
        )
    return payloads


def infer_harness_dataset_name(dataset_path: str | Path) -> str:
    records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    if records:
        value = records[0].metadata.get("harness_dataset_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    stem = Path(dataset_path).parent.name.lower()
    if "verified" in stem:
        return "princeton-nlp/SWE-bench_Verified"
    if "lite" in stem:
        return "princeton-nlp/SWE-bench_Lite"
    return "princeton-nlp/SWE-bench"


def _extract_last_fenced_block(text: str) -> str | None:
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _last_completion(payload: Mapping[str, Any]) -> str:
    best_stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            best_stage = max(best_stage, int(key.removeprefix("completion")))
    return str(payload.get(f"completion{best_stage}", "") or "")


def _latest_matching_path(root: Path, run_id: str, name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.rglob(name) if run_id in str(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_instance_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            if instance_id:
                results[instance_id] = row
    return results


def _normalize_harness_metrics(
    result_payload: Mapping[str, Any],
    instance_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    submitted = _numeric_metric(result_payload, ("submitted", "instances_submitted", "total_instances"))
    completed = _numeric_metric(result_payload, ("completed", "instances_completed"))
    resolved = _numeric_metric(result_payload, ("resolved", "instances_resolved"))
    if instance_results:
        submitted = submitted or float(len(instance_results))
        resolved = resolved or float(sum(1 for row in instance_results.values() if _is_instance_resolved(row)))
        completed = completed or float(sum(1 for row in instance_results.values() if not _instance_failed_to_run(row)))
    rate = _numeric_metric(result_payload, ("resolved_rate", "resolution_rate"))
    if rate == 0.0 and submitted:
        rate = resolved / submitted
    return {
        "swebench_instances_submitted": float(submitted),
        "swebench_instances_completed": float(completed),
        "swebench_instances_resolved": float(resolved),
        "swebench_resolution_rate": float(rate),
    }


def _numeric_metric(payload: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return float(len(value))
    return 0.0


def _is_instance_resolved(row: Mapping[str, Any]) -> bool:
    for key in ("resolved", "is_resolved", "passed", "is_passed"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "resolved"}:
            return True
    return False


def _instance_failed_to_run(row: Mapping[str, Any]) -> bool:
    for key in ("completed", "ran", "success"):
        value = row.get(key)
        if isinstance(value, bool):
            return not value
    return False


def _instance_fail_reason(row: Mapping[str, Any]) -> str:
    for key in ("fail_reason", "error", "status", "result"):
        value = row.get(key)
        if value:
            return str(value)
    return "unresolved"


__all__ = [
    "SweBenchHarnessResult",
    "build_swebench_eval_payloads",
    "build_swebench_prompt",
    "evaluate_swebench_predictions",
    "extract_swebench_patch",
    "infer_harness_dataset_name",
    "load_swebench_harness_result",
    "make_swebench_prediction",
    "run_swebench_harness",
    "write_swebench_predictions",
]
