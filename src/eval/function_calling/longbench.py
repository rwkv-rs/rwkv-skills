from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import normalize_rwkv_text
from src.eval.function_calling.final_answer import (
    build_final_answer_json_call_prompt,
    parse_final_answer_call,
    render_final_answer_call,
)
from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_long_text
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext


LONG_BENCH_DATASETS = frozenset(
    {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "gov_report",
        "qmsum",
        "multi_news",
        "vcsum",
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "passage_count",
        "passage_retrieval_en",
        "passage_retrieval_zh",
        "lcc",
        "repobench-p",
    }
)

LONG_BENCH_QA_DATASETS = frozenset(
    {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "triviaqa",
    }
)

_LONG_BENCH_ZH_DATASETS = frozenset({"multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"})
_WORD_OR_CJK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+answer|answer|答案|最终答案)\s*[:：]\s*", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_LONGBENCH_FINAL_ANSWER_DESCRIPTION = "The concise answer to the LongBench question."
_LONGBENCH_PROMPT_HISTORY_SLACK = 4096


@dataclass(frozen=True, slots=True)
class LongBenchRecord:
    task_id: str
    dataset: str
    input: str
    context: str
    answers: tuple[str, ...]
    all_classes: tuple[str, ...] = ()
    language: str = "en"
    length: int = 0
    category: str = ""
    source_path: str = ""


@dataclass(frozen=True, slots=True)
class LongBenchScore:
    exact_match: bool
    f1: float
    best_reference: str

    @property
    def passed(self) -> bool:
        return self.exact_match or self.f1 >= 0.8


def load_longbench_manifest_records(path: str | Path) -> list[LongBenchRecord]:
    records: list[LongBenchRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(_longbench_record_from_payload(payload, fallback_index=index, source_path=str(target)))
    return records


def load_longbench_rows_from_source(
    source: str | Path,
    *,
    split: str = "test",
    include_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    files = _longbench_source_files(source, split=split)
    if not files:
        raise FileNotFoundError(f"missing LongBench JSONL files under {Path(source).expanduser()}")

    rows: list[dict[str, Any]] = []
    include = {item.lower() for item in include_datasets} if include_datasets is not None else None
    for file_path in files:
        dataset = _infer_longbench_dataset_name(file_path)
        if include is not None and dataset.lower() not in include:
            continue
        with file_path.open("r", encoding="utf-8") as fh:
            for line_index, line in enumerate(fh):
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                payload.setdefault("dataset", dataset)
                record = _longbench_record_from_payload(payload, fallback_index=line_index, source_path=str(file_path))
                rows.append(_longbench_record_to_row(record))
    if not rows:
        filter_note = "" if include is None else f" after filtering datasets={sorted(include)}"
        raise FileNotFoundError(f"no LongBench rows found{filter_note} under {Path(source).expanduser()}")
    return rows


def normalize_longbench_manifest_row(
    payload: Mapping[str, Any],
    *,
    fallback_index: int,
    source_path: str,
) -> dict[str, Any]:
    return _longbench_record_to_row(
        _longbench_record_from_payload(payload, fallback_index=fallback_index, source_path=source_path)
    )


def build_longbench_prompt(
    record: LongBenchRecord,
    *,
    context_text: str | None = None,
) -> str:
    context = normalize_rwkv_text(context_text if context_text is not None else record.context)
    question = normalize_rwkv_text(record.input)
    lines = [
        "You are evaluating a long-context reading task.",
        "Answer using only the provided context.",
        "Return a concise final answer. Do not include analysis.",
    ]
    if record.all_classes:
        lines.append("If labels/classes are provided, answer with exactly one allowed label.")
        lines.append("Allowed labels/classes: " + ", ".join(record.all_classes))
    if record.language.lower().startswith("zh"):
        lines.append("If the question is Chinese, answer in Chinese.")
    lines.extend(
        [
            "",
            "Context:",
            context,
            "",
            "Question:",
            question,
        ]
    )
    instruction = normalize_rwkv_text("\n".join(lines))
    return build_final_answer_json_call_prompt(
        instruction,
        answer_description=_LONGBENCH_FINAL_ANSWER_DESCRIPTION,
        history_max_chars=len(instruction) + _LONGBENCH_PROMPT_HISTORY_SLACK,
        extra_system_lines=(
            "Put only the concise benchmark answer in arguments.answer.",
            "Do not include analysis or citations unless the dataset question asks for them.",
        ),
    )


def build_longbench_budgeted_prompt(
    record: LongBenchRecord,
    *,
    long_doc_config: LongDocEvidenceConfig,
    prompt_max_chars: int,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    query = normalize_rwkv_text(record.input)
    context = normalize_rwkv_text(record.context)
    compaction = compact_long_text(
        context,
        query=query,
        config=long_doc_config,
        label=f"longbench:{record.dataset}",
        engine=engine,
        sampling=sampling,
        progress_desc="LongBench-LongDoc",
        prompt_seed=prompt_seed,
    )
    prompt = build_longbench_prompt(record, context_text=compaction.text)
    trimmed_context_chars = 0
    if len(prompt) > prompt_max_chars:
        overhead = len(build_longbench_prompt(record, context_text=""))
        context_budget = max(256, int(prompt_max_chars) - overhead - 16)
        fitted_context = _middle_truncate_text(compaction.text, context_budget)
        trimmed_context_chars = max(0, len(compaction.text) - len(fitted_context))
        prompt = build_longbench_prompt(record, context_text=fitted_context)
        if len(prompt) > prompt_max_chars:
            extra = len(prompt) - int(prompt_max_chars)
            fitted_context = _middle_truncate_text(fitted_context, max(0, len(fitted_context) - extra - 32))
            trimmed_context_chars = max(0, len(compaction.text) - len(fitted_context))
            prompt = build_longbench_prompt(record, context_text=fitted_context)
    trace = {
        "mode": long_doc_config.mode if long_doc_config.enabled else "off",
        "enabled": bool(long_doc_config.enabled),
        "original_context_chars": len(context),
        "rendered_context_chars": len(compaction.text) - trimmed_context_chars,
        "trimmed_context_chars": trimmed_context_chars,
        "compacted": bool(compaction.compacted),
        "chunk_count": int(compaction.chunk_count),
        "selected_chunk_ids": list(compaction.selected_chunk_ids),
        "prompt_chars": len(prompt),
        "output_format": "rwkv_final_answer_json_call",
    }
    if compaction.router_error:
        trace["router_error"] = compaction.router_error
    return prompt, trace


def normalize_longbench_answer(text: str) -> str:
    body = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    if not body:
        return ""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_PREFIX_RE.match(line)
        if match:
            return line[match.end() :].strip().strip("`")
    return _ANSWER_PREFIX_RE.sub("", lines[-1]).strip().strip("`") if lines else ""


def score_longbench_answer(prediction: str, references: Sequence[str]) -> LongBenchScore:
    normalized_prediction = normalize_longbench_answer(prediction)
    best = LongBenchScore(exact_match=False, f1=0.0, best_reference="")
    for reference in references:
        ref = str(reference or "").strip()
        if not ref:
            continue
        exact = _normalized_exact_match(normalized_prediction, ref)
        f1 = _token_f1(normalized_prediction, ref)
        if exact or f1 > best.f1:
            best = LongBenchScore(exact_match=exact, f1=f1, best_reference=ref)
        if exact:
            break
    return best


def compute_longbench_completion_metrics(completions_payloads: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not completions_payloads:
        return {}
    f1_values: list[float] = []
    exact_count = 0
    compaction_count = 0
    prompt_chars: list[float] = []
    for payload in completions_payloads:
        info = payload.get("agent_info")
        if not isinstance(info, Mapping):
            continue
        try:
            f1_values.append(float(info.get("f1", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        if bool(info.get("exact_match", False)):
            exact_count += 1
        trace = info.get("long_doc")
        if isinstance(trace, Mapping):
            if bool(trace.get("compacted", False)):
                compaction_count += 1
            try:
                prompt_chars.append(float(trace.get("prompt_chars", 0) or 0))
            except (TypeError, ValueError):
                pass
    metrics: dict[str, float] = {}
    total = len(completions_payloads)
    if f1_values:
        metrics["longbench_avg_f1"] = sum(f1_values) / len(f1_values)
    metrics["longbench_exact_match_rate"] = exact_count / total
    metrics["longbench_compaction_rate"] = compaction_count / total
    if prompt_chars:
        metrics["longbench_avg_prompt_chars"] = sum(prompt_chars) / len(prompt_chars)
    return metrics


def _longbench_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    f1 = float(agent_info.get("f1", 0.0) or 0.0)
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason="" if passed else f"f1={f1:.4f}",
        answer=str(agent_info.get("prediction") or ""),
        ref_answer=str(agent_info.get("best_reference") or ""),
    )


def _run_longbench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_longbench_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("LongBench manifest is empty")

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="final",
        fallback_templates="free_response_cot_default",
    )
    if sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    sampling = sampling.clamp(args.answer_max_tokens)
    batch_size = max(1, int(args.batch_size or 8))
    prompt_max_chars = int(args.prompt_max_chars or 8192)
    long_doc_config = _longbench_long_doc_config(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        prompt_max_chars=prompt_max_chars,
    )
    selected_entries = [(int(sample_index), records[int(sample_index)]) for sample_index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [
            build_longbench_budgeted_prompt(
                record,
                long_doc_config=long_doc_config,
                prompt_max_chars=prompt_max_chars,
                engine=run.engine,
                sampling=sampling,
            )[0]
            for _, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=sampling,
            batch_size=len(prompts),
            progress_desc="LongBench-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_longbench", run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_longbench_completion_to_eval_payload,
        runner_name="longbench",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_flush_partial_eval,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for start in range(0, len(pending), batch_size):
                    chunk = pending[start : start + batch_size]
                    prompt_rows = [
                        build_longbench_budgeted_prompt(
                            record,
                            long_doc_config=long_doc_config,
                            prompt_max_chars=prompt_max_chars,
                            engine=run.engine,
                            sampling=sampling,
                            prompt_seed=sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=0,
                            ),
                        )
                        for key, record in chunk
                    ]
                    prompts = [prompt for prompt, _trace in prompt_rows]
                    outputs = run.engine.generate(
                        prompts,
                        sampling=sampling,
                        batch_size=len(prompts),
                        progress_desc="LongBench",
                        prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=1,
                            )
                            for key, _record in chunk
                        ],
                    )
                    outputs_by_index = {int(output.prompt_index): output for output in outputs}
                    for index, (key, record) in enumerate(chunk):
                        output = outputs_by_index[index]
                        prompt, trace = prompt_rows[index]
                        parse_error = ""
                        parsed_call: dict[str, Any] = {}
                        parsed_call_id = ""
                        parsed_answer = ""
                        try:
                            final_call = parse_final_answer_call(
                                output.text,
                                context_label="longbench final answer",
                            )
                            parsed_answer = final_call.answer
                            parsed_call = dict(final_call.call)
                            parsed_call_id = final_call.call_id
                        except Exception as exc:  # noqa: BLE001
                            parse_error = str(exc)
                        prediction = normalize_longbench_answer(parsed_answer)
                        score = score_longbench_answer(prediction, record.answers)
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            stages=[
                                StageRecord(
                                    prompt=prompt,
                                    completion=output.text,
                                    stop_reason=output.finish_reason,
                                )
                            ],
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": score.f1,
                            "num_turns": 1,
                            "cost": 0.0,
                            "is_passed": score.passed,
                            "error": parse_error or None,
                        }
                        sandbox_return = (
                            render_final_answer_call(prediction, call_id=parsed_call_id) if prediction else ""
                        )
                        payload["agent_info"] = {
                            "dataset": record.dataset,
                            "task_id": record.task_id,
                            "question": record.input,
                            "answers": list(record.answers),
                            "prediction": prediction,
                            "best_reference": score.best_reference,
                            "f1": score.f1,
                            "exact_match": score.exact_match,
                            "final_answer_call": sandbox_return,
                            "decoded_final_answer_call": parsed_call,
                            "parse_error": parse_error,
                            "category": record.category,
                            "language": record.language,
                            "length": record.length,
                            "long_doc": trace,
                        }
                        payload["agent_trace"] = [
                            {
                                "stage": "answer",
                                "text": prediction,
                                "raw_completion": output.text,
                                "sandbox_return": sandbox_return,
                                "parse_error": parse_error,
                            }
                        ]
                        payload["task_id"] = record.task_id
                        payload["domain"] = "long_context"
                        payload["instruction"] = record.input
                        writer.enqueue(payload)
            except Exception:  # noqa: BLE001
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_longbench_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: _longbench_score_payload(
                completions_payloads,
                metrics,
                run=run,
                model_name=run.model_name,
                plan=plan,
                job_name=job_name,
                sampling_payload=sampling_payload,
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"longbench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


def _longbench_score_payload(
    completions_payloads: Sequence[dict[str, object]],
    metrics: dict[str, float],
    *,
    run: ResolvedFunctionCallingRun,
    model_name: str,
    plan: Any,
    job_name: str,
    sampling_payload: dict[str, object],
) -> Mapping[str, object]:
    metrics.update(compute_longbench_completion_metrics(completions_payloads))
    return make_score_payload(
        run.dataset_slug,
        is_cot=True,
        model_name=model_name,
        metrics=metrics,
        samples=len(completions_payloads),
        problems=plan.sample_size,
        task=job_name,
        task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
        extra={
            "sampling_config": sampling_payload,
            "cot_mode": CoTMode.COT.value,
            "scoring": "best_reference_token_f1_exact_match",
        },
    )


def _longbench_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
    mode = str(getattr(args, "long_doc_mode", "lexical") or "lexical").strip().lower()
    enabled = mode != "off"
    if mode == "off":
        mode = "lexical"
    return LongDocEvidenceConfig(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=max(1, int(getattr(args, "long_doc_max_chars", 1000) or 1000)),
        overlap_lines=max(0, int(getattr(args, "long_doc_overlap_lines", 3) or 0)),
        min_long_text_chars=max(1, int(getattr(args, "long_doc_min_chars", 6000) or 6000)),
        max_evidence_chunks=max(1, int(getattr(args, "long_doc_max_evidence_chunks", 4) or 4)),
        max_evidence_chars=max(1, int(getattr(args, "long_doc_max_evidence_chars", 6000) or 6000)),
    )


def _longbench_source_files(source: str | Path, *, split: str) -> tuple[Path, ...]:
    root = Path(source).expanduser()
    if root.is_file():
        return (root.resolve(),)
    candidates: list[Path] = []
    search_roots = [
        root / split,
        root / "data" / split,
        root / "data",
        root,
    ]
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.jsonl"):
            resolved = path.resolve()
            if resolved in seen or ".manifest" in resolved.name:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return tuple(sorted(candidates))


def _longbench_record_from_payload(payload: Mapping[str, Any], *, fallback_index: int, source_path: str) -> LongBenchRecord:
    dataset = str(payload.get("dataset") or payload.get("subset") or Path(source_path).stem).strip() or "longbench"
    task_id = str(payload.get("task_id") or payload.get("_id") or payload.get("id") or f"{dataset}_{fallback_index:05d}")
    context = str(payload.get("context") or payload.get("document") or payload.get("passage") or "")
    question = str(payload.get("input") or payload.get("question") or payload.get("query") or payload.get("instruction") or "")
    answers = _coerce_string_tuple(
        payload.get("answers", payload.get("answer", payload.get("expected_answer", payload.get("target"))))
    )
    all_classes = _coerce_string_tuple(payload.get("all_classes") or payload.get("classes") or payload.get("choices"))
    language = str(payload.get("language") or _infer_longbench_language(dataset, question, context)).strip() or "en"
    length = _coerce_int(payload.get("length"), default=len(context))
    return LongBenchRecord(
        task_id=task_id,
        dataset=dataset,
        input=question,
        context=context,
        answers=answers,
        all_classes=all_classes,
        language=language,
        length=length,
        category=str(payload.get("category") or _longbench_category(dataset)),
        source_path=source_path,
    )


def _longbench_record_to_row(record: LongBenchRecord) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "dataset": record.dataset,
        "input": record.input,
        "context": record.context,
        "answers": list(record.answers),
        "all_classes": list(record.all_classes),
        "language": record.language,
        "length": record.length,
        "category": record.category,
        "source_path": record.source_path,
    }


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                items.extend(_coerce_string_tuple(item))
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return tuple(items)
    text = str(value).strip()
    return (text,) if text else ()


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _infer_longbench_dataset_name(path: Path) -> str:
    stem = path.stem
    if stem.lower() in {"test", "validation", "dev"} and path.parent.name:
        return path.parent.name
    return stem


def _infer_longbench_language(dataset: str, question: str, context: str) -> str:
    if dataset.lower() in _LONG_BENCH_ZH_DATASETS:
        return "zh"
    sample = f"{question}\n{context[:2000]}"
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= 20 else "en"


def _longbench_category(dataset: str) -> str:
    name = dataset.lower()
    if name in {"narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"}:
        return "single_doc_qa"
    if name in {"hotpotqa", "2wikimqa", "musique", "dureader"}:
        return "multi_doc_qa"
    if name in {"gov_report", "qmsum", "multi_news", "vcsum"}:
        return "summarization"
    if name in {"trec", "triviaqa", "samsum", "lsht"}:
        return "few_shot"
    if name in {"passage_count", "passage_retrieval_en", "passage_retrieval_zh"}:
        return "synthetic"
    if name in {"lcc", "repobench-p"}:
        return "code"
    return "unknown"


def _middle_truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    notice = "\n[... middle truncated to fit prompt budget ...]\n"
    if max_chars <= len(notice) + 8:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - len(notice)
    return text[:head] + notice + text[-tail:]


def _normalized_exact_match(prediction: str, reference: str) -> bool:
    return _normalize_for_match(prediction) == _normalize_for_match(reference)


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _answer_tokens(prediction)
    ref_tokens = _answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _answer_tokens(text: str) -> list[str]:
    normalized = _normalize_for_match(text)
    return _WORD_OR_CJK_RE.findall(normalized)


__all__ = [
    "LONG_BENCH_QA_DATASETS",
    "LONG_BENCH_DATASETS",
    "LongBenchRecord",
    "LongBenchScore",
    "_run_longbench",
    "build_longbench_budgeted_prompt",
    "build_longbench_prompt",
    "compute_longbench_completion_metrics",
    "load_longbench_manifest_records",
    "load_longbench_rows_from_source",
    "normalize_longbench_answer",
    "normalize_longbench_manifest_row",
    "score_longbench_answer",
]
