from __future__ import annotations

import argparse
import json
import re
import string
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

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
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_long_text, normalize_newlines
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext


LONGCODEBENCH_HF_REPO = "Steefano/LCB"
LONGCODEQA_ARCHIVE = "LongCodeQA.zip"
LONGCODEQA_DATASET = "longcodeqa"

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|答案|最终答案)\s*(?:is|为)?\s*[:：]?\s*(?:\*\*)?\s*\(?([A-Z])\)?(?:\*\*)?\b",
    re.IGNORECASE,
)
_CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])\)", re.MULTILINE)
_FENCED_BLOCK_RE = re.compile(r"^\s*```(?:json|text)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_INLINE_BOLD_CHOICE_RE = re.compile(r"\*\*\s*\(?([A-Z])\)?\s*[\).:]", re.IGNORECASE)
_LONGCODEQA_ANSWER_CONTRACT_RE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*$", re.IGNORECASE)
_LONGCODEQA_ANSWER_SUFFIX = "\n\nReturn exactly one option letter only. Do not include explanation.\nAnswer:"


@dataclass(frozen=True, slots=True)
class LongCodeQARecord:
    task_id: str
    prompt: str
    repo_text: str
    question: str
    correct_letter: str
    repo: str = ""
    context_bucket: str = ""
    context_size: int = 0
    prompt_goal: str = ""
    is_hard: str = ""
    is_hard_label: str = ""
    source_path: str = ""


@dataclass(frozen=True, slots=True)
class LongCodeQAScore:
    exact_match: bool
    prediction: str
    correct_letter: str

    @property
    def reward(self) -> float:
        return 1.0 if self.exact_match else 0.0


def load_longcodeqa_manifest_records(path: str | Path) -> list[LongCodeQARecord]:
    records: list[LongCodeQARecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(_longcodeqa_record_from_payload(payload, fallback_index=index, source_path=str(target)))
    return records


def load_longcodeqa_rows_from_source(source: str | Path) -> list[dict[str, Any]]:
    root = Path(source).expanduser()
    if root.is_file():
        if root.suffix.lower() == ".zip":
            return list(_rows_from_longcodeqa_zip(root))
        return list(_rows_from_json_file(root, context_bucket=root.stem))

    if not root.exists():
        raise FileNotFoundError(root)

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.zip"), key=_source_sort_key):
        rows.extend(_rows_from_longcodeqa_zip(path))
    for path in sorted(root.rglob("*.json"), key=_source_sort_key):
        rows.extend(_rows_from_json_file(path, context_bucket=path.stem))
    for path in sorted(root.rglob("*.jsonl"), key=_source_sort_key):
        rows.extend(_rows_from_jsonl_file(path, context_bucket=path.stem))
    if not rows:
        raise FileNotFoundError(f"no LongCodeQA JSON/JSONL files found under {root}")
    return rows


def normalize_longcodeqa_manifest_row(
    payload: Mapping[str, Any],
    *,
    fallback_index: int,
    source_path: str,
) -> dict[str, Any]:
    return _longcodeqa_record_to_row(
        _longcodeqa_record_from_payload(payload, fallback_index=fallback_index, source_path=source_path)
    )


def build_longcodeqa_prompt(
    record: LongCodeQARecord,
    *,
    repo_text: str | None = None,
) -> tuple[str, bool]:
    if repo_text is None:
        return normalize_newlines(record.prompt), False
    source_prompt = normalize_newlines(record.prompt)
    source_repo = normalize_newlines(record.repo_text)
    replacement = normalize_newlines(repo_text)
    if source_repo and source_repo in source_prompt:
        return _ensure_longcodeqa_answer_contract(source_prompt.replace(source_repo, replacement, 1)), True
    prompt_goal = normalize_newlines(record.prompt_goal).strip() or (
        "You are going to be provided the content of a repository and a question about it. "
        "Provide the answer to the question by stating ONLY the letter associated to the question."
    )
    question = normalize_newlines(record.question).strip()
    return _ensure_longcodeqa_answer_contract(f"{prompt_goal}\nRepository: {replacement}\n{question}"), False


def build_longcodeqa_budgeted_prompt(
    record: LongCodeQARecord,
    *,
    long_doc_config: LongDocEvidenceConfig,
    prompt_max_chars: int | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    original_prompt = normalize_newlines(record.prompt)
    repo_text = normalize_newlines(record.repo_text)
    if not long_doc_config.enabled:
        trace = {
            "mode": "off",
            "enabled": False,
            "original_context_chars": len(repo_text),
            "rendered_context_chars": len(repo_text),
            "trimmed_context_chars": 0,
            "compacted": False,
            "chunk_count": 0,
            "selected_chunk_ids": [],
            "prompt_chars": len(original_prompt),
            "replacement_found": bool(repo_text and repo_text in original_prompt),
        }
        return original_prompt, trace

    compaction = compact_long_text(
        repo_text,
        query=normalize_newlines(record.question),
        config=long_doc_config,
        label=f"longcodeqa:{record.context_bucket or record.repo or record.task_id}",
        engine=engine,
        sampling=sampling,
        progress_desc="LongCodeBench-LongDoc",
        prompt_seed=prompt_seed,
    )
    prompt, replacement_found = build_longcodeqa_prompt(record, repo_text=compaction.text)
    trimmed_context_chars = 0
    if prompt_max_chars is not None and len(prompt) > prompt_max_chars:
        overhead_prompt, _ = build_longcodeqa_prompt(record, repo_text="")
        context_budget = max(0, int(prompt_max_chars) - len(overhead_prompt) - 16)
        fitted_context = _middle_truncate_text(compaction.text, context_budget)
        trimmed_context_chars = max(0, len(compaction.text) - len(fitted_context))
        prompt, replacement_found = build_longcodeqa_prompt(record, repo_text=fitted_context)
        if len(prompt) > prompt_max_chars:
            extra = len(prompt) - int(prompt_max_chars)
            fitted_context = _middle_truncate_text(fitted_context, max(0, len(fitted_context) - extra - 32))
            trimmed_context_chars = max(0, len(compaction.text) - len(fitted_context))
            prompt, replacement_found = build_longcodeqa_prompt(record, repo_text=fitted_context)
    trace = {
        "mode": long_doc_config.mode,
        "enabled": True,
        "original_context_chars": len(repo_text),
        "rendered_context_chars": max(0, len(compaction.text) - trimmed_context_chars),
        "trimmed_context_chars": trimmed_context_chars,
        "compacted": bool(compaction.compacted),
        "chunk_count": int(compaction.chunk_count),
        "selected_chunk_ids": list(compaction.selected_chunk_ids),
        "prompt_chars": len(prompt),
        "replacement_found": replacement_found,
    }
    if compaction.router_error:
        trace["router_error"] = compaction.router_error
    return prompt, trace


def normalize_longcodeqa_answer(text: str, *, allowed_letters: Sequence[str] = ()) -> str:
    body = _clean_longcodeqa_answer_text(text)
    if not body:
        return ""
    allowed = {letter.upper() for letter in allowed_letters if str(letter).strip()} or set(string.ascii_uppercase[:8])
    json_letter = _extract_longcodeqa_json_answer(body, allowed_letters=tuple(sorted(allowed)))
    if json_letter:
        return json_letter
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_PREFIX_RE.search(line)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter
    for line in reversed(lines):
        stripped = line.strip().strip("`").strip()
        match = re.fullmatch(r"\(?([A-Z])\)?[\.\)]?", stripped, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in allowed:
                return letter
    stripped = body.lstrip()
    match = re.match(r"\(?([A-Z])\)?[\.\):,\s]", stripped, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    bold_letters = [
        match.group(1).upper()
        for match in _INLINE_BOLD_CHOICE_RE.finditer(body)
        if match.group(1).upper() in allowed
    ]
    if bold_letters and len(set(bold_letters)) == 1:
        return bold_letters[0]
    return ""


def _ensure_longcodeqa_answer_contract(prompt: str) -> str:
    normalized = normalize_newlines(prompt).rstrip()
    if _LONGCODEQA_ANSWER_CONTRACT_RE.search(normalized):
        return normalized
    return f"{normalized}{_LONGCODEQA_ANSWER_SUFFIX}"


def _clean_longcodeqa_answer_text(text: str) -> str:
    body = normalize_newlines(str(text or "")).strip()
    if body.startswith("Assistant:"):
        body = body[len("Assistant:") :].strip()
    body = _THINK_BLOCK_RE.sub("", body).strip()
    fence_match = _FENCED_BLOCK_RE.match(body)
    if fence_match:
        body = fence_match.group(1).strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json", "```text"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    if body.endswith("```"):
        body = body[: -len("```")].strip()
    return body


def _extract_longcodeqa_json_answer(text: str, *, allowed_letters: Sequence[str]) -> str:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ""
    return _extract_longcodeqa_letter_from_value(payload, allowed_letters=set(allowed_letters))


def _extract_longcodeqa_letter_from_value(value: Any, *, allowed_letters: set[str]) -> str:
    if isinstance(value, str):
        candidate = value.strip().upper()
        return candidate if len(candidate) == 1 and candidate in allowed_letters else ""
    if isinstance(value, Mapping):
        for key in ("answer", "choice", "letter", "prediction", "final_answer"):
            if key in value:
                letter = _extract_longcodeqa_letter_from_value(value.get(key), allowed_letters=allowed_letters)
                if letter:
                    return letter
        arguments = value.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, Mapping):
            letter = _extract_longcodeqa_letter_from_value(arguments, allowed_letters=allowed_letters)
            if letter:
                return letter
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            letter = _extract_longcodeqa_letter_from_value(item, allowed_letters=allowed_letters)
            if letter:
                return letter
    return ""


def score_longcodeqa_answer(
    prediction_text: str,
    correct_letter: str,
    *,
    allowed_letters: Sequence[str] = (),
) -> LongCodeQAScore:
    expected = str(correct_letter or "").strip().upper()
    prediction = normalize_longcodeqa_answer(
        prediction_text,
        allowed_letters=allowed_letters or tuple(string.ascii_uppercase[:8]),
    )
    return LongCodeQAScore(
        exact_match=bool(expected and prediction == expected),
        prediction=prediction,
        correct_letter=expected,
    )


def compute_longcodeqa_completion_metrics(completions_payloads: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not completions_payloads:
        return {}
    exact_count = 0
    parsed_count = 0
    compaction_count = 0
    prompt_chars: list[float] = []
    original_context_chars: list[float] = []
    rendered_context_chars: list[float] = []
    replacement_found = 0
    for payload in completions_payloads:
        info = payload.get("agent_info")
        if not isinstance(info, Mapping):
            continue
        if bool(info.get("exact_match", False)):
            exact_count += 1
        if str(info.get("prediction") or "").strip():
            parsed_count += 1
        trace = info.get("long_doc")
        if isinstance(trace, Mapping):
            if bool(trace.get("compacted", False)):
                compaction_count += 1
            if bool(trace.get("replacement_found", False)):
                replacement_found += 1
            _append_float(prompt_chars, trace.get("prompt_chars"))
            _append_float(original_context_chars, trace.get("original_context_chars"))
            _append_float(rendered_context_chars, trace.get("rendered_context_chars"))
    total = len(completions_payloads)
    metrics: dict[str, float] = {
        "longcodeqa_accuracy": exact_count / total,
        "longcodeqa_exact_match_rate": exact_count / total,
        "longcodeqa_answer_parse_rate": parsed_count / total,
        "longcodeqa_compaction_rate": compaction_count / total,
        "longcodeqa_repo_replacement_rate": replacement_found / total,
    }
    if prompt_chars:
        metrics["longcodeqa_avg_prompt_chars"] = sum(prompt_chars) / len(prompt_chars)
    if original_context_chars:
        metrics["longcodeqa_avg_original_context_chars"] = sum(original_context_chars) / len(original_context_chars)
    if rendered_context_chars:
        metrics["longcodeqa_avg_rendered_context_chars"] = sum(rendered_context_chars) / len(rendered_context_chars)
    return metrics


def _longcodeqa_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason="" if passed else "wrong_letter",
        answer=str(agent_info.get("prediction") or ""),
        ref_answer=str(agent_info.get("correct_letter") or ""),
    )


def _run_longcodebench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_longcodeqa_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("LongCodeQA manifest is empty")

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
    prompt_max_chars = int(args.prompt_max_chars) if args.prompt_max_chars else None
    long_doc_config = _longcodebench_long_doc_config(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        prompt_max_chars=prompt_max_chars,
    )
    selected_entries = [(int(sample_index), records[int(sample_index)]) for sample_index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [
            build_longcodeqa_budgeted_prompt(
                record,
                long_doc_config=long_doc_config,
                prompt_max_chars=prompt_max_chars,
                engine=run.engine,
                sampling=sampling,
            )[0]
            for _, record in repeated
        ]
        run.engine.generate(prompts, sampling=sampling, batch_size=len(prompts), progress_desc="LongCodeBench-Probe")
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_longcodebench", run_context=run_context)
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
        completion_to_eval=_longcodeqa_completion_to_eval_payload,
        runner_name="longcodebench",
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
                        build_longcodeqa_budgeted_prompt(
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
                        progress_desc="LongCodeBench",
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
                        allowed_letters = _choice_letters(record.question)
                        score = score_longcodeqa_answer(
                            output.text,
                            record.correct_letter,
                            allowed_letters=allowed_letters,
                        )
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
                            "reward": score.reward,
                            "num_turns": 1,
                            "cost": 0.0,
                            "is_passed": score.exact_match,
                            "error": None,
                        }
                        payload["agent_info"] = {
                            "dataset": LONGCODEQA_DATASET,
                            "task_id": record.task_id,
                            "repo": record.repo,
                            "context_bucket": record.context_bucket,
                            "context_size": record.context_size,
                            "question": record.question,
                            "prediction": score.prediction,
                            "correct_letter": score.correct_letter,
                            "exact_match": score.exact_match,
                            "allowed_letters": list(allowed_letters),
                            "prompt_goal": record.prompt_goal,
                            "is_hard_label": record.is_hard_label,
                            "long_doc": trace,
                        }
                        payload["agent_trace"] = [{"stage": "answer", "text": score.prediction}]
                        payload["task_id"] = record.task_id
                        payload["domain"] = "long_code"
                        payload["instruction"] = record.question
                        writer.enqueue(payload)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_longcodeqa_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: _longcodeqa_score_payload(
                completions_payloads,
                metrics,
                run=run,
                model_name=run.model_name,
                plan=plan,
                job_name=job_name,
                sampling_payload=sampling_payload,
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"longcodebench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


def _longcodeqa_score_payload(
    completions_payloads: Sequence[dict[str, object]],
    metrics: dict[str, float],
    *,
    run: ResolvedFunctionCallingRun,
    model_name: str,
    plan: Any,
    job_name: str,
    sampling_payload: dict[str, object],
) -> Mapping[str, object]:
    metrics.update(compute_longcodeqa_completion_metrics(completions_payloads))
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
            "scoring": "exact_letter_match",
        },
    )


def _longcodebench_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
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
        model_max_tokens=max(1, int(getattr(args, "long_doc_model_max_tokens", 96) or 96)),
        model_parallel_batch_size=max(1, int(getattr(args, "long_doc_model_parallel_batch_size", 8) or 8)),
    )


def _rows_from_longcodeqa_zip(path: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".json")]
        for name in sorted(names, key=_source_sort_key):
            payload = json.loads(archive.read(name).decode("utf-8"))
            if not isinstance(payload, list):
                continue
            context_bucket = Path(name).stem
            for index, row in enumerate(payload):
                if isinstance(row, Mapping):
                    yield normalize_longcodeqa_manifest_row(
                        {
                            **dict(row),
                            "context_bucket": context_bucket,
                            "context_size": _context_bucket_chars(context_bucket),
                        },
                        fallback_index=index,
                        source_path=f"{path}:{name}",
                    )


def _rows_from_json_file(path: Path, *, context_bucket: str) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            yield normalize_longcodeqa_manifest_row(
                {
                    **dict(row),
                    "context_bucket": str(row.get("context_bucket") or context_bucket),
                    "context_size": row.get("context_size") or _context_bucket_chars(context_bucket),
                },
                fallback_index=index,
                source_path=str(path),
            )


def _rows_from_jsonl_file(path: Path, *, context_bucket: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if isinstance(row, Mapping):
                yield normalize_longcodeqa_manifest_row(
                    {
                        **dict(row),
                        "context_bucket": str(row.get("context_bucket") or context_bucket),
                        "context_size": row.get("context_size") or _context_bucket_chars(context_bucket),
                    },
                    fallback_index=index,
                    source_path=str(path),
                )


def _longcodeqa_record_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_index: int,
    source_path: str,
) -> LongCodeQARecord:
    context_bucket = str(payload.get("context_bucket") or payload.get("bucket") or "").strip()
    repo = str(payload.get("repo") or "").strip()
    task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
    if not task_id:
        prefix = (context_bucket or LONGCODEQA_DATASET).lower().replace("/", "_")
        task_id = f"longcodeqa_{prefix}_{fallback_index:05d}"
    question = str(payload.get("question") or "")
    correct_letter = str(payload.get("correct_letter") or payload.get("answer") or "").strip().upper()
    repo_text = str(payload.get("repo_text") or payload.get("repository") or payload.get("context") or "")
    prompt = str(payload.get("prompt") or "")
    if not prompt:
        prompt_goal = str(payload.get("prompt_goal") or "").strip() or (
            "You are going to be provided the content of a repository and a question about it. "
            "Provide the answer to the question by stating ONLY the letter associated to the question."
        )
        prompt = f"{prompt_goal}\nRepository: {repo_text}\n{question}\nAnswer:"
    is_hard = str(payload.get("is_hard") or "")
    return LongCodeQARecord(
        task_id=task_id,
        prompt=normalize_newlines(prompt),
        repo_text=normalize_newlines(repo_text),
        question=normalize_newlines(question),
        correct_letter=correct_letter[:1],
        repo=repo,
        context_bucket=context_bucket,
        context_size=_coerce_int(payload.get("context_size"), default=_context_bucket_chars(context_bucket)),
        prompt_goal=normalize_newlines(str(payload.get("prompt_goal") or "")),
        is_hard=normalize_newlines(is_hard),
        is_hard_label=_parse_is_hard_label(is_hard),
        source_path=source_path,
    )


def _longcodeqa_record_to_row(record: LongCodeQARecord) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "prompt": record.prompt,
        "repo_text": record.repo_text,
        "question": record.question,
        "correct_letter": record.correct_letter,
        "repo": record.repo,
        "context_bucket": record.context_bucket,
        "context_size": record.context_size,
        "prompt_goal": record.prompt_goal,
        "is_hard": record.is_hard,
        "is_hard_label": record.is_hard_label,
        "source_path": record.source_path,
    }


def _parse_is_hard_label(text: str) -> str:
    stripped = str(text or "").strip().lower()
    if not stripped:
        return ""
    tail = stripped.splitlines()[-1].strip().strip(".")
    if tail in {"yes", "y", "true"}:
        return "yes"
    if tail in {"no", "n", "false"}:
        return "no"
    return ""


def _choice_letters(question: str) -> tuple[str, ...]:
    matches = [match.group(1).upper() for match in _CHOICE_LINE_RE.finditer(str(question or ""))]
    seen: set[str] = set()
    ordered: list[str] = []
    for letter in matches:
        if letter in seen:
            continue
        seen.add(letter)
        ordered.append(letter)
    return tuple(ordered or list(string.ascii_uppercase[:4]))


def _append_float(values: list[float], raw: object) -> None:
    try:
        values.append(float(raw or 0))
    except (TypeError, ValueError):
        return


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _context_bucket_chars(bucket: str) -> int:
    text = str(bucket or "").strip().upper()
    match = re.fullmatch(r"(\d+)\s*([KMG])?", text)
    if not match:
        return 0
    value = int(match.group(1))
    unit = match.group(2) or ""
    if unit == "K":
        return value * 1024
    if unit == "M":
        return value * 1024 * 1024
    if unit == "G":
        return value * 1024 * 1024 * 1024
    return value


def _source_sort_key(value: str | Path) -> tuple[int, str]:
    path = Path(str(value))
    bucket_value = _context_bucket_chars(path.stem)
    return (bucket_value if bucket_value > 0 else 10**12, str(value))


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


__all__ = [
    "LONGCODEBENCH_HF_REPO",
    "LONGCODEQA_ARCHIVE",
    "LONGCODEQA_DATASET",
    "LongCodeQARecord",
    "LongCodeQAScore",
    "_run_longcodebench",
    "build_longcodeqa_budgeted_prompt",
    "build_longcodeqa_prompt",
    "compute_longcodeqa_completion_metrics",
    "load_longcodeqa_manifest_records",
    "load_longcodeqa_rows_from_source",
    "normalize_longcodeqa_answer",
    "normalize_longcodeqa_manifest_row",
    "score_longcodeqa_answer",
]
