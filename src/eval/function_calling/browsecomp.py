from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import resolve_judge_model_config
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta

from .context_budget import normalize_rwkv_text

if TYPE_CHECKING:
    import argparse

    from src.eval.evaluating.contracts import RunContext


@dataclass(frozen=True, slots=True)
class BrowseCompRecord:
    task_id: str
    question: str
    answer: str
    locale: str
    topic: str | None = None


@dataclass(frozen=True, slots=True)
class BrowseCompJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_workers: int = 4
    max_retries: int = 3
    backoff_base_s: float = 0.5


@dataclass(frozen=True, slots=True)
class BrowseCompJudgeOutcome:
    is_passed: bool
    reason: str


def decrypt_xor_base64(ciphertext_b64: str, password: str) -> str:
    ciphertext = base64.b64decode(ciphertext_b64.strip())
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    plaintext = bytes(lhs ^ rhs for lhs, rhs in zip(ciphertext, _repeat_bytes(digest, len(ciphertext))))
    return plaintext.decode("utf-8")


def load_browsecomp_rows_from_csv(path: str | Path) -> list[BrowseCompRecord]:
    target = Path(path)
    rows: list[BrowseCompRecord] = []
    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            canary = str(row.get("canary") or "")
            rows.append(
                BrowseCompRecord(
                    task_id=f"browsecomp_{index:04d}",
                    question=decrypt_xor_base64(str(row.get("problem") or ""), canary),
                    answer=decrypt_xor_base64(str(row.get("answer") or ""), canary),
                    locale="en",
                    topic=str(row.get("problem_topic") or "").strip() or None,
                )
            )
    return rows


def load_browsecomp_zh_rows_from_xlsx(path: str | Path) -> list[BrowseCompRecord]:
    target = Path(path)
    with zipfile.ZipFile(target) as zf:
        shared_strings = _load_shared_strings(zf)
        rows = _load_sheet_rows(zf, shared_strings)

    records: list[BrowseCompRecord] = []
    for index, row in enumerate(rows):
        canary = str(row.get("canary") or "")
        question = _decrypt_optional_field(row.get("Question"), canary)
        answer = _decrypt_optional_field(row.get("Answer"), canary)
        if not question.strip() or not answer.strip():
            continue
        records.append(
            BrowseCompRecord(
                task_id=f"browsecomp_zh_{index:04d}",
                question=question,
                answer=answer,
                locale="zh",
            )
        )
    return records


def load_browsecomp_manifest_records(path: str | Path) -> list[BrowseCompRecord]:
    items: list[BrowseCompRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            items.append(
                BrowseCompRecord(
                    task_id=str(payload.get("task_id") or ""),
                    question=str(payload.get("question") or ""),
                    answer=str(payload.get("answer") or ""),
                    locale=str(payload.get("locale") or "en"),
                    topic=str(payload.get("topic") or "").strip() or None,
                )
            )
    return items


def build_browsecomp_user_prompt(question: str, *, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "zh":
        return normalize_rwkv_text(
            "你是一个浏览基准测试助手。请先仔细思考，再直接回答问题。\n\n"
            "请基于你自己的知识回答下面这个需要较强检索能力的问题。\n"
            "不要通过让用户自己去搜索网页来回避作答。\n"
            "即使你不完全确定，也要给出你当前最具体的答案。\n\n"
            f"问题:\n{question}\n\n"
            "请严格按下面格式回复最终答案：\n"
            "解释: <简短说明>\n"
            "最终答案: <简洁最终答案>\n"
            "置信度: <0% 到 100%>"
        )
    return normalize_rwkv_text(
        "You are a browsing benchmark assistant. Think through the question carefully and then answer directly.\n\n"
        "Answer the following browsing-intensive question using your own knowledge.\n"
        "Do not refuse by asking the user to search the web themselves.\n"
        "If you are uncertain, still provide your best concrete answer.\n\n"
        f"Question:\n{question}\n\n"
        "Return your final answer in this format:\n"
        "Explanation: <brief explanation>\n"
        "Exact Answer: <succinct final answer>\n"
        "Confidence: <0% to 100%>"
    )


def build_browsecomp_expected_context(user_prompt: str) -> str:
    return f"User: {normalize_rwkv_text(user_prompt)}\n\nAssistant: <think>"


def build_browsecomp_answer_prompt(expected_context: str, cot: str, *, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "zh":
        suffix = "\n".join(
            [
                "现在继续补完最终答案，且严格使用如下格式：",
                "解释: <简短说明>",
                "最终答案: <简洁最终答案>",
                "置信度: <0% 到 100%>",
                "解释: ",
            ]
        )
    else:
        suffix = "\n".join(
            [
                "Now continue by completing the final answer in this exact format:",
                "Explanation: <brief explanation>",
                "Exact Answer: <succinct final answer>",
                "Confidence: <0% to 100%>",
                "Explanation: ",
            ]
        )
    return f"{expected_context}{normalize_rwkv_text(cot)}</think>\n{suffix}"


def judge_browsecomp_answers(
    items: Sequence[tuple[BrowseCompRecord, str]],
    *,
    config: BrowseCompJudgeConfig,
) -> list[BrowseCompJudgeOutcome]:
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def worker(entry: tuple[BrowseCompRecord, str]) -> BrowseCompJudgeOutcome:
        record, response_text = entry
        locale = record.locale.strip().lower()
        if locale == "zh":
            prompt = (
                "你是严格的答案判定器。\n"
                "请根据 correct_answer 判断 response 是否正确回答了 question。\n"
                "数值题可接受很小的误差。\n"
                "只返回 JSON，字段为 is_passed(bool) 和 reason(string)。\n\n"
                f"[question]\n{record.question}\n\n"
                f"[response]\n{response_text}\n\n"
                f"[correct_answer]\n{record.answer}\n"
            )
        else:
            prompt = (
                "You are a rigorous answer judge.\n"
                "Decide whether the response correctly answers the question according to the correct answer.\n"
                "Treat small numerical tolerance as acceptable.\n"
                "Return only JSON with fields is_passed(bool) and reason(string).\n\n"
                f"[question]\n{record.question}\n\n"
                f"[response]\n{response_text}\n\n"
                f"[correct_answer]\n{record.answer}\n"
            )

        last_error = "judge failed"
        for attempt in range(max(1, int(config.max_retries))):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    temperature=0.0,
                    top_p=1.0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = (response.choices[0].message.content or "").strip()
                payload = _parse_json_object(content)
                return BrowseCompJudgeOutcome(
                    is_passed=bool(payload.get("is_passed", False)),
                    reason=str(payload.get("reason") or "").strip(),
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt + 1 >= max(1, int(config.max_retries)):
                    break
                time.sleep(float(config.backoff_base_s) * (2**attempt))
        return BrowseCompJudgeOutcome(is_passed=False, reason=last_error or "judge failed")

    results: list[BrowseCompJudgeOutcome] = [BrowseCompJudgeOutcome(False, "judge failed")] * len(items)
    max_workers = max(1, min(int(config.max_workers), len(items) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, entry): index for index, entry in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _repeat_bytes(raw: bytes, target_len: int) -> Iterable[int]:
    if not raw:
        return ()
    return (raw[index % len(raw)] for index in range(target_len))


def _decrypt_optional_field(value: str | None, password: str) -> str:
    if not value:
        return ""
    return decrypt_xor_base64(value, password)


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        text = zf.read("xl/sharedStrings.xml").decode("utf-8")
    except KeyError:
        return []
    root = ET.fromstring(text)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared: list[str] = []
    for item in root.findall("x:si", ns):
        parts: list[str] = []
        for text_node in item.findall(".//x:t", ns):
            parts.append(text_node.text or "")
        shared.append("".join(parts))
    return shared


def _load_sheet_rows(zf: zipfile.ZipFile, shared_strings: Sequence[str]) -> list[dict[str, str]]:
    text = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
    root = ET.fromstring(text)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = root.findall(".//x:sheetData/x:row", ns)
    if not rows:
        return []

    header_row = _parse_sheet_row(rows[0], shared_strings, ns)
    ordered_headers = {column: value for column, value in header_row.items() if value}
    parsed_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = _parse_sheet_row(row, shared_strings, ns)
        payload: dict[str, str] = {}
        for column, header in ordered_headers.items():
            payload[header] = cells.get(column, "")
        if any(value.strip() for value in payload.values()):
            parsed_rows.append(payload)
    return parsed_rows


def _parse_sheet_row(
    row: ET.Element,
    shared_strings: Sequence[str],
    ns: dict[str, str],
) -> dict[str, str]:
    cells: dict[str, str] = {}
    for cell in row.findall("x:c", ns):
        ref = str(cell.attrib.get("r") or "")
        column = _cell_column_name(ref)
        cell_type = str(cell.attrib.get("t") or "")
        value = ""
        if cell_type == "s":
            raw = cell.findtext("x:v", default="", namespaces=ns)
            if raw.isdigit():
                index = int(raw)
                if 0 <= index < len(shared_strings):
                    value = shared_strings[index]
        elif cell_type == "inlineStr":
            parts = [node.text or "" for node in cell.findall(".//x:t", ns)]
            value = "".join(parts)
        else:
            value = cell.findtext("x:v", default="", namespaces=ns) or ""
        cells[column] = _xml_unescape(value)
    return cells


def _cell_column_name(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _xml_unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"judge did not return an object: {text!r}")
    return payload


def _normalize_final_answer(text: str, *, locale: str) -> str:
    body = text.strip()
    if not body:
        return ""
    prefix = "解释:" if locale == "zh" else "Explanation:"
    return body if body.startswith(prefix) else f"{prefix} {body}"


def _browsecomp_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("judge_reason") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("response") or ""),
        ref_answer=str(agent_info.get("reference_answer") or ""),
    )


def _run_browsecomp(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_browsecomp_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("BrowseComp manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    answer_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="final",
        fallback_templates="free_response_cot_default",
    )
    if cot_sampling is None or answer_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    answer_sampling = answer_sampling.clamp(args.answer_max_tokens)

    batch_size = max(1, int(args.batch_size or 32))
    selected_entries = [(int(sample_index), records[int(sample_index)]) for sample_index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [
            build_browsecomp_expected_context(
                build_browsecomp_user_prompt(record.question, locale=record.locale)
            )
            for _, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=cot_sampling,
            batch_size=len(prompts),
            progress_desc="BrowseComp-Probe",
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("BrowseComp requires JUDGE_MODEL / judge_model_name and judge API key")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
    )

    job_name = _resolve_job_name("function_browsecomp", run_context=run_context)
    sampling_payload = normalize_sampling_config_by_stage([(1, cot_sampling), (2, answer_sampling)])
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
        judger_model_name=judge.model,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_browsecomp_completion_to_eval_payload,
        runner_name="browsecomp",
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
                    cot_prompts = [
                        build_browsecomp_expected_context(
                            build_browsecomp_user_prompt(record.question, locale=record.locale)
                        )
                        for _key, record in chunk
                    ]
                    cot_outputs = run.engine.generate(
                        cot_prompts,
                        sampling=cot_sampling,
                        batch_size=len(cot_prompts),
                        progress_desc="BrowseComp-CoT",
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
                    cot_by_index = {int(output.prompt_index): output for output in cot_outputs}
                    answer_prompts: list[str] = []
                    answer_stage_prompts: list[str] = []
                    for index, (_key, record) in enumerate(chunk):
                        cot_output = cot_by_index[index]
                        answer_prompt = build_browsecomp_answer_prompt(
                            cot_prompts[index],
                            cot_output.text,
                            locale=record.locale,
                        )
                        answer_prompts.append(answer_prompt)
                        answer_stage_prompts.append(prompt_delta(answer_prompt, f"{cot_output.prompt}{cot_output.text}"))
                    answer_outputs = run.engine.generate(
                        answer_prompts,
                        sampling=answer_sampling,
                        batch_size=len(answer_prompts),
                        progress_desc="BrowseComp-Answer",
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=2,
                            )
                            for key, _record in chunk
                        ],
                    )
                    answer_by_index = {int(output.prompt_index): output for output in answer_outputs}
                    judged = judge_browsecomp_answers(
                        [
                            (
                                record,
                                _normalize_final_answer(answer_by_index[index].text, locale=record.locale),
                            )
                            for index, (_key, record) in enumerate(chunk)
                        ],
                        config=judge,
                    )
                    for index, ((key, record), outcome) in enumerate(zip(chunk, judged)):
                        cot_output = cot_by_index[index]
                        answer_output = answer_by_index[index]
                        final_answer = _normalize_final_answer(answer_output.text, locale=record.locale)
                        stages = [
                            StageRecord(
                                prompt=cot_prompts[index],
                                completion=cot_output.text,
                                stop_reason=cot_output.finish_reason,
                            ),
                            StageRecord(
                                prompt=answer_stage_prompts[index],
                                completion=answer_output.text,
                                stop_reason=answer_output.finish_reason,
                            ),
                        ]
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 1.0 if outcome.is_passed else 0.0,
                            "num_turns": 2,
                            "cost": 0.0,
                            "is_passed": bool(outcome.is_passed),
                        }
                        payload["agent_info"] = {
                            "question": record.question,
                            "reference_answer": record.answer,
                            "response": final_answer,
                            "judge_reason": outcome.reason,
                            "locale": record.locale,
                            "cot_mode": CoTMode.COT.value,
                            "topic": record.topic or "",
                        }
                        payload["agent_trace"] = [
                            {"stage": "cot", "text": cot_output.text},
                            {"stage": "answer", "text": final_answer},
                        ]
                        payload["task_id"] = record.task_id
                        payload["domain"] = "function_call"
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
            completion_to_eval=_browsecomp_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(records),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge.model,
                    "cot_mode": CoTMode.COT.value,
                },
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0
