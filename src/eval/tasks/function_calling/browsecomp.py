from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import resolve_judge_max_workers, resolve_judge_model_config, resolve_judge_timeout_s
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.tasks.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.final_answer import (
    final_answer_tool_schema,
    parse_final_answer_call,
    render_final_answer_call,
)
from src.eval.tasks.function_calling.agent_loop import (
    AgentLoopRecord,
    _agent_loop_candidate_router_config_from_args,
    _agent_loop_candidate_router_mode,
    _agent_loop_long_doc_config,
    run_agent_loop_episode,
)
from src.eval.tasks.function_calling.agent_loop_executors import WebSearchExecutor
from src.eval.tasks.function_calling.agent_loop_executors import ExecutorSpec
from src.eval.tasks.function_calling.agent_loop_verifiers import AgentLoopVerdict, VerifierSpec
from src.eval.tasks.function_calling.native_tool_calls import openai_tool_schema
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_long_text
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta

from .context_budget import normalize_rwkv_text, trim_history

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
    max_retries: int = 6
    backoff_base_s: float = 3.0
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class BrowseCompJudgeOutcome:
    is_passed: bool
    reason: str


class BrowseCompJudgeRuntimeError(RuntimeError):
    """Raised when the external judge cannot be reached or returns an unusable response."""


_BROWSECOMP_DEFAULT_PROMPT_MAX_CHARS = 8192
_BROWSECOMP_NATIVE_TOOL_MAX_RETRIES = 2
_BROWSECOMP_AGENTIC_ENV = "RWKV_BROWSECOMP_AGENTIC"


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
    return f"User:{normalize_rwkv_text(user_prompt)}\n\nAssistant: <think>"


def build_browsecomp_answer_prompt(expected_context: str, cot: str, *, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "zh":
        suffix = "\n".join(
            [
                "现在根据上面的思考调用 final_answer。",
                "只返回一个 RWKV JSON function-call 对象。",
                '格式: {"name":"final_answer","arguments":{"answer":"<简洁最终答案>"},"id":"final_answer"}',
                "不要输出解释、置信度或 JSON 之外的文字。",
            ]
        )
    else:
        suffix = "\n".join(
            [
                "Now call final_answer using the answer from the reasoning above.",
                "Return exactly one RWKV JSON function-call object.",
                'Format: {"name":"final_answer","arguments":{"answer":"<succinct final answer>"},"id":"final_answer"}',
                "Do not output explanation, confidence, or any text outside the JSON value.",
            ]
        )
    return f"{expected_context}{normalize_rwkv_text(cot)}</think>\nUser: {suffix}\n\nAssistant: ```json\n{{"


def build_browsecomp_budgeted_answer_prompt(
    expected_context: str,
    cot: str,
    *,
    question: str,
    locale: str,
    long_doc_config: LongDocEvidenceConfig,
    prompt_max_chars: int,
    engine: Any | None = None,
    sampling: Any | None = None,
    prompt_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    source_cot = normalize_rwkv_text(cot)
    compaction = compact_long_text(
        source_cot,
        query=normalize_rwkv_text(question),
        config=long_doc_config,
        label="browsecomp:cot",
        engine=engine,
        sampling=sampling,
        progress_desc="BrowseComp-CoT-LongDoc",
        prompt_seed=prompt_seed,
    )
    rendered_cot = compaction.text
    trimmed_context_chars = 0
    prompt = build_browsecomp_answer_prompt(expected_context, rendered_cot, locale=locale)
    if prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
        empty_prompt = build_browsecomp_answer_prompt(expected_context, "", locale=locale)
        cot_budget = max(0, int(prompt_max_chars) - len(empty_prompt) - 16)
        fitted_cot = trim_history(rendered_cot, cot_budget)
        trimmed_context_chars = max(0, len(rendered_cot) - len(fitted_cot))
        rendered_cot = fitted_cot
        prompt = build_browsecomp_answer_prompt(expected_context, rendered_cot, locale=locale)
    trace = {
        "mode": long_doc_config.mode if long_doc_config.enabled else "off",
        "enabled": bool(long_doc_config.enabled),
        "original_context_chars": len(source_cot),
        "rendered_context_chars": len(rendered_cot),
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


def _answer_stage_prompt_payload(full_prompt: str, prior_context: str) -> tuple[str, bool]:
    """Store a compact stage prompt when possible, otherwise store the re-prompted context."""
    try:
        return prompt_delta(full_prompt, prior_context), False
    except ValueError:
        return full_prompt, True


def judge_browsecomp_answers(
    items: Sequence[tuple[BrowseCompRecord, str]],
    *,
    config: BrowseCompJudgeConfig,
) -> list[BrowseCompJudgeOutcome]:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=max(1.0, float(config.timeout_s)),
        max_retries=0,
    )

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

        for attempt in range(max(1, int(config.max_retries))):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    temperature=0.8,
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
                if attempt + 1 >= max(1, int(config.max_retries)):
                    raise BrowseCompJudgeRuntimeError(f"browsecomp judge failed after retries: {exc}") from exc
                time.sleep(float(config.backoff_base_s) * (2**attempt))
        raise BrowseCompJudgeRuntimeError("browsecomp judge failed")

    results: list[BrowseCompJudgeOutcome] = [BrowseCompJudgeOutcome(False, "judge failed")] * len(items)
    max_workers = max(1, min(int(config.max_workers), len(items) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, entry): index for index, entry in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = BrowseCompJudgeOutcome(False, f"judge failed after retries: {exc}")
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


def decode_browsecomp_final_answer(response: str, *, locale: str) -> tuple[str, dict[str, Any]]:
    final_call = parse_final_answer_call(response, context_label="browsecomp final answer")
    answer = _normalize_final_answer(final_call.answer, locale=locale)
    return answer, dict(final_call.call)


def _browsecomp_final_answer_tool() -> dict[str, Any]:
    return openai_tool_schema(final_answer_tool_schema(answer_description="The succinct final answer."))


def _browsecomp_final_answer_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "final_answer"}}


def _native_tool_raw_completion(output: Any) -> str:
    raw_message = getattr(output, "raw_message", {}) or {}
    if isinstance(raw_message, Mapping) and raw_message:
        return json.dumps(raw_message, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    calls = []
    for call in getattr(output, "tool_calls", []) or []:
        calls.append(
            {
                "id": str(getattr(call, "id", "") or "final_answer"),
                "type": "function",
                "function": {
                    "name": str(getattr(call, "name", "") or ""),
                    "arguments": json.dumps(
                        dict(getattr(call, "arguments", {}) or {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    return json.dumps(
        {"role": "assistant", "content": getattr(output, "content", "") or None, "tool_calls": calls},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _native_browsecomp_final_answer_row(output: Any, *, locale: str, retry_attempt: int = 0) -> dict[str, Any]:
    raw_completion = _native_tool_raw_completion(output)
    tool_calls = list(getattr(output, "tool_calls", []) or [])
    decoded_call: dict[str, Any] = {}
    final_answer = ""
    parse_error = ""
    if tool_calls:
        call = tool_calls[0]
        call_name = str(getattr(call, "name", "") or "").strip()
        arguments = dict(getattr(call, "arguments", {}) or {})
        if call_name != "final_answer":
            parse_error = f"browsecomp final answer expected final_answer tool, got {call_name!r}"
        else:
            answer = str(arguments.get("answer") or "").strip()
            call_id = str(getattr(call, "id", "") or "final_answer")
            decoded_call = {"name": "final_answer", "arguments": arguments, "id": call_id}
            if answer:
                final_answer = _normalize_final_answer(answer, locale=locale)
            else:
                parse_error = "browsecomp final answer final_answer call missing answer"
    else:
        content = str(getattr(output, "content", "") or "")
        try:
            final_answer, decoded_call = decode_browsecomp_final_answer(content, locale=locale)
        except Exception as exc:  # noqa: BLE001 - sample-level parser failure
            parse_error = str(exc)
    raw_arguments = decoded_call.get("arguments") if isinstance(decoded_call, Mapping) else {}
    raw_answer = str(raw_arguments.get("answer") or "").strip() if isinstance(raw_arguments, Mapping) else ""
    raw_call_id = str(decoded_call.get("id") or "").strip() if isinstance(decoded_call, Mapping) else ""
    final_call = render_final_answer_call(raw_answer, call_id=raw_call_id) if raw_answer else ""
    return {
        "response": final_answer,
        "decoded_final_answer_call": decoded_call,
        "final_answer_call": final_call,
        "parse_error": parse_error,
        "raw_completion": raw_completion,
        "finish_reason": str(getattr(output, "finish_reason", "stop") or "stop"),
        "native_tool": {
            "enabled": True,
            "response_source": str(getattr(output, "response_source", "") or ""),
            "retry_attempt": int(retry_attempt),
        },
    }


def _browsecomp_answer_runtime_sampling(sampling: Any) -> Any:
    return replace(sampling, alpha_presence=0.0, alpha_frequency=0.0, alpha_decay=0.0)


def _generate_native_browsecomp_final_answers(
    engine: Any,
    answer_prompts: Sequence[str],
    *,
    sampling: Any,
    prompt_seeds: Sequence[int | None],
) -> tuple[list[Any] | None, str]:
    generate_tool_calls = getattr(engine, "generate_tool_calls", None)
    if not callable(generate_tool_calls):
        return None, "engine does not expose generate_tool_calls"
    messages = [[{"role": "user", "content": prompt}] for prompt in answer_prompts]
    tools = [[_browsecomp_final_answer_tool()] for _ in answer_prompts]
    native_sampling = _browsecomp_answer_runtime_sampling(sampling)
    try:
        outputs = generate_tool_calls(
            messages,
            tools,
            sampling=native_sampling,
            batch_size=len(answer_prompts),
            progress_desc="BrowseComp-Answer-NativeTool",
            tool_choice=_browsecomp_final_answer_tool_choice(),
            parallel_tool_calls=False,
            prompt_seeds=list(prompt_seeds),
        )
    except NotImplementedError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - keep old text path when native tools are unavailable server-side
        return None, f"{type(exc).__name__}: {exc}"
    return list(outputs), ""


def _browsecomp_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
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


def _browsecomp_agentic_enabled(args: argparse.Namespace) -> bool:
    raw = str(os.environ.get(_BROWSECOMP_AGENTIC_ENV) or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "agent", "agentic", "web"}


def _browsecomp_agentic_instruction(record: BrowseCompRecord) -> str:
    if record.locale.strip().lower() == "zh":
        return normalize_rwkv_text(
            "这是 BrowseComp 浏览检索题。请使用 web_search 搜索，并只对搜索结果中明确出现或可核验的 URL 使用 fetch_url。\n"
            "不要只凭记忆作答。需要多次检索时请拆分查询并交叉验证。\n"
            "最终完成时调用 final_answer，answer 字段只填写简洁、精确的最终答案，不要放解释或置信度。\n\n"
            f"问题:\n{record.question}"
        )
    return normalize_rwkv_text(
        "This is a BrowseComp browsing task. Use web_search to investigate, and use fetch_url only for URLs "
        "that came from search results or visible evidence.\n"
        "Do not answer from memory alone. Split the search into narrower queries when needed and cross-check evidence.\n"
        "When finished, call final_answer with only the succinct exact answer in the answer field; do not include "
        "explanation or confidence in final_answer.\n\n"
        f"Question:\n{record.question}"
    )


def _browsecomp_agentic_record(record: BrowseCompRecord) -> AgentLoopRecord:
    return AgentLoopRecord(
        task_id=record.task_id,
        instruction=_browsecomp_agentic_instruction(record),
        tools=(),
        executor=ExecutorSpec(kind="web_search", config={}),
        verifier=VerifierSpec(kind="browsecomp_agentic", config={}),
        metadata={
            "source_benchmark": "browsecomp",
            "locale": record.locale,
            "topic": record.topic or "",
        },
    )


class _BrowseCompAgenticVerifier:
    def __init__(self, records: Sequence[BrowseCompRecord], judge: BrowseCompJudgeConfig) -> None:
        self._records = {record.task_id: record for record in records}
        self._judge = judge

    def verify(
        self,
        record: AgentLoopRecord,
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        del trace, executor_snapshot
        source = self._records.get(record.task_id)
        if source is None:
            raise ValueError(f"BrowseComp source record not found: {record.task_id}")
        response = _normalize_final_answer(final_answer, locale=source.locale)
        if not response.strip():
            return AgentLoopVerdict(
                reward=0.0,
                is_passed=False,
                fail_reason="browsecomp agent produced no final answer",
                details={"judge_reason": "browsecomp agent produced no final answer", "response": ""},
            )
        outcome = judge_browsecomp_answers([(source, response)], config=self._judge)[0]
        return AgentLoopVerdict(
            reward=1.0 if outcome.is_passed else 0.0,
            is_passed=bool(outcome.is_passed),
            fail_reason="" if outcome.is_passed else outcome.reason,
            details={"judge_reason": outcome.reason, "response": response},
        )


def _run_browsecomp_agentic(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    records: Sequence[BrowseCompRecord],
    *,
    run_context: "RunContext | None" = None,
) -> int:
    if args.probe_only:
        raise ValueError(f"{_BROWSECOMP_AGENTIC_ENV}=1 does not support --probe-only")

    search_error = WebSearchExecutor.config_error() or WebSearchExecutor.health_error()
    if search_error:
        raise ValueError(f"BrowseComp agentic mode requires a healthy web_search executor: {search_error}")

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="function_call_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = clamp_function_calling_sampling(
        tool_sampling,
        max(1, int(getattr(args, "decision_max_tokens", None) or 1024)),
    )

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("BrowseComp agentic mode requires JUDGE_MODEL + JUDGE_API_KEY")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
        max_workers=resolve_judge_max_workers(getattr(args, "judge_max_workers", None), default=4),
        timeout_s=resolve_judge_timeout_s(default=60.0),
    )
    verifier = _BrowseCompAgenticVerifier(records, judge)
    long_doc_config = _agent_loop_long_doc_config(args)
    candidate_router_mode = _agent_loop_candidate_router_mode(args)
    candidate_router_config = _agent_loop_candidate_router_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, tool_sampling)]),
        long_doc_config=long_doc_config,
        candidate_router_config=candidate_router_config,
    )
    sampling_payload["browsecomp_adapter"] = {
        "mode": "agentic_web_search",
        "candidate_router_mode": candidate_router_mode,
        "decision_io": "rwkv_json_or_parallel_candidate",
    }

    history_max_chars = max(0, int(getattr(args, "history_max_chars", None) or 24000))
    max_steps = max(1, int(getattr(args, "max_steps", None) or 16))
    max_tool_errors = max(1, int(getattr(args, "max_tool_errors", None) or 5))
    max_output_chars = int(getattr(args, "agent_loop_max_output_chars", None) or 8000)

    job_name = _resolve_job_name("function_browsecomp", run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 8),
        run_context=run_context,
        judger_model_name=judge.model,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_browsecomp_completion_to_eval_payload,
        runner_name="browsecomp_agentic",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for key, source_record in pending:
                    agent_record = _browsecomp_agentic_record(source_record)
                    executor = WebSearchExecutor(max_output_chars=max_output_chars)
                    try:
                        episode = run_agent_loop_episode(
                            record=agent_record,
                            engine=run.engine,
                            tool_sampling=tool_sampling,
                            executor=executor,
                            verifier=verifier,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            max_steps=max_steps,
                            max_tool_errors=max_tool_errors,
                            history_max_chars=history_max_chars,
                            max_output_chars=max_output_chars,
                            long_doc_config=long_doc_config,
                            candidate_router_mode=candidate_router_mode,
                            candidate_router_config=candidate_router_config,
                            progress_prefix="BrowseComp-Agentic",
                        )
                    except Exception as exc:  # noqa: BLE001 - sample-level infrastructure failures become failed rows.
                        episode = {
                            "stages": [],
                            "trace": [],
                            "final_answer": "",
                            "termination_reason": "episode_error",
                            "error": str(exc),
                            "verdict": AgentLoopVerdict(0.0, False, str(exc), {}),
                            "fail_reason": str(exc),
                            "num_turns": 0,
                        }
                    finally:
                        executor.close()

                    verdict: AgentLoopVerdict = episode["verdict"]
                    response = str(verdict.details.get("response") or "")
                    judge_reason = str(verdict.details.get("judge_reason") or verdict.fail_reason or "")
                    payload = SampleRecord(
                        benchmark_name=run.benchmark_name,
                        dataset_split=run.dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        stages=episode["stages"],
                        sampling_config=sampling_payload,
                    ).as_payload()
                    payload["agent_result"] = {
                        "reward": float(verdict.reward),
                        "num_turns": int(episode["num_turns"]),
                        "cost": 0.0,
                        "is_passed": bool(verdict.is_passed),
                        "error": episode["error"] or (None if verdict.is_passed else verdict.fail_reason or None),
                    }
                    payload["agent_info"] = {
                        "question": source_record.question,
                        "reference_answer": source_record.answer,
                        "response": response,
                        "final_answer": episode["final_answer"],
                        "judge_reason": judge_reason,
                        "fail_reason": episode["fail_reason"],
                        "termination_reason": episode["termination_reason"],
                        "locale": source_record.locale,
                        "cot_mode": CoTMode.NO_COT.value,
                        "topic": source_record.topic or "",
                        "agentic": True,
                    }
                    payload["agent_trace"] = episode["trace"]
                    payload["task_id"] = source_record.task_id
                    payload["domain"] = "function_call"
                    payload["instruction"] = source_record.question
                    writer.enqueue(payload)
            except Exception:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
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
                is_cot=False,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(records),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge.model,
                    "cot_mode": CoTMode.NO_COT.value,
                    "browsecomp_agentic": True,
                },
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp agentic done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


def _run_browsecomp(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_browsecomp_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("BrowseComp manifest is empty")
    if _browsecomp_agentic_enabled(args):
        return _run_browsecomp_agentic(args, run, records, run_context=run_context)

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
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
    answer_runtime_sampling = _browsecomp_answer_runtime_sampling(answer_sampling)

    batch_size = max(1, int(args.batch_size or 32))
    prompt_max_chars = int(args.prompt_max_chars or _BROWSECOMP_DEFAULT_PROMPT_MAX_CHARS)
    long_doc_config = _browsecomp_long_doc_config(args)
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
        raise ValueError("BrowseComp requires JUDGE_MODEL + JUDGE_API_KEY")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
        max_workers=resolve_judge_max_workers(getattr(args, "judge_max_workers", None), default=4),
        timeout_s=resolve_judge_timeout_s(default=60.0),
    )

    job_name = _resolve_job_name("function_browsecomp", run_context=run_context)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, cot_sampling), (2, answer_runtime_sampling)]),
        long_doc_config=long_doc_config,
        prompt_max_chars=prompt_max_chars,
    )
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
                    answer_long_doc_traces: list[dict[str, Any]] = []
                    for index, (_key, record) in enumerate(chunk):
                        cot_output = cot_by_index[index]
                        answer_prompt, answer_trace = build_browsecomp_budgeted_answer_prompt(
                            cot_prompts[index],
                            cot_output.text,
                            question=record.question,
                            locale=record.locale,
                            long_doc_config=long_doc_config,
                            prompt_max_chars=prompt_max_chars,
                            engine=run.engine,
                            sampling=answer_runtime_sampling,
                            prompt_seed=sample_repeat_seed(
                                _key.sample_index,
                                _key.repeat_index,
                                pass_index=_key.pass_index,
                                stage=10,
                            ),
                        )
                        answer_prompt_payload, prompt_delta_fallback = _answer_stage_prompt_payload(
                            answer_prompt,
                            f"{cot_output.prompt}{cot_output.text}",
                        )
                        answer_trace["prompt_delta_fallback"] = prompt_delta_fallback
                        answer_prompts.append(answer_prompt)
                        answer_stage_prompts.append(answer_prompt_payload)
                        answer_long_doc_traces.append(answer_trace)
                    answer_prompt_seeds = [
                        sample_repeat_seed(
                            key.sample_index,
                            key.repeat_index,
                            pass_index=key.pass_index,
                            stage=2,
                        )
                        for key, _record in chunk
                    ]
                    native_outputs, native_fallback_error = _generate_native_browsecomp_final_answers(
                        run.engine,
                        answer_prompts,
                        sampling=answer_runtime_sampling,
                        prompt_seeds=answer_prompt_seeds,
                    )
                    answer_by_index: dict[int, dict[str, Any]] = {}
                    if native_outputs is not None:
                        for index, (_key, record) in enumerate(chunk):
                            output = native_outputs[index]
                            row = _native_browsecomp_final_answer_row(output, locale=record.locale)
                            retry_errors: list[str] = []
                            retry_count = 0
                            while row.get("parse_error") and retry_count < _BROWSECOMP_NATIVE_TOOL_MAX_RETRIES:
                                retry_count += 1
                                retry_outputs, retry_error = _generate_native_browsecomp_final_answers(
                                    run.engine,
                                    [answer_prompts[index]],
                                    sampling=answer_runtime_sampling,
                                    prompt_seeds=[answer_prompt_seeds[index]],
                                )
                                if retry_outputs is None:
                                    if retry_error:
                                        retry_errors.append(retry_error)
                                    break
                                row = _native_browsecomp_final_answer_row(
                                    retry_outputs[0],
                                    locale=record.locale,
                                    retry_attempt=retry_count,
                                )
                                if row.get("parse_error"):
                                    retry_errors.append(str(row.get("parse_error") or "native retry parse failed"))
                            native_tool_info = dict(row.get("native_tool") or {})
                            native_tool_info["retry_count"] = retry_count
                            if retry_errors:
                                native_tool_info["retry_errors"] = retry_errors[-3:]
                            row["native_tool"] = native_tool_info
                            if row.get("parse_error"):
                                fallback_reason = str(row.get("parse_error") or "native final_answer parse failed")
                                try:
                                    fallback_outputs = run.engine.generate(
                                        [answer_prompts[index]],
                                        sampling=answer_runtime_sampling,
                                        batch_size=1,
                                        progress_desc="BrowseComp-Answer-Fallback",
                                        prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                                        prompt_seeds=[answer_prompt_seeds[index]],
                                        openai_sampling_compat=True,
                                    )
                                    fallback_output = fallback_outputs[0]
                                    row = {
                                        "response": None,
                                        "decoded_final_answer_call": None,
                                        "final_answer_call": "",
                                        "parse_error": "",
                                        "raw_completion": fallback_output.text,
                                        "finish_reason": fallback_output.finish_reason,
                                        "native_tool": {
                                            **native_tool_info,
                                            "fallback_to_text": True,
                                            "fallback_reason": fallback_reason,
                                        },
                                    }
                                except Exception as exc:  # noqa: BLE001 - keep native failure if text fallback also fails
                                    native_tool_info["fallback_to_text"] = False
                                    native_tool_info["fallback_error"] = f"{type(exc).__name__}: {exc}"
                                    row["native_tool"] = native_tool_info
                            answer_by_index[int(getattr(output, "prompt_index", index))] = row
                    else:
                        if native_fallback_error:
                            for trace in answer_long_doc_traces:
                                trace["native_tool_fallback_error"] = native_fallback_error
                        answer_outputs = run.engine.generate(
                            answer_prompts,
                            sampling=answer_runtime_sampling,
                            batch_size=len(answer_prompts),
                            progress_desc="BrowseComp-Answer",
                            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in answer_prompts],
                            prompt_seeds=answer_prompt_seeds,
                            openai_sampling_compat=True,
                        )
                        for output in answer_outputs:
                            answer_by_index[int(output.prompt_index)] = {
                                "response": None,
                                "decoded_final_answer_call": None,
                                "final_answer_call": "",
                                "parse_error": "",
                                "raw_completion": output.text,
                                "finish_reason": output.finish_reason,
                                "native_tool": {"enabled": False},
                            }
                    final_rows: list[dict[str, Any]] = []
                    judge_inputs: list[tuple[BrowseCompRecord, str]] = []
                    judge_positions: list[int] = []
                    judged: list[BrowseCompJudgeOutcome] = [
                        BrowseCompJudgeOutcome(is_passed=False, reason="browsecomp final answer not parsed")
                        for _ in chunk
                    ]
                    for index, (_key, record) in enumerate(chunk):
                        answer_row = answer_by_index[index]
                        parse_error = str(answer_row.get("parse_error") or "")
                        final_answer = str(answer_row.get("response") or "")
                        decoded_call = answer_row.get("decoded_final_answer_call")
                        if not isinstance(decoded_call, Mapping):
                            decoded_call = {}
                            try:
                                final_answer, decoded_call = decode_browsecomp_final_answer(
                                    str(answer_row.get("raw_completion") or ""),
                                    locale=record.locale,
                                )
                            except Exception as exc:  # noqa: BLE001
                                parse_error = str(exc)
                        final_call = str(answer_row.get("final_answer_call") or "")
                        if not final_call:
                            raw_arguments = decoded_call.get("arguments") if isinstance(decoded_call, Mapping) else {}
                            raw_answer = str(raw_arguments.get("answer") or "").strip() if isinstance(raw_arguments, Mapping) else ""
                            raw_call_id = (
                                str(decoded_call.get("id") or "").strip()
                                if isinstance(decoded_call, Mapping)
                                else ""
                            )
                            final_call = render_final_answer_call(raw_answer, call_id=raw_call_id) if raw_answer else ""
                        final_rows.append(
                            {
                                "response": final_answer,
                                "decoded_final_answer_call": decoded_call,
                                "final_answer_call": final_call,
                                "parse_error": parse_error,
                                "raw_completion": str(answer_row.get("raw_completion") or ""),
                                "finish_reason": str(answer_row.get("finish_reason") or "stop"),
                                "native_tool": dict(answer_row.get("native_tool") or {}),
                            }
                        )
                        if parse_error:
                            judged[index] = BrowseCompJudgeOutcome(is_passed=False, reason=parse_error)
                        else:
                            judge_positions.append(index)
                            judge_inputs.append((record, final_answer))
                    if judge_inputs:
                        for position, outcome in zip(judge_positions, judge_browsecomp_answers(judge_inputs, config=judge)):
                            judged[position] = outcome
                    for index, ((key, record), outcome) in enumerate(zip(chunk, judged)):
                        cot_output = cot_by_index[index]
                        final_row = final_rows[index]
                        final_answer = str(final_row["response"])
                        parse_error = str(final_row["parse_error"])
                        stages = [
                            StageRecord(
                                prompt=cot_prompts[index],
                                completion=cot_output.text,
                                stop_reason=cot_output.finish_reason,
                            ),
                            StageRecord(
                                prompt=answer_stage_prompts[index],
                                completion=str(final_row["raw_completion"]),
                                stop_reason=str(final_row["finish_reason"]),
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
                            "error": parse_error or None,
                        }
                        payload["agent_info"] = {
                            "question": record.question,
                            "reference_answer": record.answer,
                            "response": final_answer,
                            "judge_reason": outcome.reason,
                            "locale": record.locale,
                            "cot_mode": CoTMode.NO_COT.value,
                            "topic": record.topic or "",
                            "final_answer_call": final_row["final_answer_call"],
                            "decoded_final_answer_call": final_row["decoded_final_answer_call"],
                            "parse_error": parse_error,
                            "long_doc": answer_long_doc_traces[index],
                            "native_tool": final_row["native_tool"],
                        }
                        payload["agent_trace"] = [
                            {"stage": "cot", "text": cot_output.text},
                            {
                                "stage": "answer",
                                "text": final_answer,
                                "raw_completion": final_row["raw_completion"],
                                "sandbox_return": final_row["final_answer_call"],
                                "parse_error": parse_error,
                                "native_tool": final_row["native_tool"],
                                "long_doc": answer_long_doc_traces[index],
                            },
                        ]
                        payload["task_id"] = record.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = record.question
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
            completion_to_eval=_browsecomp_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=False,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(records),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge.model,
                    "cot_mode": CoTMode.NO_COT.value,
                },
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0
