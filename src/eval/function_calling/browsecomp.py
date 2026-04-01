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
from typing import Any, Iterable, Sequence


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
        return (
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
    return (
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
    return f"User: {user_prompt}\n\nAssistant: <think>"


def build_browsecomp_answer_prompt(expected_context: str, cot: str, *, locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "zh":
        suffix = (
            "现在继续补完最终答案，且严格使用如下格式：\n"
            "解释: <简短说明>\n"
            "最终答案: <简洁最终答案>\n"
            "置信度: <0% 到 100%>\n\n"
            "解释: "
        )
    else:
        suffix = (
            "Now continue by completing the final answer in this exact format:\n"
            "Explanation: <brief explanation>\n"
            "Exact Answer: <succinct final answer>\n"
            "Confidence: <0% to 100%>\n\n"
            "Explanation: "
        )
    return f"{expected_context}{cot}</think>\n{suffix}"


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
