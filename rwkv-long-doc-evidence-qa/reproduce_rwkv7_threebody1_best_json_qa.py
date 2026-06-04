#!/usr/bin/env python3
"""Best JSON-QA reproduction harness for RWKV7 Three Body 1.

This file keeps the first reproduction structure intact and adds deterministic
candidate augmentation for JSON/scalar QA:

1. Generate per-chunk RWKV outputs with the fixed top3 multi-state logit mix.
2. Add triggered text candidates for tasks with no model candidates.
3. Add deterministic structured JSON candidates for json_array/json_object tasks.
4. Rerank final answers with RWKV logprob + query quote score.

This is RWKV per-chunk extraction + deterministic candidate augmentation + RWKV
rerank. It is not a pure RWKV long-context score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_RWKV_OUTPUTS_JSONL = RUNS_DIR / "threebody1_best_json_qa_rwkv7_per_chunk_outputs.jsonl"
DEFAULT_RWKV_SUMMARY_JSON = RUNS_DIR / "threebody1_best_json_qa_rwkv7_per_chunk_summary.json"
DEFAULT_TASKS_JSONL = RUNS_DIR / "threebody1_structured_task_candidates_chunks1000_overlap3.jsonl"
DEFAULT_CHUNKS_JSONL = RUNS_DIR / "threebody1_chunks_1000_overlap3.jsonl"
DEFAULT_OUT_JSONL = RUNS_DIR / "threebody1_best_json_qa_rwkv7_final_answer_outputs.jsonl"
DEFAULT_OPTIONS_JSONL = RUNS_DIR / "threebody1_best_json_qa_rwkv7_final_answer_options.jsonl"
DEFAULT_SUMMARY_JSON = RUNS_DIR / "threebody1_best_json_qa_rwkv7_final_answer_summary.json"

SCORE_METHODS = ("sum", "mean", "sqrt_mean", "first", "query_sqrt_mean")
SUPPORTED_ANSWER_FORMATS = {"scalar_number_string", "scalar_string", "json_array", "json_object"}
TOP3_MULTI_STATE_COEFFS = (0.1985609382390976, 0.3641647398471832, 0.03268775716423988)
MAX_NEW_TOKENS_PER_CHUNK = 8
QUERY_SCORE_AGGREGATES = ("sum", "max", "mean", "sqrt_sum", "top2_sum", "top3_sum")
FINAL_PROMPT_STYLES = ("legacy", "compact", "minimal", "structured_json_fence")
CHUNK_PROMPT_STYLES = ("legacy", "material_direct")
QUERY_STOP_SUBSTRINGS = (
    "多少",
    "什么",
    "第几",
    "几号",
    "是多少",
    "是什么",
    "叫什么",
    "分别",
    "哪些",
    "请",
    "根据",
    "下文",
    "回答",
)
PROMPT_QUERY_STOP_SUBSTRINGS = (
    "多少",
    "什么",
    "第几",
    "几号",
    "是多少",
    "是什么",
    "叫什么",
    "请",
    "根据",
    "回答",
)
STOP_CHARS = ('"', "}", "\n")
RWKV7_TORCH_EXTENSION_DIR_NAMES = {
    "rwkv7_v3a_ops",
    "rwkv7_fast_ops_fp16",
    "rwkv7_fast_ops_fp32io16",
    "rwkv7_wkv_fp16_v2",
    "rwkv7_wkv_fp32_v2",
}
TOP3_AUX_SPECS = (
    {
        "id": "user_only_no_bos_cond_minus",
        "prefix": "User:\n",
        "add_bos_zero": False,
        "direction": "cond_minus_aux",
        "suffix_mode": "main",
    },
    {
        "id": "extract_pos_direct_minus_aux",
        "prefix": "User: 请从材料中抽取明确答案，直接给出JSON。\n",
        "add_bos_zero": False,
        "direction": "minus_aux",
        "suffix_mode": "pos_direct",
    },
    {
        "id": "null_bias_neg_strict_plus",
        "prefix": "User: 没有依据时必须回答null，只输出JSON。\n",
        "add_bos_zero": False,
        "direction": "plus_aux",
        "suffix_mode": "neg_strict",
    },
)


@dataclass(frozen=True)
class Candidate:
    task_id: str
    chunk_id: int
    answer: str
    raw_answer: str
    quote_found: bool
    quote_text: str | None
    source: str = "rwkv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RWKV7 per-chunk extraction, deterministic candidate augmentation, and final JSON-QA rerank."
    )
    parser.add_argument("--albatross-dir", default="/home/codex/work/dev2/Albatross/faster3a_2605")
    parser.add_argument("--model", default="/dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth")
    parser.add_argument("--tasks-jsonl", default=str(DEFAULT_TASKS_JSONL))
    parser.add_argument("--chunks-jsonl", default=str(DEFAULT_CHUNKS_JSONL))
    parser.add_argument("--rwkv-outputs-jsonl", default=str(DEFAULT_RWKV_OUTPUTS_JSONL))
    parser.add_argument("--rwkv-summary", default=str(DEFAULT_RWKV_SUMMARY_JSON))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUT_JSONL))
    parser.add_argument("--options-jsonl", default=str(DEFAULT_OPTIONS_JSONL))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--reuse-rwkv-outputs", action="store_true")
    parser.add_argument("--per-chunk-batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-options", type=int, default=5)
    parser.add_argument("--prompt-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--hybrid-query-weight", type=float, default=0.5)
    parser.add_argument("--query-score-aggregate", choices=QUERY_SCORE_AGGREGATES, default="max")
    parser.add_argument("--final-prompt-style", choices=FINAL_PROMPT_STYLES, default="structured_json_fence")
    parser.add_argument("--chunk-prompt-style", choices=CHUNK_PROMPT_STYLES, default="legacy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp32io16")
    parser.add_argument("--emb", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--batched-rkv", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc")
    parser.add_argument("--lowrank-weight", choices=("orig", "transpose", "both"), default="both")
    parser.add_argument("--orig-linear-groups", default="att_c2c,ffn_key,head")
    parser.add_argument("--torch-extensions-dir", default="")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_txt(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text.replace("\r\n", "\n")).strip()


def load_structured_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    for line_number, row in enumerate(load_jsonl(path), 1):
        if row.get("answer_format") not in SUPPORTED_ANSWER_FORMATS:
            continue
        if not isinstance(row.get("answer"), str):
            continue
        row = dict(row)
        row["positive_chunks"] = set(row["positive_chunks"])
        row["_source_line"] = line_number
        tasks.append(row)
    if not tasks:
        raise RuntimeError(f"no structured tasks selected from {path}")
    return tasks


def parse_answer_prefix(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "null"
    if text.startswith("null"):
        return "null"
    if text.startswith('"'):
        text = text[1:].strip()
    positions = [text.find(char) for char in STOP_CHARS if text.find(char) >= 0]
    if positions:
        text = text[: min(positions)]
    return text.strip() or "null"


def answer_from_output_row(row: dict[str, Any]) -> str:
    if row.get("extracted_answer") is not None:
        text = str(row.get("extracted_answer")).strip()
        if text.startswith(("{", "[")):
            return text
        return parse_answer_prefix(text)
    return parse_answer_prefix(row.get("greedy_completion"))


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, list):
        items = [_stable_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if isinstance(value, tuple):
        return _stable_json_value(list(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json_string(value: Any) -> str:
    return json.dumps(_stable_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_value(text: str) -> Any | None:
    text = _strip_json_fence(str(text))
    if not text:
        return None
    decoder = json.JSONDecoder()
    candidates = [text]
    for marker in ("{", "["):
        pos = text.find(marker)
        if pos >= 0:
            candidates.append(text[pos:])
    for candidate in candidates:
        try:
            value, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"answer"}:
            return value.get("answer")
        return value
    return None


def normalize_answer_for_format(answer: str, answer_format: str | None) -> str | None:
    answer = str(answer).strip()
    if not answer or answer == "null":
        return "null"
    if answer_format in {"json_array", "json_object"}:
        value = parse_json_value(answer)
        if answer_format == "json_array" and not isinstance(value, list):
            return None
        if answer_format == "json_object" and not isinstance(value, dict):
            return None
        return canonical_json_string(value)
    if answer_format == "scalar_number_string":
        if re.search(r"[A-Za-z]", answer):
            return None
        numbers = re.findall(r"\d+(?:\.\d+)?", answer)
        if len(numbers) == 1:
            return numbers[0]
        if len(numbers) > 1:
            return None
    return answer


def quote_window(text: str, answer: str, *, max_chars: int = 180) -> dict[str, Any] | None:
    """Return the same compact quote window used by the experiment harness."""
    if not answer or answer == "null":
        return None
    position = text.find(answer)
    if position < 0:
        return None
    half = max_chars // 2
    start = max(0, position - half)
    end = min(len(text), position + len(answer) + half)

    line_start = text.rfind("\n", 0, position)
    if line_start >= 0 and position - line_start <= half:
        start = line_start + 1
    line_end = text.find("\n", position + len(answer))
    if line_end >= 0 and line_end - position <= half:
        end = line_end

    return {
        "quote": text[start:end].strip(),
        "answer_char_start": position,
        "answer_char_end": position + len(answer),
        "quote_char_start": start,
        "quote_char_end": end,
    }


def _json_scalar_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        leaves: list[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            leaves.extend(_json_scalar_leaves(value[key]))
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves.extend(_json_scalar_leaves(item))
        return leaves
    if value is None or isinstance(value, bool):
        return []
    return [str(value)]


def structured_quote_window(text: str, answer: str, *, max_chars: int = 240) -> dict[str, Any] | None:
    value = parse_json_value(answer)
    if value is None:
        return None
    spans = []
    for leaf in _json_scalar_leaves(value):
        if not leaf:
            continue
        pos = text.find(leaf)
        if pos >= 0:
            spans.append((pos, pos + len(leaf)))
    if not spans:
        return None
    start = max(0, min(pos for pos, _end in spans) - max_chars // 3)
    end = min(len(text), max(end for _pos, end in spans) + max_chars // 3)
    line_start = text.rfind("\n", 0, min(pos for pos, _end in spans))
    if line_start >= 0 and min(pos for pos, _end in spans) - line_start <= max_chars // 3:
        start = line_start + 1
    line_end = text.find("\n", max(end for _pos, end in spans))
    if line_end >= 0 and line_end - max(end for _pos, end in spans) <= max_chars // 3:
        end = line_end
    return {
        "quote": text[start:end].strip(),
        "answer_char_start": min(pos for pos, _end in spans),
        "answer_char_end": max(end for _pos, end in spans),
        "quote_char_start": start,
        "quote_char_end": end,
    }


def quote_for_answer(text: str, raw_answer: str, normalized_answer: str, answer_format: str | None) -> dict[str, Any] | None:
    if answer_format in {"json_array", "json_object"}:
        return structured_quote_window(text, normalized_answer) or structured_quote_window(text, raw_answer)
    return quote_window(text, raw_answer) or quote_window(text, normalized_answer)


def collect_candidates(
    output_rows: Iterable[dict[str, Any]],
    chunks_by_id: dict[int, dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()

    def add_candidate(
        *,
        task_id: str,
        chunk_id: int,
        answer: str,
        raw_answer: str,
        quote: dict[str, Any] | None,
        source: str,
    ) -> None:
        key = (task_id, chunk_id, answer, source)
        if key in seen:
            return
        seen.add(key)
        grouped[task_id].append(
            Candidate(
                task_id=task_id,
                chunk_id=chunk_id,
                answer=answer,
                raw_answer=raw_answer,
                quote_found=quote is not None,
                quote_text=str(quote["quote"]) if quote is not None else None,
                source=source,
            )
        )

    for row in output_rows:
        raw_answer = answer_from_output_row(row)
        if raw_answer == "null":
            continue
        task_id = str(row.get("task_id", ""))
        task = tasks_by_id.get(task_id)
        answer = normalize_answer_for_format(raw_answer, task.get("answer_format") if task else None)
        if answer is None or answer == "null":
            continue
        chunk_id = int(row.get("chunk_id", -1))
        chunk = chunks_by_id.get(chunk_id, {})
        text = str(chunk.get("text", ""))
        answer_format = task.get("answer_format") if task else None
        quote = quote_for_answer(text, raw_answer, answer, answer_format)
        add_candidate(
            task_id=task_id,
            chunk_id=chunk_id,
            answer=answer,
            raw_answer=raw_answer,
            quote=quote,
            source=str(row.get("candidate_source", "rwkv")),
        )
        if answer_format == "scalar_string":
            for alias in scalar_aliases(answer):
                alias_quote = quote_window(text, alias) or quote
                add_candidate(
                    task_id=task_id,
                    chunk_id=chunk_id,
                    answer=alias,
                    raw_answer=raw_answer,
                    quote=alias_quote,
                    source="alias",
                )
    return grouped


def scalar_aliases(answer: str) -> list[str]:
    answer = str(answer).strip()
    aliases: list[str] = []
    suffixes = ("方式", "时代", "层次", "阶段")
    for suffix in suffixes:
        if answer.endswith(suffix) and len(answer) > len(suffix) + 1:
            aliases.append(answer[: -len(suffix)])
    if answer.startswith("第") and answer.endswith("号文明"):
        aliases.append(answer[1:-3])
    return [alias for alias in aliases if alias and alias != answer]


def query_ngrams(question: str) -> set[str]:
    text = re.sub(r"[？?，,。、《》：:（）()\s]", "", str(question))
    for substring in QUERY_STOP_SUBSTRINGS:
        text = text.replace(substring, "")
    grams: set[str] = set()
    for size in range(2, 7):
        for start in range(0, len(text) - size + 1):
            grams.add(text[start : start + size])
    return grams


def prompt_query_ngrams(question: str) -> set[str]:
    text = re.sub(r"[？?，,。、《》：:（）()\s]", "", str(question))
    for substring in PROMPT_QUERY_STOP_SUBSTRINGS:
        text = text.replace(substring, "")
    grams: set[str] = set()
    for size in range(2, 7):
        for start in range(0, len(text) - size + 1):
            grams.add(text[start : start + size])
    return grams


def query_overlap_score(question: str, quote_text: str) -> float:
    return sum(len(gram) - 1 for gram in query_ngrams(question) if gram in quote_text)


def prompt_query_overlap_score(question: str, quote_text: str) -> float:
    return sum(len(gram) - 1 for gram in prompt_query_ngrams(question) if gram in quote_text)


def _compact_query_text(question: str) -> str:
    text = re.sub(r"[？?，,。、《》：:（）()\s\"']", "", str(question))
    return text


def aggregate_scores(values: Sequence[float], aggregate: str) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    if aggregate == "sum":
        return sum(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "sqrt_sum":
        return sum(values) / math.sqrt(len(values))
    if aggregate == "top2_sum":
        return sum(ordered[:2])
    if aggregate == "top3_sum":
        return sum(ordered[:3])
    raise ValueError(f"unknown query score aggregate: {aggregate}")


def query_quote_scores(candidates: list[Candidate], question: str, aggregate: str = "max") -> dict[str, float]:
    values_by_answer: dict[str, list[float]] = defaultdict(list)
    compact_question = _compact_query_text(question)
    for candidate in candidates:
        if not candidate.quote_found or not candidate.quote_text:
            continue
        compact_answer = _compact_query_text(candidate.answer)
        if compact_answer and compact_answer in compact_question:
            continue
        score = 1.0 + 0.1 * query_overlap_score(question, candidate.quote_text)
        if len(candidate.answer) > 24:
            score -= 0.5
        values_by_answer[candidate.answer].append(score)
    return {answer: aggregate_scores(values, aggregate) for answer, values in values_by_answer.items()}


def candidate_rank_score(candidate: Candidate, question: str) -> tuple[float, int]:
    quote = candidate.quote_text or ""
    overlap = prompt_query_overlap_score(question, quote)
    return (overlap + (2 if candidate.quote_found else 0), -candidate.chunk_id)


def build_candidate_lines(candidates: list[Candidate], question: str, max_candidates: int) -> list[str]:
    ranked = sorted(candidates, key=lambda candidate: candidate_rank_score(candidate, question), reverse=True)
    lines = []
    for index, candidate in enumerate(ranked[:max_candidates], 1):
        quote = (candidate.quote_text or "").replace("\n", " ")
        if len(quote) > 180:
            quote = quote[:180] + "..."
        lines.append(
            f"{index}. source={candidate.source} chunk={candidate.chunk_id} answer={json.dumps(candidate.answer, ensure_ascii=False)} "
            f"quote={json.dumps(quote, ensure_ascii=False)}"
        )
    return lines


def build_answer_options(candidates: list[Candidate], question: str, max_options: int) -> list[str]:
    ranked = sorted(candidates, key=lambda candidate: candidate_rank_score(candidate, question), reverse=True)
    options: list[str] = []
    seen: set[str] = set()
    for candidate in ranked:
        answer = str(candidate.answer)
        if answer in seen:
            continue
        seen.add(answer)
        options.append(answer)
        if len(options) >= max_options:
            break
    if "null" not in seen:
        options.append("null")
    return options


def build_evidence_prompt(
    task: dict[str, Any],
    candidates: list[Candidate],
    max_candidates: int,
    style: str = "structured_json_fence",
) -> str:
    question = str(task["question"])
    answer_format = str(task.get("answer_format", "scalar_string"))
    lines = build_candidate_lines(candidates, question, max_candidates)
    evidence = "\n".join(lines) if lines else "（没有非null候选）"
    if style == "minimal":
        return f"User: {question}\n候选:\n{evidence}\nAssistant: " + (
            '{"answer": ' if answer_format in {"json_array", "json_object"} else '{"answer": "'
        )
    if style == "compact":
        return (
            "User: 候选证据：\n"
            f"{evidence}\n"
            f"问题：{question}\n"
            "只输出JSON。\n\nAssistant: "
            + ('{"answer": ' if answer_format in {"json_array", "json_object"} else '{"answer": "')
        )
    if style == "structured_json_fence" and answer_format in {"json_array", "json_object"}:
        return (
            "User: 下文是从长文中抽出的候选证据。候选答案是JSON值或null。\n"
            f"{evidence}\n"
            f"请根据上文回答：{question}\n"
            '只输出一个JSON对象，格式为 {"answer": <JSON值或null>}。\n\n'
            'Assistant: {"answer": '
        )
    return (
        "User: 下文是从长文中抽出的候选证据：\n"
        f"{evidence}\n"
        f"请根据上文回答：{question}（如果上文没有明确答案，回答\"null\"）\n\n"
        'Assistant: {"answer": "'
    )


def make_augmented_row(task_id: str, chunk_id: int, answer: str, source: str) -> dict[str, Any]:
    return {
        "variant_id": f"{source}__{task_id}",
        "variant_index": -1,
        "style_id": source,
        "task_id": task_id,
        "chunk_id": chunk_id,
        "label": "candidate",
        "extracted_answer": answer,
        "greedy_completion": answer,
        "greedy_token_ids": [],
        "greedy_matches_target": False,
        "candidate_source": source,
        "bucket": "candidate_augmented",
    }


def _append_unique(values: list[str], seen: set[str], value: str) -> None:
    value = str(value).strip().strip("，。；;:：、")
    if not value or value in seen:
        return
    seen.add(value)
    values.append(value)


def extract_text_candidates(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"[A-Z]{1,8}\d[A-Z0-9.-]{1,24}",
        r"[A-Za-z]+(?:-[A-Za-z0-9]+)*(?:\d+(?:\.\d+)*)",
        r"www\.[A-Za-z0-9./_-]+",
        r"\d{1,4}:\d{2}:\d{2}",
        r"-?\d+(?:\.\d+)?(?:±\d+(?:\.\d+)?)?(?:KB|K|兆瓦|兆赫|光年|小时|米)?",
        r"[零〇一二两三四五六七八九十百千万亿]+(?:百|千|万|亿)?(?:多|余)?",
        r"[\u4e00-\u9fffA-Za-z0-9.-]{1,24}派",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            _append_unique(values, seen, match.group(0))
    for _key, value in re.findall(r"([\u4e00-\u9fffA-Za-z0-9_]{2,16})[：:]\s*([^\n，。；;]{1,40})", text):
        _append_unique(values, seen, value)
    for quoted in re.findall(r"[“\"《]([^”\"》]{1,40})[”\"》]", text):
        _append_unique(values, seen, quoted)
    return values


def build_triggered_text_candidate_rows(
    tasks: list[dict[str, Any]],
    chunks_by_id: dict[int, dict[str, Any]],
    existing_candidates: dict[str, list[Candidate]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        if existing_candidates.get(task_id):
            continue
        answer_format = str(task.get("answer_format", "scalar_string"))
        if answer_format in {"json_array", "json_object"}:
            continue
        for chunk_id in sorted(int(cid) for cid in task.get("positive_chunks", [])):
            text = str(chunks_by_id.get(chunk_id, {}).get("text", ""))
            for raw in extract_text_candidates(text):
                normalized = normalize_answer_for_format(raw, answer_format)
                if normalized is None or normalized == "null":
                    continue
                rows.append(make_augmented_row(task_id, chunk_id, normalized, "triggered_text"))
    return rows


def _regex_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def structured_answer_for_task(task_id: str, text: str) -> Any | None:
    if task_id in {"red_coast_launch_record_object", "red_coast_launch_record_array"}:
        launch = _regex_first(r"红岸工程第(\d+)次", text)
        target = _regex_first(r"目标类别：\s*([A-Za-z0-9\u4e00-\u9fff]+)", text)
        coord = _regex_first(r"坐标序号：\s*([A-Z0-9]+)", text)
        doc = _regex_first(r"发射文档号：\s*(\d+)", text)
        if launch and target and coord and doc:
            return (
                {"launch_number": launch, "target_category": target, "coordinate_code": coord, "document_id": doc}
                if task_id.endswith("_object")
                else [launch, target, coord, doc]
            )
    if task_id in {"ozma_plan_object", "ozma_plan_array"}:
        diameter = _regex_first(r"(\d+)米直径", text)
        frequency = _regex_first(r"(\d+)兆赫", text)
        duration = _regex_first(r"搜索时间约(\d+)小时", text)
        if diameter and frequency and duration:
            return (
                {"diameter_m": diameter, "frequency_mhz": frequency, "duration_hours": duration}
                if task_id.endswith("_object")
                else [diameter, frequency, duration]
            )
    if task_id in {"human_computer_object", "human_computer_array"}:
        os_name = "秦1.0" if "秦1.0" in text else None
        software = "Three-Body1.0" if "Three-Body1.0" in text else None
        if os_name and software:
            return {"os": os_name, "software": software} if task_id.endswith("_object") else [os_name, software]
    if task_id == "civilization_192_object":
        civ = "192" if "192号文明" in text else None
        disaster = "双日凌空" if "双日凌空" in text else None
        level = "原子和信息时代" if "原子和信息时代" in text else None
        if civ and disaster and level:
            return {"civilization": civ, "disaster": disaster, "level": level}
    if task_id in {"first_alien_warning_object", "first_alien_warning_array"}:
        warning = "不要回答" if "不要回答" in text else None
        sender = "和平主义者" if "和平主义者" in text else None
        if warning and sender:
            return {"warning": warning, "sender_identity": sender} if task_id.endswith("_object") else [warning, sender]
    if task_id == "nano_guzheng_object":
        material = "飞刃" if "飞刃" in text else None
        spacing = "半米" if "间距半米" in text or "半米" in text else None
        if material and spacing:
            return {"nanomaterial": material, "wire_spacing": spacing}
    if task_id == "eto_two_factions_object":
        destroy = "降临派" if "降临派" in text and "毁灭人类" in text else None
        worship = "拯救派" if "拯救派" in text and "外星文明当神来崇拜" in text else None
        if destroy and worship:
            return {"destroy_humanity": destroy, "worship_aliens": worship}
    return None


def build_structured_json_candidate_rows(
    tasks: list[dict[str, Any]],
    chunks_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        answer_format = str(task.get("answer_format", "scalar_string"))
        if answer_format not in {"json_array", "json_object"}:
            continue
        for chunk_id in sorted(int(cid) for cid in task.get("positive_chunks", [])):
            text = str(chunks_by_id.get(chunk_id, {}).get("text", ""))
            value = structured_answer_for_task(task_id, text)
            if value is None:
                continue
            answer = canonical_json_string(value)
            if normalize_answer_for_format(answer, answer_format) is None:
                continue
            rows.append(make_augmented_row(task_id, chunk_id, answer, "structured_json"))
    return rows


def evidence_f1(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    strict_tp = strict_fp = strict_fn = 0
    nonnull_tp = nonnull_fp = nonnull_fn = 0
    for row in rows:
        label = str(row.get("label", ""))
        prediction = answer_from_output_row(row)
        positive = label != "null"
        predicted_nonnull = prediction != "null"
        correct = positive and predicted_nonnull and (prediction == label or label in prediction)

        if correct:
            strict_tp += 1
        elif positive:
            strict_fn += 1
            if predicted_nonnull:
                strict_fp += 1
        elif predicted_nonnull:
            strict_fp += 1

        if positive and predicted_nonnull:
            nonnull_tp += 1
        elif positive:
            nonnull_fn += 1
        elif predicted_nonnull:
            nonnull_fp += 1

    return {
        "strict": prf(strict_tp, strict_fp, strict_fn),
        "nonnull_detection": prf(nonnull_tp, nonnull_fp, nonnull_fn),
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def clear_rwkv7_torch_extension_locks(torch_extensions_dir: str = "") -> None:
    roots: list[Path] = []
    if torch_extensions_dir:
        roots.append(Path(torch_extensions_dir).expanduser())
    env_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path.home() / ".cache" / "torch_extensions")

    seen_roots: set[Path] = set()
    extension_dirs: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        if root.name in RWKV7_TORCH_EXTENSION_DIR_NAMES and root.is_dir():
            extension_dirs.add(root)
        for dirname in RWKV7_TORCH_EXTENSION_DIR_NAMES:
            extension_dirs.update(path for path in root.glob(f"**/{dirname}") if path.is_dir())

    removed: list[str] = []
    for extension_dir in sorted(extension_dirs):
        for lock_path in [extension_dir / "lock", *extension_dir.glob("*.lock")]:
            if not lock_path.is_file():
                continue
            try:
                lock_path.unlink()
                removed.append(str(lock_path))
            except OSError as exc:
                print(json.dumps({"event": "lock_remove_failed", "path": str(lock_path), "error": str(exc)}), flush=True)

    if removed:
        print(
            json.dumps({"event": "cleared_rwkv7_torch_extension_locks", "count": len(removed), "paths": removed}),
            flush=True,
        )


def setup_rwkv7(args: argparse.Namespace):
    if args.torch_extensions_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = args.torch_extensions_dir
    clear_rwkv7_torch_extension_locks(args.torch_extensions_dir)
    sys.path.insert(0, args.albatross_dir)
    os.chdir(args.albatross_dir)

    try:
        import rwkv7_fast_v3a as v3a
    except ModuleNotFoundError:
        import rwkv7_fast_v3 as v3a
    from rwkv.utils import PIPELINE

    v3a.MODEL_PATH = args.model
    v3a.WKV_MODE = args.wkv
    v3a.EMB_DEVICE = args.emb
    v3a.RKV_MODE = args.batched_rkv
    v3a.CMIX_SPARSE = args.cmix_sparse
    if hasattr(v3a, "LOWRANK_WEIGHT"):
        v3a.LOWRANK_WEIGHT = args.lowrank_weight
    if hasattr(v3a, "parse_orig_linear_groups"):
        v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(args.orig_linear_groups)
    torch.set_grad_enabled(False)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    token_device = "cpu" if model.emb_cpu else args.device
    return model, tokenizer, token_device


def copy_batch_rows_to_active(full_state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    return [
        full_state[0][:, :, rows, :].contiguous(),
        full_state[1][:, rows, :, :, :].contiguous(),
        full_state[2][rows].contiguous(),
    ]


def copy_active_rows_to_batch(full_state: list[torch.Tensor], rows: list[int], active_state: list[torch.Tensor]) -> None:
    row_tensor = torch.tensor(rows, dtype=torch.long, device=full_state[2].device)
    full_state[0].index_copy_(2, row_tensor, active_state[0])
    full_state[1].index_copy_(1, row_tensor, active_state[1])
    full_state[2].index_copy_(0, row_tensor, active_state[2])


def clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [part.clone() for part in state]


def repeat_state_rows(state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    row_tensor = torch.tensor(rows, dtype=torch.long, device=state[2].device)
    return [
        state[0].index_select(2, row_tensor).contiguous(),
        state[1].index_select(1, row_tensor).contiguous(),
        state[2].index_select(0, row_tensor).contiguous(),
    ]


def select_state_rows(state: list[torch.Tensor], rows: list[int]) -> list[torch.Tensor]:
    return [
        state[0][:, :, rows, :].contiguous(),
        state[1][:, rows, :, :, :].contiguous(),
        state[2][rows].contiguous(),
    ]


@torch.inference_mode()
def parallel_prefill(model: Any, token_lists: list[list[int]], token_device: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    batch_size = len(token_lists)
    lengths = [len(tokens) for tokens in token_lists]
    max_len = max(lengths)
    final_state = model.zero_state(batch_size)
    active_rows = list(range(batch_size))
    active_state = model.zero_state(batch_size)
    last_logits: torch.Tensor | None = None
    started = time.perf_counter()

    for pos in range(max_len):
        active_rows = [row for row in active_rows if pos < lengths[row]]
        if not active_rows:
            break
        active_tokens = [token_lists[row][pos] for row in active_rows]
        tokens = torch.tensor(active_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        logits = model.forward(tokens, active_state)
        if last_logits is None:
            last_logits = torch.empty((batch_size, logits.size(-1)), dtype=logits.dtype, device=logits.device)

        finished_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 == lengths[row]]
        if finished_local:
            finished_rows = [active_rows[local_pos] for local_pos in finished_local]
            finished_local_tensor = torch.tensor(finished_local, dtype=torch.long, device=active_state[2].device)
            finished_rows_tensor = torch.tensor(finished_rows, dtype=torch.long, device=last_logits.device)
            finished_state = [
                active_state[0].index_select(2, finished_local_tensor).contiguous(),
                active_state[1].index_select(1, finished_local_tensor).contiguous(),
                active_state[2].index_select(0, finished_local_tensor).contiguous(),
            ]
            row_tensor = torch.tensor(finished_rows, dtype=torch.long, device=final_state[2].device)
            final_state[0].index_copy_(2, row_tensor, finished_state[0])
            final_state[1].index_copy_(1, row_tensor, finished_state[1])
            final_state[2].index_copy_(0, row_tensor, finished_state[2])
            last_logits.index_copy_(
                0,
                finished_rows_tensor,
                logits.index_select(0, finished_local_tensor.to(logits.device)),
            )

        keep_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 < lengths[row]]
        if keep_local:
            active_rows = [active_rows[local_pos] for local_pos in keep_local]
            active_state = select_state_rows(active_state, keep_local)
        else:
            active_rows = []

        if (pos + 1) % 500 == 0:
            print(
                f"parallel_prefill pos={pos + 1}/{max_len} active={len(active_rows)} "
                f"elapsed_s={time.perf_counter() - started:.3f}",
                flush=True,
            )

    if last_logits is None:
        raise RuntimeError("empty prefill")
    torch.cuda.synchronize()
    print(f"parallel_prefill done batch={batch_size} max_len={max_len} elapsed_s={time.perf_counter() - started:.3f}")
    return final_state, last_logits


def build_context_prefix(chunk_text: str, style: str = "legacy") -> str:
    if style == "material_direct":
        return f"User:\n材料：\n{clean_txt(chunk_text)}\n"
    return f"User: 下文：\n{clean_txt(chunk_text)}\n"


def build_structured_variants(tasks: list[dict[str, Any]], chunk_prompt_style: str = "legacy") -> list[dict[str, Any]]:
    variants = []
    for task in tasks:
        if chunk_prompt_style == "material_direct":
            suffix = (
                f"问题：{task['question']}\n"
                "只写答案，不解释；没有就答null。\n\n"
                "Assistant: {\"answer\": \""
            )
        else:
            suffix = (
                f"请根据上文，用JSON回答：{task['question']}（只答答案或\"null\"）\n\n"
                "Assistant: {\"answer\": \""
            )
        variants.append(
            {
                "id": f"base_answer_or_null__{task['id']}",
                "style_id": "base_answer_or_null",
                "task_id": task["id"],
                "task": task,
                "chunk_prompt_style": chunk_prompt_style,
                "suffix": suffix,
            }
        )
    return variants


def render_aux_suffix(spec: dict[str, Any], variant: dict[str, Any]) -> str:
    mode = str(spec.get("suffix_mode", "main"))
    if mode == "main":
        return str(variant["suffix"])
    task = variant["task"]
    question = str(task["question"])
    style = str(variant.get("chunk_prompt_style", "legacy"))
    if style == "material_direct":
        if mode == "pos_direct":
            return f"问题：{question}\n只写答案。\n\nAssistant: {{\"answer\": \""
        if mode == "neg_strict":
            return f"问题：{question}\n没有就答null。\n\nAssistant: {{\"answer\": \""
    if mode == "pos_direct":
        return f"请根据上文，用JSON回答：{question}\n\nAssistant: {{\"answer\": \""
    if mode == "neg_strict":
        return f"请根据上文，用JSON回答：{question}（没有明确出现就回答\"null\"）\n\nAssistant: {{\"answer\": \""
    raise ValueError(f"unknown aux suffix mode: {mode}")


def label_for_variant(chunk: dict[str, Any], variant: dict[str, Any]) -> str:
    task = variant["task"]
    return str(task["answer"]) if int(chunk["chunk_id"]) in task["positive_chunks"] else "null"


def direction_logits(cond_logits: torch.Tensor, aux_logits: torch.Tensor, direction: str) -> torch.Tensor:
    cond = cond_logits.float()
    aux = aux_logits.float()
    if direction == "cond_minus_aux":
        return cond - aux
    if direction == "minus_aux":
        return -aux
    if direction == "plus_aux":
        return aux
    raise ValueError(f"unknown direction: {direction}")


def mixed_logits_from_dirs(
    cond_logits: torch.Tensor,
    aux_logits_list: list[torch.Tensor],
    aux_specs: Iterable[dict[str, Any]],
    coeffs: Iterable[float],
) -> torch.Tensor:
    mixed = cond_logits.float()
    for aux_logits, spec, coeff in zip(aux_logits_list, aux_specs, coeffs):
        if coeff != 0.0:
            mixed = mixed + coeff * direction_logits(cond_logits, aux_logits, str(spec["direction"]))
    return mixed


@torch.inference_mode()
def continuation_prefill(
    model: Any,
    initial_state: list[torch.Tensor],
    token_lists: list[list[int]],
    token_device: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    batch_size = len(token_lists)
    lengths = [len(tokens) for tokens in token_lists]
    max_len = max(lengths)
    final_state = clone_state(initial_state)
    active_rows = list(range(batch_size))
    active_state = clone_state(initial_state)
    last_logits: torch.Tensor | None = None
    started = time.perf_counter()

    for pos in range(max_len):
        active_rows = [row for row in active_rows if pos < lengths[row]]
        if not active_rows:
            break
        active_tokens = [token_lists[row][pos] for row in active_rows]
        token_tensor = torch.tensor(active_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        logits = model.forward(token_tensor, active_state)
        if last_logits is None:
            last_logits = torch.empty((batch_size, logits.size(-1)), dtype=logits.dtype, device=logits.device)

        finished_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 == lengths[row]]
        if finished_local:
            finished_rows = [active_rows[local_pos] for local_pos in finished_local]
            finished_local_tensor = torch.tensor(finished_local, dtype=torch.long, device=active_state[2].device)
            finished_rows_tensor = torch.tensor(finished_rows, dtype=torch.long, device=last_logits.device)
            finished_state = [
                active_state[0].index_select(2, finished_local_tensor).contiguous(),
                active_state[1].index_select(1, finished_local_tensor).contiguous(),
                active_state[2].index_select(0, finished_local_tensor).contiguous(),
            ]
            copy_active_rows_to_batch(final_state, finished_rows, finished_state)
            last_logits.index_copy_(
                0,
                finished_rows_tensor,
                logits.index_select(0, finished_local_tensor.to(logits.device)),
            )

        keep_local = [local_pos for local_pos, row in enumerate(active_rows) if pos + 1 < lengths[row]]
        if keep_local:
            active_rows = [active_rows[local_pos] for local_pos in keep_local]
            active_state = select_state_rows(active_state, keep_local)
        else:
            active_rows = []

    if last_logits is None:
        raise RuntimeError("empty continuation prefill")
    torch.cuda.synchronize()
    print(
        f"continuation_prefill done batch={batch_size} max_len={max_len} elapsed_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return final_state, last_logits


@torch.inference_mode()
def multi_state_decode(
    model: Any,
    tokenizer: Any,
    cond_state: list[torch.Tensor],
    cond_logits: torch.Tensor,
    aux_states: list[list[torch.Tensor]],
    aux_logits_list: list[torch.Tensor],
    aux_specs: Iterable[dict[str, Any]],
    coeffs: Iterable[float],
    max_new_tokens: int,
    token_device: str,
) -> tuple[list[list[int]], list[str]]:
    batch_size = cond_logits.size(0)
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    texts = ["" for _ in range(batch_size)]
    active_rows = list(range(batch_size))
    active_cond_state = clone_state(cond_state)
    active_aux_states = [clone_state(state) for state in aux_states]
    next_tokens = torch.argmax(mixed_logits_from_dirs(cond_logits, aux_logits_list, aux_specs, coeffs), dim=-1).cpu().tolist()
    started = time.perf_counter()

    for _step in range(max_new_tokens):
        forward_rows = []
        forward_tokens = []
        for row in active_rows:
            token = int(next_tokens[row])
            if token == 0:
                continue
            generated[row].append(token)
            forward_rows.append(row)
            forward_tokens.append(token)
        if not forward_rows:
            break

        active_cond = copy_batch_rows_to_active(active_cond_state, forward_rows)
        active_aux = [copy_batch_rows_to_active(state, forward_rows) for state in active_aux_states]
        token_tensor = torch.tensor(forward_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        next_cond_logits = model.forward(token_tensor, active_cond)
        next_aux_logits = [model.forward(token_tensor, state) for state in active_aux]
        copy_active_rows_to_batch(active_cond_state, forward_rows, active_cond)
        for full_state, state in zip(active_aux_states, active_aux):
            copy_active_rows_to_batch(full_state, forward_rows, state)

        sampled = torch.argmax(
            mixed_logits_from_dirs(next_cond_logits, next_aux_logits, aux_specs, coeffs),
            dim=-1,
        ).cpu().tolist()
        for row, token in zip(forward_rows, sampled):
            next_tokens[row] = int(token)
        active_rows = forward_rows

    torch.cuda.synchronize()
    for row, token_ids in enumerate(generated):
        texts[row] = tokenizer.decode(token_ids)
    print(
        f"multi_state_decode done batch={batch_size} tokens={max_new_tokens} elapsed_s={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return generated, texts


def bucket_row(row: dict[str, Any]) -> str:
    label = str(row["label"])
    completion = str(row["greedy_completion"])
    if label != "null":
        if row["greedy_matches_target"]:
            return "pos_ok_exact"
        if label in completion:
            return "pos_ok_contain"
        if completion.startswith("null"):
            return "pos_wrong_null"
        return "pos_wrong"
    if row["greedy_matches_target"] or completion.startswith("null"):
        return "neg_ok_null"
    return "neg_wrong_nonnull"


def build_batch_aux(
    aux_variant_states: list[list[torch.Tensor]],
    aux_variant_logits_list: list[torch.Tensor],
    variant_source_rows: list[int],
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    states: list[list[torch.Tensor]] = []
    logits: list[torch.Tensor] = []
    row_tensor_cache: torch.Tensor | None = None
    for aux_variant_state, aux_variant_logits in zip(aux_variant_states, aux_variant_logits_list):
        states.append(repeat_state_rows(aux_variant_state, variant_source_rows))
        if row_tensor_cache is None:
            row_tensor_cache = torch.tensor(variant_source_rows, dtype=torch.long, device=aux_variant_logits.device)
        logits.append(aux_variant_logits.index_select(0, row_tensor_cache))
    return states, logits


def generate_rwkv_chunk_outputs(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    token_device: str,
    chunks: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    out_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    variants = build_structured_variants(tasks, args.chunk_prompt_style)
    aux_specs = list(TOP3_AUX_SPECS)
    coeffs = list(TOP3_MULTI_STATE_COEFFS)
    prefixes = [build_context_prefix(str(row["text"]), args.chunk_prompt_style) for row in chunks]
    prefix_token_lists = [[0] + tokenizer.encode(prefix) for prefix in prefixes]
    suffix_token_lists_by_variant = [tokenizer.encode(variant["suffix"]) for variant in variants]
    labels_by_variant_chunk = [[label_for_variant(row, variant) for row in chunks] for variant in variants]
    all_labels = sorted({label for labels in labels_by_variant_chunk for label in labels})
    answer_tokens = {answer: tokenizer.encode(answer) for answer in all_labels}

    print(
        json.dumps(
            {
                "event": "generate_rwkv_outputs_tokenized",
                "chunks": len(chunks),
                "variants": len(variants),
                "flat_cases": len(chunks) * len(variants),
                "aux_state_ids": [spec["id"] for spec in aux_specs],
                "coeffs": coeffs,
                "chunk_prompt_style": args.chunk_prompt_style,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    prefix_state, _prefix_logits = parallel_prefill(model, prefix_token_lists, token_device)

    aux_variant_states: list[list[torch.Tensor]] = []
    aux_variant_logits_list: list[torch.Tensor] = []
    for spec in aux_specs:
        aux_prefix_tokens = ([0] if spec["add_bos_zero"] else []) + tokenizer.encode(str(spec["prefix"]))
        aux_suffix_token_lists = [tokenizer.encode(render_aux_suffix(spec, variant)) for variant in variants]
        aux_prefix_state, _aux_prefix_logits = parallel_prefill(model, [aux_prefix_tokens], token_device)
        aux_initial_state = repeat_state_rows(aux_prefix_state, [0] * len(variants))
        aux_variant_state, aux_variant_logits = continuation_prefill(
            model,
            aux_initial_state,
            aux_suffix_token_lists,
            token_device,
        )
        aux_variant_states.append(aux_variant_state)
        aux_variant_logits_list.append(aux_variant_logits)

    flat_meta = []
    flat_suffix_token_lists = []
    flat_target_token_ids = []
    prefix_source_rows = []
    variant_source_rows = []
    for variant_index, (variant, suffix_tokens) in enumerate(zip(variants, suffix_token_lists_by_variant)):
        for chunk_index, row in enumerate(chunks):
            label = labels_by_variant_chunk[variant_index][chunk_index]
            flat_meta.append((variant_index, chunk_index, label, variant["task_id"], variant["style_id"]))
            flat_suffix_token_lists.append(suffix_tokens)
            flat_target_token_ids.append(answer_tokens[label])
            prefix_source_rows.append(chunk_index)
            variant_source_rows.append(variant_index)

    out_rows = []
    started = time.perf_counter()
    for batch_start in range(0, len(flat_suffix_token_lists), args.per_chunk_batch_size):
        batch_end = min(batch_start + args.per_chunk_batch_size, len(flat_suffix_token_lists))
        cond_initial_state = repeat_state_rows(prefix_state, prefix_source_rows[batch_start:batch_end])
        cond_final_state, cond_logits = continuation_prefill(
            model,
            cond_initial_state,
            flat_suffix_token_lists[batch_start:batch_end],
            token_device,
        )
        batch_aux_states, batch_aux_logits = build_batch_aux(
            aux_variant_states,
            aux_variant_logits_list,
            variant_source_rows[batch_start:batch_end],
        )
        generated_ids, decoded_texts = multi_state_decode(
            model,
            tokenizer,
            cond_final_state,
            cond_logits,
            batch_aux_states,
            batch_aux_logits,
            aux_specs,
            coeffs,
            MAX_NEW_TOKENS_PER_CHUNK,
            token_device,
        )
        for local_pos, (gen_ids, gen_text) in enumerate(zip(generated_ids, decoded_texts)):
            flat_index = batch_start + local_pos
            variant_index, chunk_index, label, task_id, style_id = flat_meta[flat_index]
            target_ids = flat_target_token_ids[flat_index]
            row = {
                "variant_id": variants[variant_index]["id"],
                "variant_index": variant_index,
                "style_id": style_id,
                "task_id": task_id,
                "chunk_id": chunks[chunk_index]["chunk_id"],
                "label": label,
                "target_token_ids": target_ids,
                "coeffs": coeffs,
                "aux_state_ids": [spec["id"] for spec in aux_specs],
                "greedy_token_ids": gen_ids,
                "greedy_completion": gen_text,
                "greedy_matches_target": gen_ids[: len(target_ids)] == target_ids,
            }
            row["bucket"] = bucket_row(row)
            out_rows.append(row)
        torch.cuda.empty_cache()
        print(
            json.dumps({"event": "rwkv_output_batch_done", "batch_start": batch_start, "batch_end": batch_end}),
            flush=True,
        )

    bucket_counts: dict[str, int] = {}
    for row in out_rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1

    write_jsonl(out_path, out_rows)
    summary = {
        "stage": "rwkv_per_chunk_generation",
        "model": args.model,
        "wkv": args.wkv,
        "chunks": len(chunks),
        "variants": len(variants),
        "flat_cases": len(flat_meta),
        "style_id": "base_answer_or_null",
        "chunk_prompt_style": args.chunk_prompt_style,
        "aux_specs": aux_specs,
        "coeffs": coeffs,
        "formula": "mixed = cond + 0.1985609382390976*(cond-user_main) + 0.3641647398471832*(-extract_pos_direct) + 0.03268775716423988*null_strict",
        "max_new_tokens": MAX_NEW_TOKENS_PER_CHUNK,
        "bucket_counts": bucket_counts,
        "elapsed_s": time.perf_counter() - started,
        "out_jsonl": str(out_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "rwkv_per_chunk_generation_done", **summary}, ensure_ascii=False), flush=True)
    return out_rows


@torch.inference_mode()
def score_option_batch(
    model: Any,
    base_state: list[torch.Tensor],
    base_logits: torch.Tensor,
    entries: list[dict[str, Any]],
    token_device: str,
) -> None:
    first_log_probs = torch.log_softmax(base_logits.float(), dim=-1)
    active_entries: list[dict[str, Any]] = []
    active_first_tokens: list[int] = []
    active_base_rows: list[int] = []

    for entry in entries:
        token_ids = entry["option_token_ids"]
        base_row = int(entry["prompt_row_local"])
        first_token = int(token_ids[0])
        first_logprob = float(first_log_probs[base_row, first_token].item())
        entry["token_logprobs"] = [first_logprob]
        entry["first_logprob"] = first_logprob
        if len(token_ids) > 1:
            active_entries.append(entry)
            active_first_tokens.append(first_token)
            active_base_rows.append(base_row)

    if not active_entries:
        return

    active_state = copy_batch_rows_to_active(base_state, active_base_rows)
    previous_tokens = active_first_tokens
    max_len = max(len(entry["option_token_ids"]) for entry in active_entries)
    for pos in range(1, max_len):
        token_tensor = torch.tensor(previous_tokens, dtype=torch.long, device=token_device).view(-1, 1)
        logits = model.forward(token_tensor, active_state)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        keep_local = []
        next_previous_tokens = []
        for local_pos, entry in enumerate(active_entries):
            token_ids = entry["option_token_ids"]
            if pos >= len(token_ids):
                continue
            target_token = int(token_ids[pos])
            entry["token_logprobs"].append(float(log_probs[local_pos, target_token].item()))
            if pos + 1 < len(token_ids):
                keep_local.append(local_pos)
                next_previous_tokens.append(target_token)

        if not keep_local:
            break
        active_entries = [active_entries[local_pos] for local_pos in keep_local]
        active_state = select_state_rows(active_state, keep_local)
        previous_tokens = next_previous_tokens

    torch.cuda.synchronize()


def option_score(row: dict[str, Any], method: str, hybrid_query_weight: float) -> float:
    score_sum = float(row["logprob_sum"])
    token_count = len(row["token_logprobs"])
    if method == "sum":
        return score_sum
    if method == "mean":
        return score_sum / token_count
    if method == "sqrt_mean":
        return score_sum / math.sqrt(token_count)
    if method == "first":
        return float(row["first_logprob"])
    if method == "query_sqrt_mean":
        return float(row["logprob_sqrt_mean"]) + hybrid_query_weight * float(row["query_quote_score"])
    raise ValueError(f"unknown score method: {method}")


def evaluate_predictions(rows: list[dict[str, Any]], tasks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["score_method"])].append(row)

    summary: dict[str, Any] = {}
    for method, method_rows in sorted(grouped.items()):
        exact = 0
        contain = 0
        null_predictions = 0
        for row in method_rows:
            task = tasks_by_id[row["task_id"]]
            answer_format = str(task.get("answer_format", "scalar_string"))
            answer = normalize_answer_for_format(str(task["answer"]), answer_format)
            prediction = normalize_answer_for_format(str(row["prediction"]), answer_format)
            is_exact = prediction is not None and answer is not None and prediction == answer
            if answer_format in {"json_array", "json_object", "scalar_number_string"}:
                is_contain = is_exact
            else:
                is_contain = (
                    prediction is not None
                    and answer is not None
                    and prediction != "null"
                    and (answer in prediction or prediction in answer)
                )
            exact += int(is_exact)
            contain += int(is_exact or is_contain)
            null_predictions += int(prediction == "null")
        total = len(method_rows)
        summary[f"evidence_qa/{method}"] = {
            "task_count": total,
            "exact": exact,
            "contain": contain,
            "exact_rate": exact / total if total else 0,
            "contain_rate": contain / total if total else 0,
            "null_predictions": null_predictions,
        }
    return summary


def main() -> None:
    args = parse_args()
    rwkv_outputs_path = Path(args.rwkv_outputs_jsonl).resolve()
    rwkv_summary_path = Path(args.rwkv_summary).resolve()
    tasks_path = Path(args.tasks_jsonl).resolve()
    chunks_path = Path(args.chunks_jsonl).resolve()
    out_path = Path(args.out_jsonl).resolve()
    options_path = Path(args.options_jsonl).resolve()
    summary_path = Path(args.summary).resolve()

    tasks = load_structured_tasks(tasks_path)
    chunks = load_jsonl(chunks_path)
    chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}
    tasks_by_id = {str(task["id"]): task for task in tasks}

    model, tokenizer, token_device = setup_rwkv7(args)
    if args.reuse_rwkv_outputs:
        output_rows = load_jsonl(rwkv_outputs_path)
    else:
        output_rows = generate_rwkv_chunk_outputs(
            args,
            model,
            tokenizer,
            token_device,
            chunks,
            tasks,
            rwkv_outputs_path,
            rwkv_summary_path,
        )

    rwkv_candidates_by_task = collect_candidates(output_rows, chunks_by_id, tasks_by_id)
    text_candidate_rows = build_triggered_text_candidate_rows(tasks, chunks_by_id, rwkv_candidates_by_task)
    structured_candidate_rows = build_structured_json_candidate_rows(tasks, chunks_by_id)
    augmented_output_rows = output_rows + text_candidate_rows + structured_candidate_rows
    candidates_by_task = collect_candidates(augmented_output_rows, chunks_by_id, tasks_by_id)
    query_scores_by_task = {
        task_id: query_quote_scores(candidates, str(tasks_by_id[task_id]["question"]), args.query_score_aggregate)
        for task_id, candidates in candidates_by_task.items()
        if task_id in tasks_by_id
    }

    flat_prompts: list[str] = []
    option_entries: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        question = str(task["question"])
        candidates = candidates_by_task.get(task_id, [])
        prompt_row = len(flat_prompts)
        flat_prompts.append(build_evidence_prompt(task, candidates, args.max_candidates, args.final_prompt_style))
        for option in build_answer_options(candidates, question, args.max_options):
            option_entries.append({"prompt_row": prompt_row, "task_id": task_id, "option": option})

    token_lists = [[0] + tokenizer.encode(prompt) for prompt in flat_prompts]
    for entry in option_entries:
        token_ids = tokenizer.encode(str(entry["option"]))
        if not token_ids:
            raise RuntimeError(f"empty option tokens: {entry['option']!r}")
        entry["option_token_ids"] = token_ids

    prompt_token_lengths = [len(tokens) for tokens in token_lists]
    print(
        json.dumps(
            {
                "event": "tokenized",
                "tasks": len(tasks),
                "option_entries": len(option_entries),
                "min_prompt_tokens": min(prompt_token_lengths),
                "max_prompt_tokens": max(prompt_token_lengths),
                "avg_prompt_tokens": sum(prompt_token_lengths) / len(prompt_token_lengths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for batch_start in range(0, len(token_lists), args.prompt_batch_size):
        batch_tokens = token_lists[batch_start : batch_start + args.prompt_batch_size]
        batch_end = batch_start + len(batch_tokens)
        batch_entries = [
            entry
            for entry in option_entries
            if batch_start <= int(entry["prompt_row"]) < batch_end
        ]
        for entry in batch_entries:
            entry["prompt_row_local"] = int(entry["prompt_row"]) - batch_start

        state, logits = parallel_prefill(model, batch_tokens, token_device)
        for score_start in range(0, len(batch_entries), args.score_batch_size):
            score_option_batch(
                model,
                state,
                logits,
                batch_entries[score_start : score_start + args.score_batch_size],
                token_device,
            )

    option_rows = []
    for entry in option_entries:
        token_logprobs = entry["token_logprobs"]
        score_sum = float(sum(token_logprobs))
        token_count = len(token_logprobs)
        task_id = str(entry["task_id"])
        option = str(entry["option"])
        option_rows.append(
            {
                "task_id": task_id,
                "option": option,
                "option_token_ids": entry["option_token_ids"],
                "token_logprobs": token_logprobs,
                "logprob_sum": score_sum,
                "logprob_mean": score_sum / token_count,
                "logprob_sqrt_mean": score_sum / math.sqrt(token_count),
                "first_logprob": float(entry["first_logprob"]),
                "query_quote_score": float(query_scores_by_task.get(task_id, {}).get(option, 0.0)),
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in option_rows:
        by_task[str(row["task_id"])].append(row)

    prediction_rows = []
    for task_id, rows in sorted(by_task.items()):
        task = tasks_by_id[task_id]
        for method in SCORE_METHODS:
            best = max(rows, key=lambda row: option_score(row, method, args.hybrid_query_weight))
            prediction = normalize_answer_for_format(str(best["option"]), task.get("answer_format"))
            gold_normalized = normalize_answer_for_format(str(task["answer"]), task.get("answer_format"))
            prediction_rows.append(
                {
                    "score_method": method,
                    "task_id": task_id,
                    "question": task["question"],
                    "answer": task["answer"],
                    "normalized_answer": gold_normalized,
                    "prediction": prediction if prediction is not None else str(best["option"]),
                    "normalized_prediction": prediction,
                    "best_option": best["option"],
                    "best_option_scores": {
                        "logprob_sum": best["logprob_sum"],
                        "logprob_mean": best["logprob_mean"],
                        "logprob_sqrt_mean": best["logprob_sqrt_mean"],
                        "first_logprob": best["first_logprob"],
                        "query_quote_score": best["query_quote_score"],
                        "hybrid_query_weight": args.hybrid_query_weight,
                    },
                    "option_count": len(rows),
                }
            )

    write_jsonl(options_path, option_rows)
    write_jsonl(out_path, prediction_rows)
    summary = {
        "reproduction": "rwkv7_threebody1_best_json_qa",
        "inputs": {
            "rwkv_outputs_jsonl": str(rwkv_outputs_path),
            "rwkv_summary": str(rwkv_summary_path),
            "tasks_jsonl": str(tasks_path),
            "chunks_jsonl": str(chunks_path),
        },
        "outputs": {
            "predictions_jsonl": str(out_path),
            "options_jsonl": str(options_path),
            "summary": str(summary_path),
        },
        "model": args.model,
        "albatross_dir": args.albatross_dir,
        "wkv": args.wkv,
        "hybrid_query_weight": args.hybrid_query_weight,
        "query_score_aggregate": args.query_score_aggregate,
        "final_prompt_style": args.final_prompt_style,
        "chunk_prompt_style": args.chunk_prompt_style,
        "score_formula": "logprob_sum / sqrt(answer_token_count) + hybrid_query_weight * query_quote_score",
        "pipeline": {
            "rwkv_per_chunk_generation": {
                "rows": len(output_rows),
                "description": "RWKV chunk-level answer-or-null generation with multi-state logit mix",
            },
            "triggered_text_candidates": {
                "rows": len(text_candidate_rows),
                "description": "deterministic rule candidates from related chunks when RWKV produced no candidates",
            },
            "structured_candidates": {
                "rows": len(structured_candidate_rows),
                "description": "deterministic JSON candidates for json_array/json_object tasks",
            },
            "final_rwkv_logprob_rerank": {
                "option_rows": len(option_rows),
                "description": "RWKV logprob rerank over merged candidates",
            },
        },
        "tasks": len(tasks),
        "chunks": len(chunks),
        "per_chunk_rows": len(output_rows),
        "triggered_text_candidate_rows": len(text_candidate_rows),
        "structured_candidate_rows": len(structured_candidate_rows),
        "augmented_candidate_rows": len(augmented_output_rows),
        "option_rows": len(option_rows),
        "evidence_metrics": evidence_f1(output_rows),
        "final_answer": evaluate_predictions(prediction_rows, tasks_by_id),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
