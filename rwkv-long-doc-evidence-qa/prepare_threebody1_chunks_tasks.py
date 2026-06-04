#!/usr/bin/env python3
"""Prepare Three Body 1 chunks and structured QA tasks for RWKV7 reproduction.

This script takes the raw text file and generates the two JSONL files consumed
by reproduce_rwkv7_threebody1_final_answer.py:

    runs/threebody1_chunks_1000_overlap3.jsonl
    runs/threebody1_structured_task_candidates_chunks1000_overlap3.jsonl

Example:

    cd /home/codex/work/dev2
    /home/codex/miniconda3/bin/python \
      subprojects/rwkv7-long-context/examples/prepare_threebody1_chunks_tasks.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_INPUT_TXT = SCRIPT_PATH.parent / "三体1.txt"
DEFAULT_CHUNKS_JSONL = RUNS_DIR / "threebody1_chunks_1000_overlap3.jsonl"
DEFAULT_TASKS_JSONL = RUNS_DIR / "threebody1_structured_task_candidates_chunks1000_overlap3.jsonl"

TASK_DEFINITIONS_JSONL = r"""
{"id":"red_coast_launch_number","difficulty":"easy","answer_format":"scalar_number_string","question":"红岸工程这次常规发射是第几次？","answer":"147","positive_rule":{"all":["红岸工程第147次"]}}
{"id":"red_coast_target_category","difficulty":"easy","answer_format":"scalar_string","question":"红岸工程第147次常规发射的目标类别是什么？","answer":"甲三","positive_rule":{"all":["目标类别：甲三","红岸工程第147次"]}}
{"id":"red_coast_coordinate_code","difficulty":"easy","answer_format":"scalar_string","question":"红岸工程第147次常规发射的坐标序号是什么？","answer":"BN20197F","positive_rule":{"all":["坐标序号：BN20197F","红岸工程第147次"]}}
{"id":"red_coast_document_id","difficulty":"easy","answer_format":"scalar_number_string","question":"红岸工程第147次常规发射的发射文档号是多少？","answer":"22","positive_rule":{"all":["发射文档号：22","红岸工程第147次"]}}
{"id":"ozma_telescope_diameter_m","difficulty":"easy","answer_format":"scalar_number_string","question":"OZMA计划使用的射电望远镜直径是多少米？","answer":"26","positive_rule":{"all":["OZMA计划","26米直径"]}}
{"id":"ozma_frequency_mhz","difficulty":"easy","answer_format":"scalar_number_string","question":"OZMA计划的单通道接收频率是多少兆赫？","answer":"1420","positive_rule":{"all":["OZMA计划","1420兆赫"]}}
{"id":"ozma_search_duration_hours","difficulty":"easy","answer_format":"scalar_number_string","question":"OZMA计划的搜索时间约多少小时？","answer":"200","positive_rule":{"all":["OZMA计划","搜索时间约200小时"]}}
{"id":"red_coast_monitored_channels","difficulty":"easy","answer_format":"scalar_number_string","question":"文中提到某射电设施可同时监视多少个频道？","answer":"65000","positive_rule":{"all":["同时监视65000个频道"]}}
{"id":"countdown_remaining_after_photo","difficulty":"easy","answer_format":"scalar_number_string","question":"拍完胶卷最后一张时，倒计时还剩多少小时？","answer":"1194","positive_rule":{"all":["拍完胶卷最后一张","1194小时"]}}
{"id":"cosmic_background_countdown_remaining","difficulty":"medium","answer_format":"scalar_number_string","question":"宇宙尺度上的倒计时还剩多少小时？","answer":"1108","positive_rule":{"all":["宇宙尺度上继续","1108小时"]}}
{"id":"home_countdown_remaining","difficulty":"medium","answer_format":"scalar_number_string","question":"汪淼回家附近情节中，倒计时已减到多少小时？","answer":"1091","positive_rule":{"all":["倒计时已减到","1091小时"]}}
{"id":"civilization_cold_night_number","difficulty":"easy","answer_format":"scalar_number_string","question":"持续四十八年长夜后毁灭的是第几号文明？","answer":"137","positive_rule":{"all":["四十八年","第137号文明"]}}
{"id":"civilization_flame_number","difficulty":"easy","answer_format":"scalar_number_string","question":"在烈焰中毁灭的是第几号文明？","answer":"141","positive_rule":{"all":["第141号文明","烈焰中毁灭"]}}
{"id":"civilization_three_suns_in_sky_number","difficulty":"easy","answer_format":"scalar_number_string","question":"在“三日凌空”中毁灭的是第几号文明？","answer":"183","positive_rule":{"all":["183号文明","三日凌空"]}}
{"id":"civilization_flying_stars_static_number","difficulty":"medium","answer_format":"scalar_number_string","question":"“飞星不动”灾难对应的是第几号文明？","answer":"191","positive_rule":{"all":["191号文明","飞星不动"]}}
{"id":"civilization_dual_sun_number","difficulty":"easy","answer_format":"scalar_number_string","question":"在双日凌空的烈焰中毁灭的是第几号文明？","answer":"192","positive_rule":{"all":["192号文明","双日凌空"]}}
{"id":"civilization_192_level","difficulty":"medium","answer_format":"scalar_string","question":"第192号文明进化到什么时代？","answer":"原子和信息时代","positive_rule":{"all":["192号文明","原子和信息时代"]}}
{"id":"human_computer_os","difficulty":"easy","answer_format":"scalar_string","question":"三体世界的人列计算机运行的操作系统叫什么？","answer":"秦1.0","positive_rule":{"all":["人列计算机","秦1.0"]}}
{"id":"human_computer_software","difficulty":"easy","answer_format":"scalar_string","question":"在人列计算机上启动的太阳轨道计算软件叫什么？","answer":"Three-Body1.0","positive_rule":{"all":["太阳轨道计算软件","Three-Body1.0"]}}
{"id":"human_computer_external_storage_people","difficulty":"medium","answer_format":"scalar_string","question":"人列计算机的“硬盘”由多少名文化程度较高的人构成？","answer":"三百万","positive_rule":{"all":["硬盘","三百万名文化程度较高的人"]}}
{"id":"human_computer_min_people","difficulty":"medium","answer_format":"scalar_string","question":"牛顿等人说进行这种计算最少需要多少人？","answer":"三千万人","positive_rule":{"all":["最少要三千万人","数学的人海战术"]}}
{"id":"nanomaterial_codename","difficulty":"easy","answer_format":"scalar_string","question":"汪淼团队制造的超强度纳米材料代号叫什么？","answer":"飞刃","positive_rule":{"all":["飞刃","纳米材料"]}}
{"id":"guzheng_wire_spacing","difficulty":"easy","answer_format":"scalar_string","question":"古筝行动设想中，细丝间距大约是多少？","answer":"半米","positive_rule":{"all":["细丝","间距半米"]}}
{"id":"guzheng_target_ship","difficulty":"easy","answer_format":"scalar_string","question":"古筝行动要夺取信息的目标船叫什么？","answer":"审判日","positive_rule":{"all":["审判日","古筝行动"]}}
{"id":"second_red_coast_ship","difficulty":"medium","answer_format":"scalar_string","question":"第二红岸基地所在的巨轮叫什么？","answer":"审判日","positive_rule":{"all":["第二红岸基地","审判日"]}}
{"id":"first_alien_warning","difficulty":"easy","answer_format":"scalar_string","question":"来自另一个世界的第一条警告核心内容是什么？","answer":"不要回答","positive_rule":{"all":["不要回答"]}}
{"id":"first_alien_sender_identity","difficulty":"medium","answer_format":"scalar_string","question":"发出“不要回答”警告者自称是什么身份？","answer":"和平主义者","positive_rule":{"all":["和平主义者","不要回答"]}}
{"id":"eto_third_faction","difficulty":"easy","answer_format":"scalar_string","question":"三体叛军中后来出现的第三个派别叫什么？","answer":"幸存派","positive_rule":{"all":["第三个派别","幸存派"]}}
{"id":"eto_faction_destroy_humanity","difficulty":"medium","answer_format":"scalar_string","question":"三体叛军中想借助外星力量毁灭人类的是哪一派？","answer":"降临派","positive_rule":{"all":["降临派","毁灭人类"]}}
{"id":"eto_faction_worship_aliens","difficulty":"medium","answer_format":"scalar_string","question":"三体叛军中把外星文明当神来崇拜的是哪一派？","answer":"拯救派","positive_rule":{"all":["拯救派","外星文明当神来崇拜"]}}
{"id":"red_coast_launch_record_object","difficulty":"medium","answer_format":"json_object","question":"把红岸工程第147次常规发射的关键信息整理成JSON对象，包含launch_number、target_category、coordinate_code、document_id。","answer":"{\"coordinate_code\":\"BN20197F\",\"document_id\":\"22\",\"launch_number\":\"147\",\"target_category\":\"甲三\"}","positive_rule":{"all":["红岸工程第147次","目标类别：甲三","坐标序号：BN20197F","发射文档号：22"]}}
{"id":"red_coast_launch_record_array","difficulty":"medium","answer_format":"json_array","question":"按发射次数、目标类别、坐标序号、发射文档号的顺序，给出红岸工程第147次常规发射的信息数组。","answer":"[\"147\",\"甲三\",\"BN20197F\",\"22\"]","positive_rule":{"all":["红岸工程第147次","目标类别：甲三","坐标序号：BN20197F","发射文档号：22"]}}
{"id":"ozma_plan_object","difficulty":"medium","answer_format":"json_object","question":"把OZMA计划的望远镜直径、接收频率、搜索时间整理成JSON对象，key为diameter_m、frequency_mhz、duration_hours。","answer":"{\"diameter_m\":\"26\",\"duration_hours\":\"200\",\"frequency_mhz\":\"1420\"}","positive_rule":{"all":["OZMA计划","26米直径","1420兆赫","搜索时间约200小时"]}}
{"id":"ozma_plan_array","difficulty":"medium","answer_format":"json_array","question":"按望远镜直径米、接收频率兆赫、搜索时间小时的顺序，给出OZMA计划的信息数组。","answer":"[\"26\",\"1420\",\"200\"]","positive_rule":{"all":["OZMA计划","26米直径","1420兆赫","搜索时间约200小时"]}}
{"id":"human_computer_object","difficulty":"medium","answer_format":"json_object","question":"把人列计算机的操作系统和太阳轨道计算软件整理成JSON对象，key为os、software。","answer":"{\"os\":\"秦1.0\",\"software\":\"Three-Body1.0\"}","positive_rule":{"all":["人列计算机","秦1.0","Three-Body1.0"]}}
{"id":"human_computer_array","difficulty":"medium","answer_format":"json_array","question":"按操作系统、太阳轨道计算软件的顺序，给出人列计算机相关名称数组。","answer":"[\"秦1.0\",\"Three-Body1.0\"]","positive_rule":{"all":["人列计算机","秦1.0","Three-Body1.0"]}}
{"id":"civilization_192_object","difficulty":"medium","answer_format":"json_object","question":"把第192号文明的信息整理成JSON对象，包含civilization、disaster、level。","answer":"{\"civilization\":\"192\",\"disaster\":\"双日凌空\",\"level\":\"原子和信息时代\"}","positive_rule":{"all":["192号文明","双日凌空","原子和信息时代"]}}
{"id":"first_alien_warning_object","difficulty":"medium","answer_format":"json_object","question":"把第一条外星警告和发送者身份整理成JSON对象，key为warning、sender_identity。","answer":"{\"sender_identity\":\"和平主义者\",\"warning\":\"不要回答\"}","positive_rule":{"all":["不要回答","和平主义者"]}}
{"id":"first_alien_warning_array","difficulty":"medium","answer_format":"json_array","question":"按警告核心内容、发送者身份的顺序，给出第一条外星警告信息数组。","answer":"[\"不要回答\",\"和平主义者\"]","positive_rule":{"all":["不要回答","和平主义者"]}}
{"id":"nano_guzheng_object","difficulty":"medium","answer_format":"json_object","question":"把古筝行动相关的纳米材料代号和细丝间距整理成JSON对象，key为nanomaterial、wire_spacing。","answer":"{\"nanomaterial\":\"飞刃\",\"wire_spacing\":\"半米\"}","positive_rule":{"all":["飞刃","间距半米"]}}
{"id":"eto_two_factions_object","difficulty":"medium","answer_format":"json_object","question":"把三体叛军中毁灭人类和崇拜外星文明的两个派别整理成JSON对象，key为destroy_humanity、worship_aliens。","answer":"{\"destroy_humanity\":\"降临派\",\"worship_aliens\":\"拯救派\"}","positive_rule":{"all":["降临派","毁灭人类","拯救派","外星文明当神来崇拜"]}}
""".strip()


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    line_start: int
    line_end: int
    overlap_lines: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "char_count": len(self.text),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "overlap_lines": self.overlap_lines,
            "text": self.text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chunks and structured tasks for Three Body 1 reproduction.")
    parser.add_argument(
        "input_txt",
        nargs="?",
        default=str(DEFAULT_INPUT_TXT),
        help="Raw 三体1.txt path; defaults to examples/三体1.txt",
    )
    parser.add_argument("--chunks-jsonl", default=str(DEFAULT_CHUNKS_JSONL))
    parser.add_argument("--tasks-jsonl", default=str(DEFAULT_TASKS_JSONL))
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-lines", type=int, default=3)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def base_chunks(lines: list[tuple[int, str]], max_chars: int) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    current: list[tuple[int, str]] = []
    current_len = 0
    for line_no, line in lines:
        line_len = len(line)
        if line_len > max_chars:
            raise ValueError(f"line {line_no} has {line_len} chars > max_chars={max_chars}")
        if current and current_len + line_len > max_chars:
            chunks.append((current[0][0], current[-1][0], "".join(item[1] for item in current)))
            current = []
            current_len = 0
        current.append((line_no, line))
        current_len += line_len
    if current:
        chunks.append((current[0][0], current[-1][0], "".join(item[1] for item in current)))
    return chunks


def chunk_text_by_newline(text: str, max_chars: int, overlap_lines: int) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    normalized = normalize_newlines(text)
    lines = list(enumerate(normalized.splitlines(keepends=True), start=1))
    base = base_chunks(lines, max_chars)
    chunks: list[TextChunk] = []
    for idx, (line_start, line_end, chunk_text) in enumerate(base):
        effective_start = line_start
        emitted_text = chunk_text
        effective_overlap = 0
        if idx > 0 and overlap_lines:
            prev_start, prev_end, prev_text = base[idx - 1]
            tail = prev_text.splitlines(keepends=True)[-overlap_lines:]
            effective_start = max(prev_start, prev_end - len(tail) + 1)
            emitted_text = "".join(tail) + chunk_text
            effective_overlap = overlap_lines
        chunks.append(TextChunk(idx, emitted_text, effective_start, line_end, effective_overlap))
    return chunks


def load_task_definitions() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TASK_DEFINITIONS_JSONL.splitlines() if line.strip()]


def match_positive_rule(text: str, rule: dict[str, Any]) -> bool:
    if "all" in rule:
        return all(str(term) in text for term in rule["all"])
    if "any" in rule:
        return any(str(term) in text for term in rule["any"])
    if "not" in rule and isinstance(rule["not"], dict):
        return not match_positive_rule(text, rule["not"])
    raise ValueError(f"unsupported positive_rule: {rule!r}")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_chunks(input_name: str, text: str, chunks: list[TextChunk], max_chars: int, overlap_lines: int) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    normalized = normalize_newlines(text)
    line_count = normalized.count("\n") + (0 if not normalized or normalized.endswith("\n") else 1)
    return {
        "input": input_name,
        "max_chars": max_chars,
        "overlap_lines": overlap_lines,
        "total_chars": len(normalized),
        "line_count": line_count,
        "chunk_count": len(chunks),
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "avg_chunk_chars": (sum(lengths) / len(lengths)) if lengths else 0,
    }


def build_tasks(chunks: list[TextChunk], max_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    chunk_rows = [chunk.to_json() for chunk in chunks]
    output_tasks = []
    empty_task_ids = []
    for task in load_task_definitions():
        rule = task["positive_rule"]
        positive_chunks = [
            int(chunk["chunk_id"])
            for chunk in chunk_rows
            if match_positive_rule(str(chunk["text"]), rule)
        ]
        if not positive_chunks:
            empty_task_ids.append(str(task["id"]))
        output = {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "answer_format": task["answer_format"],
            "question": task["question"],
            "answer": task["answer"],
            "positive_chunks": positive_chunks,
            "positive_rule": rule,
            "null_rule": "chunk does not contain the positive_rule terms",
            "chunking": {
                "source": f"threebody1_chunks_{max_chars}_overlap{chunks[1].overlap_lines if len(chunks) > 1 else 0}",
                "max_chars": max_chars,
                "positive_chunks_recomputed_from": "positive_rule",
            },
        }
        output_tasks.append(output)
    return output_tasks, empty_task_ids


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_txt)
    text = input_path.read_text(encoding=args.encoding)
    chunks = chunk_text_by_newline(text, args.max_chars, args.overlap_lines)
    chunk_rows = [chunk.to_json() for chunk in chunks]
    tasks, empty_task_ids = build_tasks(chunks, args.max_chars)

    write_jsonl(args.chunks_jsonl, chunk_rows)
    write_jsonl(args.tasks_jsonl, tasks)

    summary = {
        "chunks": summarize_chunks(str(input_path), text, chunks, args.max_chars, args.overlap_lines),
        "tasks": len(tasks),
        "empty_positive_tasks": empty_task_ids,
        "chunks_jsonl": str(Path(args.chunks_jsonl).resolve()),
        "tasks_jsonl": str(Path(args.tasks_jsonl).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if empty_task_ids and not args.allow_empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
