"""Probe BrowseComp-Plus answer synthesis using evidence frozen in a DB task."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import psycopg
from psycopg.rows import dict_row


DEFAULT_DATASET = (
    Path(os.environ.get("BROWSECOMP_PLUS_ROOT", "/path/to/BrowseComp-Plus"))
    / "data"
    / "browsecomp_plus_decrypted.jsonl"
)
EXACT_ANSWER_RE = re.compile(r"(?im)^\s*Exact Answer\s*:\s*(.+?)\s*$")
SEARCH_QUERY_RE = re.compile(r"(?im)^\s*Search Query\s*:\s*(.+?)\s*$")


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _conninfo() -> str:
    return " ".join(
        (
            f"host={os.environ.get('PG_HOST', '127.0.0.1')}",
            f"port={os.environ.get('PG_PORT', '5432')}",
            f"dbname={os.environ.get('PG_DBNAME', 'chase_rwkv_skills')}",
            f"user={os.environ.get('PG_USER', 'postgres')}",
            f"password={os.environ.get('PG_PASSWORD', '')}",
        )
    )


def _load_task_rows(task_id: int) -> list[dict[str, Any]]:
    with psycopg.connect(_conninfo(), row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT sample_index, context
            FROM completions
            WHERE task_id = %s
            ORDER BY sample_index, avg_repeat_index, pass_index
            """,
            (task_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _document_ids(record: dict[str, Any]) -> set[str]:
    docids: set[str] = set()
    for key in ("gold_docs", "evidence_docs"):
        values = record.get(key)
        if not isinstance(values, list):
            continue
        docids.update(
            str(item.get("docid") or item.get("id") or "").strip()
            for item in values
            if isinstance(item, dict)
        )
    docids.discard("")
    return docids


def _load_gold_records(dataset_path: Path, query_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            query_id = str(record.get("query_id") or "")
            if query_id in query_ids:
                records[query_id] = {
                    "answer": str(record.get("answer") or "").strip(),
                    "qrel_docids": sorted(_document_ids(record)),
                }
                if len(records) == len(query_ids):
                    break
    return records


def _extract_frozen_evidence(context: dict[str, Any]) -> str:
    stages = context.get("stages")
    if not isinstance(stages, list):
        return ""
    for stage in reversed(stages):
        if not isinstance(stage, dict):
            continue
        prompt = str(stage.get("prompt") or "")
        marker = "User✿Evidence memory:\n"
        start = prompt.rfind(marker)
        if start < 0:
            continue
        evidence = prompt[start + len(marker) :]
        for suffix in ("\nNext action:", "✿\n\nBot✿", "✿\n\nAssistant:"):
            end = evidence.find(suffix)
            if end >= 0:
                evidence = evidence[:end]
        return evidence.strip()
    return ""


def _read_document_evidence(evidence: str) -> str:
    marker = "Documents read (most relevant last):"
    start = evidence.find(marker)
    if start < 0:
        return evidence
    selected = evidence[start:]
    for suffix in ("\nFinal step:", "\nAnswer with only"):
        end = selected.find(suffix)
        if end >= 0:
            selected = selected[:end]
    return selected.strip()


def _build_prompt(question: str, evidence: str) -> str:
    read_evidence = _read_document_evidence(evidence)
    return (
        f"User✿Question:\n{question.strip()}\n\nDocuments read:\n{read_evidence}\n\n"
        "Find the one subject shared by the clues. Do not list documents. Use `Exact Answer` only when a read "
        "document explicitly links that subject to the requested answer. Otherwise end with one focused "
        "`Search Query: ...` using that subject and the requested answer type.✿\n"
        "Bot✿<think>"
    )


def _build_decision_prompt(cot_prompt: str, cot_completion: str) -> str:
    return (
        f"{cot_prompt}{cot_completion.rstrip()}</think>✿\n"
        "User✿Return only one line. Use `Exact Answer: ...` only for an explicit subject-answer link; otherwise "
        "`Search Query: ...` using the identified subject.✿\n"
        "Bot✿<think></think>\n"
    )


def _generate(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    response = client.post(
        f"{api_base.rstrip('/')}/completions",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.35,
            "presence_penalty": 0.65,
            "repetition_penalty": 0.25,
            "penalty_decay": 0.99,
            "stop_token_ids": [0, 10060],
            "include_stop_str_in_output": False,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0].get("text") or "")


def _normalize_answer(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--api-base", default=os.environ.get("INFER_BASE_URL", "http://127.0.0.1:18073/v1"))
    parser.add_argument("--api-key", default=os.environ.get("INFER_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("INFER_MODEL", ""))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--decision-max-tokens", type=int, default=128)
    parser.add_argument("--sample-index", type=int, action="append")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.model:
        parser.error("--model or INFER_MODEL is required")

    _load_dotenv(Path(".env"))
    rows = _load_task_rows(args.task_id)
    selected = set(args.sample_index or [])
    if selected:
        rows = [row for row in rows if int(row["sample_index"]) in selected]
    query_ids = {
        str(row["context"].get("browsecomp_plus_run", {}).get("query_id") or "") for row in rows
    }
    gold_records = _load_gold_records(args.dataset, query_ids)
    results: list[dict[str, Any]] = []

    with httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        for row in rows:
            context = row["context"]
            run = context.get("browsecomp_plus_run", {})
            query_id = str(run.get("query_id") or "")
            evidence = _extract_frozen_evidence(context)
            prompt = _build_prompt(str(context.get("instruction") or ""), evidence)
            cot_completion = ""
            completion = ""
            if not args.inspect_only:
                cot_completion = _generate(
                    client,
                    args.api_base,
                    args.api_key,
                    args.model,
                    prompt,
                    args.max_tokens,
                )
                completion = _generate(
                    client,
                    args.api_base,
                    args.api_key,
                    args.model,
                    _build_decision_prompt(prompt, cot_completion),
                    args.decision_max_tokens,
                )
            match = EXACT_ANSWER_RE.search(completion)
            answer = match.group(1).strip() if match else ""
            query_match = SEARCH_QUERY_RE.search(completion)
            search_query = query_match.group(1).strip() if query_match else ""
            gold_record = gold_records.get(query_id, {})
            gold = str(gold_record.get("answer") or "")
            qrel_docids = {str(item) for item in gold_record.get("qrel_docids", [])}
            retrieved_docids = {str(item) for item in run.get("retrieved_docids", [])}
            read_docids = {str(item) for item in run.get("document_read_docids", [])}
            final_evidence_docids = set(re.findall(r"\[([0-9]+)\]", evidence))
            exact = bool(answer and _normalize_answer(answer) == _normalize_answer(gold))
            existing_answer = str(context.get("agent_info", {}).get("final_answer") or "").strip()
            result = {
                "sample_index": int(row["sample_index"]),
                "query_id": query_id,
                "evidence_chars": len(evidence),
                "prompt_chars": len(prompt),
                "qrel_docids": sorted(qrel_docids),
                "retrieved_qrel_docids": sorted(qrel_docids & retrieved_docids),
                "read_qrel_docids": sorted(qrel_docids & read_docids),
                "final_evidence_qrel_docids": sorted(qrel_docids & final_evidence_docids),
                "empty_think": cot_completion.lstrip().startswith("</think>"),
                "answer": answer,
                "search_query": search_query,
                "existing_answer": existing_answer,
                "existing_exact": bool(
                    existing_answer and _normalize_answer(existing_answer) == _normalize_answer(gold)
                ),
                "gold": gold,
                "exact": exact,
                "cot_completion": cot_completion,
                "completion": completion,
            }
            results.append(result)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "completion"},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY exact={sum(int(item['exact']) for item in results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
