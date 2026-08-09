#!/usr/bin/env python3
"""Read-only probe for constrained multiple-choice sampling.

The probe replays stored Knowledge prompts against the same OpenAI-compatible
endpoint, requests raw log-probabilities for every legal option token, and
checks whether constrained top-k=1 decoding selected the raw-logit argmax.
It never writes to the evaluation database.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from typing import Any

import httpx
import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
CHOICE_RE = re.compile(r"(?m)^([A-Z])\. ")


def _service_api_key(service: str) -> str:
    override = os.environ.get("RWKV_SKILLS_INFER_API_KEY", "").strip()
    if override:
        return override
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "--property=ExecStart",
            "--value",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    words = shlex.split(result.stdout)
    for index, word in enumerate(words[:-1]):
        if word == "--api-key":
            return words[index + 1]
    return "rwkv-skills"


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return body


def _token_ids(
    client: httpx.Client,
    api_root: str,
    model: str,
    letters: tuple[str, ...],
) -> dict[str, int]:
    def tokenize(text: str) -> tuple[int, ...]:
        result = _post(
            client,
            f"{api_root}/tokenize",
            {"model": model, "prompt": text, "add_special_tokens": False},
        )
        return tuple(int(value) for value in result["tokens"])

    prefix = tokenize("")
    resolved: dict[str, int] = {}
    for letter in letters:
        literal = f" {letter}"
        ids = tokenize(literal)
        if prefix and ids[: len(prefix)] == prefix:
            ids = ids[len(prefix) :]
        if len(ids) != 1:
            raise RuntimeError(f"{literal!r} is not one token: {ids}")
        resolved[letter] = ids[0]
    return resolved


def _stored_rows(task_id: int, limit: int) -> tuple[str, list[dict[str, Any]]]:
    # One row per benchmark sample gives substantially broader evidence for
    # Avg@K tasks than taking the first K repeat coordinates of sample zero.
    # The probe is checking prompt-to-choice transport, so repeated stochastic
    # coordinates of the same prompt do not add independent coverage here.
    query = """
        SELECT DISTINCT ON (c.sample_index)
            m.model_name,
            c.sample_index,
            c.avg_repeat_index,
            c.pass_index,
            c.context #>> '{stages,0,prompt}' AS prompt,
            c.context #>> '{stages,0,completion}' AS stored_completion,
            e.answer AS stored_answer,
            e.ref_answer
        FROM completions c
        JOIN task t ON t.task_id = c.task_id
        JOIN model m ON m.model_id = t.model_id
        LEFT JOIN eval e ON e.completions_id = c.completions_id
        WHERE c.task_id = %s
        ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
        LIMIT %s
    """
    db_config = replace(DEFAULT_DB_CONFIG, dbname=DB_NAME)
    with psycopg.connect(_build_conninfo(db_config), row_factory=dict_row) as connection:
        rows = list(connection.execute(query, (task_id, limit)).fetchall())
    if not rows:
        raise RuntimeError(f"task {task_id} has no stored completions")
    return str(rows[0]["model_name"]), rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    model, rows = _stored_rows(args.task_id, args.limit)
    api_root = args.base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    api_key = _service_api_key(args.service)
    headers = {"Authorization": f"Bearer {api_key}"}
    probes: list[dict[str, Any]] = []
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        for row in rows:
            prompt = str(row["prompt"] or "")
            letters = tuple(dict.fromkeys(CHOICE_RE.findall(prompt)))
            if len(letters) < 2:
                raise RuntimeError(
                    f"could not derive choices for sample {row['sample_index']}"
                )
            ids = _token_ids(client, api_root, model, letters)
            response = _post(
                client,
                f"{api_root}/v1/completions",
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "min_tokens": 1,
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "allowed_token_ids": list(ids.values()),
                    "logprobs": 0,
                    "logprob_token_ids": list(ids.values()),
                },
            )
            choice = response["choices"][0]
            sampled = str(choice["text"]).strip()
            top = choice["logprobs"]["top_logprobs"][0]
            raw_logprobs = {
                letter: float(top.get(f" {letter}", top.get(letter, -9999.0)))
                for letter in letters
            }
            raw_argmax = max(raw_logprobs, key=raw_logprobs.__getitem__)
            probes.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "repeat_index": int(row["avg_repeat_index"]),
                    "pass_index": int(row["pass_index"]),
                    "stored_answer": row["stored_answer"],
                    "reference_answer": row["ref_answer"],
                    "replayed_answer": sampled,
                    "raw_argmax": raw_argmax,
                    "raw_logprobs": raw_logprobs,
                    "sampler_matches_raw_argmax": sampled == raw_argmax,
                }
            )

    replay_counts = Counter(str(row["replayed_answer"]) for row in probes)
    return {
        "task_id": args.task_id,
        "model": model,
        "endpoint": args.base_url,
        "rows_probed": len(probes),
        "stored_replay_mismatches": sum(
            str(row["stored_answer"] or "").strip() != row["replayed_answer"]
            for row in probes
        ),
        "sampler_argmax_mismatches": sum(
            not row["sampler_matches_raw_argmax"] for row in probes
        ),
        "replayed_label_counts": dict(sorted(replay_counts.items())),
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_probe(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
