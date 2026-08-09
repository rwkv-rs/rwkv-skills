#!/usr/bin/env python3
"""Read-only status and protocol audit for the G1i strict-46 run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.benchmark_config import (
    resolve_benchmark_model_config,
    resolve_sampling_config,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import detect_job_from_dataset
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.results.schema import sampling_config_to_dict
from src.eval.metrics.free_response import (
    DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
    STRATEGY_A,
    STRATEGY_C,
    _extract_judgement_label,
    _is_judgement_reference,
    _strategy_judgement_text,
    llm_judge_protocol_stats_reasons,
)
from src.eval.env_config import load_env_file, resolve_judge_max_tokens

from ops.g1i_strict46.math_replay_provenance import (
    ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
    ATTESTATION_SCHEMA_VERSION,
    FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    FINAL_MATH_VERIFY_VERSION,
    FINAL_REPLAY_GIT_HASH,
    FINAL_REPLAY_PYTHONHASHSEED,
    PROVENANCE_VERSION,
    FinalMathReplayContract,
    canonical_json_sha256,
    parse_task_desc,
)
from ops.g1i_strict46.judge_transcript import TRANSCRIPT_SCHEMA_VERSION


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
AUDIT_LOCK_PATH = (
    Path(__file__).resolve().parents[2] / "logs" / "audits" / "g1i_strict46_audit.lock"
)
STRICT_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "g1h"
# Runners created before this deployment loaded the old Math final-stage stop
# policy.  It could miss BPE-merged boxed closers (for example ``}}\\``), so any
# such task that actually used the recovery/final stage must be rerun.
MATH_FINAL_TEXT_STOP_DEPLOYED_AT = datetime(2026, 8, 5, 23, 51, 55)
# Before this deployment, protocol="vllm" ordinary generation used
# /v1/chat/completions even though benchmark pipelines supplied an already
# rendered User/Assistant prompt.  The server therefore applied its chat
# template twice.  Keep all older rows as evidence, but never accept them as a
# strict-46 cell.  The backend fix makes vllm ordinary generation use raw
# /v1/completions; tool-call generation remains on chat completions.
RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT = datetime(2026, 8, 6, 5, 10, 0)
# The fail-closed scorer for an existing-but-empty Math recovery stage was
# rolled out atomically to the 157 evaluation source before this conservative
# boundary.  Tasks created earlier remain valid evidence, but any one of them
# that contains a blank recovery stage must be replayed with the fixed scorer.
BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT = datetime(2026, 8, 7, 17, 42, 0)
# The external free-response Judge was made deterministic at this deployment:
# requests now use temperature=0.0 instead of the historical sampling default.
# Root Judge tasks created before this boundary remain immutable evidence, but
# they are not protocol-compatible with the current strict-46 result cells and
# must be replayed from their stored completions.
JUDGE_DETERMINISM_DEPLOYED_AT = datetime(2026, 8, 7, 19, 20, 38)
MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)
MODEL_SIZES = ("1.5B", "2.9B", "7.2B", "13.3B")
MODEL_SIZE_BY_NAME = dict(zip(MODELS, MODEL_SIZES, strict=True))
# For these strict-46 runner families, ``stages[0]`` is the evaluator-facing
# final generation.  A length stop there is therefore a real final-output
# truncation.  Math uses a separate recovery/final-answer stage and must not be
# folded into this set; plain multiple-choice uses ``generated_choice`` and its
# one-token transport-level ``max_length`` marker is not semantic truncation.
STAGE0_FINAL_TRUNCATION_EVALUATORS = (
    "code_human_eval_naive",
    "code_mbpp_naive",
    "code_livecodebench_plain_naive",
    "instruction_following_naive",
)
# Parsing historical completion JSON dominates the read-only audit.  Split
# disjoint task IDs across a small number of read-only PostgreSQL connections
# so the server can use several cores without changing any aggregate or task
# selection semantics.
CONTENT_STATS_QUERY_WORKERS = 4
REFERENCE_MODELS = {
    "G1g": {
        "1.5B": "rwkv7-g1g-1.5b-20260526-ctx8192",
        "2.9B": "rwkv7-g1g-2.9b-20260526-ctx8192",
        "7.2B": "rwkv7-g1g-7.2b-20260523-ctx8192",
        "13.3B": "rwkv7-g1g-13.3b-20260523-ctx8192",
    },
    "G1h": {
        "1.5B": "rwkv7-g1h-1.5b-20260710-ctx10240",
        "2.9B": "rwkv7-g1h-2.9b-20260710-ctx10240",
        "7.2B": "rwkv7-g1h-7.2b-20260710-ctx10240",
        "13.3B": "rwkv7-g1h-13.3b-20260710-ctx10240",
    },
}


@contextmanager
def _exclusive_audit_lock(lock_path: Path) -> Iterator[None]:
    """Serialize audit processes with a crash-safe, blocking kernel lock."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(output_path: Path, text: str) -> None:
    """Atomically replace a text file using a temporary in the same directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = output_path.stat().st_mode & 0o777 if output_path.exists() else 0o644
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            os.fchmod(temporary_file.fileno(), output_mode)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(
            output_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


KNOWLEDGE = (
    ("mmlu", "test"),
    ("mmlu_pro", "test"),
    ("mmlu_redux", "test"),
    ("mmlu_sr_question_and_answer", "test"),
    ("gpqa", "diamond"),
    ("gpqa", "main"),
    ("gpqa", "extended"),
    ("arc_challenge", "test"),
    ("arc_easy", "test"),
    ("hellaswag", "validation"),
    ("bbh_mcq", "test"),
    ("agieval_mcq", "test"),
    ("truthfulqa_mc1", "validation"),
    ("winogrande", "validation"),
    ("openbookqa", "test"),
    ("commonsense_qa", "validation"),
    ("ceval", "test"),
    ("cmmlu", "test"),
    ("kmmlu", "test"),
    ("medqa", "test"),
    ("medmcqa", "validation"),
)
MATH = (
    ("aime24", "test"),
    ("aime25", "test"),
    ("amc23", "test"),
    ("answer_judge", "test"),
    ("beyond_aime", "test"),
    ("brumo25", "test"),
    ("comp_math_24_25", "test"),
    ("gaokao2023en", "test"),
    ("gsm8k", "test"),
    ("hmmt_feb25", "test"),
    ("math_500", "test"),
    ("math_odyssey", "test"),
    ("minerva_math", "test"),
    ("olympiadbench", "test"),
    ("simpleqa", "test"),
    ("svamp", "test"),
)
CODING = (
    ("human_eval", "test"),
    ("human_eval_cn", "test"),
    ("human_eval_fix", "test"),
    ("human_eval_plus", "test"),
    ("mbpp", "test"),
    ("mbpp_plus", "test"),
    ("livecodebench", "test"),
)
INSTRUCTION = (("ifeval", "test"), ("ifbench", "test"))

TARGETS = {
    **{key: ("knowledge", "no_cot") for key in KNOWLEDGE},
    **{key: ("math", "cot") for key in MATH},
    **{key: ("coding", "no_cot") for key in CODING},
    **{key: ("instruction_following", "no_cot") for key in INSTRUCTION},
}

# Some registered datasets persist their physical split name while strict-46
# uses a stable logical split in its target matrix.  Canonicalize at the audit
# boundary so coverage, active/failed classification, reference comparisons,
# and recovery scheduling all refer to the same cell.  Keep the physical
# source fields on each row for provenance.
TARGET_ALIASES = {
    ("simpleqa", "verified"): ("simpleqa", "test"),
}

# These two historical rows are explicitly excluded from final Math selection:
# 28869 predates the frozen comparator/provenance contract and 28872 was an
# invalid experimental replay.  Keeping the IDs here makes the gate immune to
# an accidentally edited description.
DISALLOWED_FINAL_MATH_TASK_IDS = frozenset({28869, 28872})


def canonical_target_benchmark(
    benchmark_name: str, benchmark_split: str
) -> tuple[str, str]:
    source = (str(benchmark_name), str(benchmark_split))
    return TARGET_ALIASES.get(source, source)


def canonicalize_task_benchmark(row: dict[str, Any]) -> tuple[str, str]:
    source = (str(row["benchmark_name"]), str(row["benchmark_split"]))
    benchmark = canonical_target_benchmark(*source)
    row["source_benchmark_name"] = source[0]
    row["source_benchmark_split"] = source[1]
    row["benchmark_name"] = benchmark[0]
    row["benchmark_split"] = benchmark[1]
    return benchmark


def _record_valid_candidate(
    latest_valid: dict[tuple[str, str, str], dict[str, Any]],
    superseded_valid: list[dict[str, Any]],
    cell: tuple[str, str, str],
    row: dict[str, Any],
) -> None:
    """Keep one newest valid row while retaining every superseded valid row."""

    previous = latest_valid.get(cell)
    if previous is None or int(row["task_id"]) > int(previous["task_id"]):
        if previous is not None:
            superseded_valid.append(previous)
        latest_valid[cell] = row
    else:
        superseded_valid.append(row)


TASK_QUERY = """
SELECT
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    b.num_samples AS benchmark_num_samples,
    t.task_id,
    t.evaluator,
    t.status,
    t.git_hash AS task_git_hash,
    t."desc" AS task_desc,
    t.is_tmp AS task_is_tmp,
    t.is_param_search AS task_is_param_search,
    t.created_at AS task_created_at,
    t.sampling_config,
    s.score_id,
    s.cot_mode,
    s.metrics,
    s.created_at AS score_created_at
FROM model m
JOIN task t ON t.model_id = m.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE m.model_name = ANY(%s)
ORDER BY t.task_id
"""

SOURCE_TASK_PROVENANCE_QUERY = """
SELECT
    t.task_id,
    t.status,
    t.evaluator,
    t.git_hash AS task_git_hash,
    t."desc" AS task_desc,
    t.is_tmp AS task_is_tmp,
    t.is_param_search AS task_is_param_search,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
WHERE t.task_id = ANY(%s)
ORDER BY t.task_id
"""

COMPLETION_COORDINATE_STATS_QUERY = """
SELECT
    c.task_id,
    COUNT(*) AS completion_count,
    -- uq_completions_sample guarantees one row per coordinate, so this is
    -- exactly equivalent to COUNT(DISTINCT ROW(...)) for non-empty tasks and
    -- avoids the expensive composite DISTINCT sort.  The no-completion LEFT
    -- JOIN legacy value is restored by COMPLETION_STAT_DEFAULTS below.
    COUNT(*) AS distinct_completion_coordinates,
    COUNT(DISTINCT c.sample_index) AS distinct_sample_indices,
    MIN(c.sample_index) AS min_sample_index,
    MAX(c.sample_index) AS max_sample_index,
    COUNT(DISTINCT c.avg_repeat_index) AS distinct_avg_repeat_indices,
    MIN(c.avg_repeat_index) AS min_avg_repeat_index,
    MAX(c.avg_repeat_index) AS max_avg_repeat_index
FROM completions c
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
ORDER BY c.task_id
"""

BLANK_RECOVERY_STAGE_SQL_PREDICATE = """
jsonb_typeof(c.context->'stages') = 'array'
AND jsonb_array_length(c.context->'stages') > 1
AND (
    BTRIM(COALESCE(c.context #>> '{stages,1,completion}', '')) = ''
    OR (
        COALESCE(c.context #>> '{stages,1,prompt}', '') ~ '(User✿|Bot✿)'
        AND COALESCE(c.context #>> '{stages,1,completion}', '')
            ~ '^[[:space:]]*(User✿|Bot✿|Assistant:|✿|User:)'
    )
    OR (
        COALESCE(c.context #>> '{stages,1,prompt}', '') !~ '(User✿|Bot✿)'
        AND COALESCE(c.context #>> '{stages,1,completion}', '')
            ~ E'^[[:space:]]*\\nUser:'
    )
)
""".strip()


COMPLETION_CONTENT_STATS_QUERY = """
SELECT
    c.task_id,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND COALESCE(c.context->>'direct_raw_completion', '') = ''
    ) AS blank_raw_count,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND COALESCE(c.context #>> '{stages,0,completion}', '') = ''
    ) AS blank_stage0_count,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND COALESCE(
              NULLIF(c.context->>'direct_raw_completion', ''),
              NULLIF(c.context #>> '{strategy_a,completion}', ''),
              NULLIF(c.context #>> '{stages,0,completion}', ''),
              ''
          ) = ''
    ) AS blank_primary_generation_count,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND COALESCE(
              NULLIF(c.context->>'direct_raw_completion', ''),
              NULLIF(c.context #>> '{stages,0,completion}', ''),
              NULLIF(c.context #>> '{strategy_a,completion}', ''),
              ''
          ) ~ '^[[:space:]]*></think>'
    ) AS leading_orphan_close_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context->>'direct_raw_finish_reason', '') = 'length'
    ) AS length_finish_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN ('length', 'max_tokens')
    ) AS stage0_length_stop_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
           OR COALESCE(c.context->>'direct_raw_finish_reason', '') IN ('length', 'max_tokens')
           OR (
               t.evaluator = ANY(%s)
               AND COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
                   'length', 'max_length', 'max_tokens'
               )
           )
    ) AS overall_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,strategy_a,truncated}')::boolean, FALSE)
           OR COALESCE((c.context #>> '{stats,stage1,truncated}')::boolean, FALSE)
           OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
               'length', 'max_length', 'max_tokens'
           )
    ) AS initial_generation_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,stage2,truncated}')::boolean, FALSE)
    ) AS final_stage_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
                  'length', 'max_length', 'max_tokens'
              )
          AND COALESCE(c.context #>> '{stages,0,completion}', '') LIKE '%%```%%'
    ) AS truncated_primary_with_code_fence_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{strategy_a,completion}', '') <> ''
    ) AS strategy_a_completion_count,
    COUNT(*) FILTER (
        WHERE jsonb_typeof(c.context->'strategy_a') = 'object'
          AND (
              BTRIM(COALESCE(c.context #>> '{strategy_a,completion}', '')) = ''
              OR (
                  COALESCE(c.context #>> '{strategy_a,prompt}', '') ~ '(User✿|Bot✿)'
                  AND COALESCE(c.context #>> '{strategy_a,completion}', '')
                      ~ '^[[:space:]]*(User✿|Bot✿|Assistant:|✿|User:)'
              )
              OR (
                  COALESCE(c.context #>> '{strategy_a,prompt}', '')
                      !~ '(User✿|Bot✿)'
                  AND COALESCE(c.context #>> '{strategy_a,completion}', '')
                      ~ E'^[[:space:]]*\\nUser:'
              )
          )
    ) AS blank_strategy_a_generation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,completion}', '') <> ''
    ) AS staged_generation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,1,completion}', '') <> ''
    ) AS recovery_stage_count,
    COUNT(*) FILTER (
        WHERE __BLANK_RECOVERY_STAGE_SQL_PREDICATE__
    ) AS blank_recovery_stage_count
FROM completions c
JOIN task t ON t.task_id = c.task_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
ORDER BY c.task_id
""".replace(
    "__BLANK_RECOVERY_STAGE_SQL_PREDICATE__",
    BLANK_RECOVERY_STAGE_SQL_PREDICATE,
)

EVAL_STATS_QUERY = """
SELECT
    c.task_id,
    COUNT(e.eval_id) AS eval_count,
    COUNT(*) FILTER (WHERE e.is_passed) AS passed_eval_count,
    COUNT(*) FILTER (
        WHERE e.fail_reason = 'missing_prediction'
    ) AS legacy_missing_prediction_count,
    COUNT(*) FILTER (
        WHERE e.fail_reason = 'missing_recovery_prediction'
    ) AS missing_recovery_prediction_count
FROM completions c
LEFT JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
ORDER BY c.task_id
"""

STRATEGY_A_EVAL_STATS_QUERY = """
SELECT
    candidate.parent_task_id AS task_id,
    COUNT(*) FILTER (
        WHERE e.fail_reason = 'missing_strategy_a_prediction'
    ) AS missing_strategy_a_prediction_count
FROM UNNEST(%s::bigint[], %s::bigint[])
    AS candidate(parent_task_id, strategy_a_task_id)
LEFT JOIN completions c ON c.task_id = candidate.strategy_a_task_id
LEFT JOIN eval e ON e.completions_id = c.completions_id
GROUP BY candidate.parent_task_id
ORDER BY candidate.parent_task_id
"""

BLANK_RECOVERY_INHERITANCE_STATS_QUERY = """
SELECT
    candidate.parent_task_id AS task_id,
    COUNT(*) FILTER (
        WHERE __BLANK_RECOVERY_STAGE_SQL_PREDICATE__
          AND parent_eval.is_passed IS TRUE
          AND strategy_a_eval.is_passed IS TRUE
          AND BTRIM(COALESCE(strategy_a_eval.answer, '')) <> ''
          AND parent_eval.answer IS NOT DISTINCT FROM strategy_a_eval.answer
          AND parent_eval.ref_answer IS NOT DISTINCT FROM strategy_a_eval.ref_answer
          AND COALESCE(parent_eval.fail_reason, '') = ''
          AND COALESCE(strategy_a_eval.fail_reason, '') = ''
    ) AS blank_recovery_strategy_a_inheritance_count
FROM UNNEST(%s::bigint[], %s::bigint[])
    AS candidate(parent_task_id, strategy_a_task_id)
LEFT JOIN completions parent_completion
    ON parent_completion.task_id = candidate.parent_task_id
LEFT JOIN eval parent_eval
    ON parent_eval.completions_id = parent_completion.completions_id
LEFT JOIN completions strategy_a_completion
    ON strategy_a_completion.task_id = candidate.strategy_a_task_id
   AND strategy_a_completion.sample_index = parent_completion.sample_index
   AND strategy_a_completion.avg_repeat_index = parent_completion.avg_repeat_index
   AND strategy_a_completion.pass_index = parent_completion.pass_index
LEFT JOIN eval strategy_a_eval
    ON strategy_a_eval.completions_id = strategy_a_completion.completions_id
GROUP BY candidate.parent_task_id
ORDER BY candidate.parent_task_id
""".replace(
    "__BLANK_RECOVERY_STAGE_SQL_PREDICATE__",
    BLANK_RECOVERY_STAGE_SQL_PREDICATE.replace("c.context", "parent_completion.context"),
)

RAW_BATCH_QUERY = """
SELECT
    c.task_id,
    ARRAY_AGG(
        c.context->>'direct_raw_completion'
        ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
    ) AS raw_values
FROM completions c
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
ORDER BY c.task_id
"""

PROMPT_BATCH_QUERY = """
SELECT candidate.task_id, COALESCE(probe.prompt, '') AS prompt
FROM UNNEST(%s::bigint[]) WITH ORDINALITY AS candidate(task_id, ordinal)
LEFT JOIN LATERAL (
    SELECT COALESCE(
        NULLIF(c.context #>> '{stages,0,prompt}', ''),
        NULLIF(c.context #>> '{strategy_a,prompt}', ''),
        ''
    ) AS prompt
    FROM completions c
    WHERE c.task_id = candidate.task_id
    ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
    LIMIT 1
) AS probe ON TRUE
ORDER BY candidate.ordinal
"""

CHOICE_LABEL_AUDIT_QUERY = """
SELECT
    c.task_id,
    e.answer,
    e.ref_answer,
    COUNT(*) AS row_count,
    COUNT(*) FILTER (WHERE e.is_passed) AS passed_count
FROM completions c
JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id, e.answer, e.ref_answer
ORDER BY c.task_id, e.answer, e.ref_answer
"""

JUDGEMENT_TASK_PROBE_QUERY = """
SELECT
    candidate.task_id,
    probe.ref_answer
FROM UNNEST(%s::bigint[]) AS candidate(task_id)
CROSS JOIN LATERAL (
    SELECT e.ref_answer
    FROM completions c
    JOIN eval e ON e.completions_id = c.completions_id
    WHERE c.task_id = candidate.task_id
    ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
    LIMIT 1
) AS probe
ORDER BY candidate.task_id
"""

JUDGEMENT_OUTPUT_AUDIT_QUERY = """
SELECT
    c.task_id,
    c.context,
    e.answer,
    e.ref_answer
FROM completions c
JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
ORDER BY c.task_id, c.sample_index, c.avg_repeat_index, c.pass_index
"""

TRUNCATION_EXAMPLE_BATCH_QUERY = """
SELECT
    candidate.task_id,
    example.sample_index,
    example.avg_repeat_index,
    example.pass_index,
    example.prompt_head,
    example.prompt_tail,
    example.completion_head,
    example.completion_tail,
    example.stage0_stop_reason,
    example.stage1_stop_reason,
    example.stats
FROM UNNEST(%s::bigint[]) WITH ORDINALITY AS candidate(task_id, ordinal)
CROSS JOIN LATERAL (
    SELECT
        c.sample_index,
        c.avg_repeat_index,
        c.pass_index,
        LEFT(
            COALESCE(c.context #>> '{stages,0,prompt}', ''),
            400
        ) AS prompt_head,
        RIGHT(
            COALESCE(c.context #>> '{stages,0,prompt}', ''),
            400
        ) AS prompt_tail,
        LEFT(
            COALESCE(
                NULLIF(c.context #>> '{strategy_a,completion}', ''),
                NULLIF(c.context #>> '{stages,0,completion}', ''),
                NULLIF(c.context->>'direct_raw_completion', ''),
                ''
            ),
            600
        ) AS completion_head,
        RIGHT(
            COALESCE(
                NULLIF(c.context #>> '{strategy_a,completion}', ''),
                NULLIF(c.context #>> '{stages,0,completion}', ''),
                NULLIF(c.context->>'direct_raw_completion', ''),
                ''
            ),
            600
        ) AS completion_tail,
        c.context #>> '{stages,0,stop_reason}' AS stage0_stop_reason,
        c.context #>> '{stages,1,stop_reason}' AS stage1_stop_reason,
        c.context #> '{stats}' AS stats
    FROM completions c
    WHERE c.task_id = candidate.task_id
      AND (
          COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
          OR COALESCE(
              (c.context #>> '{stats,strategy_a,truncated}')::boolean,
              FALSE
          )
          OR COALESCE(
              (c.context #>> '{stats,stage1,truncated}')::boolean,
              FALSE
          )
          OR COALESCE(
              (c.context #>> '{stats,stage2,truncated}')::boolean,
              FALSE
          )
          OR COALESCE(c.context->>'direct_raw_finish_reason', '') IN (
              'length', 'max_length', 'max_tokens'
          )
          OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
              'length', 'max_length', 'max_tokens'
          )
          OR COALESCE(c.context #>> '{stages,1,stop_reason}', '') IN (
              'length', 'max_length', 'max_tokens'
          )
      )
    ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
    LIMIT 5
) AS example
ORDER BY
    candidate.ordinal,
    example.sample_index,
    example.avg_repeat_index,
    example.pass_index
"""

REFERENCE_QUERY = """
SELECT
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    t.task_id,
    t.evaluator,
    t.status,
    t.created_at AS task_created_at,
    t.sampling_config,
    s.cot_mode,
    s.metrics,
    s.created_at AS score_created_at
FROM scores s
JOIN task t ON t.task_id = s.task_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
JOIN model m ON m.model_id = t.model_id
WHERE m.model_name = ANY(%s)
ORDER BY t.task_id
"""


COMPLETION_STAT_DEFAULTS: dict[str, int | None] = {
    "completion_count": 0,
    "eval_count": 0,
    # The former LEFT JOIN aggregate saw one all-NULL composite row for a
    # task with no completions, and PostgreSQL COUNT(DISTINCT ROW(...)) counts
    # that composite as one.  Preserve that observable audit value exactly.
    "distinct_completion_coordinates": 1,
    "distinct_sample_indices": 0,
    "min_sample_index": None,
    "max_sample_index": None,
    "distinct_avg_repeat_indices": 0,
    "min_avg_repeat_index": None,
    "max_avg_repeat_index": None,
    "blank_raw_count": 0,
    "blank_stage0_count": 0,
    "blank_primary_generation_count": 0,
    "leading_orphan_close_count": 0,
    "passed_eval_count": 0,
    "legacy_missing_prediction_count": 0,
    "missing_recovery_prediction_count": 0,
    "blank_recovery_strategy_a_inheritance_count": 0,
    "missing_strategy_a_prediction_count": 0,
    "missing_prediction_count": 0,
    "length_finish_count": 0,
    "stage0_length_stop_count": 0,
    "overall_truncation_count": 0,
    "initial_generation_truncation_count": 0,
    "final_stage_truncation_count": 0,
    "truncated_primary_with_code_fence_count": 0,
    "strategy_a_completion_count": 0,
    "blank_strategy_a_generation_count": 0,
    "staged_generation_count": 0,
    "recovery_stage_count": 0,
    "blank_recovery_stage_count": 0,
}


def _content_audit_candidate_task_ids(
    rows: list[dict[str, Any]],
    *,
    diagnostic_knowledge_replay_ids: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    """Return tasks whose completion JSON can affect the strict audit.

    Completion content is only consumed for strict-46 protocol gates,
    diagnostics, or active/failed progress.  Historical tasks outside the
    matrix and terminal scoreless tasks cannot affect the report, so parsing
    their JSON is both unnecessary and disproportionately expensive.

    A failed task with stored completions remains a candidate because its
    blank/truncation evidence is surfaced in the failed-task diagnostics.
    Explicit diagnostic replays remain candidates even without a score.
    """

    candidates: list[int] = []
    seen: set[int] = set()
    for row in rows:
        task_id = int(row["task_id"])
        benchmark = canonical_target_benchmark(
            str(row["benchmark_name"]),
            str(row["benchmark_split"]),
        )
        if benchmark not in TARGETS:
            continue
        status = str(row.get("status") or "")
        has_audit_semantics = (
            row.get("score_created_at") is not None
            or status == "Running"
            or task_id in diagnostic_knowledge_replay_ids
            or (status == "Failed" and int(row.get("completion_count") or 0) > 0)
        )
        if has_audit_semantics and task_id not in seen:
            candidates.append(task_id)
            seen.add(task_id)
    return candidates


def _partition_content_task_ids(
    rows: list[dict[str, Any]],
    task_ids: list[int],
    *,
    max_workers: int,
) -> list[list[int]]:
    """Greedily balance task IDs by their observed completion count."""

    if not task_ids:
        return []
    worker_count = max(1, min(int(max_workers), len(task_ids)))
    completion_count_by_task = {
        int(row["task_id"]): int(row.get("completion_count") or 0) for row in rows
    }
    original_order = {task_id: index for index, task_id in enumerate(task_ids)}
    weighted_task_ids = sorted(
        task_ids,
        key=lambda task_id: (
            -completion_count_by_task.get(task_id, 0),
            original_order[task_id],
        ),
    )
    partitions: list[list[int]] = [[] for _ in range(worker_count)]
    partition_weights = [0 for _ in range(worker_count)]
    for task_id in weighted_task_ids:
        partition_index = min(
            range(worker_count),
            key=lambda index: (partition_weights[index], index),
        )
        partitions[partition_index].append(task_id)
        partition_weights[partition_index] += completion_count_by_task.get(task_id, 0)
    return partitions


def _query_content_stats_partition(
    conninfo: str,
    task_ids: list[int],
) -> list[dict[str, Any]]:
    """Aggregate one disjoint content partition in a read-only transaction."""

    with psycopg.connect(conninfo, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                COMPLETION_CONTENT_STATS_QUERY,
                (list(STAGE0_FINAL_TRUNCATION_EVALUATORS), task_ids),
            )
            return [dict(row) for row in cursor.fetchall()]


def _load_content_stats(
    cursor: Any,
    rows: list[dict[str, Any]],
    task_ids: list[int],
    *,
    conninfo: str | None,
    max_workers: int,
) -> dict[int, dict[str, Any]]:
    if not task_ids:
        return {}
    if conninfo and max_workers > 1:
        partitions = _partition_content_task_ids(
            rows,
            task_ids,
            max_workers=max_workers,
        )
        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            futures = [
                executor.submit(_query_content_stats_partition, conninfo, partition)
                for partition in partitions
            ]
            content_rows = [row for future in futures for row in future.result()]
    else:
        cursor.execute(
            COMPLETION_CONTENT_STATS_QUERY,
            (list(STAGE0_FINAL_TRUNCATION_EVALUATORS), task_ids),
        )
        content_rows = [dict(row) for row in cursor.fetchall()]

    content_by_task: dict[int, dict[str, Any]] = {}
    for item in content_rows:
        stats = dict(item)
        task_id = int(stats.pop("task_id"))
        content_by_task[task_id] = stats
    return content_by_task


def _load_task_rows(
    cursor: Any,
    *,
    diagnostic_knowledge_replay_ids: set[int] | None = None,
    content_conninfo: str | None = None,
    content_stats_workers: int = CONTENT_STATS_QUERY_WORKERS,
) -> list[dict[str, Any]]:
    """Load task metadata and completion aggregates without a wide join.

    Keeping the high-cardinality completion/eval aggregation keyed only by
    ``task_id`` avoids sorting and grouping copies of the wide task sampling
    config and score metrics for every completion.  The merge below preserves
    the single-row shape and zero/NULL defaults of the former LEFT JOIN query.
    """

    cursor.execute(TASK_QUERY, (list(MODELS),))
    rows = [dict(row) for row in cursor.fetchall()]
    task_ids = _ordered_unique_task_ids(rows)
    stats_by_task: dict[int, dict[str, Any]] = {}
    if task_ids:
        lightweight_stats_queries = (
            (COMPLETION_COORDINATE_STATS_QUERY, (task_ids,)),
            (EVAL_STATS_QUERY, (task_ids,)),
        )
        for query, params in lightweight_stats_queries:
            cursor.execute(query, params)
            for item in cursor.fetchall():
                stats = dict(item)
                task_id = int(stats.pop("task_id"))
                stats_by_task.setdefault(task_id, {}).update(stats)

    for row in rows:
        row.update(COMPLETION_STAT_DEFAULTS)
        row.update(stats_by_task.get(int(row["task_id"]), {}))

    content_task_ids = _content_audit_candidate_task_ids(
        rows,
        diagnostic_knowledge_replay_ids=diagnostic_knowledge_replay_ids or set(),
    )
    content_by_task = _load_content_stats(
        cursor,
        rows,
        content_task_ids,
        conninfo=content_conninfo,
        max_workers=content_stats_workers,
    )
    for row in rows:
        row.update(content_by_task.get(int(row["task_id"]), {}))
        # Keep transport/evaluator omissions separate from a model that
        # genuinely emitted an empty final-recovery stage.  The latter is a
        # real (failed) prediction once the fail-closed scorer has persisted
        # ``missing_recovery_prediction`` for the same raw row; treating it as
        # an infrastructure-level missing prediction would make an otherwise
        # complete task impossible to accept even after a protocol-correct
        # replay.  Raw/eval agreement is enforced by
        # ``_blank_recovery_protocol_reasons`` instead.
        row["missing_prediction_count"] = int(
            row.get("legacy_missing_prediction_count") or 0
        )
    _load_strategy_a_eval_stats(cursor, rows)
    _load_blank_recovery_inheritance_stats(cursor, rows)
    return rows


def _ordered_unique_task_ids(rows: list[dict[str, Any]]) -> list[int]:
    return list(dict.fromkeys(int(row["task_id"]) for row in rows))


def _load_strategy_a_eval_stats(cursor: Any, rows: list[dict[str, Any]]) -> None:
    """Load blank-A eval evidence from each parent's Strategy-A companion.

    Strict Math persists Strategy C evals on the parent task and Strategy A
    evals on a companion task recorded in the parent's score metrics.  Query
    only parents that actually contain an explicit blank A; all other rows
    keep the zero default and pay no extra database cost.
    """

    strategy_a_by_parent: dict[int, int] = {}
    for row in rows:
        if int(row.get("blank_strategy_a_generation_count") or 0) <= 0:
            continue
        metrics = row.get("metrics")
        strategy_ids = metrics.get("strategy_task_ids") if isinstance(metrics, dict) else None
        strategy_a_task_id = (
            strategy_ids.get(STRATEGY_A) if isinstance(strategy_ids, dict) else None
        )
        try:
            resolved_strategy_a_task_id = int(strategy_a_task_id)
        except (TypeError, ValueError):
            continue
        parent_task_id = int(row["task_id"])
        strategy_a_by_parent[parent_task_id] = max(
            resolved_strategy_a_task_id,
            strategy_a_by_parent.get(parent_task_id, resolved_strategy_a_task_id),
        )

    if not strategy_a_by_parent:
        return
    parent_ids = sorted(strategy_a_by_parent)
    strategy_a_task_ids = [strategy_a_by_parent[task_id] for task_id in parent_ids]
    cursor.execute(
        STRATEGY_A_EVAL_STATS_QUERY,
        (parent_ids, strategy_a_task_ids),
    )
    counts = {
        int(item["task_id"]): int(item["missing_strategy_a_prediction_count"] or 0)
        for item in cursor.fetchall()
    }
    for row in rows:
        row["missing_strategy_a_prediction_count"] = counts.get(
            int(row["task_id"]),
            int(row.get("missing_strategy_a_prediction_count") or 0),
        )


def _load_blank_recovery_inheritance_stats(
    cursor: Any,
    rows: list[dict[str, Any]],
) -> None:
    """Count blank recovery coordinates legitimately inherited from A.

    Strategy B/C intentionally reuse a successful Strategy-A record before
    evaluating their own recovery generation.  Consequently, an empty stage
    2 does not produce ``missing_recovery_prediction`` when the matching
    Strategy-A companion coordinate already passed.  Treat that as explained
    only when the parent and companion eval rows both pass and their answer,
    reference, and empty failure reason agree exactly.  A score-level task-id
    link plus a coordinate-level eval match is deliberately required so an
    unrelated successful Strategy A cannot hide polluted parent evals.
    """

    strategy_a_by_parent: dict[int, int] = {}
    for row in rows:
        if int(row.get("blank_recovery_stage_count") or 0) <= 0:
            continue
        metrics = row.get("metrics")
        strategy_ids = (
            metrics.get("strategy_task_ids") if isinstance(metrics, dict) else None
        )
        strategy_a_task_id = (
            strategy_ids.get(STRATEGY_A) if isinstance(strategy_ids, dict) else None
        )
        try:
            resolved_strategy_a_task_id = int(strategy_a_task_id)
        except (TypeError, ValueError):
            continue
        parent_task_id = int(row["task_id"])
        strategy_a_by_parent[parent_task_id] = max(
            resolved_strategy_a_task_id,
            strategy_a_by_parent.get(parent_task_id, resolved_strategy_a_task_id),
        )

    if not strategy_a_by_parent:
        return
    parent_ids = sorted(strategy_a_by_parent)
    strategy_a_task_ids = [strategy_a_by_parent[task_id] for task_id in parent_ids]
    cursor.execute(
        BLANK_RECOVERY_INHERITANCE_STATS_QUERY,
        (parent_ids, strategy_a_task_ids),
    )
    counts = {
        int(item["task_id"]): int(
            item["blank_recovery_strategy_a_inheritance_count"] or 0
        )
        for item in cursor.fetchall()
    }
    for row in rows:
        row["blank_recovery_strategy_a_inheritance_count"] = counts.get(
            int(row["task_id"]),
            int(row.get("blank_recovery_strategy_a_inheritance_count") or 0),
        )


def _load_completion_audit_maps(
    cursor: Any,
    rows: list[dict[str, Any]],
    *,
    diagnostic_knowledge_replay_ids: set[int] | None = None,
) -> tuple[
    dict[int, list[str | None]],
    dict[int, str],
    dict[int, list[dict[str, Any]]],
]:
    """Load per-task completion evidence with a constant number of queries.

    The former implementation issued one prompt query for every task, plus
    one raw-completion query for each relevant Knowledge task and one example
    query for every task with truncation telemetry.  These maps preserve the
    same per-task ordering, empty defaults, and five-example limit while
    reducing that N+1 access pattern to at most three bulk queries.
    """

    task_ids = _ordered_unique_task_ids(rows)
    replay_ids = diagnostic_knowledge_replay_ids or set()
    knowledge_rows = [
        row
        for row in rows
        if (
            row.get("score_created_at") is not None
            or str(row.get("status") or "") == "Running"
            or int(row["task_id"]) in replay_ids
        )
        and canonical_target_benchmark(
            str(row["benchmark_name"]), str(row["benchmark_split"])
        )
        in KNOWLEDGE
    ]
    knowledge_task_ids = _ordered_unique_task_ids(knowledge_rows)
    truncation_rows = [
        row
        for row in rows
        if int(row.get("overall_truncation_count") or 0)
        or int(row.get("initial_generation_truncation_count") or 0)
        or int(row.get("final_stage_truncation_count") or 0)
    ]
    truncation_task_ids = _ordered_unique_task_ids(truncation_rows)

    raw_by_task: dict[int, list[str | None]] = {
        task_id: [] for task_id in knowledge_task_ids
    }
    if knowledge_task_ids:
        cursor.execute(RAW_BATCH_QUERY, (knowledge_task_ids,))
        for item in cursor.fetchall():
            raw_by_task[int(item["task_id"])] = list(item["raw_values"] or [])

    prompt_by_task = {task_id: "" for task_id in task_ids}
    if task_ids:
        cursor.execute(PROMPT_BATCH_QUERY, (task_ids,))
        for item in cursor.fetchall():
            prompt_by_task[int(item["task_id"])] = str(item["prompt"] or "")

    truncation_examples_by_task: dict[int, list[dict[str, Any]]] = {
        task_id: [] for task_id in truncation_task_ids
    }
    if truncation_task_ids:
        cursor.execute(
            TRUNCATION_EXAMPLE_BATCH_QUERY,
            (truncation_task_ids,),
        )
        for item in cursor.fetchall():
            example = dict(item)
            task_id = int(example.pop("task_id"))
            truncation_examples_by_task[task_id].append(example)

    return raw_by_task, prompt_by_task, truncation_examples_by_task


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _completion_payload_from_context(context: Any) -> dict[str, Any]:
    """Flatten stored completion context like EvalDbService does.

    Keeping this tiny conversion local makes the read-only auditor independent
    from repository/service state while still feeding the exact same global
    free-response judgement extractor used by evaluation.
    """

    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            context = None
    if not isinstance(context, dict):
        return {}
    payload: dict[str, Any] = {"context": context}
    stages = context.get("stages")
    if isinstance(stages, list):
        for index, stage in enumerate(stages, start=1):
            if not isinstance(stage, dict):
                continue
            payload[f"prompt{index}"] = stage.get("prompt")
            payload[f"completion{index}"] = stage.get("completion")
            payload[f"stop_reason{index}"] = stage.get("stop_reason")
    strategy_a = context.get("strategy_a")
    if isinstance(strategy_a, dict):
        payload["strategy_a_prompt"] = strategy_a.get("prompt")
        payload["strategy_a_completion"] = strategy_a.get("completion")
        payload["strategy_a_stop_reason"] = strategy_a.get("stop_reason")
    return payload


def _judgement_output_source_mismatch(row: dict[str, Any]) -> bool:
    """Detect stored labels that disagree with the evaluator's source lane.

    ``evaluate_free_response`` scores strategy A first and inherits an already
    correct A record into strategies B/C.  Only when A is not correct does C's
    final recovery output become authoritative.  Auditing every stored row
    against C alone therefore creates false mismatches whenever a correct A
    answer and a different C answer coexist.
    """

    reference = str(row.get("ref_answer") or "")
    if not _is_judgement_reference(reference):
        return False
    payload = _completion_payload_from_context(row.get("context"))
    reference_label = _extract_judgement_label(reference)
    strategy_a_label = _extract_judgement_label(
        _strategy_judgement_text(STRATEGY_A, payload)
    )
    if reference_label is not None and strategy_a_label == reference_label:
        generated_label = strategy_a_label
    else:
        generated_label = _extract_judgement_label(
            _strategy_judgement_text(STRATEGY_C, payload)
        )
    stored_label = _extract_judgement_label(str(row.get("answer") or ""))
    # A missing label is a legitimate model/evaluator failure and remains a
    # wrong answer; it is not a provenance mismatch when the stored evaluator
    # row also contains no label.  Reject only disagreement between the source
    # lane and the stored representation.
    return generated_label != stored_label


def _load_math_replay_source_tasks(
    cursor: Any,
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Load only source roots referenced by scored Math replay descriptions."""

    source_ids: set[int] = set()
    for row in rows:
        benchmark = canonical_target_benchmark(
            str(row.get("benchmark_name") or ""),
            str(row.get("benchmark_split") or ""),
        )
        if TARGETS.get(benchmark, (None, None))[0] != "math":
            continue
        parsed = parse_task_desc(row.get("task_desc"))
        try:
            source_id = int(parsed.get("replay_source_task_id") or 0)
        except ValueError:
            continue
        if source_id > 0:
            source_ids.add(source_id)
    if not source_ids:
        return {}
    cursor.execute(SOURCE_TASK_PROVENANCE_QUERY, (sorted(source_ids),))
    return {int(item["task_id"]): dict(item) for item in cursor.fetchall()}


def _answer_judge_bypasses_math_verify(row: dict[str, Any]) -> bool:
    """Prove every eval row used the judgement-reference short circuit."""

    benchmark = canonical_target_benchmark(
        str(row.get("benchmark_name") or ""),
        str(row.get("benchmark_split") or ""),
    )
    eval_count = int(row.get("eval_count") or 0)
    return (
        benchmark == ("answer_judge", "test")
        and str(row.get("evaluator") or "").lower() == "free_response_naive"
        and eval_count > 0
        and int(row.get("judgement_output_source_row_count") or 0) == eval_count
        and int(row.get("judgement_reference_row_count") or 0) == eval_count
        and int(row.get("judgement_output_source_mismatch_count") or 0) == 0
    )


def _math_judge_transcript_reasons(
    *,
    parsed: dict[str, str],
    score_provenance: dict[str, Any],
    score_attestation: object,
    source_task_id: int,
    judge_mode: str,
) -> list[str]:
    """Validate external-Judge evidence, or prove its complete absence."""

    reasons: list[str] = []
    desc_sha = str(parsed.get("judge_transcript_sha256") or "").lower()
    transcript = score_provenance.get("judge_transcript")
    attested_map = (
        score_attestation.get("judge_transcript_sha256_by_task")
        if isinstance(score_attestation, dict)
        else None
    )
    if judge_mode == "exact":
        if "judge_transcript_sha256" in parsed:
            reasons.append("math_replay_exact_desc_has_judge_transcript")
        if transcript is not None:
            reasons.append("math_replay_exact_score_has_judge_transcript")
        if not isinstance(attested_map, dict) or attested_map:
            reasons.append("math_replay_exact_attestation_has_judge_transcript")
        return reasons

    if not re.fullmatch(r"[0-9a-f]{64}", desc_sha):
        reasons.append("math_replay_judge_transcript_desc_sha_missing_or_invalid")
    if not isinstance(transcript, dict):
        reasons.append("math_replay_score_judge_transcript_missing")
    else:
        if transcript.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
            reasons.append("math_replay_score_judge_transcript_schema_mismatch")
        if str(transcript.get("sha256") or "").lower() != desc_sha:
            reasons.append("math_replay_score_judge_transcript_sha_mismatch")
        protocols = transcript.get("protocol_fingerprint_sha256")
        if (
            not isinstance(protocols, list)
            or not protocols
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(value))
                for value in protocols
            )
        ):
            reasons.append("math_replay_score_judge_transcript_protocol_invalid")
        statistics = transcript.get("statistics")
        statistic_fields = {
            "protocol_count",
            "unique_input_count",
            "actual_judge_call_count",
            "coordinate_count",
            "true_coordinate_count",
            "false_coordinate_count",
            "scope_count",
        }
        if (
            not isinstance(statistics, dict)
            or not statistic_fields.issubset(statistics)
            or any(
                isinstance(statistics.get(key), bool)
                or not isinstance(statistics.get(key), int)
                or int(statistics[key]) < 0
                for key in statistic_fields
            )
            or _safe_int(statistics.get("scope_count")) != 1
            or _safe_int(statistics.get("protocol_count")) < 1
            or _safe_int(statistics.get("actual_judge_call_count"))
            != _safe_int(statistics.get("unique_input_count"))
            or _safe_int(statistics.get("true_coordinate_count"))
            + _safe_int(statistics.get("false_coordinate_count"))
            != _safe_int(statistics.get("coordinate_count"))
            or _safe_int(statistics.get("actual_judge_call_count"))
            > _safe_int(statistics.get("coordinate_count"))
        ):
            reasons.append("math_replay_score_judge_transcript_statistics_invalid")
    if (
        not isinstance(attested_map, dict)
        or set(str(key) for key in attested_map) != {str(source_task_id)}
        or str(attested_map.get(str(source_task_id)) or "").lower()
        != desc_sha
    ):
        reasons.append("math_replay_score_attested_judge_transcript_mismatch")
    return reasons


def _math_final_provenance_reasons(
    row: dict[str, Any],
    *,
    contract: FinalMathReplayContract,
    source_tasks: dict[int, dict[str, Any]],
) -> list[str]:
    """Fail closed unless a Math score is a frozen append-only replay root."""

    task_id = int(row.get("task_id") or 0)
    if task_id in DISALLOWED_FINAL_MATH_TASK_IDS:
        return [f"explicitly_disallowed_final_math_task:{task_id}"]
    if _answer_judge_bypasses_math_verify(row):
        row["math_provenance_gate"] = {
            "passed": True,
            "mode": "answer_judge_reference_short_circuit",
        }
        return []

    reasons = list(contract.blockers())
    parsed = parse_task_desc(row.get("task_desc"))
    required_desc = {
        "provenance_version": PROVENANCE_VERSION,
        "reason_tag": contract.reason_tag,
        "extractor_lineage_sha256": contract.extractor_lineage_sha256,
        "imported_free_response_sha256": contract.imported_free_response_sha256,
        "comparator_implementation_sha256": (
            contract.comparator_implementation_sha256
        ),
        "math_verify_version": contract.math_verify_version,
        "replay_git_hash": contract.replay_git_hash,
    }
    for key, expected in required_desc.items():
        actual = str(parsed.get(key) or "").lower() if "sha256" in key or key == "replay_git_hash" else str(parsed.get(key) or "")
        expected_value = expected.lower() if "sha256" in key or key == "replay_git_hash" else expected
        if actual != expected_value:
            reasons.append(
                f"math_replay_desc.{key}:{actual or 'empty'}"
                f"!=expected:{expected_value or 'unset'}"
            )

    replay_git_hash = str(row.get("task_git_hash") or "").lower()
    if replay_git_hash != contract.replay_git_hash:
        reasons.append(
            f"math_replay_task_git_hash:{replay_git_hash or 'empty'}"
            f"!=expected:{contract.replay_git_hash or 'unset'}"
        )
    if parsed.get("replay_git_hash", "").lower() != replay_git_hash:
        reasons.append("math_replay_desc_git_hash_mismatch")

    attestation_sha = str(parsed.get("determinism_attestation_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", attestation_sha):
        reasons.append("math_replay_determinism_attestation_sha256_missing_or_invalid")
    source_evidence_sha = str(parsed.get("source_evidence_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_evidence_sha):
        reasons.append("math_replay_source_evidence_sha256_missing_or_invalid")
    pythonhashseed = str(parsed.get("pythonhashseed") or "")
    if pythonhashseed != FINAL_REPLAY_PYTHONHASHSEED:
        reasons.append(
            f"math_replay_pythonhashseed:{pythonhashseed or 'empty'}"
            f"!=expected:{FINAL_REPLAY_PYTHONHASHSEED}"
        )

    try:
        source_task_id = int(parsed.get("replay_source_task_id") or 0)
    except ValueError:
        source_task_id = 0
    if source_task_id <= 0:
        reasons.append("math_replay_source_task_id_missing_or_invalid")
        source = None
    else:
        source = source_tasks.get(source_task_id)
    if source_task_id >= task_id > 0:
        reasons.append(
            f"math_replay_source_task_id:{source_task_id}"
            f"_not_older_than_replay:{task_id}"
        )
    if source is None:
        if source_task_id > 0:
            reasons.append(f"math_replay_source_task_missing:{source_task_id}")
    else:
        source_git_hash = str(source.get("task_git_hash") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{7,64}", source_git_hash):
            reasons.append("math_replay_source_git_hash_missing_or_invalid")
        if str(parsed.get("source_git_hash") or "").lower() != source_git_hash:
            reasons.append("math_replay_source_git_hash_mismatch")
        if parse_task_desc(source.get("task_desc")).get("replay_source_task_id"):
            reasons.append("math_replay_source_is_replay_chain")
        if bool(source.get("task_is_tmp")) or bool(
            source.get("task_is_param_search")
        ):
            reasons.append("math_replay_source_is_auxiliary")
        if str(source.get("status") or "").lower() != "completed":
            reasons.append(
                "math_replay_source_status:"
                f"{source.get('status') or 'empty'}!=Completed"
            )
        if str(source.get("model_name") or "") != str(row.get("model_name") or ""):
            reasons.append("math_replay_source_model_mismatch")
        source_benchmark = canonical_target_benchmark(
            str(source.get("benchmark_name") or ""),
            str(source.get("benchmark_split") or ""),
        )
        replay_benchmark = canonical_target_benchmark(
            str(row.get("benchmark_name") or ""),
            str(row.get("benchmark_split") or ""),
        )
        if source_benchmark != replay_benchmark:
            reasons.append("math_replay_source_benchmark_mismatch")

    judge_mode = (
        "llm"
        if source is not None
        and "judge" in str(source.get("evaluator") or "").lower()
        else "exact"
    )

    metrics = row.get("metrics")
    score_provenance = (
        metrics.get("replay_provenance") if isinstance(metrics, dict) else None
    )
    if not isinstance(score_provenance, dict):
        reasons.append("math_replay_score_provenance_missing")
    else:
        if _safe_int(score_provenance.get("source_task_id")) != source_task_id:
            reasons.append("math_replay_score_source_task_id_mismatch")
        score_source_evidence = score_provenance.get("source_evidence")
        if not isinstance(score_source_evidence, dict):
            reasons.append("math_replay_score_source_evidence_missing")
        else:
            computed_source_evidence_sha = canonical_json_sha256(
                score_source_evidence
            )
            if computed_source_evidence_sha != source_evidence_sha:
                reasons.append("math_replay_score_source_evidence_sha_mismatch")
            expected_source_evidence = {
                "source_task_id": source_task_id,
                "source_git_hash": (
                    str(source.get("task_git_hash") or "").lower()
                    if source is not None
                    else ""
                ),
                "source_evaluator": (
                    str(source.get("evaluator") or "")
                    if source is not None
                    else ""
                ),
                "model_name": str(row.get("model_name") or ""),
                "dataset_slug": canonical_slug(
                    f"{row.get('source_benchmark_name') or row.get('benchmark_name') or ''}_"
                    f"{row.get('source_benchmark_split') or row.get('benchmark_split') or ''}"
                ),
            }
            if any(
                score_source_evidence.get(key) != value
                for key, value in expected_source_evidence.items()
            ):
                reasons.append("math_replay_score_source_evidence_identity_mismatch")
        if (
            str(score_provenance.get("source_evidence_sha256") or "").lower()
            != source_evidence_sha
        ):
            reasons.append("math_replay_score_source_evidence_field_mismatch")
        if score_provenance.get("contract") != contract.as_dict():
            reasons.append("math_replay_score_contract_mismatch")
        score_runtime = score_provenance.get("runtime")
        if not isinstance(score_runtime, dict):
            reasons.append("math_replay_score_runtime_missing")
        else:
            runtime_expected = {
                "imported_free_response_sha256": (
                    contract.imported_free_response_sha256
                ),
                "comparator_implementation_sha256": (
                    contract.comparator_implementation_sha256
                ),
                "math_verify_version": contract.math_verify_version,
                "replay_git_hash": contract.replay_git_hash,
                "pythonhashseed": pythonhashseed,
            }
            if any(
                str(score_runtime.get(key) or "").lower() != value.lower()
                for key, value in runtime_expected.items()
            ):
                reasons.append("math_replay_score_runtime_mismatch")
        score_attestation = score_provenance.get("determinism_attestation")
        if not isinstance(score_attestation, dict):
            reasons.append("math_replay_score_attestation_missing")
        else:
            attested_seeds = score_attestation.get("seeds")
            normalized_attested_seeds = (
                {str(seed) for seed in attested_seeds}
                if isinstance(attested_seeds, list)
                else set()
            )
            if (
                score_attestation.get("schema_version")
                != ATTESTATION_SCHEMA_VERSION
                or score_attestation.get("passed") is not True
                or len(normalized_attested_seeds) < 4
                or not {"0", "1", "42"}.issubset(
                    normalized_attested_seeds
                )
                or any(not seed.isdigit() for seed in normalized_attested_seeds)
            ):
                reasons.append("math_replay_score_attestation_metadata_invalid")
            if str(score_attestation.get("sha256") or "").lower() != attestation_sha:
                reasons.append("math_replay_score_attestation_sha_mismatch")
            attested_source_evidence = score_attestation.get(
                "source_evidence_sha256_by_task"
            )
            if (
                not isinstance(attested_source_evidence, dict)
                or str(
                    attested_source_evidence.get(str(source_task_id)) or ""
                ).lower()
                != source_evidence_sha
            ):
                reasons.append("math_replay_score_attested_source_evidence_mismatch")
            attested_results = score_attestation.get("task_result_sha256")
            result_sha = str(
                score_provenance.get("evaluation_result_sha256") or ""
            ).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", result_sha):
                reasons.append("math_replay_score_result_sha_missing_or_invalid")
            if (
                not isinstance(attested_results, dict)
                or str(attested_results.get(str(source_task_id)) or "").lower()
                != result_sha
            ):
                reasons.append("math_replay_score_attested_result_mismatch")
        reasons.extend(
            _math_judge_transcript_reasons(
                parsed=parsed,
                score_provenance=score_provenance,
                score_attestation=score_attestation,
                source_task_id=source_task_id,
                judge_mode=judge_mode,
            )
        )

    row["math_provenance_gate"] = {
        "passed": not reasons,
        "mode": "append_only_replay",
        "source_task_id": source_task_id or None,
        "task_desc_fields": parsed,
        "source_task": source,
        "judge_mode": judge_mode,
    }
    return list(dict.fromkeys(reasons))


def _knowledge_protocol_ok(
    task: dict[str, Any], raw_values: list[str | None], representative_prompt: str
) -> bool:
    if task["score_created_at"] is None or not raw_values:
        return False
    # Before task 28457, Knowledge NoCoT used unconstrained prose generation and
    # an unsafe last-letter extractor. Those rows remain historical evidence only.
    if int(task["task_id"]) < 28457:
        return False
    prompt_ok = (
        "Assistant: <think></think>" in representative_prompt
        and representative_prompt.endswith("The answer is")
    )
    outputs_ok = all(re.fullmatch(r"\s*[A-Z]\s*", raw or "") for raw in raw_values)
    return prompt_ok and outputs_ok


def _instruction_protocol_ok(task: dict[str, Any], representative_prompt: str) -> bool:
    if task["score_created_at"] is None or not representative_prompt:
        return False
    return representative_prompt.endswith("Assistant: <think></think>\n")


def _coding_protocol_ok(task: dict[str, Any], representative_prompt: str) -> bool:
    if task["score_created_at"] is None or not representative_prompt:
        return False
    return bool(
        re.search(
            r"Assistant: <think>\n?</think>\n```python\n?\Z",
            representative_prompt,
        )
    )


def _math_protocol_ok(task: dict[str, Any], representative_prompt: str) -> bool:
    if task["score_created_at"] is None or not representative_prompt:
        return False
    return representative_prompt.endswith("Assistant: <think")


def _normalize_mode(value: Any) -> str:
    normalized = re.sub(r"[^a-z]", "", str(value).lower())
    if normalized == "nocot":
        return "no_cot"
    if normalized == "cot":
        return "cot"
    return normalized


def _primary_numeric_metric(metrics: Any) -> tuple[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    preferred = [
        key
        for key in metrics
        if str(key).startswith(("avg@", "pass@")) or str(key) in {"accuracy", "score"}
    ]
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return str(key), float(value)
    return None


def _task_source_slug(task: dict[str, Any]) -> str:
    """Resolve runner/config contracts from the physical dataset identity.

    Strict-46 may canonicalize a physical dataset split (currently
    ``simpleqa/verified``) to a stable logical matrix cell
    (``simpleqa/test``).  Scheduler job detection and benchmark configuration
    are registered against the physical dataset slug, so using the logical
    alias here incorrectly makes an otherwise valid task look unregistered.
    Rows without provenance fields retain the historical fallback.
    """

    benchmark_name = task.get("source_benchmark_name") or task.get(
        "benchmark_name", ""
    )
    benchmark_split = task.get("source_benchmark_split") or task.get(
        "benchmark_split", ""
    )
    return canonical_slug(f"{benchmark_name}_{benchmark_split}")


def _expected_evaluator(task: dict[str, Any]) -> str | None:
    slug = _task_source_slug(task)
    base_job = detect_job_from_dataset(slug, task.get("domain") == "math")
    return f"{base_job}_naive" if base_job else None


def _expected_avg_k(task: dict[str, Any]) -> float:
    slug = _task_source_slug(task)
    config = resolve_benchmark_model_config(
        slug,
        str(task.get("model_name") or ""),
        stage=None,
    )
    values = config.avg_k if config is not None and config.avg_k is not None else ()
    integer_values = [
        value
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    if integer_values:
        return float(max(integer_values))
    if values:
        return float(values[-1])
    return 1.0


def _expected_effective_sample_count(task: dict[str, Any]) -> int | None:
    benchmark_samples = _safe_int(task.get("benchmark_num_samples"))
    if benchmark_samples <= 0:
        return None
    expected = benchmark_samples * _expected_avg_k(task)
    if not float(expected).is_integer():
        return None
    return int(expected)


def _expected_sampling_stages(task: dict[str, Any]) -> dict[str, dict[str, object]]:
    """Rebuild the persisted stage contract used by each strict-46 runner."""

    domain = str(task.get("domain") or "")
    if domain == "knowledge":
        # Knowledge direct answer sampling is model/record-dependent because
        # the exact allowed token IDs depend on the tokenizer and choice count.
        # Its protocol is audited from the stored prompt and exact raw output.
        return {}
    slug = _task_source_slug(task)
    model_name = str(task.get("model_name") or "")

    def resolved(stage: str | None) -> dict[str, object] | None:
        config = resolve_sampling_config(slug, model_name, stage=stage)
        return sampling_config_to_dict(config) if config is not None else None

    if domain == "math":
        cot = resolved("cot")
        final = resolved("final")
        if cot is None or final is None:
            return {}
        strategy_a = dict(cot)
        bad_words = strategy_a.get("bad_words")
        if isinstance(bad_words, list) and "</think>" in bad_words:
            filtered = [word for word in bad_words if str(word).strip() != "</think>"]
            if filtered:
                strategy_a["bad_words"] = filtered
            else:
                strategy_a.pop("bad_words", None)
            strategy_a.pop("min_think_tokens", None)
        return {"stage1": cot, "stage2": final, "strategy_a": strategy_a}
    if domain == "coding":
        source_benchmark_name = task.get("source_benchmark_name") or task.get(
            "benchmark_name"
        )
        stage = "final" if str(source_benchmark_name) == "livecodebench" else None
        direct = resolved(stage)
        return {"stage1": direct} if direct is not None else {}
    if domain == "instruction_following":
        direct = resolved(None)
        return {"stage1": direct} if direct is not None else {}
    return {}


def _sampling_protocol_reasons(task: dict[str, Any]) -> list[str]:
    domain = str(task.get("domain") or "")
    if domain == "knowledge":
        return []
    expected = _expected_sampling_stages(task)
    if not expected:
        return ["missing_expected_sampling_contract"]
    outer = task.get("sampling_config")
    actual = outer.get("sampling_config") if isinstance(outer, dict) else None
    if not isinstance(actual, dict):
        return ["missing_persisted_sampling_config"]
    reasons: list[str] = []
    for stage, expected_config in expected.items():
        actual_config = actual.get(stage)
        if not isinstance(actual_config, dict):
            reasons.append(f"missing_sampling_stage:{stage}")
            continue
        for key, expected_value in expected_config.items():
            actual_value = actual_config.get(key)
            if actual_value != expected_value:
                reasons.append(
                    f"sampling:{stage}.{key}:{actual_value!r}"
                    f"!=expected:{expected_value!r}"
                )
    return reasons


def _blank_recovery_protocol_reasons(
    task: dict[str, Any],
    *,
    require_eval_match: bool = True,
) -> list[str]:
    """Validate empty final-recovery rows without rejecting model abstention.

    A structurally present but empty stage 2 is a genuine model output.  The
    fail-closed evaluator normally scores it false, persists an empty answer,
    labels it ``missing_recovery_prediction``, and never inspects stage 1 or
    the prompt.  The one intentional exception is the strategy cascade: B/C
    reuse a Strategy-A coordinate which already passed (including its judge
    result) before evaluating stage 2.  Such inheritance is accepted only
    when the score-linked Strategy-A companion has a matching coordinate and
    both eval rows pass with identical non-empty answer/reference evidence.
    The raw blank count must be fully explained by the sum of explicit misses
    and those verified inheritances.

    Historical rows from before the scorer deployment still require replay:
    their stored eval may contain prompt-derived answers.  Active tasks skip
    the raw/eval equality check because eval rows are only written after
    generation finishes, but the deployment cutoff remains enforceable while
    they run.
    """

    blank_count = int(task.get("blank_recovery_stage_count") or 0)
    explicit_count = int(task.get("missing_recovery_prediction_count") or 0)
    inherited_count = int(
        task.get("blank_recovery_strategy_a_inheritance_count") or 0
    )
    explained_count = explicit_count + inherited_count
    if not blank_count:
        if require_eval_match and explained_count:
            return [
                "blank_recovery_eval_mismatch:"
                f"raw=0,missing={explicit_count},"
                f"inherited_a={inherited_count}"
            ]
        return []
    task_created_at = task.get("task_created_at")
    if not isinstance(task_created_at, datetime):
        return [f"blank_recovery_stage_missing_task_timestamp:{blank_count}"]
    if task_created_at < BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT:
        return [
            "blank_recovery_stage_predates_missing_protocol_fix:"
            f"{blank_count}"
        ]
    if require_eval_match and explained_count != blank_count:
        return [
            "blank_recovery_eval_mismatch:"
            f"raw={blank_count},missing={explicit_count},"
            f"inherited_a={inherited_count}"
        ]
    return []


def _blank_strategy_a_protocol_reasons(
    task: dict[str, Any],
    *,
    require_eval_match: bool = True,
) -> list[str]:
    """Require explicit empty Strategy-A generations to fail closed.

    A dedicated Strategy A is authoritative when its object exists.  An empty
    completion therefore cannot fall back to stage 1, be sent to an LLM judge,
    or be inherited by Strategy B/C.  Raw/eval equality makes that invariant
    auditable without treating a genuine model abstention as missing storage.
    """

    blank_count = int(task.get("blank_strategy_a_generation_count") or 0)
    explicit_count = int(task.get("missing_strategy_a_prediction_count") or 0)
    if require_eval_match and explicit_count != blank_count:
        return [
            "blank_strategy_a_eval_mismatch:"
            f"raw={blank_count},eval={explicit_count}"
        ]
    return []


def _judge_determinism_protocol_reasons(task: dict[str, Any]) -> list[str]:
    """Reject root external-Judge rows created before deterministic sampling.

    Strategy-A/B diagnostic evaluators include ``free_response_judge_naive``
    as a prefix, but they are not the score-bearing root task.  Exact equality
    keeps this gate cell-level and avoids invalidating auxiliary evidence.
    """

    if str(task.get("evaluator") or "").lower() != "free_response_judge_naive":
        return []
    task_created_at = task.get("task_created_at")
    if not isinstance(task_created_at, datetime):
        return ["judge_sampling_missing_task_timestamp"]
    if task_created_at < JUDGE_DETERMINISM_DEPLOYED_AT:
        return ["judge_sampling_predates_deterministic_fix"]
    return []


def _active_protocol_reasons(
    task: dict[str, Any],
    representative_prompt: str,
    raw_values: list[str | None],
) -> list[str]:
    """Catch recoverable protocol drift before an active task writes a score."""

    reasons = _sampling_protocol_reasons(task)
    reasons.extend(
        _blank_recovery_protocol_reasons(task, require_eval_match=False)
    )
    reasons.extend(
        _blank_strategy_a_protocol_reasons(task, require_eval_match=False)
    )
    reasons.extend(_judge_determinism_protocol_reasons(task))
    if int(task.get("leading_orphan_close_count") or 0):
        reasons.append(
            f"leading_orphan_close:{int(task['leading_orphan_close_count'])}"
        )
    if int(task.get("judgement_output_source_mismatch_count") or 0):
        reasons.append(
            "judgement_output_source_mismatch:"
            f"{int(task['judgement_output_source_mismatch_count'])}"
        )
    task_created_at = task.get("task_created_at")
    if (
        isinstance(task_created_at, datetime)
        and task_created_at < RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT
    ):
        reasons.append("generation_predates_raw_completions_protocol_fix")
    benchmark = (
        str(task.get("benchmark_name") or ""),
        str(task.get("benchmark_split") or ""),
    )
    target = TARGETS.get(benchmark)
    if target is None:
        return reasons
    domain, expected_mode = target
    outer = task.get("sampling_config")
    configured_mode = outer.get("cot_mode") if isinstance(outer, dict) else None
    if _normalize_mode(configured_mode) != expected_mode:
        reasons.append(
            f"configured_mode:{_normalize_mode(configured_mode) or 'empty'}"
            f"!=expected:{expected_mode}"
        )
    if representative_prompt:
        if domain == "knowledge":
            if (
                "Assistant: <think></think>" not in representative_prompt
                or not representative_prompt.endswith("The answer is")
            ):
                reasons.append("knowledge_prompt_protocol")
            if any(not re.fullmatch(r"\s*[A-Z]\s*", raw or "") for raw in raw_values):
                reasons.append("knowledge_raw_choice_protocol")
        elif domain == "instruction_following" and not representative_prompt.endswith(
            "Assistant: <think></think>\n"
        ):
            reasons.append("instruction_nocot_empty_think_protocol")
        elif domain == "coding" and not re.search(
            r"Assistant: <think>\n?</think>\n```python\n?\Z",
            representative_prompt,
        ):
            reasons.append("coding_nocot_empty_think_protocol")
        elif domain == "math" and not representative_prompt.endswith(
            "Assistant: <think"
        ):
            reasons.append("math_cot_open_think_protocol")
    if (
        domain == "math"
        and isinstance(task_created_at, datetime)
        and task_created_at < MATH_FINAL_TEXT_STOP_DEPLOYED_AT
        and int(task.get("recovery_stage_count") or 0) > 0
    ):
        reasons.append("math_final_text_stop_predates_global_fix")
    return reasons


def _general_protocol_reasons(task: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reasons.extend(_blank_recovery_protocol_reasons(task))
    reasons.extend(_blank_strategy_a_protocol_reasons(task))
    reasons.extend(_judge_determinism_protocol_reasons(task))
    task_created_at = task.get("task_created_at")
    if (
        isinstance(task_created_at, datetime)
        and task_created_at < RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT
    ):
        reasons.append("generation_predates_raw_completions_protocol_fix")
    status = str(task.get("status") or "")
    if status != "Completed":
        reasons.append(f"scored_task_status:{status or 'empty'}")

    sampling_config = task.get("sampling_config") or {}
    evaluator = str(task.get("evaluator") or "").lower()
    expected_evaluator = _expected_evaluator(task)
    if expected_evaluator is None:
        reasons.append("missing_expected_evaluator_contract")
    elif evaluator != expected_evaluator:
        reasons.append(f"evaluator:{evaluator}!=expected:{expected_evaluator}")

    expected_avg_k = _expected_avg_k(task)
    try:
        actual_avg_k = float(sampling_config.get("avg_k"))
    except (TypeError, ValueError):
        reasons.append("missing_or_invalid_avg_k")
    else:
        if abs(actual_avg_k - expected_avg_k) > 1e-12:
            reasons.append(f"avg_k:{actual_avg_k}!=expected:{expected_avg_k}")
    prompt_profile = str(sampling_config.get("prompt_profile") or "").lower()
    if prompt_profile != "naive":
        reasons.append("not_naive_protocol")
    reasons.extend(_sampling_protocol_reasons(task))

    if sampling_config.get("sample_limit") is not None:
        reasons.append(f"sample_limit:{sampling_config.get('sample_limit')}")

    completion_count = int(task.get("completion_count") or 0)
    eval_count = int(task.get("eval_count") or 0)
    recorded_expected_count = int(sampling_config.get("effective_sample_count") or 0)
    expected_count = _expected_effective_sample_count(task)
    if expected_count is None:
        reasons.append("missing_or_invalid_benchmark_num_samples")
    elif recorded_expected_count != expected_count:
        reasons.append(
            "effective_sample_count:"
            f"{recorded_expected_count}!=expected:{expected_count}"
        )
    if recorded_expected_count <= 0:
        reasons.append("missing_effective_sample_count")
    if expected_count is not None and completion_count != expected_count:
        reasons.append(
            f"completion_count:{completion_count}!=expected:{expected_count}"
        )
    if completion_count <= 0:
        reasons.append("zero_completions")
    if eval_count != completion_count:
        reasons.append(f"eval_count:{eval_count}!=completions:{completion_count}")
    distinct_coordinates = int(task.get("distinct_completion_coordinates") or 0)
    if distinct_coordinates != completion_count:
        reasons.append(
            "distinct_completion_coordinates:"
            f"{distinct_coordinates}!=completions:{completion_count}"
        )
    benchmark_samples = _safe_int(task.get("benchmark_num_samples"))
    distinct_samples = int(task.get("distinct_sample_indices") or 0)
    if benchmark_samples > 0 and distinct_samples != benchmark_samples:
        reasons.append(
            f"distinct_sample_indices:{distinct_samples}!=expected:{benchmark_samples}"
        )
    if benchmark_samples > 0 and (
        _safe_int(task.get("min_sample_index")) != 0
        or _safe_int(task.get("max_sample_index")) != benchmark_samples - 1
    ):
        reasons.append(
            "sample_index_range:"
            f"{_safe_int(task.get('min_sample_index'))}.."
            f"{_safe_int(task.get('max_sample_index'))}"
            f"!=expected:0..{benchmark_samples - 1}"
        )
    expected_avg_repeats = int(expected_avg_k)
    distinct_avg_repeats = int(task.get("distinct_avg_repeat_indices") or 0)
    if distinct_avg_repeats != expected_avg_repeats:
        reasons.append(
            "distinct_avg_repeat_indices:"
            f"{distinct_avg_repeats}!=expected:{expected_avg_repeats}"
        )
    if (
        _safe_int(task.get("min_avg_repeat_index")) != 0
        or _safe_int(task.get("max_avg_repeat_index")) != expected_avg_repeats - 1
    ):
        reasons.append(
            "avg_repeat_index_range:"
            f"{_safe_int(task.get('min_avg_repeat_index'))}.."
            f"{_safe_int(task.get('max_avg_repeat_index'))}"
            f"!=expected:0..{expected_avg_repeats - 1}"
        )
    if int(task.get("blank_primary_generation_count") or 0):
        reasons.append(
            f"blank_primary_generation:{int(task['blank_primary_generation_count'])}"
        )
    if int(task.get("missing_prediction_count") or 0):
        reasons.append(f"missing_prediction:{int(task['missing_prediction_count'])}")
    if int(task.get("leading_orphan_close_count") or 0):
        reasons.append(
            f"leading_orphan_close:{int(task['leading_orphan_close_count'])}"
        )
    if int(task.get("judgement_output_source_mismatch_count") or 0):
        reasons.append(
            "judgement_output_source_mismatch:"
            f"{int(task['judgement_output_source_mismatch_count'])}"
        )

    metrics = task.get("metrics") or {}
    if task.get("domain") == "math" and "judge" in evaluator:
        judge_stats = metrics.get("judge_stats") if isinstance(metrics, dict) else None
        if not isinstance(judge_stats, dict):
            reasons.append("missing_persisted_judge_stats")
        else:
            judge_total = _safe_int(judge_stats.get("total"))
            judge_parsed = _safe_int(judge_stats.get("parsed_count"))
            judge_invalid = _safe_int(judge_stats.get("invalid_output_count"))
            judge_request = _safe_int(judge_stats.get("request_error_count"))
            judge_errors = _safe_int(judge_stats.get("error_count"))
            if judge_parsed != judge_total:
                reasons.append(
                    f"judge_parsed_count:{judge_parsed}!=total:{judge_total}"
                )
            if judge_invalid or judge_request or judge_errors:
                reasons.append(
                    "judge_errors:"
                    f"invalid:{judge_invalid},request:{judge_request},"
                    f"total:{judge_errors}"
                )
            sampling_config = task.get("sampling_config") or {}
            expected_model = str(
                sampling_config.get("judger_model_name")
                if isinstance(sampling_config, dict)
                else ""
            ) or os.environ.get("JUDGE_MODEL", "")
            slug = canonical_slug(
                f"{task.get('benchmark_name', '')}_"
                f"{task.get('benchmark_split', '')}"
            )
            config = resolve_benchmark_model_config(
                slug,
                str(task.get("model_name") or ""),
                stage=None,
            )
            expected_prompt = (
                config.judge_prompt_template
                if config is not None and config.judge_prompt_template
                else DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE
            )
            reasons.extend(
                llm_judge_protocol_stats_reasons(
                    judge_stats,
                    expected_model=expected_model or None,
                    expected_prompt_template=expected_prompt,
                    expected_max_completion_tokens=resolve_judge_max_tokens(None),
                )
            )
    if (
        task.get("domain") == "math"
        and isinstance(task_created_at, datetime)
        and task_created_at < MATH_FINAL_TEXT_STOP_DEPLOYED_AT
        and int(task.get("recovery_stage_count") or 0) > 0
    ):
        reasons.append("math_final_text_stop_predates_global_fix")

    primary_metric = _primary_numeric_metric(metrics)
    if primary_metric is None:
        reasons.append("missing_primary_numeric_metric")
    elif eval_count:
        metric_name, metric_value = primary_metric
        eval_pass_rate = int(task.get("passed_eval_count") or 0) / eval_count
        if abs(metric_value - eval_pass_rate) > 1e-12:
            reasons.append(
                f"metric_eval_mismatch:{metric_name}:{metric_value}!={eval_pass_rate}"
            )
    return reasons


def _final_protocol_assessment(
    task: dict[str, Any],
    diagnostic_knowledge_replay_ids: set[int],
) -> tuple[list[str], bool]:
    """Return immutable final gates plus any diagnostic replay provenance.

    Replaying a historical completion through the current answer adapter is
    useful evidence about extraction, but it cannot repair the prompt that was
    sent to the model.  In particular, a task created before the raw
    completions deployment may contain a double-applied chat template.  Keep
    that provenance visible without ever removing a current-protocol failure.
    """

    task_id = int(task.get("task_id") or 0)
    has_diagnostic_replay = (
        str(task.get("domain") or "") == "knowledge"
        and task_id in diagnostic_knowledge_replay_ids
    )
    return _general_protocol_reasons(task), has_diagnostic_replay


_KNOWLEDGE_REPLAY_DIAGNOSTIC_MARKERS = (
    "diagnostic_only",
    "knowledge_replay_diagnostic_evidence",
    "replay_eligible_except_cutoff",
)


def _is_diagnostic_knowledge_replay_row(row: dict[str, Any]) -> bool:
    """Accept current and legacy replay provenance markers as evidence only."""

    return any(bool(row.get(marker)) for marker in _KNOWLEDGE_REPLAY_DIAGNOSTIC_MARKERS)


def _main_unlocked() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary-since-task-id", type=int)
    parser.add_argument(
        "--knowledge-replay-report",
        type=Path,
        default=Path("logs/audits/g1h_g1i_knowledge_replay_frontend46_20260806.json"),
        help=(
            "Read-only diagnostic replay evidence. It is reported as provenance "
            "only and never waives the current raw-completions protocol cutoff."
        ),
    )
    parser.add_argument(
        "--require-model-complete",
        choices=MODELS,
        help="Exit non-zero unless the selected model passes all 46 acceptance gates.",
    )
    parser.add_argument(
        "--final-extractor-lineage-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_EXTRACTOR_LINEAGE_SHA256",
            ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
        ),
    )
    parser.add_argument(
        "--final-imported-free-response-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_IMPORTED_FREE_RESPONSE_SHA256",
            FINAL_IMPORTED_FREE_RESPONSE_SHA256,
        ),
    )
    parser.add_argument(
        "--final-comparator-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_COMPARATOR_SHA256",
            FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
        ),
    )
    parser.add_argument(
        "--final-math-verify-version",
        default=os.environ.get(
            "RWKV_FINAL_MATH_VERIFY_VERSION",
            FINAL_MATH_VERIFY_VERSION,
        ),
    )
    parser.add_argument(
        "--final-git-hash",
        default=os.environ.get(
            "RWKV_FINAL_MATH_REPLAY_GIT_HASH",
            FINAL_REPLAY_GIT_HASH,
        ),
    )
    parser.add_argument(
        "--final-reason-tag",
        default=os.environ.get("RWKV_FINAL_MATH_REPLAY_REASON_TAG", ""),
    )
    args = parser.parse_args()

    math_replay_contract = FinalMathReplayContract.from_values(
        extractor_lineage_sha256=args.final_extractor_lineage_sha256,
        imported_free_response_sha256=args.final_imported_free_response_sha256,
        comparator_implementation_sha256=args.final_comparator_sha256,
        math_verify_version=args.final_math_verify_version,
        replay_git_hash=args.final_git_hash,
        reason_tag=args.final_reason_tag,
    )

    diagnostic_knowledge_replay_ids: set[int] = set()
    if args.knowledge_replay_report and args.knowledge_replay_report.exists():
        replay_report = json.loads(
            args.knowledge_replay_report.read_text(encoding="utf-8")
        )
        diagnostic_knowledge_replay_ids = {
            int(row["task_id"])
            for row in replay_report.get("tasks", [])
            if row.get("task_id") is not None
            and _is_diagnostic_knowledge_replay_row(row)
        }

    # This strict-46 campaign deliberately uses the approved G1h/G1i field
    # protocol root for every size.  Pin the audit to the same source of truth
    # instead of inheriting a caller's unrelated config-root environment.
    load_env_file(Path(__file__).resolve().parents[2] / ".env")
    os.environ["RWKV_BENCHMARK_CONFIG_ROOT"] = str(STRICT_CONFIG_ROOT)

    db_config = replace(DEFAULT_DB_CONFIG, dbname=DB_NAME)
    conninfo = _build_conninfo(db_config)
    with psycopg.connect(conninfo, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            rows = _load_task_rows(
                cursor,
                diagnostic_knowledge_replay_ids=diagnostic_knowledge_replay_ids,
                content_conninfo=conninfo,
            )
            math_replay_source_tasks = _load_math_replay_source_tasks(cursor, rows)
            (
                raw_by_task,
                prompt_by_task,
                truncation_examples_by_task,
            ) = _load_completion_audit_maps(
                cursor,
                rows,
                diagnostic_knowledge_replay_ids=diagnostic_knowledge_replay_ids,
            )
            for row in rows:
                representative_prompt = prompt_by_task[int(row["task_id"])]
                row["representative_prompt_tail"] = representative_prompt[-160:]
            scored_knowledge_task_ids = [
                int(row["task_id"])
                for row in rows
                if row["score_created_at"] is not None
                and (str(row["benchmark_name"]), str(row["benchmark_split"]))
                in KNOWLEDGE
            ]
            choice_label_rows_by_task: dict[int, list[dict[str, Any]]] = {}
            if scored_knowledge_task_ids:
                cursor.execute(
                    CHOICE_LABEL_AUDIT_QUERY,
                    (scored_knowledge_task_ids,),
                )
                for item in cursor.fetchall():
                    choice_label_rows_by_task.setdefault(
                        int(item["task_id"]), []
                    ).append(dict(item))
            scored_free_response_task_ids = [
                int(row["task_id"])
                for row in rows
                if row["score_created_at"] is not None
                and str(row.get("evaluator") or "").startswith("free_response")
            ]
            judgement_mismatches_by_task: Counter[int] = Counter()
            judgement_rows_by_task: Counter[int] = Counter()
            judgement_reference_rows_by_task: Counter[int] = Counter()
            if scored_free_response_task_ids:
                cursor.execute(
                    JUDGEMENT_TASK_PROBE_QUERY,
                    (scored_free_response_task_ids,),
                )
                judgement_task_ids = [
                    int(item["task_id"])
                    for item in cursor.fetchall()
                    if _is_judgement_reference(str(item.get("ref_answer") or ""))
                ]
                if judgement_task_ids:
                    cursor.execute(
                        JUDGEMENT_OUTPUT_AUDIT_QUERY,
                        (judgement_task_ids,),
                    )
                    for item in cursor.fetchall():
                        item_dict = dict(item)
                        task_id = int(item_dict["task_id"])
                        judgement_rows_by_task[task_id] += 1
                        if _is_judgement_reference(
                            str(item_dict.get("ref_answer") or "")
                        ):
                            judgement_reference_rows_by_task[task_id] += 1
                        if _judgement_output_source_mismatch(item_dict):
                            judgement_mismatches_by_task[task_id] += 1
            for row in rows:
                task_id = int(row["task_id"])
                row["judgement_output_source_row_count"] = int(
                    judgement_rows_by_task.get(task_id, 0)
                )
                row["judgement_output_source_mismatch_count"] = int(
                    judgement_mismatches_by_task.get(task_id, 0)
                )
                row["judgement_reference_row_count"] = int(
                    judgement_reference_rows_by_task.get(task_id, 0)
                )
            reference_model_names = [
                model_name
                for architecture in REFERENCE_MODELS.values()
                for model_name in architecture.values()
            ]
            cursor.execute(REFERENCE_QUERY, (reference_model_names,))
            reference_rows = [dict(row) for row in cursor.fetchall()]

    latest_valid: dict[tuple[str, str, str], dict[str, Any]] = {}
    superseded_valid: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    invalid_scored: list[dict[str, Any]] = []
    for row in rows:
        benchmark = canonicalize_task_benchmark(row)
        target = TARGETS.get(benchmark)
        if target is None:
            continue
        domain, expected_mode = target
        row["domain"] = domain
        status = str(row["status"])
        if status == "Running":
            active.append(row)
        elif status == "Failed":
            failed.append(row)
        if row["score_created_at"] is None:
            continue
        mode_ok = _normalize_mode(row["cot_mode"]) == expected_mode
        general_protocol_reasons, has_diagnostic_knowledge_replay = (
            _final_protocol_assessment(row, diagnostic_knowledge_replay_ids)
        )
        if domain == "math":
            general_protocol_reasons.extend(
                _math_final_provenance_reasons(
                    row,
                    contract=math_replay_contract,
                    source_tasks=math_replay_source_tasks,
                )
            )
        row["knowledge_replay_diagnostic_evidence"] = has_diagnostic_knowledge_replay
        protocol_ok = mode_ok and not general_protocol_reasons
        knowledge_protocol_ok = True
        instruction_protocol_ok = True
        coding_protocol_ok = True
        math_protocol_ok = True
        if domain == "knowledge":
            knowledge_protocol_ok = _knowledge_protocol_ok(
                row,
                raw_by_task.get(int(row["task_id"]), []),
                prompt_by_task.get(int(row["task_id"]), ""),
            )
            protocol_ok = protocol_ok and knowledge_protocol_ok
        elif domain == "instruction_following":
            instruction_protocol_ok = _instruction_protocol_ok(
                row,
                prompt_by_task.get(int(row["task_id"]), ""),
            )
            protocol_ok = protocol_ok and instruction_protocol_ok
        elif domain == "coding":
            coding_protocol_ok = _coding_protocol_ok(
                row,
                prompt_by_task.get(int(row["task_id"]), ""),
            )
            protocol_ok = protocol_ok and coding_protocol_ok
        elif domain == "math":
            math_protocol_ok = _math_protocol_ok(
                row,
                prompt_by_task.get(int(row["task_id"]), ""),
            )
            protocol_ok = protocol_ok and math_protocol_ok
        if not protocol_ok:
            invalid_row = dict(row)
            invalid_reasons = []
            invalid_reasons.extend(general_protocol_reasons)
            if not mode_ok:
                invalid_reasons.append(
                    f"mode:{_normalize_mode(row['cot_mode']) or 'empty'}!=expected:{expected_mode}"
                )
            if not knowledge_protocol_ok:
                invalid_reasons.append("knowledge_generation_or_extraction_protocol")
            if not instruction_protocol_ok:
                invalid_reasons.append("instruction_nocot_empty_think_protocol")
            if not coding_protocol_ok:
                invalid_reasons.append("coding_nocot_empty_think_protocol")
            if not math_protocol_ok:
                invalid_reasons.append("math_cot_open_think_protocol")
            invalid_row["invalid_reasons"] = invalid_reasons
            if domain == "knowledge":
                task_id = int(row["task_id"])
                raw_values = raw_by_task.get(task_id, [])
                invalid_raw = [
                    raw
                    for raw in raw_values
                    if not re.fullmatch(r"\s*[A-Z]\s*", raw or "")
                ]
                prompt = prompt_by_task.get(task_id, "")
                invalid_row["protocol_audit"] = {
                    "task_id_is_current": task_id >= 28457,
                    "representative_prompt_tail": prompt[-160:],
                    "prompt_has_empty_think": "Assistant: <think></think>" in prompt,
                    "prompt_ends_with_answer_prefix": prompt.endswith("The answer is"),
                    "raw_count": len(raw_values),
                    "invalid_raw_count": len(invalid_raw),
                    "invalid_raw_examples": invalid_raw[:10],
                }
            elif domain == "instruction_following":
                prompt = prompt_by_task.get(int(row["task_id"]), "")
                invalid_row["protocol_audit"] = {
                    "representative_prompt_tail": prompt[-160:],
                    "prompt_has_empty_think": prompt.endswith(
                        "Assistant: <think></think>\n"
                    ),
                }
            elif domain in {"coding", "math"}:
                invalid_row["protocol_audit"] = {
                    "representative_prompt_tail": prompt_by_task.get(
                        int(row["task_id"]), ""
                    )[-160:],
                }
            invalid_scored.append(invalid_row)
            continue
        cell = (str(row["model_name"]), benchmark[0], benchmark[1])
        _record_valid_candidate(latest_valid, superseded_valid, cell, row)

    cells_by_model: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        missing = []
        completed = []
        for benchmark, (domain, expected_mode) in TARGETS.items():
            cell = (model, benchmark[0], benchmark[1])
            row = latest_valid.get(cell)
            descriptor = {
                "benchmark": f"{benchmark[0]}__{benchmark[1]}",
                "domain": domain,
                "mode": expected_mode,
            }
            if row is None:
                missing.append(descriptor)
            else:
                completed.append({**descriptor, "task_id": int(row["task_id"])})
        cells_by_model[model] = {
            "complete": len(completed),
            "missing": len(missing),
            "completed_cells": completed,
            "missing_cells": missing,
        }

    status_counts = Counter(str(row["status"]) for row in rows)
    invalid_reason_counts = Counter(
        reason for row in invalid_scored for reason in row.get("invalid_reasons", [])
    )
    valid_task_rows = []
    for row in sorted(latest_valid.values(), key=lambda item: int(item["task_id"])):
        task_id = int(row["task_id"])
        raw_counts = Counter(
            (raw or "").strip() for raw in raw_by_task.get(task_id, [])
        )
        choice_rows = choice_label_rows_by_task.get(task_id, [])
        predicted_label_counts = Counter()
        reference_label_counts = Counter()
        confusion_counts = Counter()
        for choice_row in choice_rows:
            answer = str(choice_row.get("answer") or "")
            reference = str(choice_row.get("ref_answer") or "")
            count = int(choice_row.get("row_count") or 0)
            predicted_label_counts[answer] += count
            reference_label_counts[reference] += count
            confusion_counts[f"{reference}->{answer}"] += count
        valid_task_rows.append(
            {
                "model_name": row["model_name"],
                "benchmark": f"{row['benchmark_name']}__{row['benchmark_split']}",
                "source_benchmark_name": row.get(
                    "source_benchmark_name", row["benchmark_name"]
                ),
                "source_benchmark_split": row.get(
                    "source_benchmark_split", row["benchmark_split"]
                ),
                "domain": row["domain"],
                "task_id": task_id,
                "score_id": _safe_int(row.get("score_id")),
                "status": row["status"],
                "task_git_hash": row.get("task_git_hash"),
                "task_desc": row.get("task_desc"),
                "math_provenance_gate": row.get("math_provenance_gate"),
                "task_created_at": _json_value(row["task_created_at"]),
                "evaluator": row["evaluator"],
                "cot_mode": row["cot_mode"],
                "sampling_config": row["sampling_config"],
                "metrics": row["metrics"],
                "score_created_at": _json_value(row["score_created_at"]),
                "representative_prompt_tail": prompt_by_task.get(task_id, "")[-160:],
                "expected_completion_count": int(
                    _expected_effective_sample_count(row) or 0
                ),
                "completion_count": int(row["completion_count"]),
                "eval_count": int(row["eval_count"]),
                "distinct_completion_coordinates": int(
                    row["distinct_completion_coordinates"]
                ),
                "distinct_sample_indices": int(row["distinct_sample_indices"]),
                "sample_index_range": [
                    _safe_int(row["min_sample_index"]),
                    _safe_int(row["max_sample_index"]),
                ],
                "distinct_avg_repeat_indices": int(row["distinct_avg_repeat_indices"]),
                "avg_repeat_index_range": [
                    _safe_int(row["min_avg_repeat_index"]),
                    _safe_int(row["max_avg_repeat_index"]),
                ],
                "completion_eval_count_match": int(row["completion_count"])
                == int(row["eval_count"]),
                "completion_expected_count_match": bool(
                    _expected_effective_sample_count(row)
                )
                and int(row["completion_count"])
                == _expected_effective_sample_count(row),
                "blank_raw_count": (
                    int(row["blank_raw_count"])
                    if row["domain"] == "knowledge"
                    else None
                ),
                "blank_stage0_count": int(row["blank_stage0_count"]),
                "blank_primary_generation_count": int(
                    row["blank_primary_generation_count"]
                ),
                "leading_orphan_close_count": int(row["leading_orphan_close_count"]),
                "passed_eval_count": int(row["passed_eval_count"]),
                "eval_pass_rate": (
                    int(row["passed_eval_count"]) / int(row["eval_count"])
                    if int(row["eval_count"])
                    else None
                ),
                "missing_prediction_count": int(row["missing_prediction_count"]),
                "missing_recovery_prediction_count": int(
                    row["missing_recovery_prediction_count"]
                ),
                "blank_recovery_strategy_a_inheritance_count": int(
                    row["blank_recovery_strategy_a_inheritance_count"]
                ),
                "missing_strategy_a_prediction_count": int(
                    row["missing_strategy_a_prediction_count"]
                ),
                "length_finish_count": (
                    int(row["length_finish_count"])
                    if row["domain"] == "knowledge"
                    else None
                ),
                "stage0_length_stop_count": int(row["stage0_length_stop_count"]),
                "overall_truncation_count": int(row["overall_truncation_count"]),
                "overall_truncation_rate": (
                    int(row["overall_truncation_count"]) / int(row["completion_count"])
                    if int(row["completion_count"])
                    else None
                ),
                "initial_generation_truncation_count": int(
                    row["initial_generation_truncation_count"]
                ),
                "final_stage_truncation_count": int(
                    row["final_stage_truncation_count"]
                ),
                "truncated_primary_with_code_fence_count": int(
                    row["truncated_primary_with_code_fence_count"]
                ),
                "strategy_a_completion_count": int(row["strategy_a_completion_count"]),
                "blank_strategy_a_generation_count": int(
                    row["blank_strategy_a_generation_count"]
                ),
                "staged_generation_count": int(row["staged_generation_count"]),
                "recovery_stage_count": int(row["recovery_stage_count"]),
                "blank_recovery_stage_count": int(
                    row["blank_recovery_stage_count"]
                ),
                "truncation_examples": truncation_examples_by_task.get(task_id, []),
                "raw_answer_counts": dict(sorted(raw_counts.items())),
                "predicted_label_counts": dict(sorted(predicted_label_counts.items())),
                "reference_label_counts": dict(sorted(reference_label_counts.items())),
                "choice_confusion_counts": dict(sorted(confusion_counts.items())),
            }
        )
    choice_bias_signals = []
    for row in valid_task_rows:
        if row["domain"] != "knowledge" or not row["predicted_label_counts"]:
            continue
        top_label, top_count = max(
            row["predicted_label_counts"].items(),
            key=lambda item: int(item[1]),
        )
        total = int(row["completion_count"])
        top_share = int(top_count) / total if total else 0.0
        reference_share = (
            int(row["reference_label_counts"].get(top_label, 0)) / total
            if total
            else 0.0
        )
        if top_share < 0.5:
            continue
        choice_bias_signals.append(
            {
                "task_id": row["task_id"],
                "model_name": row["model_name"],
                "benchmark": row["benchmark"],
                "top_predicted_label": top_label,
                "top_prediction_share": top_share,
                "reference_share_for_same_label": reference_share,
                "delta_pp": (top_share - reference_share) * 100.0,
                "investigate": True,
                "signal_only": True,
            }
        )
    curve_comparisons = []
    for benchmark, (domain, _mode) in TARGETS.items():
        for smaller_model, larger_model in zip(MODELS, MODELS[1:], strict=False):
            smaller = latest_valid.get((smaller_model, benchmark[0], benchmark[1]))
            larger = latest_valid.get((larger_model, benchmark[0], benchmark[1]))
            if smaller is None or larger is None:
                continue
            smaller_metrics = smaller.get("metrics") or {}
            larger_metrics = larger.get("metrics") or {}
            for metric in sorted(set(smaller_metrics) & set(larger_metrics)):
                smaller_value = smaller_metrics.get(metric)
                larger_value = larger_metrics.get(metric)
                if not isinstance(smaller_value, (int, float)) or not isinstance(
                    larger_value, (int, float)
                ):
                    continue
                delta_pp = (float(larger_value) - float(smaller_value)) * 100.0
                curve_comparisons.append(
                    {
                        "benchmark": f"{benchmark[0]}__{benchmark[1]}",
                        "domain": domain,
                        "metric": metric,
                        "smaller_model": smaller_model,
                        "smaller_task_id": int(smaller["task_id"]),
                        "smaller_score": float(smaller_value),
                        "larger_model": larger_model,
                        "larger_task_id": int(larger["task_id"]),
                        "larger_score": float(larger_value),
                        "delta_pp": delta_pp,
                        "investigate": delta_pp < -5.0,
                    }
                )
    latest_references: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in reference_rows:
        benchmark = canonicalize_task_benchmark(row)
        target = TARGETS.get(benchmark)
        if target is None:
            continue
        _domain, expected_mode = target
        if _normalize_mode(row["cot_mode"]) != expected_mode:
            continue
        sampling_config = row.get("sampling_config") or {}
        prompt_profile = str(sampling_config.get("prompt_profile") or "").lower()
        evaluator = str(row.get("evaluator") or "").lower()
        if prompt_profile != "naive" and "naive" not in evaluator:
            continue
        cell = (str(row["model_name"]), benchmark[0], benchmark[1])
        previous = latest_references.get(cell)
        if previous is None or int(row["task_id"]) > int(previous["task_id"]):
            latest_references[cell] = row

    reference_comparisons = []
    for g1i_cell, g1i_row in latest_valid.items():
        g1i_model, benchmark_name, benchmark_split = g1i_cell
        size = MODEL_SIZE_BY_NAME[g1i_model]
        domain, _mode = TARGETS[(benchmark_name, benchmark_split)]
        for architecture, models_by_size in REFERENCE_MODELS.items():
            reference_model = models_by_size[size]
            reference = latest_references.get(
                (reference_model, benchmark_name, benchmark_split)
            )
            if reference is None:
                continue
            g1i_metrics = g1i_row.get("metrics") or {}
            reference_metrics = reference.get("metrics") or {}
            for metric in sorted(set(g1i_metrics) & set(reference_metrics)):
                g1i_value = g1i_metrics.get(metric)
                reference_value = reference_metrics.get(metric)
                if not isinstance(g1i_value, (int, float)) or not isinstance(
                    reference_value, (int, float)
                ):
                    continue
                delta_pp = (float(g1i_value) - float(reference_value)) * 100.0
                reference_comparisons.append(
                    {
                        "benchmark": f"{benchmark_name}__{benchmark_split}",
                        "domain": domain,
                        "metric": metric,
                        "size": size,
                        "g1i_model": g1i_model,
                        "g1i_task_id": int(g1i_row["task_id"]),
                        "g1i_score": float(g1i_value),
                        "reference_architecture": architecture,
                        "reference_model": reference_model,
                        "reference_task_id": int(reference["task_id"]),
                        "reference_score": float(reference_value),
                        "delta_pp": delta_pp,
                        "investigate": abs(delta_pp) > 5.0,
                        "reference_is_signal_only": True,
                    }
                )
    unresolved_active = []
    superseded_active = []
    for row in active:
        cell = (
            str(row["model_name"]),
            str(row["benchmark_name"]),
            str(row["benchmark_split"]),
        )
        destination = superseded_active if cell in latest_valid else unresolved_active
        destination.append(row)
    unresolved_failed = []
    superseded_failed = []
    for row in failed:
        cell = (
            str(row["model_name"]),
            str(row["benchmark_name"]),
            str(row["benchmark_split"]),
        )
        destination = superseded_failed if cell in latest_valid else unresolved_failed
        destination.append(row)
    active_protocol_issues = []
    for row in active:
        task_id = int(row["task_id"])
        reasons = _active_protocol_reasons(
            row,
            prompt_by_task.get(task_id, ""),
            raw_by_task.get(task_id, []),
        )
        if reasons:
            active_protocol_issues.append(
                {
                    "task_id": task_id,
                    "model_name": row["model_name"],
                    "benchmark": (f"{row['benchmark_name']}__{row['benchmark_split']}"),
                    "completion_count": int(row["completion_count"]),
                    "reasons": reasons,
                }
            )
    result = {
        "generated_at": datetime.now().astimezone(),
        "database": DB_NAME,
        "math_replay_provenance_contract": {
            **math_replay_contract.as_dict(),
            "blockers": math_replay_contract.blockers(),
            "explicitly_disallowed_task_ids": sorted(
                DISALLOWED_FINAL_MATH_TASK_IDS
            ),
        },
        "knowledge_replay_report": (
            str(args.knowledge_replay_report)
            if args.knowledge_replay_report and args.knowledge_replay_report.exists()
            else None
        ),
        "diagnostic_knowledge_replay_task_ids": sorted(diagnostic_knowledge_replay_ids),
        "target_cells": len(TARGETS) * len(MODELS),
        "valid_complete": len(latest_valid),
        "remaining": len(TARGETS) * len(MODELS) - len(latest_valid),
        "models": cells_by_model,
        "task_status_counts": dict(sorted(status_counts.items())),
        "valid_task_rows": valid_task_rows,
        "superseded_valid_target_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in superseded_valid
        ],
        "choice_bias_signals": choice_bias_signals,
        "curve_comparisons": curve_comparisons,
        "curve_inversions_over_5pp": [
            row for row in curve_comparisons if row["investigate"]
        ],
        "reference_comparisons": reference_comparisons,
        "reference_differences_over_5pp": [
            row for row in reference_comparisons if row["investigate"]
        ],
        "active_target_tasks": [
            {key: _json_value(value) for key, value in row.items()} for row in active
        ],
        "active_protocol_issues": active_protocol_issues,
        "unresolved_active_target_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in unresolved_active
        ],
        "superseded_active_target_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in superseded_active
        ],
        "failed_target_tasks": [
            {key: _json_value(value) for key, value in row.items()} for row in failed
        ],
        "unresolved_failed_target_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in unresolved_failed
        ],
        "superseded_failed_target_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in superseded_failed
        ],
        "invalid_scored_tasks": [
            {key: _json_value(value) for key, value in row.items()}
            for row in invalid_scored
        ],
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "truncation_examples_by_task": {
            str(task_id): examples
            for task_id, examples in sorted(truncation_examples_by_task.items())
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=_json_value)
    if args.output:
        _atomic_write_text(args.output, rendered + "\n")
    if args.require_model_complete:
        model_result = cells_by_model[args.require_model_complete]
        if int(model_result["complete"]) != len(TARGETS):
            print(
                f"{args.require_model_complete} is incomplete: "
                f"{model_result['complete']}/{len(TARGETS)}",
            )
            raise SystemExit(24)
    if args.summary:
        summary_payload = {
            "database": result["database"],
            "math_replay_provenance_contract": result[
                "math_replay_provenance_contract"
            ],
            "target_cells": result["target_cells"],
            "valid_complete": result["valid_complete"],
            "remaining": result["remaining"],
            "models": {
                model: {
                    "complete": data["complete"],
                    "missing": data["missing"],
                }
                for model, data in cells_by_model.items()
            },
            "active_target_tasks": len(active),
            "active_protocol_issues": active_protocol_issues,
            "unresolved_active_target_tasks": len(unresolved_active),
            "superseded_active_target_tasks": len(superseded_active),
            "failed_target_tasks": len(failed),
            "unresolved_failed_target_tasks": len(unresolved_failed),
            "superseded_failed_target_tasks": len(superseded_failed),
            "invalid_scored_tasks": len(invalid_scored),
            "superseded_valid_target_tasks": len(superseded_valid),
            "invalid_reason_counts": result["invalid_reason_counts"],
            "curve_comparisons": len(curve_comparisons),
            "curve_inversions_over_5pp": result["curve_inversions_over_5pp"],
            "reference_comparisons": len(reference_comparisons),
            "reference_differences_over_5pp": result["reference_differences_over_5pp"],
        }
        if args.summary_since_task_id is not None:
            summary_payload["valid_scores_since_task_id"] = [
                row
                for row in valid_task_rows
                if int(row["task_id"]) >= args.summary_since_task_id
            ]
            summary_payload["active_since_task_id"] = [
                {
                    "task_id": int(row["task_id"]),
                    "model_name": row["model_name"],
                    "benchmark": f"{row['benchmark_name']}__{row['benchmark_split']}",
                    "completion_count": int(row["completion_count"]),
                    "expected_completion_count": int(
                        (row.get("sampling_config") or {}).get("effective_sample_count")
                        or 0
                    ),
                }
                for row in active
                if int(row["task_id"]) >= args.summary_since_task_id
            ]
        print(
            json.dumps(
                summary_payload,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(rendered)


def main() -> None:
    configured_lock_path = os.environ.get("RWKV_G1I_AUDIT_LOCK_PATH")
    lock_path = Path(configured_lock_path) if configured_lock_path else AUDIT_LOCK_PATH
    with _exclusive_audit_lock(lock_path):
        _main_unlocked()


if __name__ == "__main__":
    main()
