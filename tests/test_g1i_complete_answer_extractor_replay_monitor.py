from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess

from ops.g1i_strict46 import monitor_complete_answer_extractor_replays as monitor
from ops.g1i_strict46.monitor_judge_determinism_replays import _provenance_marker


EXTRACTOR_SHA256 = "a" * 64
REASON_TAG = monitor._reason_tag_for_sha256(EXTRACTOR_SHA256)


def _candidate(task_id: int, created_at: datetime) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_created_at": created_at,
        "evaluator": "free_response_naive",
        "sampling_config": {"prompt_profile": "naive", "cot_mode": "CoT"},
        "cot_mode": "CoT",
        "model_name": "rwkv7-g1i-7.2b-20260805-ctx16384",
        "benchmark_name": "math_500",
        "benchmark_split": "test",
    }


def test_reason_marker_binds_replay_to_frozen_extractor() -> None:
    assert EXTRACTOR_SHA256[:8] in REASON_TAG
    assert _provenance_marker(28762, REASON_TAG) == (
        "replay_source_task_id=28762;" + REASON_TAG
    )


def test_source_query_is_root_naive_cot_math_lane() -> None:
    assert "t.evaluator IN ('free_response_naive', 'free_response_judge_naive')" in (
        monitor.SOURCE_QUERY
    )
    assert "sampling_config->>'prompt_profile'" in monitor.SOURCE_QUERY
    assert "sampling_config->>'cot_mode'" in monitor.SOURCE_QUERY
    assert "NOT LIKE '%replay_source_task_id=%'" not in monitor.SOURCE_QUERY


def test_split_uses_explicit_process_restart_cutoff(monkeypatch) -> None:
    deployed_at = datetime(2026, 8, 8, 12, 0, 0)
    before = _candidate(1, deployed_at - timedelta(microseconds=1))
    after = _candidate(2, deployed_at)
    observed: dict[str, list[int]] = {}

    def select(rows, *, now=None):
        _ = now
        observed["pre"] = [int(row["task_id"]) for row in rows]
        return {}, []

    def post(rows, *, now=None):
        _ = now
        observed["post"] = [int(row["task_id"]) for row in rows]
        return {}, {}

    monkeypatch.setattr(monitor, "_select_latest_valid_sources", select)
    monkeypatch.setattr(monitor, "_split_post_candidates", post)

    monitor._split_candidates([before, after], deployed_at=deployed_at)

    assert observed == {"pre": [1], "post": [2]}


def test_replay_command_has_exact_new_reason_and_append_only_commit() -> None:
    repo = Path("/srv/rwkv-skills")
    output = repo / "logs/audits/extractor/source_28762.json"
    command = monitor._build_replay_command(
        repo=repo,
        replay_script=repo / "ops/g1i_strict46/recompute_math_from_completions.py",
        source_task_id=28762,
        dbname="strict46",
        reason_tag=REASON_TAG,
        output=output,
        judge_max_workers=32,
    )

    assert command[command.index("--reason-tag") + 1] == REASON_TAG
    assert command[command.index("--judge-mode") + 1] == "auto"
    assert command[command.index("--judge-max-workers") + 1] == "32"
    assert command[command.index("--output") + 1] == str(output)
    assert "--commit" in command


def test_nonzero_replay_keeps_blocking_artifact_detail(tmp_path: Path) -> None:
    output = tmp_path / "blocked.json"
    output.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 28762,
                        "replayable": False,
                        "blocked": True,
                        "reason": "evaluation_preflight_failed",
                        "blocking_reasons": [
                            "strategy_c.unresolved_math_verify_timeouts:1"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess([], 2, stdout="", stderr="")

    replay, failure = monitor._subprocess_result(
        completed=completed,
        output=output,
        source_task_id=28762,
    )

    assert replay is not None
    assert replay["blocked"] is True
    assert failure == (
        "subprocess_returncode:2;"
        "replay_not_replayable:evaluation_preflight_failed"
    )
