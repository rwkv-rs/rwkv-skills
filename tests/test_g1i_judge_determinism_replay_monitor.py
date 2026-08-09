from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from ops.g1i_strict46.monitor_judge_determinism_replays import (
    JUDGE_DETERMINISM_DEPLOYED_AT,
    REASON_TAG,
    ROOT_EVALUATOR,
    _build_replay_command,
    _classify_existing_replays,
    _completion_contract_reasons,
    _eligible_source,
    _filter_task_candidates,
    _is_complete_post_cutoff,
    _once_exit_code,
    _plan_replays,
    _provenance_marker,
    _replay_artifact,
    _split_candidates,
    _strict_config_environment,
    _strict_protocol_reasons,
    _terminal_action,
)
from ops.g1i_strict46.audit_current import _expected_sampling_stages
from src.eval.metrics.free_response import (
    LLMJudgeConfig,
    llm_judge_protocol,
    llm_judge_protocol_fingerprint,
)


MODEL = "rwkv7-g1i-7.2b-20260805-ctx16384"
BENCHMARK_PROTOCOL = {
    "amc23": (40, 64),
    "comp_math_24_25": (256, 16),
    "gaokao2023en": (385, 8),
    "minerva_math": (272, 16),
}


def _row(
    task_id: int,
    *,
    benchmark: str = "amc23",
    created_at: datetime | None = None,
    status: str = "Completed",
    score_id: int | None = 100,
    completion_count: int | None = None,
    eval_count: int | None = None,
) -> dict[str, object]:
    benchmark_samples, avg_k = BENCHMARK_PROTOCOL[benchmark]
    expected = benchmark_samples * avg_k
    completion_count = expected if completion_count is None else completion_count
    task_identity = {
        "benchmark_name": benchmark,
        "benchmark_split": "test",
        "model_name": MODEL,
        "domain": "math",
    }
    with _strict_config_environment():
        stages = _expected_sampling_stages(task_identity)
    judge_config = LLMJudgeConfig(
        api_key="test",
        model="judge-model",
        max_workers=32,
    )
    return {
        "task_id": task_id,
        "status": status,
        "task_created_at": created_at
        or JUDGE_DETERMINISM_DEPLOYED_AT - timedelta(minutes=1),
        "evaluator": ROOT_EVALUATOR,
        "sampling_config": {
            "prompt_profile": "naive",
            "cot_mode": "CoT",
            "avg_k": avg_k,
            "effective_sample_count": expected,
            "sample_limit": None,
            "n_shot": 0,
            "judger_model_name": judge_config.model,
            "sampling_config": stages,
        },
        "task_desc": "",
        "model_name": MODEL,
        "benchmark_name": benchmark,
        "benchmark_split": "test",
        "benchmark_num_samples": benchmark_samples,
        "is_param_search": False,
        "is_tmp": False,
        "score_id": score_id,
        "cot_mode": "CoT" if score_id is not None else None,
        "metrics": {
            f"avg@{avg_k}": 0.5,
            "judge_stats": {
                "total": completion_count,
                "parsed_count": completion_count,
                "invalid_output_count": 0,
                "request_error_count": 0,
                "error_count": 0,
                **llm_judge_protocol(judge_config),
                "protocol_fingerprint_sha256": (
                    llm_judge_protocol_fingerprint(judge_config)
                ),
            },
        },
        "score_created_at": None,
        "completion_count": completion_count,
        "total_completion_count": completion_count,
        "non_completed_completion_count": 0,
        "distinct_completion_coordinates": completion_count,
        "distinct_sample_repeat_coordinates": completion_count,
        "distinct_sample_indices": (
            benchmark_samples if completion_count == expected else benchmark_samples - 1
        ),
        "min_sample_index": 0,
        "max_sample_index": (
            benchmark_samples - 1
            if completion_count == expected
            else benchmark_samples - 2
        ),
        "distinct_avg_repeat_indices": avg_k,
        "min_avg_repeat_index": 0,
        "max_avg_repeat_index": avg_k - 1,
        "distinct_pass_indices": 1,
        "min_pass_index": 0,
        "max_pass_index": 0,
        "eval_count": completion_count if eval_count is None else eval_count,
        "passed_eval_count": completion_count // 2,
    }


def test_filter_is_exact_root_naive_cot_strict46_scope() -> None:
    valid = _row(1)
    strategy = {**_row(2), "evaluator": f"{ROOT_EVALUATOR}:strategy_a"}
    exact = {**_row(3), "evaluator": "free_response_naive"}
    wrong_benchmark = {**_row(4), "benchmark_name": "math_500"}
    normal = {
        **_row(5),
        "sampling_config": {**_row(5)["sampling_config"], "prompt_profile": "normal"},
    }
    nocot = {
        **_row(6),
        "sampling_config": {**_row(6)["sampling_config"], "cot_mode": "NoCoT"},
        "cot_mode": "NoCoT",
    }

    filtered = _filter_task_candidates(
        [valid, strategy, exact, wrong_benchmark, normal, nocot]
    )

    assert [row["task_id"] for row in filtered] == [1]


def test_post_cutoff_full_scored_gaokao_resolves_latest_pre_cutoff_source() -> None:
    source = _row(
        28751,
        benchmark="gaokao2023en",
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT - timedelta(seconds=1),
    )
    replay = _row(
        28765,
        benchmark="gaokao2023en",
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
    )

    sources, resolved, pending, _invalid = _split_candidates(
        [source, replay], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    plan = _plan_replays(sources, resolved, pending, {28751: []})

    assert plan["eligible_to_replay"] == []
    assert [row["task_id"] for row in plan["resolved_post_cutoff"]] == [28751]
    assert plan["resolved_post_cutoff"][0]["post_cutoff_task_id"] == 28765


def test_known_fingerprintless_replays_are_invalid_and_sources_reenter_queue() -> None:
    known_post_task_ids = (28765, 28771, 28776, 28779)
    benchmarks = tuple(BENCHMARK_PROTOCOL)
    rows: list[dict[str, object]] = []
    source_task_ids: list[int] = []
    for offset, (post_task_id, benchmark) in enumerate(
        zip(known_post_task_ids, benchmarks, strict=True)
    ):
        source_task_id = 28600 + offset
        source_task_ids.append(source_task_id)
        source = _row(
            source_task_id,
            benchmark=benchmark,
            created_at=JUDGE_DETERMINISM_DEPLOYED_AT - timedelta(seconds=1),
        )
        post = _row(
            post_task_id,
            benchmark=benchmark,
            created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
        )
        post_metrics = deepcopy(post["metrics"])
        del post_metrics["judge_stats"]["protocol_fingerprint_sha256"]
        post["metrics"] = post_metrics
        assert not _is_complete_post_cutoff(post)
        rows.extend((source, post))

    sources, resolved, pending, _invalid = _split_candidates(
        rows,
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
    )
    plan = _plan_replays(
        sources,
        resolved,
        pending,
        {source_task_id: [] for source_task_id in source_task_ids},
    )

    assert resolved == {}
    assert pending == {}
    assert sorted(row["task_id"] for row in plan["eligible_to_replay"]) == sorted(
        source_task_ids
    )


def test_latest_pre_cutoff_amc23_source_is_eligible_once_settled_and_full() -> None:
    older = _row(28575)
    latest = _row(28625)
    sources, resolved, pending, _invalid = _split_candidates(
        [older, latest], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    plan = _plan_replays(sources, resolved, pending, {28625: []})

    assert [row["task_id"] for row in plan["eligible_to_replay"]] == [28625]


def test_running_source_and_running_post_cutoff_task_wait() -> None:
    running_source = _row(28606, benchmark="comp_math_24_25", status="Running")
    sources, resolved, pending, _invalid = _split_candidates(
        [running_source], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    plan = _plan_replays(sources, resolved, pending, {})
    assert [row["task_id"] for row in plan["waiting_sources"]] == [28606]

    post_running = _row(
        28770,
        benchmark="comp_math_24_25",
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
        status="Running",
        score_id=None,
        completion_count=512,
        eval_count=0,
    )
    sources, resolved, pending, _invalid = _split_candidates(
        [running_source, post_running], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    plan = _plan_replays(sources, resolved, pending, {})
    assert plan["waiting_sources"] == []
    assert [row["task_id"] for row in plan["pending_post_cutoff"]] == [28606]
    assert plan["pending_post_cutoff"][0]["post_cutoff_task_id"] == 28770


def test_failed_source_replays_only_when_completion_grid_is_full() -> None:
    complete = _row(28659, status="Failed", score_id=None)
    partial = _row(
        28658,
        benchmark="gaokao2023en",
        status="Failed",
        score_id=None,
        completion_count=2000,
        eval_count=0,
    )
    sources, resolved, pending, invalid = _split_candidates(
        [complete, partial], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    plan = _plan_replays(
        sources,
        resolved,
        pending,
        {28659: [], 28658: []},
        invalid,
    )

    assert [row["task_id"] for row in plan["eligible_to_replay"]] == [28659]
    assert [row["task_id"] for row in plan["ignored_invalid_sources"]] == [28658]


def test_source_reason_marker_is_second_idempotency_gate() -> None:
    source = _row(28625)
    sources, resolved, pending, _invalid = _split_candidates(
        [source], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )

    scored = _row(
        28766,
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
    )
    plan = _plan_replays(sources, resolved, pending, {28625: [scored]})
    assert plan["eligible_to_replay"] == []
    assert [row["task_id"] for row in plan["already_replayed"]] == [28625]
    assert _provenance_marker(28625, REASON_TAG) == (
        "replay_source_task_id=28625;" + REASON_TAG
    )

    running = {**scored, "status": "Running", "score_id": None}
    plan = _plan_replays(sources, resolved, pending, {28625: [running]})
    assert [row["task_id"] for row in plan["pending_existing_replay"]] == [28625]

    failed = {**running, "status": "Failed"}
    plan = _plan_replays(sources, resolved, pending, {28625: [failed]})
    assert [row["task_id"] for row in plan["blocked_existing_replay"]] == [28625]


def test_existing_replay_pending_state_is_strict_and_score_grace_is_bounded() -> None:
    created_at = JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1)
    valid_running = _row(
        28780,
        created_at=created_at,
        status="Running",
        score_id=None,
    )
    invalid_running = {
        **valid_running,
        "sampling_config": {
            **valid_running["sampling_config"],
            "prompt_profile": "normal",
        },
    }
    completed_without_score = {
        **valid_running,
        "status": "Completed",
    }

    assert _classify_existing_replays(
        [valid_running],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
        now=created_at + timedelta(minutes=20),
    )[0] == "pending"
    assert _classify_existing_replays(
        [invalid_running],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
        now=created_at + timedelta(minutes=1),
    )[0] == "blocked"
    assert _classify_existing_replays(
        [completed_without_score],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
        now=created_at + timedelta(minutes=5),
    )[0] == "pending"
    assert _classify_existing_replays(
        [completed_without_score],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
        now=created_at + timedelta(minutes=20),
    )[0] == "blocked"


def test_replay_command_commits_deterministic_judge_with_32_workers() -> None:
    repo = Path("/srv/rwkv-skills")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    output = repo / "logs/audits/judge/source_28625.json"

    command = _build_replay_command(
        repo=repo,
        replay_script=replay_script,
        source_task_id=28625,
        dbname="strict46",
        reason_tag=REASON_TAG,
        output=output,
        judge_max_workers=32,
    )

    assert command[command.index("--judge-mode") + 1] == "auto"
    assert command[command.index("--judge-max-workers") + 1] == "32"
    assert command[command.index("--reason-tag") + 1] == REASON_TAG
    assert command[command.index("--dbname") + 1] == "strict46"
    assert command[command.index("--output") + 1] == str(output)
    assert "--commit" in command


def test_strict_protocol_rejects_wrong_avg_stage_tokens_stop_and_sample_limit() -> None:
    valid = _row(1, benchmark="gaokao2023en")
    assert _strict_protocol_reasons(valid) == []

    wrong_avg = deepcopy(valid)
    wrong_avg["sampling_config"]["avg_k"] = 16
    assert any(
        reason.startswith("avg_k:16.0!=expected:8.0")
        for reason in _strict_protocol_reasons(wrong_avg)
    )

    wrong_stage = deepcopy(valid)
    stage1 = wrong_stage["sampling_config"]["sampling_config"]["stage1"]
    stage1["max_new_tokens"] = 4096
    stage1["stop_tokens"] = [999]
    reasons = _strict_protocol_reasons(wrong_stage)
    assert any("sampling:stage1.max_new_tokens" in reason for reason in reasons)
    assert any("sampling:stage1.stop_tokens" in reason for reason in reasons)

    sampled = deepcopy(valid)
    sampled["sampling_config"]["sample_limit"] = 10
    assert "sample_limit:10" in _strict_protocol_reasons(sampled)


def test_post_cutoff_resolution_requires_clean_persisted_judge_stats() -> None:
    valid = _row(
        10,
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
    )
    assert _is_complete_post_cutoff(valid)

    request_error = deepcopy(valid)
    request_error["metrics"]["judge_stats"]["request_error_count"] = 1
    request_error["metrics"]["judge_stats"]["error_count"] = 1
    assert not _is_complete_post_cutoff(request_error)

    missing_stats = deepcopy(valid)
    missing_stats["metrics"].pop("judge_stats")
    assert not _is_complete_post_cutoff(missing_stats)

    legacy_stats = deepcopy(valid)
    for key in tuple(legacy_stats["metrics"]["judge_stats"]):
        if key not in {
            "total",
            "parsed_count",
            "invalid_output_count",
            "request_error_count",
            "error_count",
        }:
            legacy_stats["metrics"]["judge_stats"].pop(key)
    assert not _is_complete_post_cutoff(legacy_stats)

    missing_primary = deepcopy(valid)
    missing_primary["metrics"].pop("avg@64")
    assert not _is_complete_post_cutoff(missing_primary)

    mismatched_primary = deepcopy(valid)
    mismatched_primary["metrics"]["avg@64"] = 0.75
    assert not _is_complete_post_cutoff(mismatched_primary)


def test_completion_contract_requires_completed_exact_cartesian_grid() -> None:
    valid = _row(1)
    assert _completion_contract_reasons(valid) == []

    non_completed = deepcopy(valid)
    non_completed["total_completion_count"] += 1
    non_completed["non_completed_completion_count"] = 1
    reasons = _completion_contract_reasons(non_completed)
    assert "non_completed_completions:1" in reasons
    assert any(reason.startswith("total_completion_count:") for reason in reasons)

    missing_pair = deepcopy(valid)
    missing_pair["distinct_sample_repeat_coordinates"] -= 1
    assert any(
        reason.startswith("distinct_sample_repeat_coordinates:")
        for reason in _completion_contract_reasons(missing_pair)
    )

    extra_pass = deepcopy(valid)
    extra_pass["distinct_pass_indices"] = 2
    extra_pass["max_pass_index"] = 1
    reasons = _completion_contract_reasons(extra_pass)
    assert "distinct_pass_indices:2!=expected:1" in reasons
    assert "pass_index_range:0..1!=expected:0..0" in reasons


def test_any_valid_marker_wins_over_newer_failed_duplicate() -> None:
    source = _row(100)
    sources, resolved, pending, _invalid = _split_candidates(
        [source], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    valid = _row(
        101,
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
    )
    failed = {
        **_row(
            102,
            created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=2),
        ),
        "status": "Failed",
        "score_id": None,
    }

    plan = _plan_replays(
        sources,
        resolved,
        pending,
        {100: [valid, failed]},
    )

    assert [row["task_id"] for row in plan["already_replayed"]] == [100]
    assert plan["already_replayed"][0]["accepted_replay_task_id"] == 101


def test_newer_protocol_drift_does_not_hide_older_valid_source() -> None:
    older = _row(100)
    newer = deepcopy(_row(101))
    newer["task_created_at"] = older["task_created_at"] + timedelta(seconds=1)
    newer["sampling_config"]["avg_k"] = 32

    sources, resolved, pending, invalid = _split_candidates(
        [older, newer],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
    )
    plan = _plan_replays(sources, resolved, pending, {100: []}, invalid)

    assert [row["task_id"] for row in plan["eligible_to_replay"]] == [100]
    assert [row["task_id"] for row in plan["ignored_invalid_sources"]] == [101]
    assert plan["blocked_invalid_source_cells"] == []


def test_completed_without_score_expires_instead_of_waiting_forever() -> None:
    source = _row(100)
    post = _row(
        101,
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
        status="Completed",
        score_id=None,
    )
    sources, resolved, pending, _invalid = _split_candidates(
        [source, post],
        deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT,
        now=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(minutes=20),
    )

    assert pending == {}
    assert list(sources.values())[0]["task_id"] == 100


def test_blocked_cell_waits_for_independent_running_or_launchable_cells() -> None:
    source = _row(1)
    plan = {
        "eligible_to_replay": [],
        "resolved_post_cutoff": [],
        "waiting_sources": [{**source, "task_id": 2, "status": "Running"}],
        "pending_post_cutoff": [],
        "already_replayed": [],
        "pending_existing_replay": [],
        "blocked_existing_replay": [{**source, "task_id": 1}],
        "blocked_incomplete_source": [],
    }
    assert _terminal_action(plan) == "wait"

    plan["waiting_sources"] = []
    assert _terminal_action(plan) == "blocked"

    plan["blocked_existing_replay"] = []
    plan["eligible_to_replay"] = [{**source, "task_id": 3}]
    assert _terminal_action(plan) == "wait"
    assert _terminal_action(plan, {3}) == "blocked"


def test_locked_rescan_can_cancel_a_now_resolved_launch() -> None:
    source = _row(1)
    sources, resolved, pending, _invalid = _split_candidates(
        [source], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    initial = _plan_replays(sources, resolved, pending, {1: []})
    assert _eligible_source(initial, 1) is not None

    post = _row(
        2,
        created_at=JUDGE_DETERMINISM_DEPLOYED_AT + timedelta(seconds=1),
    )
    sources, resolved, pending, _invalid = _split_candidates(
        [source, post], deployed_at=JUDGE_DETERMINISM_DEPLOYED_AT
    )
    refreshed = _plan_replays(sources, resolved, pending, {1: []})
    assert _eligible_source(refreshed, 1) is None


def test_once_returns_nonzero_for_subprocess_or_terminal_block() -> None:
    empty = {
        "eligible_to_replay": [],
        "resolved_post_cutoff": [],
        "waiting_sources": [],
        "pending_post_cutoff": [],
        "already_replayed": [],
        "pending_existing_replay": [],
        "blocked_existing_replay": [],
        "blocked_incomplete_source": [],
    }
    assert _once_exit_code(replay_failed=True, plan=empty) == 1
    blocked = {**empty, "blocked_existing_replay": [_row(1)]}
    assert _once_exit_code(replay_failed=False, plan=blocked) == 2
    assert _once_exit_code(replay_failed=False, plan=empty) == 0


def test_replay_artifact_rejects_malformed_or_uncommitted_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        '{"tasks":[{"task_id":"not-an-id","replayable":true,'
        '"replay_task_id":12}]}',
        encoding="utf-8",
    )
    replay, error = _replay_artifact(malformed, 28625)
    assert replay is None
    assert error == "replay_artifact_missing_source"

    uncommitted = tmp_path / "uncommitted.json"
    uncommitted.write_text(
        '{"tasks":[{"task_id":28625,"replayable":true}]}',
        encoding="utf-8",
    )
    replay, error = _replay_artifact(uncommitted, 28625)
    assert replay is not None
    assert error == "replay_artifact_missing_committed_task"
