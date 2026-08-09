from __future__ import annotations

from pathlib import Path

from ops.g1i_strict46.audit_current import _expected_sampling_stages
from ops.g1i_strict46.monitor_blank_recovery_replays import (
    BLANK_RECOVERY_COUNTS_QUERY,
    BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
    BLANK_RECOVERY_STAGE_SQL_PREDICATE,
    REASON_TAG,
    SOURCE_QUERY,
    _blank_recovery_counts,
    _build_replay_command,
    _filter_source_candidates,
    _once_exit_code,
    _plan_replays,
    _provenance_marker,
    _source_is_settled,
)
from ops.g1i_strict46.monitor_judge_determinism_replays import (
    _strict_config_environment,
)


class _Rows:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _Connection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, parameters: tuple[object, ...]) -> _Rows:
        self.calls.append((query, parameters))
        return _Rows(self.rows)


def _source(
    task_id: int,
    *,
    status: str = "Completed",
    score_id: int | None = 100,
    benchmark_name: str = "math_500",
    benchmark_split: str = "test",
    prompt_profile: str = "naive",
    cot_mode: str | None = "CoT",
    evaluator: str = "free_response_naive",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "model_name": f"model-{task_id}",
        "status": status,
        "score_id": score_id,
        "benchmark_name": benchmark_name,
        "benchmark_split": benchmark_split,
        "evaluator": evaluator,
        "sampling_config": {
            "prompt_profile": prompt_profile,
            "cot_mode": "CoT",
        },
        "cot_mode": cot_mode,
    }


def _existing_running_replay(task_id: int) -> dict[str, object]:
    model_name = "rwkv7-g1i-7.2b-20260805-ctx16384"
    identity = {
        "benchmark_name": "math_500",
        "benchmark_split": "test",
        "model_name": model_name,
        "domain": "math",
    }
    with _strict_config_environment():
        stages = _expected_sampling_stages(identity)
    return {
        "replay_task_id": task_id,
        "task_id": task_id,
        "task_created_at": BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
        "replay_status": "Running",
        "status": "Running",
        "replay_score_id": None,
        "score_id": None,
        "evaluator": "free_response_naive",
        "model_name": model_name,
        "benchmark_name": "math_500",
        "benchmark_split": "test",
        "is_param_search": False,
        "is_tmp": False,
        "sampling_config": {
            "prompt_profile": "naive",
            "cot_mode": "CoT",
            "avg_k": 8,
            "effective_sample_count": 4000,
            "sample_limit": None,
            "n_shot": 0,
            "sampling_config": stages,
        },
    }


def test_source_filter_is_strict46_math_cot_naive_only() -> None:
    rows = [
        _source(1),
        _source(2, benchmark_name="simpleqa", benchmark_split="verified"),
        _source(3, benchmark_name="mmlu"),
        _source(4, prompt_profile="normal"),
        _source(5, cot_mode="NoCoT"),
        _source(6, status="Running", score_id=None, cot_mode=None),
        _source(8, evaluator="free_response_naive:strategy_a"),
        {
            **_source(7, status="Running", score_id=None, cot_mode=None),
            "sampling_config": {"prompt_profile": "naive", "cot_mode": "NoCoT"},
        },
    ]

    selected = _filter_source_candidates(rows)

    assert [row["task_id"] for row in selected] == [1, 2, 6]
    assert (selected[1]["benchmark_name"], selected[1]["benchmark_split"]) == (
        "simpleqa",
        "test",
    )


def test_source_query_keeps_previous_replays_as_repair_sources() -> None:
    # A task may already be an append-only replay for an older repair and
    # still need the newer fail-closed recovery semantics.  The current-wave
    # time window and pre-deployment cutoff prevent this monitor from
    # recursively selecting the replay it creates itself.
    assert "NOT LIKE '%replay_source_task_id=%'" not in SOURCE_QUERY
    assert "ROW_NUMBER() OVER" not in SOURCE_QUERY
    assert "t.evaluator IN ('free_response_naive', 'free_response_judge_naive')" in SOURCE_QUERY
    assert "sampling_config->>'prompt_profile'" in SOURCE_QUERY
    assert "sampling_config->>'cot_mode'" in SOURCE_QUERY


def test_blank_count_uses_the_auditors_exact_raw_content_aggregate() -> None:
    connection = _Connection(
        [
            {"task_id": 10, "blank_recovery_stage_count": 3},
            {"task_id": 11, "blank_recovery_stage_count": 0},
        ]
    )

    counts = _blank_recovery_counts(connection, [10, 11])  # type: ignore[arg-type]

    assert counts == {10: 3, 11: 0}
    assert connection.calls == [
        (
            BLANK_RECOVERY_COUNTS_QUERY,
            ([10, 11],),
        )
    ]
    assert BLANK_RECOVERY_STAGE_SQL_PREDICATE in BLANK_RECOVERY_COUNTS_QUERY


def test_plan_replays_only_scored_completed_raw_blank_sources() -> None:
    sources = [
        _source(1),
        _source(2),
        _source(3, status="Running", score_id=None, cot_mode=None),
        _source(4, status="Completed", score_id=None),
        _source(5, status="Failed", score_id=None),
        _source(6),
        _source(7),
        _source(8),
    ]
    blank_counts = {1: 3, 2: 0, 3: 1, 4: 2, 5: 4, 6: 1, 7: 1, 8: 1}
    existing = {
        1: [],
        6: [],
        7: [{
            "replay_task_id": 701,
            "task_id": 701,
            "task_created_at": BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
            "replay_status": "Failed",
            "status": "Failed",
            "replay_score_id": None,
            "score_id": None,
        }],
        8: [_existing_running_replay(702)],
    }

    plan = _plan_replays(
        sources,
        blank_counts,
        existing,
        {("model-6", "math_500", "test"): {"task_id": 600}},
    )

    assert [row["task_id"] for row in plan["eligible_to_replay"]] == [1]
    assert [row["task_id"] for row in plan["not_affected"]] == [2]
    assert [row["task_id"] for row in plan["pending_sources"]] == [3, 4]
    assert [row["task_id"] for row in plan["terminal_failed_sources"]] == [5]
    assert [row["task_id"] for row in plan["already_replayed"]] == [6]
    assert [row["task_id"] for row in plan["blocked_existing_replay"]] == [7]
    assert [row["task_id"] for row in plan["pending_existing_replay"]] == [8]


def test_completed_without_score_is_not_settled() -> None:
    assert _source_is_settled(_source(1))
    assert not _source_is_settled(_source(2, score_id=None))
    assert _source_is_settled(_source(3, status="Failed", score_id=None))


def test_terminal_failed_source_makes_one_shot_nonzero() -> None:
    plan = _plan_replays(
        [_source(14, status="Failed", score_id=None)],
        {14: 1},
        {},
    )

    assert _once_exit_code(
        replay_failed=False,
        replay_lock_busy=False,
        plan=plan,
    ) == 2


def test_replay_marker_and_command_are_exact_and_append_only() -> None:
    repo = Path("/srv/rwkv-skills")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    output = repo / "logs/audits/blank/source_28738.json"

    command = _build_replay_command(
        repo=repo,
        replay_script=replay_script,
        source_task_id=28738,
        dbname="strict46",
        reason_tag=REASON_TAG,
        output=output,
        judge_max_workers=48,
    )

    assert _provenance_marker(28738, REASON_TAG) == (
        "replay_source_task_id=28738;blank_recovery_fail_closed_20260807"
    )
    assert command[command.index("--reason-tag") + 1] == REASON_TAG
    assert command[command.index("--judge-max-workers") + 1] == "48"
    assert command[command.index("--dbname") + 1] == "strict46"
    assert command[command.index("--output") + 1] == str(output)
    assert "--commit" in command
    assert "--summary" in command


def test_existing_unscored_replay_waits_without_duplicate_creation() -> None:
    source = _source(11)
    plan = _plan_replays(
        [source],
        {11: 2},
        {
            11: [_existing_running_replay(12)]
        },
    )

    assert plan["eligible_to_replay"] == []
    assert [row["task_id"] for row in plan["pending_existing_replay"]] == [11]


def test_invalid_running_replay_is_blocked_instead_of_waiting_forever() -> None:
    source = _source(12)
    invalid_running = {
        **_existing_running_replay(13),
        "sampling_config": {
            **_existing_running_replay(13)["sampling_config"],
            "prompt_profile": "normal",
        },
    }

    plan = _plan_replays([source], {12: 1}, {12: [invalid_running]})

    assert plan["pending_existing_replay"] == []
    assert [row["task_id"] for row in plan["blocked_existing_replay"]] == [12]


def test_fully_valid_post_task_resolves_same_cell_across_source_markers() -> None:
    source = _source(11)
    resolved = {("model-11", "math_500", "test"): {"task_id": 99}}

    plan = _plan_replays(
        [source],
        {11: 1},
        {11: []},
        resolved,
    )

    assert plan["eligible_to_replay"] == []
    assert [row["task_id"] for row in plan["already_replayed"]] == [11]
    assert plan["already_replayed"][0]["post_cutoff_task_id"] == 99
