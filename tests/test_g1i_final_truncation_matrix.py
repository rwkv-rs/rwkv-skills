from ops.g1i_strict46.report_final_truncation_matrix import (
    META_SQL,
    STATS_SCHEMA_VERSION,
    STATS_SQL,
    aggregate,
    build_summary_tables,
    load_exact_task_stats_cache,
    select_latest,
    write_summary_markdown,
)


def test_metadata_query_keeps_failed_latest_task_groups_visible():
    assert "t.status = 'Running'" not in META_SQL
    assert "s.task_id IS NOT NULL" not in META_SQL
    assert "t.evaluator NOT LIKE '%%:strategy_%%'" in META_SQL


def test_stats_cache_is_keyed_by_exact_database_and_task_id(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        '{"stats_schema_version":"' + STATS_SCHEMA_VERSION + '","selected_rows": ['
        '{"database": "current", "task_id": 7, "completion_count": 10},'
        '{"database": "other", "task_id": 7, "completion_count": 20}'
        ']}'
    )

    cache = load_exact_task_stats_cache(path)

    assert cache["current"][7]["completion_count"] == 10
    assert cache["other"][7]["completion_count"] == 20


def test_stats_cache_rejects_old_final_output_definition(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        '{"selected_rows":[{"database":"current","task_id":7,'
        '"completion_count":10}]}',
        encoding="utf-8",
    )

    assert load_exact_task_stats_cache(path) == {}


def _row(domain: str, **overrides):
    row = {
        "domain": domain,
        "task_id": 1,
        "status": "Completed",
        "score_created_at": "2026-08-06T00:00:00",
        "completion_count": 100,
        "math_final_output_count": 0,
        "math_final_truncated_count": 0,
        "knowledge_final_output_count": 0,
        "knowledge_final_truncated_count": 0,
        "ordinary_final_output_count": 0,
        "ordinary_final_truncated_count": 0,
    }
    row.update(overrides)
    return row


def test_math_uses_only_final_answer_stage_telemetry():
    result = aggregate(
        [
            _row(
                "math",
                math_final_output_count=80,
                math_final_truncated_count=8,
            )
        ]
    )

    assert result["telemetry_coverage"] == 0.8
    assert result["final_truncation_rate_all_completions"] == 0.08
    assert result["conditional_final_truncation_rate"] == 0.1


def test_math_final_sql_prefers_stage2_and_keeps_evaluator_facing_fallbacks():
    # A two-stage row must use stages[1], never its reasoning stage.  A correct
    # strategy-A response is submitted directly to the evaluator and therefore
    # remains final output even though no stages[1] record exists.  Historical
    # single-stage rows use stages[0] only as the last fallback.
    assert "WHEN c.context #> '{stages,1}' IS NOT NULL THEN" in STATS_SQL
    assert "WHEN c.context ? 'strategy_a_stop_reason' THEN" in STATS_SQL
    assert "WHEN c.context #> '{strategy_a,stop_reason}' IS NOT NULL THEN" in STATS_SQL
    assert "WHEN c.context #> '{stages,0,stop_reason}' IS NOT NULL THEN" in STATS_SQL
    assert STATS_SQL.index("WHEN c.context #> '{stages,1}' IS NOT NULL THEN") < STATS_SQL.index(
        "WHEN c.context ? 'strategy_a_stop_reason' THEN"
    ) < STATS_SQL.index("WHEN c.context #> '{stages,0,stop_reason}' IS NOT NULL THEN")


def test_failed_task_with_stale_score_is_not_a_complete_cell():
    row = _row(
        "coding",
        status="Failed",
        score_created_at="2026-08-06T00:00:00",
        ordinary_final_output_count=100,
    )

    result = aggregate([row])

    assert result["cells"] == 1
    assert result["complete_cells"] == 0
    assert result["failed_cells"] == 1


def test_missing_knowledge_telemetry_is_not_counted_as_clean_coverage():
    result = aggregate(
        [
            _row(
                "knowledge",
                knowledge_final_output_count=20,
                knowledge_final_truncated_count=10,
            )
        ]
    )

    assert result["telemetry_coverage"] == 0.2
    assert result["final_truncation_rate_all_completions"] == 0.1
    assert result["conditional_final_truncation_rate"] == 0.5


def test_non_math_domains_use_evaluator_facing_generation():
    result = aggregate(
        [
            _row(
                "coding",
                ordinary_final_output_count=100,
                ordinary_final_truncated_count=3,
            )
        ]
    )

    assert result["telemetry_coverage"] == 1.0
    assert result["final_truncation_rate_all_completions"] == 0.03
    assert result["conditional_final_truncation_rate"] == 0.03


def test_mixed_domain_aggregate_uses_each_domains_final_output_definition():
    result = aggregate(
        [
            _row("math", math_final_output_count=80, math_final_truncated_count=8),
            _row(
                "knowledge",
                task_id=2,
                knowledge_final_output_count=50,
                knowledge_final_truncated_count=5,
            ),
            _row(
                "coding",
                task_id=3,
                ordinary_final_output_count=100,
                ordinary_final_truncated_count=2,
            ),
        ]
    )

    assert result["completion_count"] == 300
    assert result["observable_final_output_count"] == 230
    assert result["final_truncated_count"] == 15
    assert result["final_truncation_rate_all_completions"] == 0.05
    assert result["conditional_final_truncation_rate"] == 15 / 230


def test_summary_tables_separate_architecture_size_and_family_domain():
    rows = [
        {
            **_row("knowledge", knowledge_final_output_count=100, knowledge_final_truncated_count=10),
            "family": "G1i",
            "size": "2.9B",
        },
        {
            **_row(
                "math",
                task_id=2,
                math_final_output_count=80,
                math_final_truncated_count=4,
            ),
            "family": "G1i",
            "size": "2.9B",
        },
        {
            **_row(
                "knowledge",
                task_id=3,
                knowledge_final_output_count=100,
                knowledge_final_truncated_count=20,
            ),
            "family": "G1h",
            "size": "2.9B",
        },
    ]

    tables = build_summary_tables(rows)
    g1i_size = next(
        row
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i" and row["parameter_size"] == "2.9B"
    )
    g1i_all = next(
        row
        for row in tables["truncation_vs_g1x"]
        if row["family"] == "G1i" and row["domain"] == "all"
    )

    assert g1i_size["completion_count"] == 200
    assert g1i_size["final_truncated_count"] == 14
    assert g1i_all["final_truncated_count"] == 14
    assert any(
        row["family"] == "G1i" and row["domain"] == "math"
        for row in tables["truncation_vs_g1x"]
    )


def test_aggregate_exposes_strictly_comparable_rate_separately():
    result = aggregate(
        [
            _row(
                "coding",
                task_id=1,
                ordinary_final_output_count=100,
                ordinary_final_truncated_count=10,
                protocol_compatible=True,
            ),
            _row(
                "coding",
                task_id=2,
                ordinary_final_output_count=100,
                ordinary_final_truncated_count=100,
                protocol_compatible=False,
            ),
        ]
    )

    assert result["final_truncation_rate_all_completions"] == 0.55
    assert result["protocol_compatible_cells"] == 1
    assert result["protocol_incompatible_cells"] == 1
    assert result[
        "protocol_compatible_final_truncation_rate_all_completions"
    ] == 0.1
    assert result[
        "protocol_compatible_conditional_final_truncation_rate"
    ] == 0.1


def test_markdown_summary_keeps_both_truncation_rates(tmp_path):
    path = tmp_path / "table.md"
    write_summary_markdown(
        path,
        [
            {
                "family": "G1i",
                "parameter_size": "7.2B",
                "cells": 46,
                "complete_cells": 46,
                "completion_count": 100,
                "observable_final_output_count": 80,
                "telemetry_coverage": 0.8,
                "final_truncated_count": 8,
                "final_truncation_rate_all_completions": 0.08,
                "conditional_final_truncation_rate": 0.1,
                "protocol_compatible_cells": 46,
                "protocol_compatible_final_truncation_rate_all_completions": 0.08,
                "protocol_compatible_conditional_final_truncation_rate": 0.1,
            }
        ],
        dimension="parameter_size",
    )

    rendered = path.read_text(encoding="utf-8")
    assert "总体截断率" in rendered
    assert "条件截断率" in rendered
    assert "80.0000%" not in rendered
    assert "8.0000%" in rendered
    assert "10.0000%" in rendered
    assert "协议一致" in rendered
    assert "可比总体截断率" in rendered


def test_latest_selection_enforces_strict46_modes_for_every_domain():
    base = {
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "mmlu",
        "benchmark_split": "test",
        "status": "Completed",
        "score_created_at": "2026-08-06T00:00:00",
        "task_created_at": "2026-08-06T00:00:00",
        "evaluator": "multi_choice_plain_naive",
        "sampling_config": {},
    }
    selected = select_latest(
        [
            {**base, "task_id": 1, "cot_mode": "CoT"},
            {**base, "task_id": 2, "cot_mode": "NoCoT"},
        ]
    )

    assert [row["task_id"] for row in selected] == [2]


def test_latest_selection_uses_newest_database_task_group():
    base = {
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "mmlu",
        "benchmark_split": "test",
        "status": "Completed",
        "score_created_at": "2026-08-06T00:00:00",
        "evaluator": "multi_choice_plain_naive",
        "sampling_config": {},
        "cot_mode": "NoCoT",
    }

    selected = select_latest(
        [
            {**base, "task_id": 100, "task_created_at": "2026-08-06T00:00:00"},
            {**base, "task_id": 101, "task_created_at": "2026-08-07T00:00:00"},
        ]
    )

    assert [row["task_id"] for row in selected] == [101]


def test_simpleqa_verified_is_canonicalized_to_strict46_test_target():
    selected = select_latest(
        [
            {
                "model_name": "rwkv7-g1h-2.9b-20260710-ctx10240",
                "benchmark_name": "simpleqa",
                "benchmark_split": "verified",
                "task_id": 10,
                "status": "Completed",
                "score_created_at": "2026-08-06T00:00:00",
                "task_created_at": "2026-08-06T00:00:00",
                "evaluator": "free_response_naive",
                "cot_mode": "CoT",
                "sampling_config": {},
            }
        ]
    )

    assert len(selected) == 1
    assert selected[0]["benchmark_split"] == "test"
    assert selected[0]["source_benchmark_split"] == "verified"
    assert selected[0]["protocol_compatible"] is True


def test_latest_livecodebench_task_wins_even_when_older_task_is_compatible():
    base = {
        "model_name": "rwkv7-g1h-2.9b-20260710-ctx10240",
        "benchmark_name": "livecodebench",
        "benchmark_split": "test",
        "status": "Completed",
        "score_created_at": "2026-08-06T00:00:00",
        "evaluator": "code_livecodebench_naive",
        "sampling_config": {},
    }
    legacy_only = select_latest(
        [{**base, "task_id": 10, "task_created_at": "2026-08-06T01:00:00", "cot_mode": "CoT"}]
    )
    selected = select_latest(
        [
            {**base, "task_id": 10, "task_created_at": "2026-08-06T01:00:00", "cot_mode": "CoT"},
            {**base, "task_id": 9, "task_created_at": "2026-08-06T00:00:00", "cot_mode": "NoCoT"},
        ]
    )

    assert legacy_only[0]["protocol_compatible"] is False
    assert selected[0]["task_id"] == 10
    assert selected[0]["protocol_compatible"] is False


def test_newest_incompatible_task_never_falls_back_to_older_compatible_task():
    base = {
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "mmlu",
        "benchmark_split": "test",
        "status": "Completed",
        "score_created_at": "2026-08-06T00:00:00",
        "evaluator": "multi_choice_plain_naive",
        "sampling_config": {},
    }

    selected = select_latest(
        [
            {
                **base,
                "task_id": 20,
                "task_created_at": "2026-08-06T00:00:00",
                "cot_mode": "NoCoT",
            },
            {
                **base,
                "task_id": 21,
                "task_created_at": "2026-08-07T00:00:00",
                "cot_mode": "CoT",
            },
        ]
    )

    assert [row["task_id"] for row in selected] == [21]
    assert selected[0]["protocol_compatible"] is False


def test_newest_failed_task_never_falls_back_to_older_completed_task():
    base = {
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "mmlu",
        "benchmark_split": "test",
        "score_created_at": None,
        "evaluator": "multi_choice_plain_naive",
        "sampling_config": {},
        "cot_mode": "NoCoT",
    }

    selected = select_latest(
        [
            {
                **base,
                "task_id": 30,
                "task_created_at": "2026-08-06T00:00:00",
                "status": "Completed",
            },
            {
                **base,
                "task_id": 31,
                "task_created_at": "2026-08-07T00:00:00",
                "status": "Failed",
            },
        ]
    )

    assert [row["task_id"] for row in selected] == [31]
    assert selected[0]["status"] == "Failed"


def test_auxiliary_strategy_task_does_not_shadow_latest_primary_group():
    base = {
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "aime24",
        "benchmark_split": "test",
        "status": "Completed",
        "score_created_at": "2026-08-07T00:00:00",
        "sampling_config": {},
        "cot_mode": "CoT",
    }

    selected = select_latest(
        [
            {
                **base,
                "task_id": 40,
                "task_created_at": "2026-08-07T00:00:00",
                "evaluator": "free_response_naive",
            },
            {
                **base,
                "task_id": 41,
                "task_created_at": "2026-08-07T00:01:00",
                "evaluator": "free_response_naive:strategy_b",
                "score_created_at": None,
            },
        ]
    )

    assert [row["task_id"] for row in selected] == [40]
