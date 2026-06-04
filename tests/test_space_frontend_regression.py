from src.space.db_profiles import math_db_enabled
from src.space.eval_records import _render_eval_records_html
from src.space.tables import _render_pivot_html


def test_eval_records_hide_eval_group_and_show_context_preview_button() -> None:
    html = _render_eval_records_html(
        meta={
            "cell_id": "cell-test",
            "benchmark_name": "math_500",
            "eval_method": "cot",
            "model": "rwkv7-test",
        },
        records=[
            {
                "sample_index": 0,
                "repeat_index": 1,
                "answer": "42",
                "ref_answer": "42",
                "is_passed": True,
                "fail_reason": "",
                "context_preview": "{}",
            }
        ],
        only_wrong=False,
    )

    assert "eval_group" not in html
    assert "strategy_a" not in html
    assert "<th>model_output</th>" in html
    assert "<th>context</th>" in html
    assert ">{}</button>" in html


def test_pivot_table_uses_grouped_header_row_after_frontend_rollback() -> None:
    html = _render_pivot_html(
        ["benchmark_name", "1.5b latest (g1g)"],
        [["math_500", "70.0%"]],
        title="test",
    )

    assert "group-row" in html


def test_space_math_db_profile_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SPACE_MATH_PG_DBNAME", raising=False)
    monkeypatch.setenv("PG_DBNAME", "mathverify")

    assert math_db_enabled() is False
