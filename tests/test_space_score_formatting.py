from __future__ import annotations

from datetime import datetime
from pathlib import Path
import unittest

from src.space.constants import ParamLineage
from src.space.data import ScoreEntry
from src.space.metrics import (
    _delta_from_display_scores,
    _format_score_1dp,
    _parse_display_number,
    _styled_delta_cell,
)
from src.space.tables import (
    _build_benchmark_detail_delta_table,
    _render_pivot_html,
    _sort_table_by_column,
)


class SpaceScoreFormattingTest(unittest.TestCase):
    def test_delta_uses_same_precision_as_displayed_scores(self) -> None:
        previous = 0.69544
        latest = 0.69551

        self.assertEqual(_format_score_1dp(previous), "69.5")
        self.assertEqual(_format_score_1dp(latest), "69.6")

        delta = _delta_from_display_scores(latest, previous)
        self.assertAlmostEqual(delta or 0.0, 0.1)
        self.assertEqual(_styled_delta_cell(delta), ("+0.1", "cell-delta-pos"))

    def test_delta_rounding_to_zero_uses_gray_cell(self) -> None:
        self.assertEqual(_styled_delta_cell(0.04), ("0.0", "cell-delta-zero"))
        self.assertEqual(_styled_delta_cell(-0.04), ("0.0", "cell-delta-zero"))

    def test_large_delta_uses_warning_cell(self) -> None:
        self.assertEqual(_styled_delta_cell(5.0), ("+5.0", "cell-delta-warn-pos"))
        self.assertEqual(_styled_delta_cell(-5.0), ("\u22125.0", "cell-delta-warn-neg"))

    def test_unicode_minus_delta_is_parsed_as_numeric(self) -> None:
        self.assertEqual(_parse_display_number(("\u22120.1", "cell-delta-neg")), -0.1)

    def test_delta_column_sort_handles_unicode_minus(self) -> None:
        headers = ["benchmark", "delta"]
        rows = [
            ["negative", ("\u22120.1", "cell-delta-neg")],
            ["positive", ("+0.2", "cell-delta-pos")],
            ["zero", ("0.0", "cell-delta-zero")],
        ]

        sorted_rows, _ = _sort_table_by_column(headers, rows, None, 1, ascending=False)

        self.assertEqual([row[0] for row in sorted_rows], ["positive", "zero", "negative"])

    def test_delta_cell_renders_with_delta_css_class(self) -> None:
        html = _render_pivot_html(
            ["benchmark", "prev", "latest", "delta"],
            [["mbpp_nocot", "69.5", "69.6", ("+0.1", "cell-delta-pos")]],
            title="delta",
        )

        self.assertIn("cell-delta-pos", html)
        self.assertIn("+0.1", html)

    def test_detail_delta_keeps_plain_regression_red(self) -> None:
        lineages = [
            ParamLineage("0_1b", "rwkv7-g1d-0.1b", "g1d", "rwkv7-g1a-0.1b", "g1a"),
        ]
        _, rows, _ = _build_benchmark_detail_delta_table(
            lineages=lineages,
            model_cache={
                "rwkv7-g1a-0.1b": [
                    _score_entry("rwkv7-g1a-0.1b", "math_500_test", 0.70)
                ],
                "rwkv7-g1d-0.1b": [
                    _score_entry("rwkv7-g1d-0.1b", "math_500_test", 0.69)
                ],
            },
            domains={"math"},
        )

        self.assertEqual(rows[0][6][1], "cell-delta-neg")
        self.assertNotIn("\u5f02\u5e38", rows[0][6][0])
        self.assertNotIn("\u26a0", rows[0][6][0])
        self.assertEqual(rows[0][6][0], "\u22121.0")

    def test_detail_delta_marks_param_curve_inversion_as_suspect(self) -> None:
        lineages = [
            ParamLineage("0_1b", "rwkv7-g1d-0.1b", "g1d", "rwkv7-g1a-0.1b", "g1a"),
            ParamLineage("0_4b", "rwkv7-g1d-0.4b", "g1d", "rwkv7-g1a-0.4b", "g1a"),
        ]
        _, rows, _ = _build_benchmark_detail_delta_table(
            lineages=lineages,
            model_cache={
                "rwkv7-g1a-0.1b": [
                    _score_entry("rwkv7-g1a-0.1b", "math_500_test", 0.69)
                ],
                "rwkv7-g1d-0.1b": [
                    _score_entry("rwkv7-g1d-0.1b", "math_500_test", 0.70)
                ],
                "rwkv7-g1a-0.4b": [
                    _score_entry("rwkv7-g1a-0.4b", "math_500_test", 0.59)
                ],
                "rwkv7-g1d-0.4b": [
                    _score_entry("rwkv7-g1d-0.4b", "math_500_test", 0.60)
                ],
            },
            domains={"math"},
        )

        self.assertEqual(rows[0][6][1], "cell-delta-pos")
        self.assertEqual(rows[0][9][1], "cell-delta-suspect")

    def test_detail_delta_marks_negative_param_curve_inversion_as_suspect(self) -> None:
        lineages = [
            ParamLineage("0_1b", "rwkv7-g1d-0.1b", "g1d", "rwkv7-g1a-0.1b", "g1a"),
            ParamLineage("0_4b", "rwkv7-g1d-0.4b", "g1d", "rwkv7-g1a-0.4b", "g1a"),
        ]
        _, rows, _ = _build_benchmark_detail_delta_table(
            lineages=lineages,
            model_cache={
                "rwkv7-g1a-0.1b": [
                    _score_entry("rwkv7-g1a-0.1b", "math_500_test", 0.70)
                ],
                "rwkv7-g1d-0.1b": [
                    _score_entry("rwkv7-g1d-0.1b", "math_500_test", 0.70)
                ],
                "rwkv7-g1a-0.4b": [
                    _score_entry("rwkv7-g1a-0.4b", "math_500_test", 0.70)
                ],
                "rwkv7-g1d-0.4b": [
                    _score_entry("rwkv7-g1d-0.4b", "math_500_test", 0.689)
                ],
            },
            domains={"math"},
        )

        self.assertEqual(rows[0][6][1], "cell-delta-zero")
        self.assertEqual(rows[0][9][1], "cell-delta-suspect")
        self.assertEqual(rows[0][9][0], "\u22121.1")


def _score_entry(model: str, dataset: str, exact_accuracy: float) -> ScoreEntry:
    return ScoreEntry(
        task_id=1,
        dataset=dataset,
        model=model,
        metrics={"exact_accuracy": exact_accuracy},
        samples=100,
        problems=100,
        created_at=datetime(2026, 1, 1),
        log_path="",
        cot=False,
        task="free_response",
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain="math",
        extra={},
        arch_version="RWKV7",
        data_version="G1D",
        num_params="0_1b",
    )


if __name__ == "__main__":
    unittest.main()
