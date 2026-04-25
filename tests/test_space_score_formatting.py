from __future__ import annotations

import unittest

from src.space.metrics import (
    _delta_from_display_scores,
    _format_score_1dp,
    _parse_display_number,
    _styled_delta_cell,
)
from src.space.tables import _render_pivot_html, _sort_table_by_column


class SpaceScoreFormattingTest(unittest.TestCase):
    def test_delta_uses_same_precision_as_displayed_scores(self) -> None:
        previous = 0.69544
        latest = 0.69551

        self.assertEqual(_format_score_1dp(previous), "69.5")
        self.assertEqual(_format_score_1dp(latest), "69.6")

        delta = _delta_from_display_scores(latest, previous)
        self.assertAlmostEqual(delta or 0.0, 0.1)
        self.assertEqual(_styled_delta_cell(delta), ("+0.1", "cell-delta-pos"))

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


if __name__ == "__main__":
    unittest.main()
