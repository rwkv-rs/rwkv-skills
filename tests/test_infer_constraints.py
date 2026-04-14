from __future__ import annotations

from src.infer.constraints import PlainTextConstraint


def test_plain_text_constraint_rejects_forbidden_control_markers() -> None:
    constraint = PlainTextConstraint(forbidden_substrings=("<tool_call>", "<think>", "```"))

    assert constraint.feed_text("Need one more detail.") is True
    assert constraint.feed_text(" Still plain text.") is True

    violating = PlainTextConstraint(forbidden_substrings=("<tool_call>", "<think>", "```"))
    assert violating.feed_text("Please wait <think>") is False
