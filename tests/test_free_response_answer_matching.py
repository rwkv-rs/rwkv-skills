from __future__ import annotations

from src.eval.metrics.free_response import (
    _format_answer_for_storage,
    _is_exact_match,
)


def test_legacy_numeric_exact_match_uses_last_number() -> None:
    assert _is_exact_match(r"Therefore, the answer is \(\\boxed{9", "9")
    assert not _is_exact_match("100", "9")


def test_legacy_text_exact_match_is_not_case_folded() -> None:
    assert _is_exact_match("Evelyn", "Evelyn")
    assert not _is_exact_match("Briana", "Evelyn")
    assert not _is_exact_match("evelyn", "Evelyn")


def test_legacy_storage_keeps_extracted_number_for_numeric_refs() -> None:
    assert _format_answer_for_storage("Final answer: 50 cents.", "0.5 dollars") == "50"
