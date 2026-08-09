from __future__ import annotations

import os

import pytest

from src.eval.metrics import free_response as fr


def _dataset(tmp_path):
    path = tmp_path / "free.jsonl"
    path.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    return path


def _scored(
    group: str,
    payload: dict,
    *,
    sample_index: int,
    repeat_index: int,
    question: str,
    reference: str,
    passed: bool,
    fail_reason: str,
) -> fr._ScoredCompletion:
    return fr._ScoredCompletion(
        source_payload=payload,
        sample_index=sample_index,
        repeat_index=repeat_index,
        question=question,
        reference=reference,
        scoring_text=str(payload.get("completion1") or ""),
        display_answer="7" if passed else "",
        math_passed=passed,
        final_passed=passed,
        fail_reason=fail_reason,
    )


def test_timeout_retry_isolated_row_resolves_and_restores_environment(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[str, str | None]] = []

    def score(group: str, payload: dict, **kwargs):
        timeout = os.environ.get("RWKV_MATH_VERIFY_TIMEOUT_S")
        calls.append((group, timeout))
        passed = timeout == "15"
        return _scored(
            group,
            payload,
            **kwargs,
            passed=passed,
            fail_reason="" if passed else "math_verify_timeout",
        )

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (object(), object()))
    monkeypatch.setattr(fr, "score_free_response_strategy", score)
    monkeypatch.setenv("RWKV_MATH_VERIFY_TIMEOUT_S", "2")

    evaluation = fr.evaluate_free_response(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "answer"}],
        dataset_path=_dataset(tmp_path),
        primary_group=fr.STRATEGY_C,
        math_verify_retry_timeout_s=15.0,
    )

    assert calls == [(fr.STRATEGY_A, "2"), (fr.STRATEGY_A, "15")]
    assert os.environ["RWKV_MATH_VERIFY_TIMEOUT_S"] == "2"
    assert evaluation.rows_by_group[fr.STRATEGY_C] == [(0, 0, True)]
    assert evaluation.math_verify_retry_stats_by_group[fr.STRATEGY_A] == {
        "attempted_count": 1,
        "resolved_count": 1,
        "unresolved_count": 0,
        "rows": [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "first_fail_reason": "math_verify_timeout",
                "retry_fail_reason": "",
                "resolved": True,
            }
        ],
    }
    assert all(
        evaluation.math_verify_retry_stats_by_group[group]["attempted_count"] == 0
        for group in (fr.STRATEGY_B, fr.STRATEGY_C)
    )


def test_timeout_retry_that_still_times_out_fails_closed_before_persistence(
    monkeypatch, tmp_path
) -> None:
    def score(group: str, payload: dict, **kwargs):
        return _scored(
            group,
            payload,
            **kwargs,
            passed=False,
            fail_reason="prediction_parse_timeout",
        )

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (object(), object()))
    monkeypatch.setattr(fr, "score_free_response_strategy", score)

    with pytest.raises(fr.UnresolvedMathVerifyTimeoutError) as exc_info:
        fr.evaluate_free_response(
            [{"sample_index": 0, "repeat_index": 0, "completion1": "answer"}],
            dataset_path=_dataset(tmp_path),
            primary_group=fr.STRATEGY_C,
            math_verify_retry_timeout_s=15.0,
        )

    exc = exc_info.value
    assert exc.group == fr.STRATEGY_A
    assert exc.sample_index == 0
    assert exc.repeat_index == 0
    assert exc.first_fail_reason == "prediction_parse_timeout"
    assert exc.retry_fail_reason == "prediction_parse_timeout"


def test_timeout_retry_is_enabled_by_default(monkeypatch, tmp_path) -> None:
    calls: list[str | None] = []

    def score(group: str, payload: dict, **kwargs):
        timeout = os.environ.get("RWKV_MATH_VERIFY_TIMEOUT_S")
        calls.append(timeout)
        return _scored(
            group,
            payload,
            **kwargs,
            passed=timeout == "15",
            fail_reason="" if timeout == "15" else "math_verify_timeout",
        )

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (object(), object()))
    monkeypatch.setattr(fr, "score_free_response_strategy", score)
    monkeypatch.setenv("RWKV_MATH_VERIFY_TIMEOUT_S", "2")

    evaluation = fr.evaluate_free_response(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "answer"}],
        dataset_path=_dataset(tmp_path),
        primary_only=True,
        primary_group=fr.STRATEGY_A,
    )

    assert calls == ["2", "15"]
    assert evaluation.rows_by_group[fr.STRATEGY_A] == [(0, 0, True)]


def test_timeout_retry_does_not_match_fail_reason_substrings(
    monkeypatch, tmp_path
) -> None:
    calls = 0

    def score(group: str, payload: dict, **kwargs):
        nonlocal calls
        calls += 1
        return _scored(
            group,
            payload,
            **kwargs,
            passed=False,
            fail_reason="non_timeout_parse_error",
        )

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (object(), object()))
    monkeypatch.setattr(fr, "score_free_response_strategy", score)

    evaluation = fr.evaluate_free_response(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "answer"}],
        dataset_path=_dataset(tmp_path),
        primary_only=True,
        primary_group=fr.STRATEGY_A,
    )

    assert calls == 1
    assert evaluation.rows_by_group[fr.STRATEGY_A] == [(0, 0, False)]
    assert evaluation.math_verify_retry_stats_by_group[
        fr.STRATEGY_A
    ]["attempted_count"] == 0
