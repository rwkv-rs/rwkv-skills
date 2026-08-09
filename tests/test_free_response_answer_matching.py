from __future__ import annotations

import json
import re
import time

import pytest

from src.eval.metrics import free_response as fr
from src.eval.metrics.free_response import (
    DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
    LLM_JUDGE_PROTOCOL_VERSION,
    LLM_JUDGE_RESPONSE_CONTRACT,
    LLMJudge,
    LLMJudgeConfig,
    STRATEGY_A,
    STRATEGY_B,
    STRATEGY_C,
    STRATEGY_GROUPS,
    build_grouped_metrics_payload,
    evaluate_free_response,
    llm_judge_prompt_sha256,
    llm_judge_protocol,
    llm_judge_protocol_fingerprint,
    llm_judge_protocol_stats_reasons,
)


def _patch_math_verify(monkeypatch) -> None:
    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [("gold", text.removeprefix("$\\boxed{").removesuffix("}$"))]
        if "requires_unclosed_repair" in text and "</think>" in text and "Therefore, the final answer is " in text:
            return [("pred", "7")]
        if "requires_truncated_repair" in text and "\nTherefore, the final answer is " in text:
            return [("pred", "7")]
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxes:
            return [("pred", boxes[-1])]
        return []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        if not gold or not pred:
            return False
        return gold[-1][-1] == pred[-1][-1]

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))


def test_math_verify_timeout_marks_completion_wrong(monkeypatch) -> None:
    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [("gold", "7")]
        return [("pred", "8")]

    def verify(_gold, _pred, *, strict: bool = False):
        _ = strict
        time.sleep(1.0)
        return True

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    monkeypatch.setenv("RWKV_MATH_VERIFY_TIMEOUT_S", "0.1")

    result = fr._math_verify("7", "Final answer: \\boxed{8}")

    assert not result.passed
    assert result.fail_reason == "math_verify_timeout"


def _patch_last_scalar_math_verify(monkeypatch) -> None:
    """Small deterministic parser for answer-window selection regressions."""

    def parse(text: str):
        values = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        return [("value", values[-1])] if values else []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1][-1] == pred[-1][-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))


def _patch_any_scalar_math_verify(monkeypatch) -> None:
    """Parser that would resurrect any old scalar if given the full text."""

    def parse(text: str):
        return re.findall(r"[+-]?\d+(?:\.\d+)?", text)

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1] in pred)

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))


def test_truncated_raw_math_prefers_complete_explicit_result_over_incomplete_tail(
    monkeypatch,
) -> None:
    """Regression for G1i Gaokao task 28814 sample 221/repeat 3."""

    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "Therefore, Area of PQRS = 9 * A = 9 * 6 = 54. "
        "That seems straightforward.\n"
        + "Repeated verification without a new answer. " * 100
        + "\nBecause the base expression is 6, multiplying by 9 gives 54. Wait,"
    )

    assert fr._math_verify_input(completion) == (
        "Because the base expression is 6, multiplying by 9 gives 54."
    )
    result = fr._math_verify("54", completion)
    assert result.passed
    assert result.answer == "54"


def test_truncated_raw_math_prefers_latest_complete_answer_over_partial_product(
    monkeypatch,
) -> None:
    """Regression for G1i Gaokao task 28814 sample 326/repeat 2."""

    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "Therefore, the answer is 6.\n"
        + "Additional sign-table verification. " * 100
        + "\nP(6) = (5)^1 * (4)^2 * (3)^3 * (-2)^"
    )

    assert fr._math_verify_input(completion) == "the answer is 6."
    result = fr._math_verify("6", completion)
    assert result.passed
    assert result.answer == "6"


def test_incomplete_tail_does_not_promote_answer_meta_statement() -> None:
    completion = (
        "The answer is 45. A later check says the answer is correct.\n"
        "One more unfinished calculation: 9 *"
    )

    assert fr._math_verify_input(completion) == "The answer is 45."


def test_incomplete_tail_ignores_problem_restatement_and_quoted_prior_answer() -> None:
    restatement = (
        "The answer is 60. The answer should be in the form m + n sqrt(p).\n"
        "Unfinished check: 2 *"
    )
    quoted_prior = (
        'The answer is 20. The supplied note says "the final answer is 0.2." '
        "That was for subproblem 0.\nUnfinished check: 2 *"
    )

    assert fr._math_verify_input(restatement) == "The answer is 60."
    assert fr._math_verify_input(quoted_prior) == "The answer is 20."


def test_incomplete_tail_ignores_single_quoted_prior_answer() -> None:
    completion = (
        "The answer is 7. The prompt says 'the answer is 5.'\n"
        "Unfinished check: 2 *"
    )

    assert fr._math_verify_input(completion) == "The answer is 7."


def test_incomplete_tail_ignores_conditional_answer_hypothesis() -> None:
    completion = (
        "The answer is 7. If the answer is 5, another branch applies.\n"
        "Unfinished check: 2 *"
    )

    assert fr._math_verify_input(completion) == "The answer is 7."


def test_incomplete_tail_ignores_conditional_answer_suffix() -> None:
    if_suffix = (
        "The answer is 7. Later, the answer is 5 if x > 0.\n"
        "Unfinished check: 2 *"
    )
    assuming_suffix = (
        "The answer is 7. Later, the answer is 2, "
        "assuming 4 dollars for 2 bars.\nUnfinished check: 2 *"
    )

    assert fr._math_verify_input(if_suffix) == "The answer is 7."
    assert fr._math_verify_input(assuming_suffix) == "The answer is 7."


def test_later_complete_text_answer_blocks_older_numeric_recovery() -> None:
    completion = (
        "The answer is 3.\n"
        + "Extended verification. " * 180
        + "\nAnswer: Each candy bar cost $2.\nUnfinished check: 2 *"
    )

    window = fr._math_verify_input(completion)
    assert window != "The answer is 3."
    assert "Each candy bar cost $2" in window


def test_terminal_hypothetical_result_is_not_recovered() -> None:
    completion = (
        "The answer is 928.\n"
        + "Extended verification. " * 180
        + "\nSuppose at 800 degrees, concentration is 1/10000.\n"
        "Unfinished check: exp("
    )

    assert fr._math_verify_input(completion) == "The answer is 928."


def test_incomplete_tail_does_not_promote_single_intermediate_equality() -> None:
    completion = (
        "The answer is 45. Therefore, W_R = 3x + 2.\n"
        "One more unfinished calculation: 9 *"
    )

    assert fr._math_verify_input(completion) == "The answer is 45."


def test_incomplete_later_box_does_not_override_complete_box(monkeypatch) -> None:
    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "Final answer: \\boxed{54}.\n"
        + "Long verification. " * 180
        + "\nA final unfinished recalculation gives \\boxed{(1/2"
    )

    assert fr._last_boxed_content(completion) == "54"
    assert fr._math_verify_input(completion) == "\\boxed{54}"
    assert fr._math_verify("54", completion).passed


def test_latest_complete_correction_wins_without_reference_driven_search(
    monkeypatch,
) -> None:
    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "The answer is 54. Actually, the final answer is 27.\n"
        + "Verification continues. " * 150
        + "\nThe last scratch expression is 2^"
    )

    assert fr._math_verify_input(completion) == "final answer is 27."
    assert fr._math_verify("27", completion).passed
    # The extractor must not walk backwards until it happens to find the
    # reference.  A complete later correction is authoritative even if wrong.
    assert not fr._math_verify("54", completion).passed


def test_incomplete_correction_falls_back_only_when_old_answer_was_not_retracted() -> None:
    non_retracted = (
        "The answer is 54.\nFurther work.\nThe final answer is (1/2"
    )
    assert fr._math_verify_input(non_retracted) == "The answer is 54."

    explicitly_retracted = (
        "The answer is 54. Wait, that is wrong. The final answer is (1/2"
    )
    # Once the model retracts 54 there is no complete authoritative answer;
    # keeping the raw text fails closed instead of resurrecting a known-old
    # candidate merely because it matches a reference.
    assert fr._math_verify_input(explicitly_retracted) == explicitly_retracted


def test_explicit_retraction_is_found_beyond_legacy_120_character_window() -> None:
    completion = (
        "The answer is 54. "
        + "Verification detail that does not revise the result. " * 8
        + "That answer was wrong. The final answer is (1/2"
    )

    assert len(completion.split("That answer", 1)[0].split("54.", 1)[1]) > 120
    assert fr._math_verify_input(completion) == completion


def test_incomplete_but_wait_without_explicit_negation_does_not_retract() -> None:
    completion = (
        "The answer is 54. "
        + "Verification detail that does not revise the result. " * 8
        + "But wait"
    )

    assert fr._math_verify_input(completion) == "The answer is 54."


@pytest.mark.parametrize(
    "marker",
    ("Correction:", "Corrected answer:", "Revised answer="),
)
def test_explicit_correction_markers_retract_after_long_suffix(marker: str) -> None:
    completion = (
        "The answer is 54. "
        + "Verification detail. " * 20
        + f"{marker} (1/2"
    )

    assert fr._math_verify_input(completion) == completion


@pytest.mark.parametrize(
    "later_text",
    (
        "Suppose a student says 42. That answer was wrong.",
        'A source says \"the previous answer was wrong\".',
        "If this answer were wrong, we would revisit it.",
        "Another route, that calculation was wrong.",
    ),
)
def test_unrelated_retraction_language_does_not_retract_candidate(
    later_text: str,
) -> None:
    completion = f"The answer is 54. {later_text} But wait"

    assert fr._math_verify_input(completion) == "The answer is 54."


@pytest.mark.parametrize(
    "completion",
    (
        "The answer is 54. But wait, if this answer were wrong, we would "
        "recompute. The final answer is (1/2",
        "The answer is 54. But wait, the other calculation is wrong, so let me "
        "verify. The final answer is (1/2",
        "The answer is 54. A source says 'the previous answer was wrong'. But wait",
        "The answer is 54. A source says \u201cthe previous answer was wrong\u201d. But wait",
    ),
)
def test_contextual_retraction_language_does_not_discard_complete_answer(
    completion: str,
) -> None:
    assert fr._math_verify_input(completion) == "The answer is 54."


def test_local_condition_does_not_hide_later_genuine_retraction() -> None:
    completion = (
        "The answer is 54. If we verify it, the check is simple. "
        "But wait, my answer is wrong. Final answer: 55"
    )

    assert fr._answer_candidate_is_retracted(completion.split("54.", 1)[1])


@pytest.mark.parametrize(
    "completion",
    (
        "The answer is 54. Correction: 55.",
        "The answer is 54. Corrected answer: 55.",
        "The answer is 54. Revised answer=55.",
        "The answer is 54. The correct result is 55.",
        "The answer is 54. Therefore, 55.",
        "The answer is 54. Actually, 55.",
        "The answer is 54. Therefore 55.",
        "The answer is 54. Actually 55.",
        "The answer is 54. Wait, that answer is wrong. Final answer: 55.",
    ),
)
def test_complete_replacement_is_the_only_scored_math_answer(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    payload = {"completion1": completion, "stop_reason1": "stop_token"}

    old_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )
    new_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert not old_result.final_passed
    assert new_result.final_passed


def test_conflicting_followup_invalidates_an_explicit_candidate() -> None:
    completion = (
        "The answer should be 4. Wait, earlier I obtained 60; that is conflicting.\n"
        "Unfinished recalculation: ("
    )

    assert fr._math_verify_input(completion) == completion


def test_terminal_questioned_number_does_not_override_confirmed_answer() -> None:
    completion = (
        "The answer is 12.\n"
        + "Extended verification. " * 180
        + "\nPerimeter = 23? No! Wait,"
    )

    assert fr._math_verify_input(completion) == "The answer is 12."


def test_repeated_tentative_numeric_answer_fails_closed_to_raw_text() -> None:
    completion = (
        "The answer is 58.\n"
        + (
            "The answer is 80, even though the givens look inconsistent? "
            "Let me check again.\n"
        )
        * 3
        + "The supplied quantities still look inconsistent.\nAlternatively,"
    )

    assert fr._math_verify_input(completion) == completion


def test_repeated_questioned_answer_is_never_promoted() -> None:
    completion = (
        "The answer is 12.\n"
        + "The answer is 23? No! Let me check again.\n" * 3
        + "Unfinished check: ("
    )

    assert fr._math_verify_input(completion) == completion


def test_complete_math_dollars_and_single_currency_dollar_are_balanced() -> None:
    assert fr._answer_candidate_is_complete("The answer is $12$.")
    assert fr._answer_candidate_is_complete("The answer is $12 + 3$.")
    assert fr._answer_candidate_is_complete("The answer is $12^2$.")
    assert fr._answer_candidate_is_complete(r"The answer is $12 \text{ cm}$.")
    assert fr._answer_candidate_is_complete("Answer: Each candy bar cost $2.")
    assert fr._answer_candidate_is_complete("*Note: total is $3.*")
    assert not fr._answer_candidate_is_complete("The answer is $2.")
    assert not fr._answer_candidate_is_complete("The answer is $-2.")
    assert not fr._answer_candidate_is_complete("The answer is $+2.")
    assert not fr._answer_candidate_is_complete(
        r"The answer is $12 + \frac{1}{2} and each item costs $2."
    )


def test_money_context_does_not_close_an_earlier_unfinished_math_span() -> None:
    assert not fr._answer_candidate_is_complete(
        r"The answer is $12 + 3 and the final item costs $2."
    )


def test_terminal_explicit_escaped_currency_scalar_uses_minimal_answer(
    monkeypatch,
) -> None:
    """Regression for SVAMP task 28004 sample 35/repeat 1."""

    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "1. Initial amount: \\$4. "
        "2. After buying the candy bar: \\$4 - \\$8 = -\\$4. "
        "3. After receiving money: -\\$4 + \\$5 = \\$1. "
        r"**Answer:** \$1"
    )

    assert fr._math_verify_input(completion) == "Answer: 1"
    assert fr._math_verify("1", completion).passed


def test_embedded_escaped_currency_is_normalized_only_at_verify_boundary(
    monkeypatch,
) -> None:
    """Regressions for task 2449/2443 final money-answer prose."""

    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [("gold", text.removeprefix("$\\boxed{").removesuffix("}$"))]
        if r"\$" in text:
            return []
        values = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        return [("value", values[-1])] if values else []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1][-1] == pred[-1][-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    completion = (
        r"**Answer:** Janet saves **\$2** per week by purchasing the pass."
    )

    assert r"\$2" in fr._math_verify_input(completion)
    result = fr._math_verify("2", completion)
    assert result.passed
    assert result.answer == "2"


def test_strong_answer_presentation_outranks_later_explanatory_weak_cue(
    monkeypatch,
) -> None:
    """Regression for SVAMP task 27724 sample 665/repeat 3."""

    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        "**Answer:** Dan spent $13 buying the items.\n"
        + "Long consistency check. " * 180
        + "\nFor completeness, the answer is $13, even though it is impossible "
        "with only $4.\n"
        "Unfinished check: ("
    )

    window = fr._math_verify_input(completion)
    assert "Dan spent $13" in window
    assert "even though" not in window
    assert fr._math_verify("13", completion).passed


def test_false_complete_latex_payloads_are_rejected_but_signed_scalar_is_kept() -> None:
    assert fr._answer_candidate_is_complete("-1")
    assert fr._answer_candidate_is_complete(r"\frac{1}{2}")
    assert not fr._answer_candidate_is_complete(r"\;")
    assert not fr._answer_candidate_is_complete(r"-\,\!")
    assert not fr._answer_candidate_is_complete(
        r"(\,\underbrace{p_{1}(x)}_{= \;\;\;\;\;\;} )"
    )


def test_false_complete_box_never_overrides_earlier_complete_box(
    monkeypatch,
) -> None:
    _patch_last_scalar_math_verify(monkeypatch)
    tails = (
        r"\boxed{\; }",
        r"\boxed{-\,\!}",
        r"\boxed{(\,\underbrace{p_{1}(x)}_{= \;\;\;\;\;\;} )}",
    )
    for tail in tails:
        completion = (
            r"Final answer: \boxed{-1}."
            + "\nLong verification. " * 180
            + f"\n{tail}\nUnfinished check: ("
        )
        assert fr._last_boxed_content(completion) == "-1"
        assert fr._math_verify_input(completion) == r"\boxed{-1}"
        assert fr._math_verify("-1", completion).passed


def test_markdown_final_answer_heading_outranks_quoted_prior_answer(
    monkeypatch,
) -> None:
    """Regression for task 27188 completion 68854941.

    A quoted earlier subproblem says ``final answer is 6.2``.  The model later
    presents both price and quantity under a real Markdown Final Answer
    heading; the last boxed quantity is the answer to the active subproblem.
    """

    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        'The prompt quoted: "Final answer: The final answer is 6.2."\n'
        "Long derivation. " * 180
        + "\n**Final Answer**\n"
        "Subproblem 0: price is \\boxed{6.2}; quantity is \\boxed{57}.\n"
        "Subproblem 1: price is \\boxed{6.2}; quantity is \\boxed{57}.\n"
        + '{"escaped": "### Final Answer\\\\nprice \\\\boxed{6.2}; '
        'quantity \\\\boxed{57}."}'
    )

    assert fr._math_verify_input(completion) == r"\boxed{57}"
    assert fr._math_verify("57", completion).passed


def test_legacy_one_stage_b_and_c_share_safe_answer_recovery(monkeypatch) -> None:
    _patch_last_scalar_math_verify(monkeypatch)
    completion = (
        r"Final answer: \boxed{-1}."
        + "\nLong verification. " * 180
        + "\n"
        + r"\boxed{-\,\!}"
        + "\nUnfinished check: ("
    )
    payload = {
        "prompt1": "User: q\nAssistant:",
        "completion1": completion,
        "stop_reason1": "stop_token",
    }

    for group in (STRATEGY_B, STRATEGY_C):
        result = fr.score_free_response_strategy(
            group,
            payload,
            sample_index=0,
            repeat_index=0,
            question="q",
            reference="-1",
        )
        assert result.math_passed
        assert result.display_answer == "-1"


def test_conclusive_mcq_label_bypasses_incomplete_math_tail(monkeypatch) -> None:
    monkeypatch.setattr(
        fr,
        "_math_verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MCQ must not enter math verification")
        ),
    )
    question = "Choose one: (A) alpha (B) beta (C) gamma (D) delta"
    payload = {
        "prompt1": f"User: {question}\nAssistant: <think",
        "completion1": (
            "That corresponds to option A. Additional arithmetic ends with "
            r"\boxed{-\,\!}"
        ),
        "stop_reason1": "max_tokens",
    }

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="A",
    )
    assert result.math_passed
    assert result.display_answer == "A"


def test_mcq_option_text_accepts_only_cosmetic_latex_variants() -> None:
    exponent_question = (
        "Factor completely.\n"
        "A) \\( (32b - 1)(32b + 1) \\)\n"
        "B) \\( (b - 8)^2 \\)\n"
        "C) \\( (8b - 1)^2 \\)\n"
        "D) \\( (8b + 1)^2 \\)"
    )
    exponent = fr._multiple_choice_verify(
        exponent_question,
        "C",
        r"\boxed{(8b-1)^{2}}",
    )
    assert exponent is not None
    assert exponent.conclusive
    assert exponent.result.passed
    assert exponent.result.answer == "C"

    inequality_question = (
        "Which set is correct?\n"
        r"A) \{x:x\geq 1\}" "\n"
        r"B) \{x:x\geq -7\}" "\n"
        r"C) \{x:x\leq 1\}" "\n"
        r"D) \{x:x\leq -7\}"
    )
    inequality = fr._multiple_choice_verify(
        inequality_question,
        "D",
        r"\boxed{\{x : x \le -7\}}",
    )
    assert inequality is not None
    assert inequality.conclusive
    assert inequality.result.passed
    assert inequality.result.answer == "D"


def test_judgement_path_bypasses_math_candidate_recovery(monkeypatch) -> None:
    monkeypatch.setattr(
        fr,
        "_math_verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("judgement must not enter math verification")
        ),
    )
    payload = {
        "prompt1": "User: judge this\nAssistant: <think",
        "completion1": r"irrelevant \boxed{-\,\!}",
        "stop_reason1": "stop_token",
        "prompt2": "\nJudgement: ",
        "completion2": "No",
        "stop_reason2": "stop_token",
    }

    for group in (STRATEGY_B, STRATEGY_C):
        result = fr.score_free_response_strategy(
            group,
            payload,
            sample_index=0,
            repeat_index=0,
            question="judge this",
            reference="Judgement: No",
        )
        assert result.math_passed
        assert result.display_answer == "Judgement: No"


def test_terminal_display_math_closer_is_not_mistaken_for_truncation() -> None:
    completion = (
        "Thus final answer: \\boxed{(3,0)} and \\boxed{(-3,1)}."
        "</think>The pairs are \\(3,0\\) and \\((-3,1)\\).\n"
        r"\[\boxed{(3,0)} \quad \text{and} \quad \boxed{(-3,1)}\]"
    )

    assert not fr._tail_is_syntactically_incomplete(completion)
    assert fr._math_verify_input(completion) == completion


def test_think_boundary_ends_an_explicit_answer_clause_before_unfinished_tail() -> None:
    completion = (
        "Thus, final answer: 4085.</think>"
        "A second presentation begins with \\(g(2011) = ("
    )

    assert fr._math_verify_input(completion) == "final answer: 4085."


def test_conclusive_option_mapping_outranks_unlabelled_trailing_scalar() -> None:
    completion = (
        "Thus answer: Rana's sample contained more students. "
        "That corresponds to option A. "
        "Checking the reported values 7.1 +/- 1.2 and 8.3 +/-"
    )

    assert fr._math_verify_input(completion) == "Final answer: A"


def test_strategy_b_mixed_mcq_uses_real_question_before_numeric_fallback() -> None:
    """Regression for Gaokao task 20664, sample 205, repeat 6.

    The completion contains several reported decimal values after identifying
    option A.  Replays must preserve the original question so the deterministic
    multiple-choice path wins instead of treating those decimals as a free-
    response prediction.
    """

    question = (
        "Micha estimated a mean of 7.1 with margin 1.2; Rana estimated 8.3 "
        "with margin 0.8. Which best explains Rana's smaller margin? "
        "(A) Rana sampled more students. "
        "(B) Rana sampled more students who drank soft drinks. "
        "(C) More students drank exactly seven servings. "
        "(D) More students drank exactly eight servings."
    )
    completion = (
        "The margin is inversely proportional to the square root of sample size. "
        "That corresponds to option A. "
        "The reported intervals are 7.1 +/- 1.2 and 8.3 +/- 0.8. "
        "Thus the correct choice is A."
    )
    payload = {
        "prompt1": f"User: {question}\n\nAssistant: <think",
        "completion1": f" reasoning</think>{completion}",
        "stop_reason1": "stop_token",
    }

    result = fr.score_free_response_strategy(
        STRATEGY_B,
        payload,
        sample_index=205,
        repeat_index=6,
        question=question,
        reference="A",
    )

    assert result.math_passed
    assert result.display_answer == "A"
    assert result.fail_reason == ""


def test_embedded_format_instruction_fails_closed_to_later_raw_text() -> None:
    completion = (
        "The answer should be 50 years.\n"
        + "Precision discussion. " * 180
        + "\nFormatting note: the answer should be presented as approximately 50.7 years, "
        "or 51 years after rounding.\nUnfinished check: log("
    )

    window = fr._math_verify_input(completion)
    assert window != "The answer should be 50 years."
    assert "approximately 50.7 years" in window


def test_line_level_format_answer_fails_closed_to_raw_text() -> None:
    completion = (
        "The answer should be 50 years.\n"
        "The answer should be presented as approximately 50.7 years, "
        "or 51 years after rounding.\nUnfinished check: log("
    )

    assert fr._math_verify_input(completion) == completion


def test_empty_box_format_instruction_never_becomes_answer_candidate() -> None:
    completion = (
        "The derivation gives \\boxed{12}.\n"
        "The problem asks to put the complete final answer inside \\boxed{}.\n"
        "Therefore, final answer is "
    )

    assert fr._math_verify_input(completion) == r"\boxed{12}"


def test_first_person_answer_commitment_outranks_older_interpretation() -> None:
    completion = (
        "That is a clean answer: $1800.\n"
        "The literal wording instead gives $2400.\n"
        "I'll answer $2400.\n"
        "But wait: the attendance clause might be used to compute"
    )

    assert fr._math_verify_input(completion) == "I'll answer 2400"


def test_format_answer_with_later_rounding_condition_still_invalidates_old_value() -> None:
    completion = (
        "The answer should be 50 years.\n"
        "The answer should be presented as approximately 50.7 years, "
        "which can be rounded to 51 years if using two significant figures.\n"
        "Unfinished check: log("
    )

    assert fr._math_verify_input(completion) == completion


def test_preopened_final_stage_keeps_b_raw_and_c_format_repair(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "wrong answer \\boxed{8}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "work</think>",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "7",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]


def test_truncated_stage2_never_inherits_stage1_boxed_answer(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "strategy_a_prompt": "User: q\nAssistant: <think",
        "strategy_a_completion": "wrong answer \\boxed{8}",
        "strategy_a_stop_reason": "stop_token",
        "prompt1": "User: q\nAssistant: <think",
        "completion1": "reasoning with \\boxed{7}</think>",
        "stop_reason1": "stop_token",
        "prompt2": "\nTherefore, the answer is \\(\\boxed{",
        "completion2": "(",
        "stop_reason2": "max_length",
    }

    evaluation = evaluate_free_response(
        [payload],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group[STRATEGY_B] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert "reasoning with" not in fr._strategy_scoring_text(STRATEGY_B, payload)
    assert "reasoning with" not in fr._strategy_scoring_text(STRATEGY_C, payload)


def test_mcq_path_recovers_complete_label_before_incomplete_tail(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) one (B) two (C) three (D) four",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    completion = (
        "Therefore, the answer is C.\n"
        + "Long optional verification. " * 150
        + "\nunfinished \\boxed{(1/2"
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": completion,
                "stop_reason1": "max_length",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "C"


def test_strategy_a_scores_raw_full_generation_and_tracks_stop_rate(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q1\n\nAssistant: <think",
                "completion1": "<think>work</think>\nFinal answer: \\boxed{7}",
                "stop_reason1": "stop_token",
                "stats": {"truncated": False, "stop_detail": "token_0"},
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "User: q2\n\nAssistant: <think",
                "completion1": "<think>work</think>\nFinal answer: \\boxed{8}",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            },
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert set(evaluation.metrics_by_group) == set(STRATEGY_GROUPS)
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_a"]["stop_rate"] == 0.5
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True), (1, 0, False)]
    assert len(evaluation.payloads) == 2
    assert all("eval_group" not in payload for payload in evaluation.payloads)

    metrics_payload, task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 0.5
    assert metrics_payload["stop_rate"] == 0.5
    assert task_details == {}


def test_g1h_prompt_clips_flower_sentinel_before_math_scoring(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User✿q✿\nBot✿<think></think>",
                "completion1": "</think>\nFinal answer: \\boxed{7}✿clean_length \\boxed{0}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]


def test_legacy_prompt_does_not_clip_flower_sentinel_for_g1g(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nFinal answer: \\boxed{7}✿clean_length \\boxed{0}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]


def test_strategy_b_c_inherit_a_passes_and_only_rescore_a_failures(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q1\n\nAssistant: <think",
                "completion1": "</think>\nlegacy answer \\boxed{7}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "0",
                "stop_reason2": "stop_token",
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "User: q2\n\nAssistant: <think",
                "completion1": "</think>\nlegacy answer \\boxed{8}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "9",
                "stop_reason2": "stop_token",
            },
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == "strategy_a"
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, True), (1, 0, False)]
    assert evaluation.rows_by_group["strategy_c"] == [(0, 0, True), (1, 0, True)]
    assert evaluation.payloads[0]["answer"].endswith("7")

    metrics_payload, _task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 0.5
    assert metrics_payload["strategy_metrics"]["strategy_c"]["avg@1"] == 1.0


def test_strategy_c_can_be_the_formal_combined_primary(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q1\nAssistant: <think",
                "strategy_a_completion": "</think>\nFinal answer: \\boxed{7}",
                "strategy_a_stop_reason": "stop_token",
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q2\nAssistant: <think",
                "strategy_a_completion": "wrong answer \\boxed{8}",
                "strategy_a_stop_reason": "max_length",
                "prompt1": "User: q2\nAssistant: <think",
                "completion1": "work</think>",
                "stop_reason1": "stop_token",
                "prompt2": "\n</think>\nTherefore, the answer is \\(\\boxed{",
                "completion2": "9",
                "stop_reason2": "stop_token",
            },
        ],
        dataset_path=dataset,
        judge=None,
        primary_group=STRATEGY_C,
    )

    metrics_payload, _ = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert evaluation.primary_group == STRATEGY_C
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True), (1, 0, True)]
    assert metrics_payload["avg@1"] == 1.0


def test_blank_recovery_stage_is_an_explicit_missing_prediction(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "wrong answer \\boxed{8}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "reasoning with \\boxed{7}</think>",
                "stop_reason1": "stop_token",
                "prompt2": "Bot✿</think>\\nTherefore, the answer is \\(\\boxed{",
                # Sentinel clipping also has to produce the same blank
                # recovery semantics as a literal empty response.
                "completion2": "   ✿ignored transport suffix",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
        primary_group=STRATEGY_C,
    )

    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert evaluation.metrics_by_group[STRATEGY_C]["exact_accuracy"] == 0.0
    assert (
        evaluation.payloads_by_group["strategy_b"][0]["fail_reason"]
        == fr.MISSING_RECOVERY_PREDICTION
    )
    assert evaluation.payloads[0]["answer"] == ""
    assert (
        evaluation.payloads[0]["fail_reason"]
        == fr.MISSING_RECOVERY_PREDICTION
    )


def test_blank_recovery_result_is_independent_of_stage1_length(
    monkeypatch, tmp_path
) -> None:
    """A blank final stage must never expose a stage-1 answer-window quirk."""

    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    payloads = []
    for padding in ("short", "x" * 4_000):
        payloads.append(
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\nAssistant: <think",
                "completion1": f"{padding} reasoning \\boxed{{7}}</think>",
                "stop_reason1": "stop_token",
                "prompt2": "</think>\nTherefore, the answer is \\(\\boxed{",
                "completion2": " \t\n",
                "stop_reason2": "stop_token",
            }
        )

    evaluation = evaluate_free_response(
        payloads,
        dataset_path=dataset,
        judge=None,
        primary_only=True,
        primary_group=STRATEGY_C,
    )

    assert evaluation.rows_by_group[STRATEGY_C] == [
        (0, 0, False),
        (0, 0, False),
    ]
    assert [item["answer"] for item in evaluation.payloads] == ["", ""]
    assert [item["fail_reason"] for item in evaluation.payloads] == [
        fr.MISSING_RECOVERY_PREDICTION,
        fr.MISSING_RECOVERY_PREDICTION,
    ]


def test_structural_blank_strategy_a_does_not_fallback_or_inherit(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "",
                "strategy_a_stop_reason": "stop_token",
                # This would make A pass under the old silent fallback.
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "reasoning with \\boxed{7}</think>",
                "stop_reason1": "stop_token",
                "prompt2": "</think>\nTherefore, the answer is \\(\\boxed{",
                "completion2": "8}",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert evaluation.payloads_by_group["strategy_a"][0]["answer"] == ""
    assert (
        evaluation.payloads_by_group["strategy_a"][0]["fail_reason"]
        == fr.MISSING_STRATEGY_A_PREDICTION
    )


def test_structural_blank_strategy_a_is_never_sent_to_llm_judge(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            backoff_base=0.0,
        )
    )
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "reasoning with \\boxed{7}</think>",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert judge.client.prompts == []
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert evaluation.payloads[0]["answer"] == ""
    assert (
        evaluation.payloads[0]["fail_reason"]
        == fr.MISSING_STRATEGY_A_PREDICTION
    )
    assert all(
        payloads[0]["fail_reason"] == fr.MISSING_STRATEGY_A_PREDICTION
        for payloads in evaluation.payloads_by_group.values()
    )


def test_sentinel_only_strategy_a_is_an_explicit_missing_prediction(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    flower = fr.G1H_GENERATION_STOP_SUFFIXES[-1]
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": f"User{flower} q\nBot{flower}<think",
                "strategy_a_completion": (
                    f"   {fr.G1H_GENERATION_STOP_SUFFIXES[-1]}ignored suffix"
                ),
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "</think>\nFinal answer: \\boxed{7}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
        primary_only=True,
        primary_group="strategy_a",
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["answer"] == ""
    assert (
        evaluation.payloads[0]["fail_reason"]
        == fr.MISSING_STRATEGY_A_PREDICTION
    )


def test_legacy_payload_without_strategy_a_keeps_stage1_scoring(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "</think>\nFinal answer: \\boxed{7}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]


def test_blank_recovery_stage_is_never_sent_to_llm_judge(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            backoff_base=0.0,
        )
    )
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "reasoning with \\boxed{7}</think>",
                "stop_reason1": "stop_token",
                "prompt2": "</think>\\nTherefore, the answer is \\(\\boxed{",
                "completion2": "",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
        primary_only=True,
        primary_group=STRATEGY_C,
    )

    assert judge.client.prompts == []
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert evaluation.payloads[0]["answer"] == ""
    assert evaluation.payloads[0]["fail_reason"] == "missing_recovery_prediction"


def test_strategy_a_pass_still_inherits_when_recovery_stage_is_blank(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "</think>\\nFinal answer: \\boxed{7}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "wrong reasoning \\boxed{8}</think>",
                "stop_reason1": "stop_token",
                "prompt2": "</think>\\nTherefore, the answer is \\(\\boxed{",
                "completion2": "",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, True)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_C][0]["answer"] == "7"


def test_strategy_c_keeps_legacy_single_stage_compatibility(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "</think>\\nFinal answer: \\boxed{7}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
        primary_only=True,
        primary_group=STRATEGY_C,
    )

    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "7"


def test_strategy_c_repairs_unclosed_think_for_scoring_only(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "requires_unclosed_repair",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_strategy_c_closes_box_before_any_synthetic_think_repair(
    monkeypatch, tmp_path
) -> None:
    """The think closure belongs before the final-stage prompt and answer."""

    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "unfinished wrong reasoning",
                "strategy_a_stop_reason": "max_length",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "unfinished reasoning",
                "stop_reason1": "max_length",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "7",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
        primary_group=STRATEGY_C,
    )

    assert evaluation.metrics_by_group[STRATEGY_C]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "7"


def test_strategy_c_repairs_truncated_answer_region(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nrequires_truncated_repair",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_two_stage_b_and_c_repair_only_a_failure_with_unclosed_think(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "wrong answer \\boxed{8}",
                "strategy_a_stop_reason": "max_length",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "requires_unclosed_repair",
                "stop_reason1": "max_length",
                # Legacy payload without a structural think closure still
                # needs the B/C synthetic repair below.
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "7}",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_two_stage_c_repairs_truncated_answer_but_b_keeps_raw_text(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "User: q\nAssistant: <think",
                "strategy_a_completion": "wrong answer \\boxed{8}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "User: q\nAssistant: <think",
                "completion1": "work</think>",
                "stop_reason1": "stop_token",
                "prompt2": "\n</think>\nTherefore, the answer is \\(\\boxed{",
                "completion2": "requires_truncated_repair",
                "stop_reason2": "max_length",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_two_stage_payload_uses_legacy_completion1_scoring(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nwrong intermediate \\boxed{8}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "7",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.primary_group == "strategy_a"
    assert evaluation.payloads[0]["answer"].endswith("8")


def test_judgement_label_dataset_scores_final_stage_without_math_verify(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fr, "_load_math_verify", lambda: None)
    dataset = tmp_path / "answer_judge.jsonl"
    dataset.write_text(
        (
            '{"question":"q","expected_answer":"reference text",'
            '"predicted_answer":"candidate text","expected_judgement":"Judgement: No"}\n'
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "answer_judge",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nJudgement: Yes",
                "stop_reason1": "stop_token",
                "prompt2": "\nReturn exactly one final label: ",
                "completion2": "Judgement: No",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == "strategy_c"
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_c"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "Judgement: No"

    metrics_payload, _task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 1.0
    assert metrics_payload["strategy_metrics"]["strategy_a"]["avg@1"] == 0.0


def test_judgement_label_blank_recovery_does_not_fallback_to_stage1(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(fr, "_load_math_verify", lambda: None)
    dataset = tmp_path / "answer_judge.jsonl"
    dataset.write_text(
        (
            '{"question":"q","expected_answer":"reference text",'
            '"predicted_answer":"candidate text",'
            '"expected_judgement":"Judgement: No"}\n'
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "Assistant: <think",
                "strategy_a_completion": "</think> Judgement: Yes",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "Assistant: <think",
                # This would pass under the old silent stage-1 fallback.
                "completion1": "</think> Judgement: No",
                "stop_reason1": "stop_token",
                "prompt2": "Return exactly one final label: ",
                "completion2": "",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == STRATEGY_C
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, False)]
    assert evaluation.payloads[0]["answer"] == ""
    assert evaluation.payloads[0]["fail_reason"] == "missing_recovery_prediction"


def test_judgement_bare_final_stage_does_not_leak_prompt_labels(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fr, "_load_math_verify", lambda: None)
    dataset = tmp_path / "answer_judge.jsonl"
    dataset.write_text(
        (
            '{"question":"q","expected_answer":"reference text",'
            '"predicted_answer":"candidate text","expected_judgement":"Judgement: Yes"}\n'
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "answer_judge",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": (
                    "Return exactly `Judgement: Yes` or `Judgement: No`.\n\n"
                    "Assistant: <think"
                ),
                "completion1": '</think>{"Judgement": "Yes"}',
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "Yes",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == "strategy_c"
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.payloads[0]["answer"] == "Judgement: Yes"


def test_judgement_label_accepts_json_style_generated_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fr, "_load_math_verify", lambda: None)
    dataset = tmp_path / "answer_judge.jsonl"
    dataset.write_text(
        (
            '{"question":"q","expected_answer":"reference text",'
            '"predicted_answer":"candidate text","expected_judgement":"Judgement: No"}\n'
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "answer_judge",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "Assistant: <think",
                "completion1": '</think>{"Judgement": "No"}',
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 1.0
    assert evaluation.payloads[0]["answer"] == "Judgement: No"


def test_a_judge_pass_skips_strategy_judge_and_inherits_correctness(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            backoff_base=0.0,
        )
    )
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "\\boxed{7}",
                "stop_reason1": "stop_token",
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "\\boxed{8}",
                "stop_reason1": "stop_token",
            },
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_a"]["judge_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_b"]["judge_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_c"]["judge_accuracy"] == 1.0
    assert [row[2] for row in evaluation.rows_by_group["strategy_a"]] == [True, True]
    assert [row[2] for row in evaluation.rows_by_group["strategy_b"]] == [True, True]
    assert [row[2] for row in evaluation.rows_by_group["strategy_c"]] == [True, True]
    assert len(judge.client.prompts) == 1
    assert all("Question: q2" in prompt for prompt in judge.client.prompts)
    assert all("Student's Answer:" in prompt for prompt in judge.client.prompts)

    metrics_payload, task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    expected_stats = {
        "total": 1,
        "parsed_count": 1,
        "invalid_output_count": 0,
        "request_error_count": 0,
        "error_count": 0,
        "invalid_output_examples": [],
        "request_error_examples": [],
        **llm_judge_protocol(judge.config),
        "protocol_fingerprint_sha256": llm_judge_protocol_fingerprint(judge.config),
    }
    assert metrics_payload["judge_stats"] == expected_stats
    assert task_details["judge_stats"] == expected_stats


def test_llm_judge_defaults_to_deterministic_temperature() -> None:
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            recovery_rounds=0,
        )
    )
    client = _CapturingJudgeClient("True")
    judge.client = client

    assert judge.judge([("question", "reference", "prediction")]) == [True]
    assert client.requests[0]["temperature"] == 0.0
    assert judge.last_run_stats is not None
    stats = judge.last_run_stats.as_dict()
    assert stats["protocol_version"] == LLM_JUDGE_PROTOCOL_VERSION
    assert stats["temperature"] == 0.0
    assert stats["response_contract"] == LLM_JUDGE_RESPONSE_CONTRACT
    assert stats["protocol_fingerprint_sha256"] == llm_judge_protocol_fingerprint(
        judge.config
    )


def test_llm_judge_protocol_fingerprint_is_stable_complete_and_secret_free() -> None:
    config = LLMJudgeConfig(
        api_key="super-secret-key",
        base_url="https://secret-relay.invalid/v1",
        model="judge-model-v2",
        max_completion_tokens=37,
        temperature=0.0,
        max_workers=32,
        max_retries=4,
        recovery_rounds=1,
        prompt_template="Question=<Q>\nReference=<REF>\nAnswer=<A>\nTrue or False only.",
    )

    protocol = llm_judge_protocol(config)
    serialized = json.dumps(protocol, sort_keys=True)

    assert protocol == {
        "protocol_version": LLM_JUDGE_PROTOCOL_VERSION,
        "model": "judge-model-v2",
        "temperature": 0.0,
        "prompt_template_sha256": llm_judge_prompt_sha256(config.prompt_template),
        "max_completion_tokens": 37,
        "response_contract": LLM_JUDGE_RESPONSE_CONTRACT,
        "stream": False,
        "qwen3_enable_thinking": None,
        "max_workers": 32,
        "max_retries": 4,
        "recovery_rounds": 1,
    }
    assert len(llm_judge_protocol_fingerprint(config)) == 64
    assert "super-secret-key" not in serialized
    assert "secret-relay" not in serialized
    assert "api_key" not in protocol
    assert "base_url" not in protocol


def test_llm_judge_protocol_fingerprint_changes_with_content_semantics() -> None:
    baseline = LLMJudgeConfig(api_key="k", model="m")
    variants = (
        LLMJudgeConfig(api_key="k", model="other"),
        LLMJudgeConfig(api_key="k", model="m", temperature=0.1),
        LLMJudgeConfig(api_key="k", model="m", max_completion_tokens=17),
        LLMJudgeConfig(api_key="k", model="m", prompt_template="custom <Q> <REF> <A>"),
    )

    baseline_fingerprint = llm_judge_protocol_fingerprint(baseline)
    assert all(
        llm_judge_protocol_fingerprint(variant) != baseline_fingerprint
        for variant in variants
    )
    assert llm_judge_protocol_fingerprint(
        LLMJudgeConfig(
            api_key="different-secret",
            base_url="https://different.invalid/v1",
            model="m",
        )
    ) == baseline_fingerprint


def test_llm_judge_protocol_stats_validation_is_exact_and_tamper_evident() -> None:
    config = LLMJudgeConfig(
        api_key="k",
        model="Qwen3-32B",
        max_workers=32,
        max_completion_tokens=64,
        prompt_template="judge <Q> <REF> <A>",
    )
    stats = {
        **llm_judge_protocol(config),
        "protocol_fingerprint_sha256": llm_judge_protocol_fingerprint(config),
    }
    assert llm_judge_protocol_stats_reasons(
        stats,
        expected_model="Qwen3-32B",
        expected_prompt_template=config.prompt_template,
        expected_max_completion_tokens=64,
        expected_max_workers=32,
    ) == []

    tampered = {**stats, "temperature": 0.8}
    reasons = llm_judge_protocol_stats_reasons(tampered)
    assert any(reason.startswith("judge_protocol_fingerprint:") for reason in reasons)
    assert "judge_temperature:0.8!=expected:0.0" in reasons

    assert llm_judge_protocol_stats_reasons({}) == [
        "judge_protocol_missing_fields:protocol_version,model,temperature,"
        "prompt_template_sha256,max_completion_tokens,response_contract,stream,"
        "qwen3_enable_thinking,max_workers,max_retries,recovery_rounds"
    ]


def test_llm_judge_prompt_sha256_uses_exact_utf8_template_bytes() -> None:
    assert llm_judge_prompt_sha256(DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE) == (
        "63d7a5079471ae9756e8f73b6586bc8375dea6dfafb90d9c1b3635dcba9b9518"
    )
    assert llm_judge_prompt_sha256("A\nB") != llm_judge_prompt_sha256("A\r\nB")


def test_qwen3_judge_protocol_records_disabled_thinking_request() -> None:
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="Qwen3-32B",
            max_workers=1,
            max_retries=0,
            recovery_rounds=0,
        )
    )
    client = _CapturingJudgeClient("False")
    judge.client = client

    assert judge.judge([("question", "reference", "prediction")]) == [False]
    assert client.requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert judge.last_run_stats is not None
    assert judge.last_run_stats.as_dict()["qwen3_enable_thinking"] is False


def test_mcq_explicit_correct_label_is_deterministic_and_not_judged(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose one: (A) zero (B) one (C) two (D) three",'
        '"answer":"A"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "Long reasoning that math-verify cannot parse.\n**Answer: A**",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "A"
    assert judge.client.prompts == []
    zero_call_stats = evaluation.judge_stats_by_group["strategy_a"]
    assert zero_call_stats["total"] == 0
    assert zero_call_stats["parsed_count"] == 0
    assert zero_call_stats["protocol_fingerprint_sha256"] == (
        llm_judge_protocol_fingerprint(judge.config)
    )
    metrics_payload, task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["judge_stats"] == zero_call_stats
    assert task_details["judge_stats"] == zero_call_stats


def test_mcq_option_text_maps_to_reference_label_without_judge(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"How many? (A) 10 (B) 15 (C) 20 (D) 25",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "Calculation omitted.\nFinal answer: 20",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "C"
    assert judge.client.prompts == []


def test_mcq_latex_align_options_map_content_to_label_without_judge(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Pick one:\\n\\\\begin{align*}\\n'
        '\\\\text{A)}\\\\ & 10 &\\n\\\\text{B)}\\\\ & 15\\\\\\\\\\n'
        '\\\\text{C)}\\\\ & 20 &\\n\\\\text{D)}\\\\ & 25\\\\\\\\\\n'
        '\\\\end{align*}","answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "Calculation omitted.\nFinal answer: 20",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "C"
    assert judge.client.prompts == []


def test_mcq_boxed_parenthesized_label_plus_option_text_maps_to_label(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Pick one: (A) y=f(x-2) (B) y=f(x+2) '
        '(C) y=f(x)-11 (D) y=f(x)+11","answer":"D"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": r"Final answer: \boxed{\text{(D) } y=f(x)+11}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "D"


def test_mcq_math_fallback_display_answer_maps_to_unique_scalar_option(
    monkeypatch, tmp_path
) -> None:
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"How many? (A) 30 (B) 90 (C) 120 (D) 150",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(fr, "_load_math_verify", lambda: (object(), object()))
    monkeypatch.setattr(
        fr,
        "_math_verify",
        lambda *_args, **_kwargs: fr._MathVerifyResult(
            passed=False,
            answer="120",
            fail_reason="math_verify_false",
        ),
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Thus 120 adult tickets were sold.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "C"


def test_mcq_latex_spacing_commands_do_not_change_option_identity(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Pick: (A) $g(x)=16(0.8)^{x-2}$ '
        '(B) $g(x)=8(0.8)^{x-2}$","answer":"A"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": r"Final answer: \boxed{g(x)=16\,(0.8)^{\,x-2}}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "A"


def test_mcq_concatenated_markers_ignore_unrelated_roman_marker(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Facts (I) and (II). Choose: (A) first(B) second(C) third(D) fourth",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "Final answer: third",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "C"


def test_mcq_repeated_option_markers_fail_closed() -> None:
    question = (
        "Example: (A) demo (B) demo. Actual: "
        "(A) alpha (B) beta (C) gamma (D) delta"
    )

    assert fr._parse_question_options(question, required_label="C") == {}


def test_mcq_overlapping_ambiguous_marker_syntax_fails_closed() -> None:
    assert fr._option_markers_are_ambiguous(
        [("A", 10, 16), ("B", 14, 20)]
    )


def test_mcq_repeated_markers_cannot_be_overturned_by_judge(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "ambiguous_mcq.jsonl"
    dataset.write_text(
        '{"question":"Example (A) alpha (B) beta. Actual (A) one '
        '(B) two (C) three (D) four","answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final answer: three",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["fail_reason"] == (
        "ambiguous_multiple_choice_question"
    )
    assert judge.client.prompts == []


def test_mcq_explicit_wrong_label_fails_closed_without_judge(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose one: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    # A stochastic judge used to be able to overturn this recognized wrong
    # choice.  The structured comparator must keep it deterministically false.
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": (
                    "Reasoning.\n(D) delta\nTherefore, the answer is \\boxed{D"
                ),
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["answer"] == "D"
    assert evaluation.payloads[0]["fail_reason"] == "multiple_choice_label_mismatch"
    assert judge.client.prompts == []


def test_mcq_non_abcd_labels_are_supported_when_question_proves_the_option_set(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose one: (F) red (G) blue (H) green (J) black (K) white",'
        '"answer":"K"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "The correct choice is **K**.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "K"
    assert judge.client.prompts == []


def test_single_letter_free_response_without_options_keeps_math_fallback(
    monkeypatch, tmp_path
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"Name the integration constant.","answer":"A"}\n',
        encoding="utf-8",
    )

    assert fr._multiple_choice_verify(
        "Name the integration constant.", "A", "Final answer: \\boxed{A}"
    ) is None
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "Final answer: \\boxed{A}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]


class _CapturingJudgeClient:
    def __init__(self, response: str) -> None:
        self.prompts: list[str] = []
        self.requests: list[dict] = []
        self.chat = _CapturingChat(self, response)


class _CapturingChat:
    def __init__(self, client: _CapturingJudgeClient, response: str) -> None:
        self.completions = _CapturingCompletions(client, response)


class _CapturingCompletions:
    def __init__(self, client: _CapturingJudgeClient, response: str) -> None:
        self._client = client
        self._response = response

    def create(self, **kwargs):
        from types import SimpleNamespace

        self._client.prompts.append(kwargs["messages"][0]["content"])
        self._client.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._response)),
            ]
        )


@pytest.mark.parametrize(
    "completion",
    (
        "The answer is 54. Wait, that answer is wrong. The final answer is (1/2",
        r"The answer is \boxed{54}. Wait, that answer is wrong. Final answer:",
    ),
)
def test_retracted_math_without_complete_replacement_cannot_be_revived(
    monkeypatch,
    tmp_path,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"54"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        'A source says "Final answer: C." I reject it.',
        "If the premise held, the answer is C. But it does not.",
        "Answer: C? No, that cannot be right.",
        "Final answer: C. Wait, that answer is wrong. Final answer:",
    ),
)
def test_contextual_or_retracted_mcq_label_cannot_be_revived_by_judge(
    monkeypatch,
    tmp_path,
    completion: str,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["fail_reason"] == (
        "authoritative_multiple_choice_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "question",
    (
        "Choose the listed option: (C) gamma",
        "Choose: (A) alpha (C) gamma (D) delta",
        "Choose: (A) alpha [B) beta (C) gamma (D) delta",
    ),
)
def test_invalid_label_schema_cannot_fall_through_to_judge(
    monkeypatch,
    tmp_path,
    question: str,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "invalid_mcq.jsonl"
    dataset.write_text(
        json.dumps({"question": question, "answer": "C"}) + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final answer: C.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.payloads[0]["fail_reason"] == (
        "invalid_multiple_choice_question"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "question",
    (
        "Choose: A: alpha B: beta C: gamma D: delta",
        "Choose: A. alpha; B. beta; C. gamma; D. delta",
        "Choose: [A] alpha [B] beta [C] gamma [D] delta",
        "Choose: {A} alpha {B} beta {C} gamma {D} delta",
        "Choose: (a) alpha (b) beta (c) gamma (d) delta",
        "Choose:\na) alpha\nb) beta\nc) gamma\nd) delta",
        "Choose: [a] alpha [b] beta [c] gamma [d] delta",
    ),
)
def test_supported_mcq_schemas_are_resolved_without_judge(
    monkeypatch,
    tmp_path,
    question: str,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        json.dumps({"question": question, "answer": "C"}) + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final answer: C.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == "C"
    assert judge.client.prompts == []


def test_lowercase_mcq_reference_and_completion_normalize_to_schema_label(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "lowercase_mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (a) alpha (b) beta (c) gamma (d) delta",'
        '"answer":"c"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "final answer: c.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == "C"


def test_two_unrelated_i_markers_do_not_invalidate_clean_a_to_d_schema(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_math_verify(monkeypatch)
    question = (
        "Facts (I) applies, while a later note also calls itself (I). "
        "Choose: (A) alpha (B) beta (C) gamma (D) delta"
    )
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        json.dumps({"question": question, "answer": "C"}) + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final answer: C.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True)]
    assert judge.client.prompts == []


def test_judge_receives_only_complete_post_correction_answer(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"56"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "The answer is 54. Correction: 55.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert judge.client.prompts
    assert all("Student's Answer: 55" in prompt for prompt in judge.client.prompts)
    assert all("54" not in prompt for prompt in judge.client.prompts)


@pytest.mark.parametrize(
    "completion",
    (
        'A source says "The answer is 53." I reject it.\n54',
        "According to Bob, the answer is 53.\n54",
        "The textbook claims the answer is 53.\n54",
        "Under that assumption, the answer is 53. The assumption fails.\n54",
        "The answer is 53. Disregard that.\n54",
        "The answer is 53. Withdraw that answer.\n54",
        "The answer is 53. Ignore it.\n54",
        "The answer is 53. That was wrong.\n54",
    ),
)
def test_contextual_or_retracted_event_allows_only_terminal_math_replacement(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert result.final_passed
    assert result.display_answer == "54"


def test_contextual_mcq_event_allows_terminal_bare_replacement_without_judge(
    monkeypatch,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": 'A source says "Final answer: D." I reject it.\nC',
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert result.final_passed
    assert result.display_answer == "C"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        '"Final answer: 54"',
        "'Final answer: 54'",
        '{"answer":"Final answer: 54"}',
        '{"response":{"answer":"Final answer: 54"}}',
    ),
)
def test_serialized_answer_envelopes_are_unwrapped_before_scoring(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert result.final_passed
    assert "54" in result.display_answer


@pytest.mark.parametrize(
    "completion",
    (
        "A source says the answer is 54. I reject it.",
        "The answer is 54. I retract that answer.",
        r"Final answer: \boxed{54}. Scratch that.",
        "Under that assumption, the answer is 54. But the assumption fails.",
        "Final answer: 54. Disregard that.",
        "Final answer: 54. Withdraw that answer.",
        "Final answer: 54. Ignore it.",
        "Final answer: 54. That was wrong.",
        "According to Bob, the answer is 54.",
        "The textbook claims the answer is 54.",
    ),
)
def test_contextual_or_retracted_old_math_value_cannot_be_revived(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert not result.final_passed
    assert not result.judge_eligible
    assert result.display_answer == ""
    assert result.fail_reason in {
        "authoritative_answer_invalidated",
        "contextual_answer_only",
    }


def test_questioned_mcq_answer_followed_by_no_is_invalidated_without_judge(
    monkeypatch,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": "Is the answer C? No", "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_multiple_choice_answer_invalidated"
    assert result.display_answer == ""
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "replacement",
    ("Correction: D.", "Actually, D.", "Therefore D."),
)
def test_latest_mcq_replacement_is_conclusive_and_never_sent_to_judge(
    monkeypatch,
    tmp_path,
    replacement: str,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": f"Final answer: C. {replacement}",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == "D"
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "multiple_choice_label_mismatch"
    )
    assert judge.client.prompts == []


def test_two_stage_mcq_recovery_does_not_inherit_strategy_a_replacement(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "p",
                "strategy_a_completion": "Final answer: C. Correction: D.",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "reasoning prompt",
                "completion1": "reasoning",
                "stop_reason1": "stop_token",
                "prompt2": "\nFinal answer: ",
                "completion2": "C.",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_B] == [(0, 0, True)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == "D"
    assert evaluation.payloads_by_group[STRATEGY_B][0]["answer"] == "C"
    assert evaluation.payloads_by_group[STRATEGY_C][0]["answer"] == "C"
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        "Correction: 55. Final answer: 54.",
        "Answer: 53. Correction: 55. Final answer: 54.",
    ),
)
def test_latest_committed_math_event_wins_by_text_position(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    payload = {"completion1": completion, "stop_reason1": "stop_token"}

    latest = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )
    superseded = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert latest.final_passed
    assert "54" in latest.display_answer
    assert not superseded.final_passed


@pytest.mark.parametrize(
    "completion",
    (
        (
            r"Final answer: 54. A later verification temporarily substitutes \boxed{55} "
            "into an intermediate expression."
        ),
        r"Final answer: 54. An intermediate substitution uses \boxed{55}",
        (
            r"Final answer: 54. An alternative is shown so readers can compare: "
            r"\boxed{55}"
        ),
    ),
)
def test_committed_final_answer_beats_later_incidental_box(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    payload = {"completion1": completion, "stop_reason1": "stop_token"}

    final_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )
    incidental_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert final_result.final_passed
    assert not incidental_result.final_passed


@pytest.mark.parametrize(
    "completion",
    (
        "The answer is 54.\n\\boxed{55}",
        "Final answer: 54. Recalculation gives 55, so \\boxed{55}",
        "The answer is 54.\n55",
        "Final answer: 54." + (" " * 360) + "\\boxed{55}",
        "Final answer: 54." + (" " * 361) + "\\boxed{55}",
        "Final answer: 54." + (" \n" * 1000) + "\\boxed{55}",
        "Final answer: 54."
        + (r" \qquad \; \quad \hspace{2em}" * 200)
        + r" \boxed{55}",
        "Final answer: 54. A fresh verification yields 55: \\boxed{55}",
        "Final answer: 54. Rechecking the arithmetic gives 55: \\boxed{55}",
        "Final answer: 54. An omitted unit changes the total to 55: \\boxed{55}",
        (
            "Final answer: 54. Thus \\boxed{55}, "
            "completing the conclusion."
        ),
    ),
)
def test_postposed_terminal_math_answer_wins_chronologically(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    payload = {"completion1": completion, "stop_reason1": "stop_token"}

    latest = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )
    superseded = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert latest.final_passed
    assert "55" in latest.display_answer
    assert not superseded.final_passed


def test_long_explicit_recalculation_terminal_answer_overrides_old_commitment(
    monkeypatch,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    review = "I inspect every derivation and compare the constraints carefully. " * 40
    completion = (
        "Final answer: 54. "
        + review
        + "After a fresh recalculation, the corrected result is 55, so \\boxed{55}"
    )

    latest = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert latest.final_passed
    assert "55" in latest.display_answer


def test_terminal_boxed_mcq_answer_wins_chronologically_without_judge(
    monkeypatch,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": "Final answer: C.\n\\boxed{D}",
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="D",
    )

    assert result.final_passed
    assert result.display_answer == "D"
    assert not result.judge_eligible


def test_postposed_mcq_answer_keeps_two_stage_groups_independent(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "p",
                "strategy_a_completion": "Final answer: C.\n\\boxed{D}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "reasoning prompt",
                "completion1": "reasoning",
                "stop_reason1": "stop_token",
                "prompt2": "\nFinal answer: ",
                "completion2": "C.",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_B] == [(0, 0, True)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == "D"
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "expression",
    (
        r"(x,y)=(\boxed{2},\boxed{3})",
        r"x=\boxed{2}, y=\boxed{3}",
        r"\vec v=(\boxed{2},\boxed{3},\boxed{4})",
        r"S=\{\boxed{2},\boxed{3}\}",
        r"[\boxed{2},\boxed{3})",
        (
            r"\begin{pmatrix}\boxed{1}&\boxed{2}\\"
            r"\boxed{3}&\boxed{4}\end{pmatrix}"
        ),
    ),
)
def test_explicit_multi_box_answer_preserves_complete_expression(
    expression: str,
) -> None:
    completion = f"Final Answer:\n{expression}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert fr._math_verify_input(completion) == completion
    assert evidence.candidate.content != fr._last_boxed_content(expression)
    assert evidence.candidate.scoring_text.count("\\boxed{") == expression.count(
        "\\boxed{"
    )


def test_multi_box_tuple_is_passed_intact_to_math_verify(monkeypatch) -> None:
    parsed_predictions: list[str] = []

    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return ["(2,3)"]
        parsed_predictions.append(text)
        normalized = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
        return ["(2,3)"] if "(x,y)=(2,3)" in normalized else []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1] == pred[-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    completion = "Final Answer:\n(x,y)=(\\boxed{2},\\boxed{3})"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="(2,3)",
    )
    assert result.final_passed
    assert parsed_predictions == [completion]


@pytest.mark.parametrize(
    "expression",
    (
        "(\n\\boxed{2},\n\\boxed{3}\n)",
        "\\{\n\\boxed{2},\n\\boxed{3}\n\\}",
        "[\n\\boxed{2},\n\\boxed{3}\n)",
        "\\left\\langle\n\\boxed{2},\n\\boxed{3}\n\\right\\rangle",
        (
            "\\begin{pmatrix}\n"
            "\\boxed{1} & \\boxed{2} \\\\\n"
            "\\boxed{3} & \\boxed{4}\n"
            "\\end{pmatrix}"
        ),
    ),
)
def test_explicit_multiline_multi_box_block_preserves_complete_expression(
    expression: str,
) -> None:
    completion = f"Final Answer:\n{expression}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert evidence.candidate.scoring_text.count(r"\boxed{") == expression.count(
        r"\boxed{"
    )


@pytest.mark.parametrize(
    "cue",
    (
        "Answer:\n",
        "### Final Answer\n",
        "**Final Answer:**\r\n",
        "- **Final Answer:**\n",
    ),
)
def test_explicit_multiline_answer_block_accepts_heading_variants(cue: str) -> None:
    completion = f"{cue}(\n\\boxed{{2}},\n\\boxed{{3}}\n)"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert fr._math_verify_input(completion) == completion


@pytest.mark.parametrize(
    ("completion", "reference"),
    (
        ("Final Answer:\n(\n\\boxed{2},\n\\boxed{3}\n)", "(2,3)"),
        (
            "Final Answer:\n\\left\\langle\n\\boxed{2},\n"
            "\\boxed{3}\n\\right\\rangle",
            r"\langle 2,3\rangle",
        ),
    ),
)
def test_multiline_compound_is_passed_intact_to_math_verify(
    monkeypatch,
    completion: str,
    reference: str,
) -> None:
    parsed_predictions: list[str] = []

    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [reference]
        parsed_predictions.append(text)
        return [reference] if text == completion else []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1] == pred[-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference=reference,
    )

    assert result.final_passed
    assert parsed_predictions == [completion]


@pytest.mark.parametrize(
    "boundary",
    (
        "\nExplanation: a later check writes \\boxed{999}",
        "\n### Explanation\na later check writes \\boxed{999}",
        "\n\nA later check writes \\boxed{999}",
    ),
)
def test_explicit_multiline_answer_block_stops_at_semantic_boundary(
    boundary: str,
) -> None:
    answer = "Final Answer:\n(\n\\boxed{2},\n\\boxed{3}\n)"
    evidence = fr._scan_answer_evidence(answer + boundary)

    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == answer
    assert "999" not in evidence.candidate.scoring_text


def test_empty_final_heading_blocks_explanation_box_fallback_and_judge(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = (
        "Final Answer: 54.\n"
        "### Final Answer\n"
        "### Explanation\n"
        "For example, \\boxed{999}"
    )
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None
    for reference in ("54", "999"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "stop_token"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible

    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"999"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


def test_distinct_naked_multiline_final_boxes_invalidate_without_judge(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = "Final Answer:\n\\boxed{54}\n\\boxed{55}"
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None
    for reference in ("54", "55"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "stop_token"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible

    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"55"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


def test_repeated_identical_naked_multiline_boxes_are_not_a_conflict() -> None:
    completion = "Final Answer:\n\\boxed{54}\n\\boxed{54}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion


def test_same_line_independent_boxes_are_authoritatively_invalidated(
    monkeypatch,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = "Final Answer:\n\\boxed{2} and independently \\boxed{3}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None
    for reference in ("2", "3"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "stop_token"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\n\\boxed{2}\nindependently \\boxed{3}",
        "Final Answer:\n\\boxed{2}\n    and separately \\boxed{3}",
        "Final Answer:\n\\boxed{2}\n- alternatively, \\boxed{3}",
        "Final Answer:\n\\boxed{2}\n> or \\boxed{3}",
        "Final Answer:\n\\boxed{2}\nand separately\n    \\boxed{3}",
        "Final Answer:\n\\boxed{2}\n1) independently \\boxed{3}",
        "Final Answer:\n\\boxed{2}\n- [ ] independently \\(\\boxed{3}\\)",
        "Final Answer:\n\\boxed{2}\nand, independently, \\boxed{3}",
        (
            "Final Answer:\n\\boxed{2}\n"
            "> 1) - [x] **independently** \\(\\boxed{3}\\)"
        ),
    ),
)
def test_relation_led_multiline_independent_boxes_invalidate_answer_block(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize("reference", ("2", "3"))
def test_relation_led_multiline_conflict_fails_e2e_without_judge(
    monkeypatch,
    tmp_path,
    reference: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = "Final Answer:\n\\boxed{2}\nindependently \\boxed{3}"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference=reference,
    )
    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible

    dataset = tmp_path / f"math-{reference}.jsonl"
    dataset.write_text(
        json.dumps({"question": "q", "answer": reference}) + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == ""
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\n\\boxed{2}\nalternatively \\boxed{2}",
        "Final Answer:\nx=\\boxed{2},\ny=\\boxed{3}",
    ),
)
def test_multiline_repeated_identity_and_named_components_are_not_conflicts(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\nx=\\boxed{2} and y=\\boxed{3}",
        "Final Answer:\nx=\\boxed{2}\ny=\\boxed{3}",
        "Final Answer:\nx=\\boxed{2}\nand y=\\boxed{3}",
        "Final Answer:\n- [x] x=\\boxed{2}\n- [x] and y=\\boxed{3}",
    ),
)
def test_distinct_named_assignment_components_form_one_answer(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "payload",
    (
        "x=\\boxed{2}\ny=",
        "a=\\boxed{2}\nand b:",
        "1) x=\\boxed{2}\n2) y=",
        "v_1=\\boxed{2}\n> - [ ] and v_2:",
        "x=\\boxed{2} and y=",
        "(x=\\boxed{2}, y=)",
        "[v_1=\\boxed{2}, v_2=]",
        "(\nx=\\boxed{2},\ny=\n)",
        "\\[\nx=\\boxed{2}\ny=\n\\]",
        "width=\\boxed{2}\nheight=",
        "component 1: \\boxed{2}\ncomponent 2:",
        "1: \\boxed{2}\n2:",
        "1) \\boxed{2}\n2)",
        "1) \\boxed{2}; 2)",
    ),
)
def test_incomplete_named_component_tail_is_one_authoritative_barrier(
    payload: str,
) -> None:
    completion = f"Final Answer:\n{payload}"
    normalized = fr._normalize_answer_block_payload(payload)
    cue = next(fr._EXPLICIT_ANSWER_CUE_RE.finditer(completion))
    end, _uses_block_layout = fr._explicit_answer_cue_span(
        completion,
        cue_start=cue.start(),
        cue_end=cue.end(),
    )

    assert end == len(completion)
    assert fr._answer_block_has_incomplete_named_component_tail(normalized)
    assert not fr._answer_candidate_is_complete(normalized)
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\nx=\\boxed{2}\ny=",
        "Final Answer:\nx=\\boxed{2}\nand y=",
        "Final Answer: x=\\boxed{2} and y=",
        "Final Answer:\n1) x=\\boxed{2}\n2) y=",
        "Final Answer:\nv_1=\\boxed{2}\nv_2:",
    ),
)
@pytest.mark.parametrize("reference", ("2", "3"))
@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_incomplete_named_component_never_leaks_an_internal_box(
    monkeypatch,
    completion: str,
    reference: str,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": stop_reason},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference=reference,
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible


@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_incomplete_named_component_fails_e2e_reference_two_without_judge(
    monkeypatch,
    tmp_path,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / f"named-component-{stop_reason}.jsonl"
    dataset.write_text('{"question":"q","answer":"2"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final Answer:\nx=\\boxed{2}\nand y=",
                "stop_reason1": stop_reason,
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == ""
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\nx: \\boxed{2}\ny: \\boxed{3}",
        (
            "Final Answer:\ncomponent 1: \\boxed{2}\n"
            "component 2: \\boxed{3}"
        ),
        "Final Answer:\n1: \\boxed{2}\n2: \\boxed{3}",
        "Final Answer:\nv_1=\\boxed{2}\nand v_2=\\boxed{3}",
        "Final Answer:\nwidth: \\boxed{2}\nheight: \\boxed{3}",
    ),
)
def test_complete_component_schemas_remain_valid_compound_answers(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert not evidence.candidate.conflicting


def test_plain_prose_assignment_does_not_extend_named_answer_block() -> None:
    completion = "Final Answer:\nx=\\boxed{2}\nwhy="
    cue = next(fr._EXPLICIT_ANSWER_CUE_RE.finditer(completion))
    end, uses_block_layout = fr._explicit_answer_cue_span(
        completion,
        cue_start=cue.start(),
        cue_end=cue.end(),
    )

    assert uses_block_layout
    assert completion[cue.end() : end].strip() == "x=\\boxed{2}"
    assert not fr._answer_block_has_incomplete_named_component_tail(
        fr._normalize_answer_block_payload("x=\\boxed{2}\nwhy=")
    )
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.content == "2"


_MARKED_OR_LATEX_INCOMPLETE_COMPONENT_ANSWERS = (
    r"Final Answer: x=\boxed{2} and **y=**",
    r"Final Answer: x=\boxed{2} and _y=_",
    r"Final Answer: x=\boxed{2}\;\text{and}\;y=",
    r"Final Answer: a=\boxed{2},\qquad b=",
    r"Final Answer: \langle x=\boxed{2},\;y=\rangle",
    (
        "Final Answer:\n"
        "\\begin{aligned}\n"
        "x &= \\boxed{2} \\\\\n"
        "y &=\n"
        "\\end{aligned}"
    ),
)


@pytest.mark.parametrize(
    "completion",
    _MARKED_OR_LATEX_INCOMPLETE_COMPONENT_ANSWERS,
)
def test_markdown_and_latex_component_syntax_keeps_incomplete_tail_authoritative(
    completion: str,
) -> None:
    cue = next(fr._EXPLICIT_ANSWER_CUE_RE.finditer(completion))
    end, _uses_block_layout = fr._explicit_answer_cue_span(
        completion,
        cue_start=cue.start(),
        cue_end=cue.end(),
    )
    normalized = fr._normalize_answer_block_payload(
        completion[cue.end() : end].strip()
    )

    assert end == len(completion)
    assert fr._answer_block_has_incomplete_named_component_tail(normalized)
    assert not fr._answer_candidate_is_complete(normalized)
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "completion",
    _MARKED_OR_LATEX_INCOMPLETE_COMPONENT_ANSWERS,
)
@pytest.mark.parametrize("reference", ("2", "3"))
@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_markdown_and_latex_incomplete_components_never_leak_box_or_judge(
    monkeypatch,
    completion: str,
    reference: str,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": stop_reason},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference=reference,
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    _MARKED_OR_LATEX_INCOMPLETE_COMPONENT_ANSWERS,
)
@pytest.mark.parametrize("reference", ("2", "3"))
@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_markdown_and_latex_incomplete_components_fail_e2e_without_judge(
    monkeypatch,
    tmp_path,
    completion: str,
    reference: str,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "marked-latex-component.jsonl"
    dataset.write_text(
        json.dumps({"question": "q", "answer": reference}) + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": stop_reason,
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == ""
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    ("completion", "normalized_payload"),
    (
        (
            r"Final Answer: **x**=\boxed{2} and **y**=\boxed{3}",
            r"x=\boxed{2} and y=\boxed{3}",
        ),
        (
            r"Final Answer: _x_=\boxed{2} and _y_=\boxed{3}",
            r"x=\boxed{2} and y=\boxed{3}",
        ),
        (
            r"Final Answer: x=\boxed{2}\;\text{and}\;y=\boxed{3}",
            r"x=\boxed{2} and y=\boxed{3}",
        ),
        (
            r"Final Answer: \langle x=\boxed{2},\;y=\boxed{3}\rangle",
            r"\langle x=\boxed{2},y=\boxed{3}\rangle",
        ),
        (
            (
                "Final Answer:\n"
                "\\begin{aligned}\n"
                "**x** &= \\boxed{2} \\\\\n"
                "**y** &= \\boxed{3}\n"
                "\\end{aligned}"
            ),
            (
                "\\begin{aligned}\n"
                "x = \\boxed{2} \\\\\n"
                "y = \\boxed{3}\n"
                "\\end{aligned}"
            ),
        ),
    ),
)
def test_complete_markdown_and_latex_named_components_remain_valid(
    completion: str,
    normalized_payload: str,
) -> None:
    cue = next(fr._EXPLICIT_ANSWER_CUE_RE.finditer(completion))
    end, _uses_block_layout = fr._explicit_answer_cue_span(
        completion,
        cue_start=cue.start(),
        cue_end=cue.end(),
    )

    assert fr._normalize_answer_block_payload(
        completion[cue.end() : end].strip()
    ) == normalized_payload
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "prose_tail",
    (
        "**why=**",
        "reasoning=",
        "verification:",
        "commentary=",
    ),
)
def test_marked_or_named_prose_tail_is_not_a_component_continuation(
    prose_tail: str,
) -> None:
    completion = f"Final Answer:\nx=\\boxed{{2}}\n{prose_tail}"
    cue = next(fr._EXPLICIT_ANSWER_CUE_RE.finditer(completion))
    end, uses_block_layout = fr._explicit_answer_cue_span(
        completion,
        cue_start=cue.start(),
        cue_end=cue.end(),
    )

    assert uses_block_layout
    assert completion[cue.end() : end].strip() == "x=\\boxed{2}"
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.content == "2"


@pytest.mark.parametrize(
    "relation",
    (
        "or",
        "independently",
        "separately",
        "alternatively",
        "and separately",
        "and, independently,",
    ),
)
def test_named_assignments_with_independent_relation_fail_closed(
    relation: str,
) -> None:
    completion = (
        "Final Answer:\n"
        f"x=\\boxed{{2}}\n{relation} y=\\boxed{{3}}"
    )
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "section_boundary",
    (
        "**Explanation**",
        "Explanation:",
        "---",
        "<!-- answer intentionally omitted -->",
    ),
)
def test_unresolved_answer_block_is_a_barrier_across_presentation_styles(
    monkeypatch,
    section_boundary: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = (
        "Final Answer: 54.\n"
        "Final Answer:\n"
        f"{section_boundary}\n"
        "For example, \\boxed{999}"
    )

    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None
    for reference in ("54", "999"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "stop_token"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\n(\n\\boxed{2},\n\\boxed{3}",
        "Final Answer: (\\boxed{2}, \\boxed{3}",
    ),
)
def test_incomplete_multi_box_answer_cannot_leak_component_boxes(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None
    for reference in ("2", "3"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "max_length"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible


def test_conflicting_strategy_a_does_not_contaminate_math_b_or_c(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"2"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "p",
                "strategy_a_completion": (
                    "Final Answer:\n\\boxed{2}\n"
                    "1) independently \\(\\boxed{3}\\)"
                ),
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "reasoning prompt",
                "completion1": "reasoning",
                "stop_reason1": "stop_token",
                "prompt2": "\nFinal answer: ",
                "completion2": "2.",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_B] == [(0, 0, True)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    ("raw_payload", "normalized_payload"),
    (
        ("1) \\boxed{2}", "\\boxed{2}"),
        (
            "1) \\boxed{2}\n2) \\boxed{2}",
            "\\boxed{2}\n\\boxed{2}",
        ),
        (
            "> 1) - [x] \\boxed{2}\n> 2) - [ ] \\boxed{2}",
            "\\boxed{2}\n\\boxed{2}",
        ),
    ),
)
def test_explicit_block_uses_one_normalized_structural_representation(
    raw_payload: str,
    normalized_payload: str,
) -> None:
    completion = f"Final Answer:\n{raw_payload}"

    assert fr._normalize_answer_block_payload(raw_payload) == normalized_payload
    assert fr._answer_candidate_is_complete(normalized_payload)
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None


@pytest.mark.parametrize(
    "wrapped_answer",
    (
        "\\[\n\\boxed{2}\n\\]",
        "$$\n\\boxed{2}\n$$",
        "\\(\n\\boxed{2}\n\\)",
        "```latex\n\\boxed{2}\n```",
        "```math\n> 1) \\boxed{2}\n```",
        "> ```math\n> 1) - [x] \\boxed{2}\n> ```",
    ),
)
def test_display_wrapped_answer_preserves_raw_block_but_not_wrapper_identity(
    monkeypatch,
    wrapped_answer: str,
) -> None:
    completion = f"Final Answer:\n{wrapped_answer}"
    parsed_predictions: list[str] = []

    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return ["2"]
        parsed_predictions.append(text)
        return ["2"] if text == completion else []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        return bool(gold and pred and gold[-1] == pred[-1])

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.content == "2"
    assert evidence.candidate.scoring_text == completion
    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="2",
    )
    assert result.final_passed
    assert parsed_predictions == [completion]


@pytest.mark.parametrize(
    "completion",
    (
        (
            "Final Answer:\n(\n\\boxed{2} independently computed,\n"
            "\\boxed{3}\n)"
        ),
        "Final Answer:\n(\\boxed{2} or \\boxed{3})",
        "Final Answer:\nS=\\{\\boxed{2} independently, \\boxed{3}\\}",
        "Final Answer:\n[\\boxed{2} alternatively, \\boxed{3})",
        (
            "Final Answer:\n\\begin{pmatrix}\\boxed{2} & "
            "\\text{independently computed} & \\boxed{3}"
            "\\end{pmatrix}"
        ),
        (
            "Final Answer:\nx=\\boxed{2} (independently computed), "
            "y=\\boxed{3}"
        ),
        (
            "Final Answer:\nx=\\boxed{2} "
            "\\text{independently computed}, y=\\boxed{3}"
        ),
    ),
)
def test_compound_enclosure_precedes_internal_relation_words(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.scoring_text == completion
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "completion",
    (
        "Final Answer:\n\\boxed{2}\nindependently \\boxed{3}",
        "Final Answer:\n\\[\n\\boxed{2} independently \\boxed{3}\n\\]",
        "Final Answer:\nx=\\boxed{2}\nindependently y=\\boxed{3}",
    ),
)
def test_only_top_level_relation_between_identities_is_invalidated(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "opening_wrapper",
    ("\\[", "$$", "\\(", "```latex"),
)
def test_unclosed_display_block_invalidates_components_without_judge(
    monkeypatch,
    tmp_path,
    opening_wrapper: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = (
        f"Final Answer:\n{opening_wrapper}\n"
        "1) \\boxed{2}\n2) independently \\boxed{3}"
    )
    for reference in ("2", "3"):
        result = fr.score_free_response_strategy(
            STRATEGY_A,
            {"completion1": completion, "stop_reason1": "max_length"},
            sample_index=0,
            repeat_index=0,
            question="q",
            reference=reference,
        )
        assert not result.final_passed
        assert result.fail_reason == "authoritative_answer_invalidated"
        assert not result.judge_eligible

    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"2"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")
    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": "max_length",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )
    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        r"Final Answer: \boxed{C} or \boxed{D}",
        "Final Answer: option C or option D",
    ),
)
def test_authoritative_mcq_with_different_labels_is_invalidated_without_judge(
    monkeypatch,
    completion: str,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert not result.final_passed
    assert result.display_answer == ""
    assert result.fail_reason == "authoritative_multiple_choice_answer_invalidated"
    assert not result.judge_eligible


def test_repeated_identical_mcq_labels_remain_one_authoritative_answer(
    monkeypatch,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": r"Final Answer: \boxed{C}, equivalently \boxed{C}",
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert result.final_passed
    assert result.display_answer == "C"
    assert not result.judge_eligible


def test_conflicting_mcq_stage_a_does_not_contaminate_stage_b_or_c(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "mcq.jsonl"
    dataset.write_text(
        '{"question":"Choose: (A) alpha (B) beta (C) gamma (D) delta",'
        '"answer":"C"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "strategy_a_prompt": "p",
                "strategy_a_completion": r"Final Answer: \boxed{C} or \boxed{D}",
                "strategy_a_stop_reason": "stop_token",
                "prompt1": "reasoning prompt",
                "completion1": "reasoning",
                "stop_reason1": "stop_token",
                "prompt2": "\nFinal answer: ",
                "completion2": "C.",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.rows_by_group[STRATEGY_B] == [(0, 0, True)]
    assert evaluation.rows_by_group[STRATEGY_C] == [(0, 0, True)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["answer"] == ""
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_multiple_choice_answer_invalidated"
    )
    assert judge.client.prompts == []


def test_latest_committed_mcq_event_wins_by_text_position(monkeypatch) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": "Final answer: B. Correction: D. Final answer: C.",
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert result.final_passed
    assert result.display_answer == "C"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "replacement",
    (
        "On second thought,55.",
        "Actually,the correct value is55.",
        "However,55 is correct.",
        "Rather,55.",
        "Instead it is55.",
        "In fact: 55.",
        "On reflection, the correct answer is 55.",
        "On reconsideration, it should be 55.",
    ),
)
def test_replacement_synonyms_expose_only_latest_payload_to_parser(
    monkeypatch,
    replacement: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    completion = f"Final answer: 53. {replacement}"
    evidence = fr._scan_answer_evidence(completion)
    assert evidence.candidate is not None
    assert "55" in evidence.candidate.scoring_text
    assert "53" not in evidence.candidate.scoring_text

    payload = {"completion1": completion, "stop_reason1": "stop_token"}
    old_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="53",
    )
    new_result = fr.score_free_response_strategy(
        STRATEGY_A,
        payload,
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert not old_result.final_passed
    assert new_result.final_passed


def test_incomplete_replacement_boundary_cannot_revive_old_value(monkeypatch) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": (
                "Final answer: 54. Actually, the correct value is (1/2"
            ),
            "stop_reason1": "max_length",
        },
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        "Final answer: 54. Do not ignore it.",
        "Final answer: 54. I will not retract that answer.",
        "Final answer: 54. There is no need to withdraw it.",
        "Final answer: 54. I refuse to disregard it.",
        "Final answer: 54. Never ignore it.",
        "Final answer: 54. It is false that that answer was wrong.",
        "Final answer: 54. It is not true that that answer was wrong.",
    ),
)
def test_negated_retraction_language_does_not_create_authority_boundary(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert result.final_passed
    assert fr._answer_authority_boundaries(completion) == []


def test_real_retraction_changes_legacy_pass_to_fail_without_judge(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"54"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": "Final answer: 54. I retract that answer.",
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    (
        "According to my calculation, the answer is 54.",
        "According to my computation, the answer is 54.",
        "According to the computation, the answer is 54.",
        'A source says "The answer is 54," and I agree.',
        "The textbook claims the answer is 54. I confirm it.",
        "The textbook states the answer is 54, which agrees with my result.",
        "I agree with the textbook; the textbook states the answer is 54.",
        "I confirm the source. The source says the answer is 54.",
        "Under that assumption, the answer is 54. The assumption holds.",
        "According to Bob, the answer is 54, and my calculation agrees.",
    ),
)
def test_explicitly_adopted_source_or_assumption_is_valid_answer_evidence(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert result.final_passed


@pytest.mark.parametrize(
    "completion",
    (
        "According to Bob, the answer is 54, but I disagree.",
        "A source says the answer is 54. I reject it.",
        "The textbook claims the answer is 54. I reject that claim.",
        "Under that assumption, the answer is 54. The assumption fails.",
        "According to my calculation, the answer is 54, but I reject it.",
        "According to the computation, the answer is 54, but the computation fails.",
        "The textbook states the answer is 54. I accept neither it nor its answer.",
        "The textbook states the answer is 54. I agree with neither source.",
        "The textbook states the answer is 54. I confirm only that it is false.",
        "The textbook states the answer is 54; accept neither it nor its answer.",
        "The textbook states the answer is 54; agree with neither source.",
        "The textbook states the answer is 54; confirm only that it is false.",
        (
            "I agree with the textbook; the textbook states the answer is 54. "
            "I agree with neither source."
        ),
    ),
)
def test_rejected_source_or_failed_assumption_is_not_answer_evidence(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert not result.final_passed
    assert not result.judge_eligible
    assert result.fail_reason in {
        "authoritative_answer_invalidated",
        "contextual_answer_only",
    }


@pytest.mark.parametrize(
    "completion",
    (
        "A source says the answer is 53, and I agree. Actually,55.",
        "A source says the answer is 53 and I reject it; actually,55.",
        "Under that assumption, the answer is 53. The assumption fails. Rather,55.",
    ),
)
def test_latest_replacement_after_source_stance_remains_authoritative(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="55",
    )

    assert result.final_passed


@pytest.mark.parametrize(
    "completion",
    (
        '{"answer":"54","final_answer":"55"}',
        '{"answer":"Final answer: 54","response":{"answer":"55"}}',
        '{"answer":54,"final_answer":55}',
        '{"answer":["54","55"]}',
        '["54","55"]',
        (
            '{"choices":[{"message":{"content":"54"}},'
            '{"message":{"content":"55"}}]}'
        ),
    ),
)
def test_conflicting_serialized_math_answer_fields_fail_closed(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        '{"answer":"54","final_answer":"Final answer: 54"}',
        '{"answer":["54","Final answer: 54"]}',
        (
            '{"choices":[{"message":{"content":"54"}},'
            '{"message":{"content":"Final answer: 54"}}]}'
        ),
    ),
)
def test_consistent_serialized_math_answer_fields_are_unwrapped(
    monkeypatch,
    completion: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": completion,
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question="q",
        reference="54",
    )

    assert result.final_passed


@pytest.mark.parametrize(
    ("completion", "expected_passed", "expected_answer"),
    (
        ('{"answer":"C","final_answer":"D"}', False, ""),
        ('{"answer":"C","final_answer":"C"}', True, "C"),
        (
            '{"choices":[{"message":{"content":"C"}},'
            '{"message":{"content":"D"}}]}',
            False,
            "",
        ),
        (
            '{"choices":[{"message":{"content":"C"}},'
            '{"message":{"content":"C"}}]}',
            True,
            "C",
        ),
    ),
)
def test_serialized_mcq_answer_fields_fail_closed_on_conflict(
    monkeypatch,
    completion: str,
    expected_passed: bool,
    expected_answer: str,
) -> None:
    _patch_math_verify(monkeypatch)
    question = "Choose: (A) alpha (B) beta (C) gamma (D) delta"

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "stop_token"},
        sample_index=0,
        repeat_index=0,
        question=question,
        reference="C",
    )

    assert result.final_passed is expected_passed
    assert result.display_answer == expected_answer
    assert not result.judge_eligible


def test_conflicting_serialized_array_is_never_sent_to_judge(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "math.jsonl"
    dataset.write_text('{"question":"q","answer":"54"}\n', encoding="utf-8")
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": (
                    '{"choices":[{"message":{"content":"54"}},'
                    '{"message":{"content":"55"}}]}'
                ),
                "stop_reason1": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


_AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS = (
    r"Final Answer: x=\boxed{2} | y=",
    "Final Answer:\nx=\\boxed{2} |\n| y= |",
    r"Final Answer: x=\boxed{2}<br>y=",
    r"Final Answer: 1. a=\boxed{2}; **2. b=**",
    r"Final Answer: [1) x=\boxed{2}; 2) y=]",
    r"Final Answer: \begin{aligned}x &= \boxed{2}, \\ y &=\end{aligned}",
    "Final Answer: ⟨x=\\boxed{2}, y=⟩",
    r"Final Answer: x=\boxed{2}\;\textbf{and}\;y=",
    r"Final Answer: x=\boxed{2}\;\mbox{and}\;y=",
    r"Final Answer: x:=\boxed{2}; y:=",
    r"Final Answer: [x=](#x)\boxed{2} and [y=](#y)",
    r"Final Answer: \mathbf{x}=\boxed{2}, \mathbf{y}=",
    r"Final Answer: \vec{x}=\boxed{2}, \vec{y}=",
    r"Final Answer: i) x=\boxed{2}; ii) y=",
)


_AUTHORITATIVE_ASSIGNMENT_COMPLETE_FORMATS = (
    r"Final Answer: x=\boxed{2} | y=\boxed{3}",
    r"Final Answer: \begin{aligned}x &= \boxed{2}, \\ y &=\boxed{3}\end{aligned}",
    "Final Answer: ⟨x=\\boxed{2}, y=\\boxed{3}⟩",
    r"Final Answer: x=\boxed{2}\;\textbf{and}\;y=\boxed{3}",
    r"Final Answer: x=\boxed{2}\;\mbox{and}\;y=\boxed{3}",
    r"Final Answer: x:=\boxed{2}; y:=\boxed{3}",
    r"Final Answer: [x=](#x)\boxed{2} and [y=](#y)\boxed{3}",
    r"Final Answer: \mathbf{x}=\boxed{2}, \mathbf{y}=\boxed{3}",
    r"Final Answer: \vec{x}=\boxed{2}, \vec{y}=\boxed{3}",
    r"Final Answer: i) x=\boxed{2}; ii) y=\boxed{3}",
    "Final Answer:\n```math\n1) \\boxed{2}\n2) \\boxed{3}\n```",
)


@pytest.mark.parametrize(
    "completion",
    _AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS,
)
@pytest.mark.parametrize("reference", ("2", "3"))
@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_authoritative_assignment_lexer_fails_closed_for_empty_rhs(
    monkeypatch,
    completion: str,
    reference: str,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)

    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": stop_reason},
        sample_index=0,
        repeat_index=0,
        question="Return every named field.",
        reference=reference,
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible


@pytest.mark.parametrize(
    "completion",
    (
        _AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS[0],
        _AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS[5],
        _AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS[6],
        _AUTHORITATIVE_ASSIGNMENT_INCOMPLETE_FORMATS[11],
    ),
)
@pytest.mark.parametrize("reference", ("2", "3"))
@pytest.mark.parametrize("stop_reason", ("stop_token", "max_length"))
def test_authoritative_assignment_empty_rhs_never_reaches_judge(
    monkeypatch,
    tmp_path,
    completion: str,
    reference: str,
    stop_reason: str,
) -> None:
    _patch_any_scalar_math_verify(monkeypatch)
    dataset = tmp_path / "assignment-lexer.jsonl"
    dataset.write_text(
        json.dumps({"question": "Return every named field.", "answer": reference})
        + "\n",
        encoding="utf-8",
    )
    judge = LLMJudge(LLMJudgeConfig(api_key="k", model="m", max_workers=1))
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "completion1": completion,
                "stop_reason1": stop_reason,
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.rows_by_group[STRATEGY_A] == [(0, 0, False)]
    assert evaluation.payloads_by_group[STRATEGY_A][0]["fail_reason"] == (
        "authoritative_answer_invalidated"
    )
    assert judge.client.prompts == []


@pytest.mark.parametrize(
    "completion",
    _AUTHORITATIVE_ASSIGNMENT_COMPLETE_FORMATS,
)
def test_authoritative_assignment_complete_formats_remain_one_answer(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "prose_label",
    ("why", "reasoning", "notes", "verification"),
)
def test_authoritative_assignment_lexer_excludes_prose_labels(
    prose_label: str,
) -> None:
    completion = f"Final Answer:\nx=\\boxed{{2}}\n{prose_label}="
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert evidence.candidate.content == "2"


_EMPTY_LATEX_PRESENTATION_WRAPPERS = (
    "text",
    "textrm",
    "textsf",
    "texttt",
    "textbf",
    "textit",
    "mathrm",
    "mathbf",
    "mathsf",
    "mathtt",
    "mathit",
    "operatorname",
    "mbox",
    "hbox",
    "boldsymbol",
    "underline",
    "overline",
    "phantom",
    "hphantom",
    "vphantom",
)


@pytest.mark.parametrize("wrapper", _EMPTY_LATEX_PRESENTATION_WRAPPERS)
@pytest.mark.parametrize("boxed", (False, True))
def test_empty_latex_presentation_rhs_is_semantically_empty(
    monkeypatch,
    wrapper: str,
    boxed: bool,
) -> None:
    rhs = f"\\{wrapper}{{}}"
    if boxed:
        rhs = f"\\boxed{{{rhs}}}"
    completion = f"Final Answer: x=\\boxed{{2}}; y={rhs}"
    verifier_calls = 0

    def forbidden_verifier():
        nonlocal verifier_calls
        verifier_calls += 1
        raise AssertionError("invalidated assignment must bypass math verifier")

    monkeypatch.setattr(fr, "_load_math_verify", forbidden_verifier)
    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "max_length"},
        sample_index=0,
        repeat_index=0,
        question="Return every field.",
        reference="2",
    )

    assert not result.final_passed
    assert result.fail_reason == "authoritative_answer_invalidated"
    assert not result.judge_eligible
    assert verifier_calls == 0


@pytest.mark.parametrize(
    "empty_rhs",
    (
        r"\text{\mathbf{}}",
        r"\boxed{\text{\mathrm{}}}",
        r"\phantom{3}",
        r"<!-- intentionally empty -->",
        r"<em></em>",
        r"<strong><!-- empty --></strong>",
    ),
)
def test_nested_or_html_presentation_only_rhs_is_empty(empty_rhs: str) -> None:
    completion = f"Final Answer: x=\\boxed{{2}}; y={empty_rhs}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "complete_rhs",
    (
        r"\text{three}",
        r"\mathrm{x}",
        r"\mathbf{y}",
        r"\operatorname{root}",
        r"\mbox{nonzero}",
        r"\alpha",
        r"\sqrt{3}",
        r"<em>three</em>",
    ),
)
def test_nonempty_text_or_symbol_assignment_rhs_remains_complete(
    complete_rhs: str,
) -> None:
    completion = f"Final Answer: x=\\boxed{{2}}; y={complete_rhs}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert all(assignment.rhs_complete for assignment in assignments)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None


@pytest.mark.parametrize(
    ("completion", "expected_state"),
    (
        (
            r"Final Answer: <em>x=</em>\boxed{2}<br><em>y=</em>",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
        (
            r"Final Answer: <em>x=</em>\boxed{2}<br><em>y=</em>\boxed{3}",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
        (
            r"Final Answer: <table><tr><td>x=</td><td>\boxed{2}</td></tr>"
            r"<tr><td>y=</td><td></td></tr></table>",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
        (
            r"Final Answer: <table><tr><td>x=</td><td>\boxed{2}</td></tr>"
            r"<tr><td>y=</td><td>\boxed{3}</td></tr></table>",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
        (
            r"Final Answer: (IV) \boxed{2}; (V)",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
        (
            r"Final Answer: (IV) \boxed{2}; (V) \boxed{3}",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
        (
            "Final Answer: ١) \\boxed{2}; ٢)",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
        (
            "Final Answer: ١) \\boxed{2}; ٢) \\boxed{3}",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
    ),
)
def test_html_and_unicode_ordered_assignment_blocks_share_one_schema(
    completion: str,
    expected_state: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == expected_state
    if expected_state == fr._ANSWER_EVIDENCE_CANDIDATE:
        assert evidence.candidate is not None
        assert not evidence.candidate.conflicting
    else:
        assert evidence.candidate is None


@pytest.mark.parametrize("last_value", ("", r"\boxed{3}"))
def test_markdown_answer_table_is_collected_as_one_authoritative_block(
    last_value: str,
) -> None:
    completion = (
        "Final Answer:\n| field | value |\n|:--|--:|\n"
        "| **x=** | \\boxed{2} |\n"
        f"| [y=](#field-y) | {last_value} |"
    )
    evidence = fr._scan_answer_evidence(completion)

    expected = (
        fr._ANSWER_EVIDENCE_CANDIDATE
        if last_value
        else fr._ANSWER_EVIDENCE_INVALIDATED
    )
    assert evidence.state == expected


_ROUND15_ASSIGNMENT_SCHEMAS = (
    r"x=\boxed{2}; y={rhs}",
    r"v_1=\boxed{2}; v_2={rhs}",
    r"width=\boxed{2}; height={rhs}",
    r"part 1=\boxed{2}; part 2={rhs}",
    r"component_1=\boxed{2}; component_2={rhs}",
)


@pytest.mark.parametrize("schema", _ROUND15_ASSIGNMENT_SCHEMAS)
@pytest.mark.parametrize(
    "empty_rhs",
    ("<u></u>", r"\ensuremath{}", r"\color{red}{}"),
)
def test_common_presentation_only_rhs_is_empty_in_every_assignment_schema(
    schema: str,
    empty_rhs: str,
) -> None:
    completion = "Final Answer: " + schema.replace("{rhs}", empty_rhs)
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert assignments[0].rhs_complete
    assert not assignments[1].rhs_complete
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "complete_rhs",
    ("<u>three</u>", r"\ensuremath{\gamma}", r"\color{red}{3}"),
)
def test_common_nonempty_presentation_rhs_retains_semantic_content(
    complete_rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={complete_rhs}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert all(assignment.rhs_complete for assignment in assignments)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "complete_rhs",
    (
        r"\boxed{\boxed{3}}",
        r"\boxed{\text{\boxed{\mathrm{three}}}}",
        r"\boxed{\boxed{\boxed{\alpha}}}",
        r"\boxed{\frac{\boxed{1}}{\boxed{2}}}",
        r"\boxed{\boxed{1},\boxed{2}}",
    ),
)
def test_recursive_box_tree_accepts_complete_nested_values_and_siblings(
    complete_rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={complete_rhs}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert all(assignment.rhs_complete for assignment in assignments)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "incomplete_rhs",
    (
        r"\boxed{}",
        r"\boxed{\boxed{}}",
        r"\boxed{\boxed{\text{}}}",
        r"\boxed{\boxed{1},\boxed{}}",
        r"\boxed{\boxed{answer}}",
        r"\boxed{\boxed{\quad}}",
        r"\boxed{\boxed{3}",
        r"\boxed{\boxed{3",
        r"\boxed{\boxed{3}}}",
    ),
)
def test_recursive_box_tree_rejects_empty_placeholder_or_unbalanced_descendants(
    incomplete_rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={incomplete_rhs}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert assignments[0].rhs_complete
    # Assignment completeness includes the enclosing LaTeX brace structure,
    # not merely the box subtree.  An extra closer is therefore incomplete at
    # the lexer boundary as well as invalid at the committed-block boundary.
    assert not assignments[1].rhs_complete
    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize("one,two", (("Ⅳ", "Ⅴ"), ("ⅳ", "ⅴ")))
@pytest.mark.parametrize("complete", (False, True))
def test_unicode_roman_ordered_items_share_the_numbered_schema(
    one: str,
    two: str,
    complete: bool,
) -> None:
    tail = r" \boxed{3}" if complete else ""
    completion = f"Final Answer: {one}) \\boxed{{2}}; {two}){tail}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert all(assignment.schema == "numbered" for assignment in assignments)
    assert evidence.state == (
        fr._ANSWER_EVIDENCE_CANDIDATE
        if complete
        else fr._ANSWER_EVIDENCE_INVALIDATED
    )
    if complete:
        assert evidence.candidate is not None
        assert not evidence.candidate.conflicting
    else:
        assert evidence.candidate is None


@pytest.mark.parametrize(
    ("completion", "expected_state"),
    (
        (
            r"Final Answer: x=\boxed{2}<br class='row' data-index='2'/>"
            r"y=\boxed{3}",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
        (
            r"Final Answer: x=\boxed{2}<br class='row' data-index='2'/>y=",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
        (
            r"Final Answer: x=\boxed{2}; y=\boxed{<div class='answer-block' "
            r"data-depth='4'><p aria-label='answer'><mark><small>3</small>"
            r"</mark></p></div>}",
            fr._ANSWER_EVIDENCE_CANDIDATE,
        ),
        (
            r"Final Answer: x=\boxed{2}; y=\boxed{<p class='answer'>"
            "&nbsp;&#160;&#xA0;&ensp;&emsp;&thinsp;\u200b</p>}",
            fr._ANSWER_EVIDENCE_INVALIDATED,
        ),
    ),
)
def test_html_tokens_isolate_attributes_and_preserve_semantic_content(
    completion: str,
    expected_state: str,
) -> None:
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert [assignment.label for assignment in assignments] == ["x", "y"]
    assert evidence.state == expected_state


@pytest.mark.parametrize(
    ("rhs", "expected_complete"),
    (
        (r"\fbox{}", False),
        (r"\fbox{\text{\mathbf{&ensp;" + "\u200b" + "}}}", False),
        (r"\fbox{3}", True),
        (r"\fbox{\text{\mathbf{three}}}", True),
        (r"\text{\mathbf{3}", False),
    ),
)
def test_recursive_presentation_wrappers_are_semantically_complete_only_with_a_leaf(
    rhs: str,
    expected_complete: bool,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={rhs}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert assignments[1].rhs_complete is expected_complete
    assert evidence.state == (
        fr._ANSWER_EVIDENCE_CANDIDATE
        if expected_complete
        else fr._ANSWER_EVIDENCE_INVALIDATED
    )


@pytest.mark.parametrize(
    ("relation", "expected_state"),
    (
        (r"\text{ or }", fr._ANSWER_EVIDENCE_INVALIDATED),
        (r"\text{ alternatively }", fr._ANSWER_EVIDENCE_INVALIDATED),
        (",", fr._ANSWER_EVIDENCE_CANDIDATE),
    ),
)
def test_nested_box_tree_distinguishes_alternatives_from_compounds(
    relation: str,
    expected_state: str,
) -> None:
    completion = (
        r"Final Answer: x=\boxed{2}; y=\boxed{\boxed{3}"
        + relation
        + r"\boxed{4}}"
    )
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == expected_state


@pytest.mark.parametrize(
    ("one", "two"),
    (
        ("1.", "2."),
        (chr(0x2163) + ".", chr(0x2164) + "."),
        (chr(0xFF11) + chr(0xFF09), chr(0xFF12) + chr(0xFF09)),
        (chr(0x2460), chr(0x2461)),
    ),
)
@pytest.mark.parametrize("complete", (False, True))
def test_ordered_marker_families_share_authority_and_empty_tail_semantics(
    one: str,
    two: str,
    complete: bool,
) -> None:
    tail = r" \boxed{3}" if complete else ""
    completion = f"Final Answer: {one} \\boxed{{2}}; {two}{tail}"
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )
    evidence = fr._scan_answer_evidence(completion)

    assert len(assignments) == 2
    assert all(assignment.schema == "numbered" for assignment in assignments)
    assert evidence.state == (
        fr._ANSWER_EVIDENCE_CANDIDATE
        if complete
        else fr._ANSWER_EVIDENCE_INVALIDATED
    )


def test_unboxed_latex_text_mcq_label_uses_the_structured_mcq_path(
    monkeypatch,
) -> None:
    def parse(text: str):
        return [text]

    def verify(*_args, **_kwargs):
        return False

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {
            "completion1": r"Final Answer: \text{(D) }",
            "stop_reason1": "stop_token",
        },
        sample_index=0,
        repeat_index=0,
        question="Choose one.\n(A) one\n(B) two\n(C) three\n(D) four",
        reference="D",
    )

    assert result.final_passed
    assert result.display_answer == "D"
    assert result.fail_reason == ""


@pytest.mark.parametrize(
    "rhs",
    (
        r"<span data-decoy='z=' title='or'>\boxed{3}</span>",
        r"<a href='https://invalid.example/?x=9&amp;y='>\boxed{3}</a>",
        r"<span data-box='\boxed{99}' title='x=7'>\boxed{3}</span>",
        r"<!-- \boxed{99}; z= -->\boxed{3}",
    ),
)
def test_visible_html_projection_excludes_metadata_from_answer_semantics(
    rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={rhs}"
    evidence = fr._scan_answer_evidence(completion)
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )

    assert [assignment.label for assignment in assignments] == ["x", "y"]
    assert all(assignment.rhs_complete for assignment in assignments)
    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "rhs",
    (
        r"\boxed{\frac{1}{}}",
        r"\boxed{\frac{}{2}}",
        r"\boxed{\sqrt{}}",
        r"\boxed{\sqrt{\phantom{3}}}",
    ),
)
def test_empty_mandatory_tex_arguments_invalidate_authoritative_answer(
    rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={rhs}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    "completion",
    (
        r"Final Answer: coordinate=\boxed{(2,3)}; order=\boxed{4}",
        r"Final Answer: x=\boxed{2}; y=\boxed{\operatorname{floor}(3)}",
        r"Final Answer: x=\boxed{2}; y=\boxed{p\lor q}",
    ),
)
def test_named_compounds_and_logical_operator_are_not_prose_alternatives(
    completion: str,
) -> None:
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_CANDIDATE
    assert evidence.candidate is not None
    assert not evidence.candidate.conflicting


@pytest.mark.parametrize(
    "rhs",
    (
        r"\boxed{3\text{ or }4}",
        r"\boxed{\boxed{3}\text{ or }\boxed{4}}",
        r"\boxed{3}\text{ alternatively }\boxed{4}",
    ),
)
def test_visible_explicit_alternatives_invalidate_single_or_nested_boxes(
    rhs: str,
) -> None:
    completion = rf"Final Answer: x=\boxed{{2}}; y={rhs}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == fr._ANSWER_EVIDENCE_INVALIDATED
    assert evidence.candidate is None


@pytest.mark.parametrize(
    ("one", "two"),
    (
        ("\uff081\uff09", "\uff082\uff09"),
        ("IV\uff1a", "V\uff1a"),
        ("\u2474", "\u2475"),
    ),
)
@pytest.mark.parametrize("complete", (False, True))
def test_fullwidth_roman_and_parenthesized_ordered_markers_share_authority(
    one: str,
    two: str,
    complete: bool,
) -> None:
    tail = r" \boxed{3}" if complete else ""
    completion = f"Final Answer: {one} \\boxed{{2}}; {two}{tail}"
    evidence = fr._scan_answer_evidence(completion)

    assert evidence.state == (
        fr._ANSWER_EVIDENCE_CANDIDATE
        if complete
        else fr._ANSWER_EVIDENCE_INVALIDATED
    )


def test_tuple_and_function_parentheses_do_not_create_ordered_fields() -> None:
    completion = (
        r"Final Answer: coordinate=\boxed{(2,3)}; "
        r"floor=\boxed{\operatorname{floor}(3)}"
    )
    assignments = fr._lex_authoritative_answer_assignments(
        completion.split(":", 1)[1]
    )

    assert [assignment.label for assignment in assignments] == [
        "coordinate",
        "floor",
    ]
    assert all(assignment.rhs_complete for assignment in assignments)


@pytest.mark.parametrize(
    ("completion", "expected_passed", "expected_reason"),
    (
        (
            r"Final Answer: <span data-choice='C'>(D)</span>",
            True,
            "",
        ),
        (
            r"Final Answer: <span data-choice='D'>(C)</span>",
            False,
            "multiple_choice_label_mismatch",
        ),
        (
            r"Final Answer: <span data-choice='D'>(D)</span> or (C)",
            False,
            "authoritative_multiple_choice_answer_invalidated",
        ),
        (
            r"Final Answer: <span>D</span> or <span>C</span>",
            False,
            "authoritative_multiple_choice_answer_invalidated",
        ),
    ),
)
def test_mcq_uses_visible_labels_and_fails_closed_on_conflict(
    monkeypatch,
    completion: str,
    expected_passed: bool,
    expected_reason: str,
) -> None:
    def forbidden_loader():
        raise AssertionError("structured MCQ result must precede math_verify")

    monkeypatch.setattr(fr, "_load_math_verify", forbidden_loader)
    result = fr.score_free_response_strategy(
        STRATEGY_A,
        {"completion1": completion, "stop_reason1": "max_length"},
        sample_index=0,
        repeat_index=0,
        question="Choose one.\n(A) one\n(B) two\n(C) three\n(D) four",
        reference="D",
    )

    assert result.final_passed is expected_passed
    assert result.fail_reason == expected_reason
    assert not result.judge_eligible
