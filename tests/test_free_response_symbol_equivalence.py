from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.eval.metrics import free_response as fr


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def strict_math_verify():
    api = fr._load_math_verify()
    assert api is not None
    return api[1]


def test_math_verify_loader_disables_nested_timeouts(monkeypatch) -> None:
    math_verify = pytest.importorskip("math_verify")
    calls: dict[str, dict[str, object]] = {}

    def raw_parse(value: str, **kwargs):
        calls["parse"] = {"value": value, **kwargs}
        return [value]

    def raw_verify(gold, pred, **kwargs):
        calls["verify"] = {"gold": gold, "pred": pred, **kwargs}
        math_verify.grader.solve("equation", {"z", "a"})
        return True

    def raw_solve(expression, *symbols, **kwargs):
        calls["solve"] = {
            "expression": expression,
            "symbols": symbols,
            **kwargs,
        }
        return []

    monkeypatch.setattr(math_verify, "parse", raw_parse)
    monkeypatch.setattr(math_verify, "verify", raw_verify)
    monkeypatch.setattr(math_verify.grader, "solve", raw_solve)
    fr._load_math_verify.cache_clear()
    try:
        api = fr._load_math_verify()
        assert api is not None
        parse, verify = api
        assert parse("answer") == ["answer"]
        assert verify(["gold"], ["pred"], strict=True)
        with pytest.raises(ValueError, match="non-strict math-verify"):
            verify(["gold"], ["pred"], strict=False)
    finally:
        fr._load_math_verify.cache_clear()

    assert calls["parse"] == {
        "value": "answer",
        "parsing_timeout": None,
        "raise_on_error": True,
    }
    assert calls["verify"] == {
        "gold": ["gold"],
        "pred": ["pred"],
        "strict": True,
        "timeout_seconds": None,
        "raise_on_error": True,
    }
    assert calls["solve"] == {
        "expression": "equation",
        "symbols": (("a", "z"),),
    }


def test_deterministic_verify_preserves_same_name_symbols(strict_math_verify) -> None:
    sympy = pytest.importorskip("sympy")
    x, y, z = sympy.symbols("x y z")

    equivalent = fr._deterministic_math_verify(
        [2 * x + 3 * y],
        [2 * x + 3 * z],
        strict_math_verify,
    )
    assert equivalent.passed

    # A name-agnostic swap could make these equal, but the shared x denotes the
    # same variable on both sides and must not be remapped merely to force a pass.
    misleading_swap = fr._deterministic_math_verify(
        [x + 2 * y],
        [2 * x + z],
        strict_math_verify,
    )
    assert not misleading_swap.passed
    assert not misleading_swap.limit_exceeded


def test_deterministic_verify_enumerates_renamed_multivariable_bijections(
    strict_math_verify,
) -> None:
    sympy = pytest.importorskip("sympy")
    x, y, a, b = sympy.symbols("x y a b")

    outcome = fr._deterministic_math_verify(
        [2 * x + 3 * y],
        [2 * a + 3 * b],
        strict_math_verify,
    )

    assert outcome.passed
    assert outcome.attempted_bijections >= 1


def test_deterministic_verify_uses_simultaneous_substitution(
    strict_math_verify,
) -> None:
    sympy = pytest.importorskip("sympy")
    c, n, r = sympy.symbols("C n R")

    outcome = fr._deterministic_math_verify(
        [2 * r * c + r + c],
        [3 * r + 1 + (n - 1) * (2 * r + 1)],
        strict_math_verify,
    )

    assert outcome.passed


def test_deterministic_verify_handles_equivalent_relations(strict_math_verify) -> None:
    sympy = pytest.importorskip("sympy")
    x, y, a, b = sympy.symbols("x y a b")

    outcome = fr._deterministic_math_verify(
        [sympy.Eq(x + y, 1)],
        [sympy.Eq(2 * a + 2 * b, 2)],
        strict_math_verify,
    )

    assert outcome.passed

    non_equivalent = fr._deterministic_math_verify(
        [sympy.Eq(x + y, 1)],
        [sympy.Eq(2 * a + 2 * b, 3)],
        strict_math_verify,
    )
    assert not non_equivalent.passed


def test_deterministic_verify_rejects_non_equivalent_expressions(
    strict_math_verify,
) -> None:
    sympy = pytest.importorskip("sympy")
    x, y, a, b = sympy.symbols("x y a b")

    outcome = fr._deterministic_math_verify(
        [2 * x + 3 * y],
        [2 * a + 4 * b],
        strict_math_verify,
    )

    assert not outcome.passed
    assert not outcome.limit_exceeded


def test_deterministic_verify_rejects_different_symbol_counts(
    strict_math_verify,
) -> None:
    sympy = pytest.importorskip("sympy")
    x, y, a, b, c = sympy.symbols("x y a b c")

    outcome = fr._deterministic_math_verify(
        [x + y],
        [a + b + c],
        strict_math_verify,
    )

    assert not outcome.passed
    assert not outcome.limit_exceeded
    assert outcome.attempted_bijections == 0


def test_deterministic_verify_rejects_incompatible_symbol_shapes(
    strict_math_verify,
) -> None:
    sympy = pytest.importorskip("sympy")
    scalar = sympy.Symbol("x")
    matrix_2x2 = sympy.MatrixSymbol("A", 2, 2)
    renamed_matrix_2x2 = sympy.MatrixSymbol("B", 2, 2)
    matrix_3x3 = sympy.MatrixSymbol("C", 3, 3)

    assert not fr._deterministic_math_verify(
        [matrix_2x2],
        [scalar],
        strict_math_verify,
    ).passed
    assert not fr._deterministic_math_verify(
        [matrix_2x2],
        [matrix_3x3],
        strict_math_verify,
    ).passed
    assert fr._deterministic_math_verify(
        [matrix_2x2],
        [renamed_matrix_2x2],
        strict_math_verify,
    ).passed


def test_symbol_sorting_fails_closed_on_stable_key_collision() -> None:
    sympy = pytest.importorskip("sympy")
    first = sympy.Dummy("x")
    second = sympy.Dummy("x")
    expression = sympy.Add(first, second, evaluate=False)

    assert fr._symbol_sort_key(first) == fr._symbol_sort_key(second)
    assert fr._stable_free_symbols(expression) is None


def test_symbol_shape_lookup_failure_is_fail_closed() -> None:
    sympy = pytest.importorskip("sympy")
    indexed = sympy.IndexedBase("B")[sympy.Symbol("i")]
    scalar = sympy.Symbol("x")

    assert fr._symbols_are_bijection_compatible(indexed, scalar) is None


def test_deterministic_verify_fails_closed_at_bijection_limit(
    monkeypatch,
    strict_math_verify,
    tmp_path,
) -> None:
    sympy = pytest.importorskip("sympy")
    a, b, c, d, e = sympy.symbols("a b c d e")
    u, v, w, x, y = sympy.symbols("u v w x y")
    gold = a + 2 * b + 3 * c + 4 * d + 5 * e
    pred = 6 * u + 7 * v + 8 * w + 9 * x + 10 * y

    outcome = fr._deterministic_math_verify(
        [gold],
        [pred],
        strict_math_verify,
    )
    assert not outcome.passed
    assert outcome.limit_exceeded
    assert outcome.attempted_bijections == fr._MAX_DETERMINISTIC_SYMBOL_BIJECTIONS

    def parse(value: str):
        return [gold] if "gold-token" in value else [pred]

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, strict_math_verify))
    result = fr._math_verify("gold-token", "Final answer: pred-token")
    assert not result.passed
    assert result.fail_reason == fr._SYMBOL_BIJECTION_LIMIT_FAIL_REASON
    assert result.fail_reason in fr._AUTHORITATIVE_ANSWER_FAIL_REASONS

    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q","answer":"gold-token"}\n',
        encoding="utf-8",
    )
    with pytest.raises(fr.UnresolvedMathVerifySymbolBijectionError):
        fr.evaluate_free_response(
            [
                {
                    "sample_index": 0,
                    "repeat_index": 0,
                    "strategy_a_prompt": "User: q\nAssistant: <think",
                    "strategy_a_completion": "Final answer: pred-token",
                    "strategy_a_stop_reason": "stop_token",
                }
            ],
            dataset_path=dataset,
            primary_only=True,
            math_verify_retry_timeout_s=None,
        )


def test_olympiad_symbol_equivalence_is_stable_across_python_hash_seeds() -> None:
    script = r"""
import json
import sympy
from src.eval.metrics import free_response as fr

equivalent = fr._math_verify(
    "2RC+R+C",
    r"Final answer: \boxed{3R + 1 + (n-1)(2R + 1)}",
)
non_equivalent = fr._math_verify(
    "2RC+R+C",
    r"Final answer: \boxed{3R + 2 + (n-1)(2R + 1)}",
)
plain_x = sympy.Symbol("x")
positive_x = sympy.Symbol("x", positive=True)
plain_y = sympy.Symbol("y")
positive_y = sympy.Symbol("y", positive=True)
gold = sympy.Add(plain_x, 2 * positive_x, evaluate=False)
pred = sympy.Add(plain_y, 2 * positive_y, evaluate=False)
_, strict_verify = fr._load_math_verify()
same_name_assumptions = fr._deterministic_math_verify(
    [gold],
    [pred],
    strict_verify,
)
x, y, a, b = sympy.symbols("x y a b")
scaled_relation = fr._deterministic_math_verify(
    [sympy.Eq(x + y, 1)],
    [sympy.Eq(2 * a + 2 * b, 2)],
    strict_verify,
)
print(json.dumps({
    "equivalent": [equivalent.passed, equivalent.fail_reason],
    "non_equivalent": [non_equivalent.passed, non_equivalent.fail_reason],
    "same_name_assumptions": same_name_assumptions.passed,
    "scaled_relation": scaled_relation.passed,
}))
"""
    expected = {
        "equivalent": [True, ""],
        "non_equivalent": [False, "math_verify_false"],
        "same_name_assumptions": True,
        "scaled_relation": True,
    }
    for seed in (0, 1, 2, 3, 7, 13, 31):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(seed)
        env["RWKV_MATH_VERIFY_TIMEOUT_S"] = "10"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert json.loads(completed.stdout) == expected, (
            seed,
            completed.stdout,
            completed.stderr,
        )
