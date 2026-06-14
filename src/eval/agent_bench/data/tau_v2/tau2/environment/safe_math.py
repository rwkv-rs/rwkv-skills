"""Small arithmetic evaluator for TAU calculate tools."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable


_ALLOWED_CHARS = frozenset("0123456789+-*/(). ")
_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_decimal_expression(expression: str) -> str:
    """Evaluate a basic arithmetic expression and match TAU's rounded string output."""

    if not expression or not all(char in _ALLOWED_CHARS for char in expression):
        raise ValueError("Invalid characters in expression")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression") from exc
    return str(round(float(_eval_node(parsed.body)), 2))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Invalid literal in expression")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported unary operator")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported operator")
        return op(_eval_node(node.left), _eval_node(node.right))
    raise ValueError("Unsupported expression")
