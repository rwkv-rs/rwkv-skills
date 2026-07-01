import abc
import ast
import operator
from collections.abc import Callable
from typing import Any


_ALLOWED_CALCULATE_CHARS = frozenset("0123456789+-*/(). ")
_CALCULATE_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_CALCULATE_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class Tool(abc.ABC):
    @staticmethod
    def invoke(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError


def calculate_decimal_expression(expression: str) -> str:
    """Evaluate a basic arithmetic expression and match TAU's rounded string output."""

    if not expression or not all(char in _ALLOWED_CALCULATE_CHARS for char in expression):
        raise ValueError("Invalid characters in expression")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid mathematical expression") from exc
    return str(round(float(_eval_calculate_node(parsed.body)), 2))


def _eval_calculate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Invalid literal in expression")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        op = _CALCULATE_UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported unary operator")
        return op(_eval_calculate_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _CALCULATE_BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("Unsupported operator")
        return op(_eval_calculate_node(node.left), _eval_calculate_node(node.right))
    raise ValueError("Unsupported expression")
