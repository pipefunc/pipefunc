from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def _prod(*args: int) -> int:
    result = 1
    for arg in args:
        result *= arg
    return result


def _shape(x: Any) -> tuple[int, ...]:
    return np.shape(x)


SAFE_FUNCTIONS: dict[str, Callable] = {
    "len": len,
    "prod": _prod,
    "shape": _shape,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "round": round,
    "slice": slice,
    "complex": complex,
}


OPERATORS: dict[type[ast.operator], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.MatMult: operator.matmul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}

COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

ALLOWED_ATTRS: set[str] = {"shape"}


def _safe_eval(node: ast.AST, inputs: dict[str, Any]) -> Any:  # noqa: PLR0911, PLR0912
    if isinstance(node, ast.Expression):
        return _eval_expression(node, inputs)
    if isinstance(node, ast.Constant):
        return _eval_constant(node)
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, inputs)
    if isinstance(node, ast.UnaryOp):
        return _eval_unaryop(node, inputs)
    if isinstance(node, ast.Name):
        return _eval_name(node, inputs)
    if isinstance(node, ast.Call):
        return _eval_call(node, inputs)
    if isinstance(node, ast.Attribute):
        return _eval_attribute(node, inputs)
    if isinstance(node, ast.Subscript):
        return _eval_subscript(node, inputs)
    if isinstance(node, ast.Slice):
        return _eval_slice(node, inputs)
    if isinstance(node, ast.List):
        return _eval_list(node, inputs)
    if isinstance(node, ast.BoolOp):
        return _eval_boolop(node, inputs)
    if isinstance(node, ast.Tuple):
        return _eval_tuple(node, inputs)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, inputs)

    msg = f"Unsupported expression type: {type(node).__name__}"
    raise TypeError(msg)


def _eval_expression(node: ast.Expression, inputs: dict[str, Any]) -> Any:
    return _safe_eval(node.body, inputs)


def _eval_constant(node: ast.Constant) -> Any:
    return node.value


def _eval_binop(node: ast.BinOp, inputs: dict[str, Any]) -> Any:
    left = _safe_eval(node.left, inputs)
    right = _safe_eval(node.right, inputs)
    op = OPERATORS[type(node.op)]
    return op(left, right)


def _eval_unaryop(node: ast.UnaryOp, inputs: dict[str, Any]) -> Any:
    operand = _safe_eval(node.operand, inputs)
    op = UNARY_OPERATORS[type(node.op)]
    return op(operand)


def _eval_name(node: ast.Name, inputs: dict[str, Any]) -> Any:
    if node.id in inputs:
        return inputs[node.id]
    if node.id in SAFE_FUNCTIONS:
        return SAFE_FUNCTIONS[node.id]
    msg = f"Undefined variable or function '{node.id}'"
    raise NameError(msg)


def _eval_call(node: ast.Call, inputs: dict[str, Any]) -> Any:
    func = _safe_eval(node.func, inputs)
    args = [_safe_eval(arg, inputs) for arg in node.args]
    kwargs = {kw.arg: _safe_eval(kw.value, inputs) for kw in node.keywords}
    return func(*args, **kwargs)


def _eval_attribute(node: ast.Attribute, inputs: dict[str, Any]) -> Any:
    value = _safe_eval(node.value, inputs)
    attr = node.attr
    if attr in ALLOWED_ATTRS:
        return getattr(value, attr)
    msg = f"Access to attribute '{attr}' is not allowed"
    raise AttributeError(msg)


def _eval_subscript(node: ast.Subscript, inputs: dict[str, Any]) -> Any:
    value = _safe_eval(node.value, inputs)
    index = _safe_eval(node.slice, inputs)
    return value[index]


def _eval_slice(node: ast.Slice, inputs: dict[str, Any]) -> slice:
    lower = _safe_eval(node.lower, inputs) if node.lower else None
    upper = _safe_eval(node.upper, inputs) if node.upper else None
    step = _safe_eval(node.step, inputs) if node.step else None
    return slice(lower, upper, step)


def _eval_list(node: ast.List, inputs: dict[str, Any]) -> list:
    return [_safe_eval(elt, inputs) for elt in node.elts]


def _eval_boolop(node: ast.BoolOp, inputs: dict[str, Any]) -> bool:
    values = [_safe_eval(value, inputs) for value in node.values]  # noqa: PD011
    if isinstance(node.op, ast.And):
        return all(values)
    if isinstance(node.op, ast.Or):
        return any(values)
    msg = f"Unsupported boolean operator: {type(node.op).__name__}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def _eval_tuple(node: ast.Tuple, inputs: dict[str, Any]) -> tuple:
    return tuple(_safe_eval(elt, inputs) for elt in node.elts)


def _eval_compare(node: ast.Compare, inputs: dict[str, Any]) -> bool:
    left = _safe_eval(node.left, inputs)
    comparisons = zip(node.ops, node.comparators)
    result = True
    for op, comparator in comparisons:
        right = _safe_eval(comparator, inputs)
        operator_func = COMPARISON_OPERATORS[type(op)]
        if not operator_func(left, right):
            result = False
            break
        left = right  # Chain comparisons
    return result

    msg = f"Unsupported expression type: {type(node).__name__}"
    raise TypeError(msg)


def evaluate_expression(expression: str, inputs: dict[str, Any]) -> Any:
    """Safely evaluate a string expression using provided inputs.

    Parameters
    ----------
    expression
        The expression to evaluate.
    inputs
        A dictionary of variables to use in the expression.

    Returns
    -------
    Any
        The result of the evaluated expression.

    Raises
    ------
    SyntaxError
        If the expression is not valid.
    NameError
        If a variable or function is not defined in the expression.
    AttributeError
        If an attribute is accessed that is not allowed.
    TypeError
        If an unsupported expression type is encountered.

    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        msg = f"Invalid syntax in expression: {expression}"
        raise SyntaxError(msg) from e
    return _safe_eval(tree, inputs)
