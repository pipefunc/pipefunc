from typing import Any

import numpy as np
import pytest

from pipefunc.map._safe_eval import evaluate_expression


def test_basic_literals() -> None:
    inputs: dict[str, Any] = {}
    assert evaluate_expression("1", inputs) == 1
    assert evaluate_expression("'string'", inputs) == "string"


def test_variable_access() -> None:
    inputs: dict[str, Any] = {"n": 5}
    assert evaluate_expression("n", inputs) == 5


def test_basic_arithmetic() -> None:
    inputs: dict[str, Any] = {"n": 5}
    assert evaluate_expression("n + 1", inputs) == 6
    assert evaluate_expression("n - 2", inputs) == 3
    assert evaluate_expression("n * 2", inputs) == 10
    assert evaluate_expression("n / 2", inputs) == 2.5
    assert evaluate_expression("n // 2", inputs) == 2
    assert evaluate_expression("n % 2", inputs) == 1
    assert evaluate_expression("n ** 2", inputs) == 25


def test_unary_operations() -> None:
    inputs: dict[str, Any] = {"n": -5}
    assert evaluate_expression("-n", inputs) == 5
    assert evaluate_expression("+n", inputs) == -5
    assert evaluate_expression("~n", inputs) == ~(-5)


def test_bitwise_operations() -> None:
    inputs: dict[str, Any] = {"a": 5, "b": 3}
    assert evaluate_expression("a & b", inputs) == 5 & 3
    assert evaluate_expression("a | b", inputs) == 5 | 3
    assert evaluate_expression("a ^ b", inputs) == 5 ^ 3
    assert evaluate_expression("a << b", inputs) == 5 << 3
    assert evaluate_expression("a >> b", inputs) == 5 >> 3


def test_function_calls() -> None:
    inputs: dict[str, Any] = {"n": 5}
    assert evaluate_expression("len([1, 2, 3])", inputs) == 3
    assert evaluate_expression("abs(-10)", inputs) == 10
    assert evaluate_expression("min(1, 2, 3)", inputs) == 1
    assert evaluate_expression("max(1, 2, 3)", inputs) == 3
    assert evaluate_expression("sum([1, 2, 3])", inputs) == 6
    assert evaluate_expression("round(3.1415, 2)", inputs) == 3.14
    assert evaluate_expression("int(3.9)", inputs) == 3
    assert evaluate_expression("float(3)", inputs) == 3.0
    assert evaluate_expression("prod(2, 3, 4)", inputs) == 24


def test_shape_function() -> None:
    inputs: dict[str, Any] = {"x": np.zeros((2, 3))}
    assert evaluate_expression("shape(x)", inputs) == (2, 3)


def test_attribute_access() -> None:
    inputs: dict[str, Any] = {"x": np.zeros((2, 3))}
    assert evaluate_expression("x.shape[0]", inputs) == 2
    assert evaluate_expression("x.shape[1]", inputs) == 3


def test_subscript_access() -> None:
    inputs: dict[str, Any] = {"x": [10, 20, 30, 40, 50]}
    assert evaluate_expression("x[0]", inputs) == 10
    assert evaluate_expression("x[-1]", inputs) == 50
    assert evaluate_expression("x[1:3]", inputs) == [20, 30]
    assert evaluate_expression("x[:3]", inputs) == [10, 20, 30]
    assert evaluate_expression("x[::2]", inputs) == [10, 30, 50]


def test_list_and_tuple_literals() -> None:
    inputs: dict[str, Any] = {"a": 1, "b": 2}
    assert evaluate_expression("[a, b, 3]", inputs) == [1, 2, 3]
    assert evaluate_expression("(a, b, 3)", inputs) == (1, 2, 3)


def test_complex_expressions() -> None:
    inputs: dict[str, Any] = {"n": 5, "x": np.array([1, 2, 3])}
    assert evaluate_expression("(n + 1) * 2", inputs) == 12
    assert evaluate_expression("prod(n, (n + 1) * 2)", inputs) == 60
    assert evaluate_expression("max(len(x), n)", inputs) == 5


def test_undefined_variable() -> None:
    inputs: dict[str, Any] = {"n": 5}
    with pytest.raises(NameError) as exc_info:
        evaluate_expression("m + 1", inputs)
    assert "Undefined variable or function 'm'" in str(exc_info.value)


def test_undefined_function() -> None:
    inputs: dict[str, Any] = {"n": 5}
    with pytest.raises(NameError) as exc_info:
        evaluate_expression("unknown_func(n)", inputs)
    assert "Undefined variable or function 'unknown_func'" in str(exc_info.value)


def test_disallowed_attribute() -> None:
    inputs: dict[str, Any] = {"x": np.array([1, 2, 3])}
    with pytest.raises(AttributeError) as exc_info:
        evaluate_expression("x.data", inputs)
    assert "Access to attribute 'data' is not allowed" in str(exc_info.value)


def test_unsupported_expression_type() -> None:
    inputs: dict[str, Any] = {}
    with pytest.raises(TypeError) as exc_info:
        evaluate_expression("lambda x: x", inputs)
    assert "Unsupported expression type: Lambda" in str(exc_info.value)


def test_syntax_error() -> None:
    inputs: dict[str, Any] = {}
    with pytest.raises(SyntaxError):
        evaluate_expression("1 +", inputs)


def test_zero_division() -> None:
    inputs: dict[str, Any] = {"a": 1, "b": 0}
    with pytest.raises(ZeroDivisionError):
        evaluate_expression("a / b", inputs)


def test_index_error() -> None:
    inputs: dict[str, Any] = {"x": [1, 2, 3]}
    with pytest.raises(IndexError):
        evaluate_expression("x[10]", inputs)


def test_no_arbitrary_code_execution() -> None:
    inputs: dict[str, Any] = {}
    with pytest.raises((AttributeError, NameError, TypeError)):
        evaluate_expression("__import__('os').system('echo hacked')", inputs)


def test_slice_object() -> None:
    inputs: dict[str, Any] = {"x": [0, 1, 2, 3, 4, 5]}
    assert evaluate_expression("x[slice(1, 4)]", inputs) == [1, 2, 3]


def test_extended_slices() -> None:
    inputs: dict[str, Any] = {"x": np.arange(27).reshape(3, 3, 3)}
    assert evaluate_expression("x[:, 1, :]", inputs).tolist() == inputs["x"][:, 1, :].tolist()


def test_negative_indices() -> None:
    inputs: dict[str, Any] = {"x": [1, 2, 3, 4, 5]}
    assert evaluate_expression("x[-1]", inputs) == 5
    assert evaluate_expression("x[-2]", inputs) == 4


def test_ellipsis() -> None:
    inputs: dict[str, Any] = {"x": np.ones((3, 3, 3))}
    r = evaluate_expression("x[..., 0]", inputs)
    assert r.tolist() == inputs["x"][..., 0].tolist()


def test_complex_numbers() -> None:
    inputs: dict[str, Any] = {}
    assert evaluate_expression("complex(1, 2)", inputs) == complex(1, 2)


def test_string_operations() -> None:
    inputs: dict[str, Any] = {"s": "hello"}
    assert evaluate_expression("s + ' world'", inputs) == "hello world"
    assert evaluate_expression("s * 2", inputs) == "hellohello"


def test_dict_access() -> None:
    inputs: dict[str, Any] = {"d": {"key": "value"}}
    assert evaluate_expression("d['key']", inputs) == "value"
    with pytest.raises(AttributeError, match="Access to attribute 'key' is not allowed"):
        evaluate_expression("d.key", inputs)


def test_float_division() -> None:
    inputs: dict[str, Any] = {"a": 5, "b": 2}
    assert evaluate_expression("a / b", inputs) == 2.5


def test_matrix_multiplication() -> None:
    inputs: dict[str, Any] = {"A": np.array([[1, 2], [3, 4]]), "B": np.array([[5, 6], [7, 8]])}
    result = evaluate_expression("A @ B", inputs)
    expected = inputs["A"] @ inputs["B"]
    assert np.array_equal(result, expected)


def test_logical_not() -> None:
    inputs: dict[str, Any] = {"a": True}
    assert evaluate_expression("not a", inputs) is False
    inputs2: dict[str, Any] = {"a": False}
    assert evaluate_expression("not a", inputs2) is True


def test_logical_and_or() -> None:
    inputs: dict[str, Any] = {"a": True, "b": False}
    assert evaluate_expression("a and b", inputs) is False
    assert evaluate_expression("a or b", inputs) is True


def test_comparisons() -> None:
    inputs: dict[str, Any] = {"a": 5, "b": 3}
    assert evaluate_expression("a > b", inputs) is True
    assert evaluate_expression("a == b", inputs) is False
    assert evaluate_expression("a >= b", inputs) is True
    assert evaluate_expression("a < b", inputs) is False
    assert evaluate_expression("a != b", inputs) is True
