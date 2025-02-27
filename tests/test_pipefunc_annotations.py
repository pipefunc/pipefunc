from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipefunc import PipeFunc, pipefunc
from pipefunc._pipeline._validation import _axis_is_reduced, _mapspec_with_internal_shape


def test_pipeline_function_annotations_single_output():
    def use_tuple(a: tuple[int, float]) -> float:
        return sum(a)

    use_tuple_func = PipeFunc(
        use_tuple,
        output_name="sum",
        renames={"a": "numbers"},
    )
    assert use_tuple_func.parameter_annotations == {"numbers": tuple[int, float]}
    assert use_tuple_func.output_annotation == {"sum": float}


def test_pipeline_function_annotations_multiple_outputs():
    def add_numbers(a: int, b: float) -> tuple[int, float]:
        return a + 1, b + 1.0

    add_func = PipeFunc(
        add_numbers,
        output_name=("a_plus_one", "b_plus_one"),
        renames={"a": "x", "b": "y"},
    )

    assert add_func.parameter_annotations == {"x": int, "y": float}
    assert add_func.output_annotation == {"a_plus_one": int, "b_plus_one": float}

    result = add_func(x=1, y=2.0)
    assert result == (2, 3.0)

    assert str(add_func) == "add_numbers(...) → a_plus_one, b_plus_one"
    assert repr(add_func) == "PipeFunc(add_numbers)"


def test_axis_is_generated():
    @pipefunc("y", mapspec="... -> y[i]")
    def f(x: int) -> list[int]:
        return list(range(x))

    @pipefunc("z", mapspec="y[i] -> z[i]")
    def g(y: int) -> int:
        return y

    assert _mapspec_with_internal_shape(f, "y")


def test_axis_is_reduced():
    @pipefunc("y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return x

    @pipefunc("z")
    def g(y: list[int]) -> list[int]:
        return y

    assert _axis_is_reduced(f, g, "y")


def test_multi_output():
    @pipefunc(output_name=("x", "y"), mapspec="... -> x[i, j], y[i, j]")
    def generate_ints(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.ones((n, n)), np.ones((n, n))

    @pipefunc(output_name="z", mapspec="x[i, :], y[i, :] -> z[i]")
    def double_it(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 2 * sum(x) + 0 * sum(y)

    assert _mapspec_with_internal_shape(generate_ints, "x")
    assert _mapspec_with_internal_shape(generate_ints, "y")


def test_pipefunc_from_class():
    class Adder:
        def __init__(self, a: int) -> None:
            self.a = a

    adder_func = PipeFunc(Adder, output_name="sum")
    assert isinstance(adder_func(a=1), Adder)
    assert adder_func.parameter_annotations == {"a": int}
    assert adder_func.output_annotation == {"sum": Adder}


@dataclass
class Foo:
    name: str

    @classmethod
    def from_name(cls, name: str) -> Foo:
        return cls(name)


def test_pipefunc_with_classmethod_dataclass() -> None:
    f = PipeFunc(Foo.from_name, output_name="foo")
    assert f.parameter_annotations == {"name": str}
    assert f.output_annotation == {"foo": Foo}


class Bar:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_name(cls, name: str) -> Bar:
        return cls(name)


def test_pipefunc_with_classmethod() -> None:
    f = PipeFunc(Bar.from_name, output_name="foo")
    assert f.parameter_annotations == {"name": str}
    assert f.output_annotation == {"foo": Bar}
