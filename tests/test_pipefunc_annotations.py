from pipefunc import PipeFunc, pipefunc
from pipefunc._pipeline import _axis_is_generated, _axis_is_reduced


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

    assert _axis_is_generated(f, g, "y")


def test_axis_is_reduced():
    @pipefunc("y", mapspec="x[i] -> y[i]")
    def f(x: int) -> list[int]:
        return x

    @pipefunc("z")
    def g(y: list[int]) -> list[int]:
        return y

    assert _axis_is_reduced(f, g, "y")
