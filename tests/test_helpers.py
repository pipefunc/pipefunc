import inspect

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.helpers import collect_kwargs


def test_collect_kwargs() -> None:
    f = collect_kwargs(("a", "b"), annotations=(int, list[int]), function_name="yolo")
    assert f(a=1, b=[1]) == {"a": 1, "b": [1]}
    sig = inspect.signature(f)
    assert sig.parameters["a"].annotation is int
    assert sig.parameters["b"].annotation == list[int]
    assert f.__name__ == "yolo"

    with pytest.raises(ValueError, match="should have equal length"):
        collect_kwargs(parameters=("a",), annotations=())


def test_collect_kwargs_in_pipefunc() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    f_agg = PipeFunc(
        collect_kwargs(
            (
                "a",  # input parameter
                "d",  # output parameter
                "e",  # output parameter
            ),
            function_name="aggregate",
        ),
        output_name="result_dict",
    )

    pipeline = Pipeline([f1, f2, f3, f_agg])
    result = pipeline(a=1, b=2)
    assert result == {"a": 1, "d": 6, "e": 18}  # same parameters as in `collect_kwargs`
