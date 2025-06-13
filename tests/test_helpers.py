import importlib.util
import inspect
import re
from unittest.mock import patch

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.helpers import collect_kwargs, get_attribute_factory, launch_maps

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None


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


class MyClass:
    def __init__(self, data: int, name: str) -> None:
        self.data = data
        self.name = name


class MyOtherClass:
    def __init__(self, value: float) -> None:
        self.value = value


def test_get_attribute_factory_basic() -> None:
    f = get_attribute_factory("data", "obj")
    obj = MyClass(data=123, name="test")
    assert f(obj) == 123


def test_get_attribute_factory_custom_names() -> None:
    f = get_attribute_factory("name", parameter_name="my_object", function_name="get_name")
    obj = MyClass(data=123, name="test")
    assert f(my_object=obj) == "test"  # type: ignore[call-arg]
    assert f.__name__ == "get_name"


def test_get_attribute_factory_annotations() -> None:
    f = get_attribute_factory(
        "value",
        "obj",
        parameter_annotation=MyOtherClass,
        return_annotation=float,
    )
    sig = inspect.signature(f)
    assert sig.parameters["obj"].annotation is MyOtherClass
    assert sig.return_annotation is float
    obj = MyOtherClass(value=3.14)
    assert f(obj) == 3.14


def test_get_attribute_factory_in_pipefunc() -> None:
    @pipefunc(output_name="extracted_data")
    def extract_data(obj: MyClass):
        f = get_attribute_factory(
            "data",
            "obj",
            parameter_annotation=MyClass,
            return_annotation=int,
        )
        return f(obj)

    pipeline = Pipeline([extract_data])
    result = pipeline(obj=MyClass(data=42, name="example"))
    assert result == 42


def test_get_attribute_factory_invalid_attribute() -> None:
    f = get_attribute_factory("invalid_attribute", "obj")
    obj = MyClass(data=123, name="test")
    with pytest.raises(AttributeError):
        f(obj)


def test_get_attribute_factory_no_parameter_name() -> None:
    f = get_attribute_factory("data", "obj")
    with pytest.raises(TypeError, match="missing a required argument: 'obj'"):
        f()  # type: ignore[call-arg]


def test_get_attribute_factory_with_defaults() -> None:
    f = get_attribute_factory("data", "obj")
    obj = MyClass(data=123, name="test")
    assert f(obj=obj) == 123  # type: ignore[call-arg]


def test_get_attribute_factory_return_annotation_inference() -> None:
    f = get_attribute_factory("data", "obj", parameter_annotation=MyClass)
    sig = inspect.signature(f)
    assert sig.return_annotation == inspect.Parameter.empty
    obj = MyClass(data=42, name="example")
    assert f(obj) == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("output_tabs", [True, False])
async def test_launch_maps(output_tabs: bool) -> None:  # noqa: FBT001
    if output_tabs and not has_ipywidgets:
        pytest.skip("ipywidgets not installed")

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double_it])
    inputs_dicts = [{"x": [1, 2, 3, 4, 5]}, {"x": [6, 7, 8, 9, 10]}]

    with (
        # This first patch is meant to ensure that maybe_async_task_status_widget
        # returns a widget, however, for some unknown reason, the patch is not working.
        patch("pipefunc._widgets.helpers.is_running_in_ipynb", return_value=True),
        patch("pipefunc.map._progress.is_running_in_ipynb", return_value=True),
    ):
        runners = [
            pipeline.map_async(inputs, start=False, display_widgets=output_tabs)
            for inputs in inputs_dicts
        ]
        task = launch_maps(*runners, max_concurrent=1)
    result0, result1 = await task
    assert result0["y"].output.tolist() == [2, 4, 6, 8, 10]
    assert result1["y"].output.tolist() == [12, 14, 16, 18, 20]


@pytest.mark.asyncio
async def test_launch_maps_already_running() -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double_it])
    runner = pipeline.map_async(inputs={"x": [1, 2, 3, 4, 5]}, start=True)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "`pipeline.map_async(..., start=False)` must be called before `launch_maps`.",
        ),
    ):
        await launch_maps(runner)
