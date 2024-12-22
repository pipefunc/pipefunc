import inspect

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.helpers import collect_kwargs, get_attribute_factory


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
