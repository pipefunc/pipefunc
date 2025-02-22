import numpy as np
from pydantic import BaseModel, Field

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.typing import safe_get_type_hints


def test_pydantic_annotations() -> None:
    class Foo(BaseModel):
        x: int
        y: int = 1
        z: dict = Field(default_factory=dict)

    foo = PipeFunc(Foo, "foo", defaults={"x": 2})

    type_hints = safe_get_type_hints(Foo)
    assert type_hints == {"x": int, "y": int, "z": dict}
    assert foo.defaults == {"x": 2, "y": 1, "z": {}}
    assert foo.parameter_annotations == {"x": int, "y": int, "z": dict}

    class Bar(BaseModel):
        foo: Foo

    bar = PipeFunc(Bar, "bar")
    assert bar.parameter_annotations == {"foo": Foo}
    assert bar.defaults == {}


def test_pydantic_model_with__call__() -> None:
    class Foo(BaseModel):
        x: int
        y: int = 1
        z: dict = Field(default_factory=dict)

        def __call__(self, a: int = 42) -> int:
            return self.x + self.y + a

    foo = PipeFunc(Foo(x=1, y=2, z={1: 2}), "foo")
    assert foo.__name__ == "Foo"
    assert foo(3) == foo(a=3) == 6
    assert foo.parameter_annotations == {"a": int}
    assert foo.output_annotation == {"foo": int}
    assert foo.defaults == {"a": 42}
    pipeline = Pipeline([foo])
    assert pipeline(a=3) == 6
    assert pipeline.parameter_annotations == {"a": int}
    assert pipeline.defaults == {"a": 42}
    assert pipeline.output_annotations == {"foo": int}


def test_dataclass_annotations() -> None:
    # Just for reference, pydantic.BaseModel should behave the same way as dataclasses.dataclass
    from dataclasses import dataclass, field

    @dataclass
    class Foo:
        x: int
        y: int = 1
        z: dict = field(default_factory=dict)

    foo = PipeFunc(Foo, "foo", defaults={"x": 2})
    assert foo.defaults == {"x": 2, "y": 1, "z": {}}
    assert foo.parameter_annotations == {"x": int, "y": int, "z": dict}
    assert foo.__name__ == "Foo"


def test_pipeline_input_as_pydantic_model() -> None:
    @pipefunc("foo")
    def foo(x: int, y: int = 1, z: dict[str, int] | None = None) -> int:
        return x + y

    pipeline = Pipeline([foo])
    Model = pipeline.pydantic_model()  # noqa: N806
    model = Model(x=1, y=2, z={"a": 1})
    expected = {"x": 1, "y": 2, "z": {"a": 1}}
    assert model.model_dump() == expected
    model = Model(x="1.0", y="2.0", z={"a": "1.0"})
    assert model.model_dump() == expected
    assert repr(model) == "InputModel(x=1, y=2, z={'a': 1})"


def test_pipeline_with_mapspec_input_as_pydantic_model() -> None:
    @pipefunc("foo", mapspec="x[i] -> foo[i]")
    def foo(x: int, y: int = 1, z: dict[str, int] | None = None) -> int:
        return x + y

    pipeline = Pipeline([foo])
    Model = pipeline.pydantic_model()  # noqa: N806
    model = Model(x=[1, 2, 3], y=2, z={"a": 1})
    expected = {"x": [1, 2, 3], "y": 2, "z": {"a": 1}}
    out = model.model_dump()
    assert isinstance(out["x"], np.ndarray)
    assert np.array_equal(out["x"], expected["x"])  # type: ignore[arg-type]
    assert out["y"] == expected["y"]


def test_pipeline_2d_mapspec_as_pydantic_model() -> None:
    @pipefunc("foo", mapspec="x[i, j] -> foo[i, j]")
    def foo(x: int, y: int = 1, z: dict[str, int] | None = None) -> int:
        return x + y

    pipeline = Pipeline([foo])
    Model = pipeline.pydantic_model()  # noqa: N806
    model = Model(x=[[1, 2], [3, 4]], y=2, z={"a": 1})
    expected = {"x": np.array([[1, 2], [3, 4]]), "y": 2, "z": {"a": 1}}
    out = model.model_dump()
    assert isinstance(out["x"], np.ndarray)
    assert np.array_equal(out["x"], expected["x"])  # type: ignore[arg-type]
    assert out["y"] == expected["y"]
    assert out["z"] == expected["z"]


class CustomObject:
    def __init__(self, x: int):
        self.x = x


def test_pipeline_3d_mapspec_as_pydantic_model_with_custom_objects() -> None:
    @pipefunc("foo", mapspec="x[i, j, k] -> foo[i, j, k]")
    def foo(x: CustomObject, y: int = 1) -> int:
        return x.x + y

    pipeline = Pipeline([foo])
    Model = pipeline.pydantic_model()  # noqa: N806
    x = [[[CustomObject(1), CustomObject(2)], [CustomObject(3), CustomObject(4)]]]
    model = Model(x=x, y=2, z={"a": 1})
    expected = {"x": np.array(x), "y": 2, "z": {"a": 1}}
    out = model.model_dump()
    assert isinstance(out["x"], np.ndarray)
    assert np.array_equal(out["x"], expected["x"])  # type: ignore[arg-type]
    assert out["x"].shape == (1, 2, 2)
    assert str(out["x"].dtype) == "object"
    assert out["y"] == expected["y"]
