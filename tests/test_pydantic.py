import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, Field

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.typing import safe_get_type_hints

has_griffe = importlib.util.find_spec("griffe") is not None


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
    # Test the `PlainSerializer`
    out_json = model.model_dump_json()
    assert out_json == '{"x":[1,2,3],"y":2,"z":{"a":1}}'


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


def test_pipeline_to_pydantic_without_griffe(monkeypatch):
    # Import the module where pipeline_to_pydantic is defined.
    # Adjust the import path according to your project structure.
    from pydantic import BaseModel

    from pipefunc import Pipeline, pipefunc
    from pipefunc._pipeline._pydantic import pipeline_to_pydantic

    # Force has_griffe to False for this test.
    monkeypatch.setattr("pipefunc._pipeline._pydantic.has_griffe", False)

    # Create a simple function with a docstring that would normally be used for field descriptions
    @pipefunc("foo")
    def foo(x: int, y: int = 1) -> int:
        """Add x and y.

        Parameters:
            x: The first number.
            y: The second number.
        """
        return x + y

    # Build a pipeline using the function.
    pipeline = Pipeline([foo])
    # Create the Pydantic model from the pipeline.
    Model: type[BaseModel] = pipeline_to_pydantic(pipeline, model_name="TestModel")  # noqa: N806

    # Instantiate the model.
    instance = Model(x=10, y=5)
    # Since has_griffe is False, the field descriptions should be None.
    field_info = instance.model_fields["x"]
    assert field_info.description is None, "Expected no description when griffe is not available."
    # Check that the instance validates the values as expected.
    assert instance.x == 10
    assert instance.y == 5


@pytest.mark.skipif(not has_griffe, reason="requires griffe")
def test_pipeline_to_pydantic_with_doc():
    @pipefunc("foo")
    def foo(x: int, y: int = 1) -> int:
        """
        Add x and y.

        Parameters
        ----------
        x : int
            The first number.
        y : int, optional
            The second number.

        Returns
        -------
        int
        """
        return x + y

    # Build a pipeline using the function.
    pipeline = Pipeline([foo])
    # Create the Pydantic model from the pipeline.
    Model: type[BaseModel] = pipeline.pydantic_model(model_name="TestModel")  # noqa: N806

    # Instantiate the model.
    instance = Model(x=10, y=5)

    # Verify that the field descriptions are populated (i.e. not None).
    field_x_info = instance.model_fields["x"]
    assert field_x_info.description is not None
    assert "The first number." in field_x_info.description

    field_y_info = instance.model_fields["y"]
    assert field_y_info.description is not None
    assert "The second number." in field_y_info.description

    # Check that the instance validates the values as expected.
    assert instance.x == 10
    assert instance.y == 5


def test_pipeline_map(tmp_path: Path) -> None:
    @pipefunc("foo", mapspec="x[i, j] -> foo[i, j]")
    def foo(x: float, y: int = 1, z: dict[str, int] | None = None) -> float:
        return x + y

    @pipefunc("bar", mapspec="foo[i, j] -> bar[i, j]")
    def bar(foo: float) -> float:
        return foo * 2

    pipeline = Pipeline([foo, bar])
    Model = pipeline.pydantic_model()  # noqa: N806
    model = Model(x=[[1, 2], [3, 4]], y=2, z={"a": 1})
    results = pipeline.map(model, run_folder=tmp_path, parallel=False, storage="dict")
    out = results["foo"].output
    assert out.shape == (2, 2)
    assert out.tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert isinstance(out[0, 0], float)
    out = results["bar"].output
    assert out.shape == (2, 2)
    assert out.tolist() == [[6.0, 8.0], [10.0, 12.0]]
    assert isinstance(out[0, 0], float)


def test_not_annotated_warning() -> None:
    @pipefunc("foo")
    def foo(x: int, y=1) -> int:
        return x + y

    pipeline = Pipeline([foo])
    with pytest.warns(UserWarning, match="Parameter 'y' is not annotated"):
        pipeline.pydantic_model()


def test_none_default() -> None:
    @pipefunc("foo")
    def foo(x: int, y: int | None = None) -> int:
        return x + (y or 1)

    pipeline = Pipeline([foo])
    Model = pipeline.pydantic_model()  # noqa: N806
    model = Model(x=1)
    assert model.y is None
    assert Model.model_fields["y"].default is None
    assert pipeline(x=1) == 2
