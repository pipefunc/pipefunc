"""Tests for NamedTuple output type annotation handling."""

from __future__ import annotations

from typing import NamedTuple

from pipefunc import Pipeline, pipefunc


class MyOutput(NamedTuple):
    """A NamedTuple with two fields for testing."""

    y: int
    z: int


class SingleFieldOutput(NamedTuple):
    """A NamedTuple with a single field."""

    value: str


class ThreeFieldOutput(NamedTuple):
    """A NamedTuple with three fields."""

    field1: int
    field2: str
    field3: float


def test_namedtuple_output_annotations():
    """Test that NamedTuple return types provide correct type annotations."""

    @pipefunc(output_name=("y", "z"))
    def double_it(x: int) -> MyOutput:
        return MyOutput(2 * x, 3 * x)

    # Check that the output annotations are correctly extracted
    assert double_it.output_annotation == {"y": int, "z": int}


def test_namedtuple_in_pipeline():
    """Test that NamedTuple annotations work correctly in a pipeline."""

    @pipefunc(output_name=("y", "z"))
    def double_it(x: int) -> MyOutput:
        return MyOutput(2 * x, 3 * x)

    @pipefunc(output_name="result")
    def half_it(y: int) -> float:
        return y / 2

    pipeline = Pipeline([double_it, half_it])

    # Check pipeline output annotations
    assert pipeline.output_annotations["y"] is int
    assert pipeline.output_annotations["z"] is int
    assert pipeline.output_annotations["result"] is float


def test_single_output_namedtuple():
    """Test that single output from NamedTuple returns the NamedTuple class."""

    @pipefunc(output_name="value")
    def get_string(x: int) -> SingleFieldOutput:
        return SingleFieldOutput(str(x))

    # For single output, it should return the NamedTuple class itself
    assert get_string.output_annotation == {"value": SingleFieldOutput}


def test_regular_tuple_still_works():
    """Test that regular tuple annotations still work correctly."""

    @pipefunc(output_name=("a", "b"))
    def returns_tuple(x: int) -> tuple[int, str]:
        return (x, str(x))

    assert returns_tuple.output_annotation == {"a": int, "b": str}


def test_mismatched_output_names_and_fields():
    """Test handling when output names don't match all NamedTuple fields."""

    @pipefunc(output_name=("x", "y"))  # Only using 2 of the 3 fields
    def mismatched_func(val: int) -> ThreeFieldOutput:
        return ThreeFieldOutput(val, str(val), float(val))

    # Should map first two fields to the output names
    assert mismatched_func.output_annotation == {"x": int, "y": str}


def test_namedtuple_with_custom_output_picker():
    """Test that custom output_picker still results in NoAnnotation."""

    def custom_picker(output: MyOutput, key: str) -> int | None:
        if key == "y":
            return output.y
        if key == "z":
            return output.z
        return None

    @pipefunc(output_name=("y", "z"), output_picker=custom_picker)
    def with_picker(x: int) -> MyOutput:
        return MyOutput(2 * x, 3 * x)

    # With custom output_picker, we can't determine the output type
    from pipefunc.typing import NoAnnotation

    assert with_picker.output_annotation == {"y": NoAnnotation, "z": NoAnnotation}


def test_namedtuple_execution():
    """Test that NamedTuple outputs work correctly during execution."""

    @pipefunc(output_name=("y", "z"))
    def double_it(x: int) -> MyOutput:
        return MyOutput(2 * x, 3 * x)

    @pipefunc(output_name="sum")
    def add_them(y: int, z: int) -> int:
        return y + z

    pipeline = Pipeline([double_it, add_them])

    # Test execution
    result = pipeline("sum", x=5)
    assert result == 25  # (2*5) + (3*5) = 10 + 15 = 25

    # Test with full_output to see intermediate values
    full_result = pipeline.run("sum", kwargs={"x": 5}, full_output=True)
    assert full_result["y"] == 10
    assert full_result["z"] == 15
    assert full_result["sum"] == 25
