import re

import pytest

from pipefunc._mapspec import (
    ArraySpec,
    MapSpec,
    _parse_index_string,
    _parse_indexed_arrays,
    _shape_to_strides,
)


def test_shape_to_strides():
    assert _shape_to_strides((3, 4, 5)) == (20, 5, 1)
    assert _shape_to_strides(()) == ()
    assert _shape_to_strides((1,)) == (1,)
    assert _shape_to_strides((1, 2)) == (2, 1)


def test_arrayspec_init():
    spec = ArraySpec("a", ("i", "j"))
    assert spec.name == "a"
    assert spec.axes == ("i", "j")

    with pytest.raises(ValueError, match="is not a valid Python identifier"):
        ArraySpec("123", ("i", "j"))

    with pytest.raises(ValueError, match="is not a valid Python identifier"):
        ArraySpec("a", ("i", "123"))


def test_arrayspec_str():
    spec = ArraySpec("a", ("i", None, "j"))
    assert str(spec) == "a[i, :, j]"


def test_arrayspec_indices():
    spec = ArraySpec("a", ("i", None, "j"))
    assert spec.indices == ("i", "j")


def test_arrayspec_rank():
    spec = ArraySpec("a", ("i", None, "j"))
    assert spec.rank == 3


def test_arrayspec_validate():
    spec = ArraySpec("a", ("i", "j"))
    spec.validate((3, 4))

    with pytest.raises(
        ValueError,
        match=re.escape("Expecting array of rank 2, but got array of shape (3, 4, 5)"),
    ):
        spec.validate((3, 4, 5))


def test_mapspec_init():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, output)
    assert spec.inputs == inputs
    assert spec.output == output

    with pytest.raises(
        ValueError,
        match=re.escape("Output array must have all axes indexed (no ':')."),
    ):
        MapSpec(inputs, ArraySpec("q", ("i", None, "k")))

    with pytest.raises(
        ValueError,
        match="Output array has indices that do not appear in the input: {'l'}",
    ):
        MapSpec(inputs, ArraySpec("q", ("i", "j", "l")))

    with pytest.raises(
        ValueError,
        match="Input array have indices that do not appear in the output: {'k'}",
    ):
        MapSpec((ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "k"))), output)


def test_mapspec_parameters():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, output)
    assert spec.parameters == ("a", "b")


def test_mapspec_indices():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, output)
    assert spec.indices == ("i", "j")


def test_mapspec_shape():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, output)
    shapes = {"a": (3, 4), "b": (3, 4)}
    assert spec.shape(shapes) == (3, 4)

    with pytest.raises(
        ValueError,
        match="Inputs expected by this map were not provided: {'b'}",
    ):
        spec.shape({"a": (3, 4)})

    with pytest.raises(ValueError, match="Dimension mismatch for arrays"):
        spec.shape({"a": (3, 4), "b": (3, 5)})

    with pytest.raises(ValueError, match="Got extra array"):
        spec.shape({"a": (3, 4), "b": (3, 4), "c": (3, 4)})


def test_mapspec_output_key():
    spec = MapSpec.from_string("x[i, j], y[j, :, k] -> z[i, j, k]")
    assert spec.output_key((5, 2, 3), 23) == (3, 1, 2)

    with pytest.raises(
        ValueError,
        match=re.escape("Expected a shape of length 3, got (5, 2)"),
    ):
        spec.output_key((5, 2), 23)


def test_mapspec_input_keys():
    spec = MapSpec.from_string("x[i, j], y[j, :, k] -> z[i, j, k]")
    assert spec.input_keys((5, 2, 3), 23) == {
        "x": (3, 1),
        "y": (1, slice(None, None, None), 2),
    }

    with pytest.raises(
        ValueError,
        match=re.escape("Expected a shape of length 3, got (5, 2)"),
    ):
        spec.input_keys((5, 2), 23)


def test_mapspec_str():
    spec = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    assert str(spec) == "a[i, j], b[i, j], c[k] -> q[i, j, k]"


def test_mapspec_from_string():
    spec = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    assert isinstance(spec, MapSpec)
    assert len(spec.inputs) == 3
    assert spec.output == ArraySpec("q", ("i", "j", "k"))

    with pytest.raises(ValueError, match="Expected expression of form"):
        MapSpec.from_string("a[i, j], b[i, j], c[k]")

    with pytest.raises(ValueError, match="Expected a single output, but got 2"):
        MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k], r[i]")


def test_mapspec_to_string():
    spec = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    assert spec.to_string() == "a[i, j], b[i, j], c[k] -> q[i, j, k]"


def test_parse_index_string():
    assert _parse_index_string("i, j, k") == ("i", "j", "k")
    assert _parse_index_string("i, :, k") == ("i", None, "k")


def test_parse_indexed_arrays():
    expr = "a[i, j], b[i, j], c[k]"
    arrays = _parse_indexed_arrays(expr)
    assert arrays == (
        ArraySpec("a", ("i", "j")),
        ArraySpec("b", ("i", "j")),
        ArraySpec("c", ("k",)),
    )
