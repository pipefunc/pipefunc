import re

import numpy as np
import pytest

from pipefunc.map._mapspec import (
    ArraySpec,
    MapSpec,
    _parse_index_string,
    _parse_indexed_arrays,
    array_mask,
    array_shape,
    expected_mask,
    shape_to_strides,
    validate_consistent_axes,
)


def testshape_to_strides():
    assert shape_to_strides((3, 4, 5)) == (20, 5, 1)
    assert shape_to_strides(()) == ()
    assert shape_to_strides((1,)) == (1,)
    assert shape_to_strides((1, 2)) == (2, 1)


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
    new_spec = spec.add_axes("k")
    assert str(new_spec) == "a[i, :, j, k]"
    assert str(spec.add_axes(None)) == "a[i, :, j, :]"
    with pytest.raises(ValueError, match="Duplicate axes"):
        new_spec = spec.add_axes("i")


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
    spec = MapSpec(inputs, (output,))
    assert spec.inputs == inputs
    assert spec.outputs == (output,)

    # Add an extra axis to the output
    spec = MapSpec(inputs, (ArraySpec("q", ("i", "j", "l")),))
    assert str(spec) == "a[i, j], b[i, j] -> q[i, j, l]"

    with pytest.raises(
        ValueError,
        match=re.escape("Output array must have all axes indexed (no ':')."),
    ):
        MapSpec(inputs, (ArraySpec("q", ("i", None, "k")),))

    with pytest.raises(
        ValueError,
        match="Input array have indices that do not appear in the output: {'k'}",
    ):
        MapSpec((ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "k"))), (output,))


def test_mapspec_input_names():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, (output,))
    assert spec.input_names == ("a", "b")
    assert spec.output_names == ("q",)


def test_mapspec_indices():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, (output,))
    assert spec.output_indices == ("i", "j")


def test_mapspec_shape():
    inputs = (ArraySpec("a", ("i", "j")), ArraySpec("b", ("i", "j")))
    output = ArraySpec("q", ("i", "j"))
    spec = MapSpec(inputs, (output,))
    shapes = {"a": (3, 4), "b": (3, 4)}
    shape, mask = spec.shape(shapes)
    assert shape == (3, 4)

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


def test_mapspec_missing_index():
    with pytest.raises(ValueError, match="includes array indices in square brackets."):
        MapSpec.from_string("a -> b")


def test_mapspec_from_string():
    spec = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    assert isinstance(spec, MapSpec)
    assert len(spec.inputs) == 3
    assert spec.outputs[0] == ArraySpec("q", ("i", "j", "k"))

    with pytest.raises(ValueError, match="Expected expression of form"):
        MapSpec.from_string("a[i, j], b[i, j], c[k]")

    with pytest.raises(
        ValueError,
        match="All output arrays must have identical indices.",
    ):
        MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k], r[i]")

    multi_output = MapSpec.from_string("a[i, j], b[i, j] -> q[i, j], r[i, j]")
    assert len(multi_output.outputs) == 2


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


def test_expected_mask():
    mapspec = MapSpec.from_string("a[i], b[i, j] -> c[i, j]")

    # Test with numpy arrays
    inputs = {
        "a": np.array([1, 2, 3]),
        "b": np.array([[4, 5], [6, 7], [8, 9]]),
    }
    expected = np.array([[False, False], [False, False], [False, False]])
    assert np.array_equal(expected_mask(mapspec, inputs), expected)

    # Test with masked arrays
    inputs = {
        "a": np.ma.array([1, 2, 3], mask=[False, True, False]),
        "b": np.ma.array(
            [[4, 5], [6, 7], [8, 9]],
            mask=[[False, False], [True, True], [False, False]],
        ),
    }
    expected = np.array([[False, False], [True, True], [False, False]])
    assert np.array_equal(expected_mask(mapspec, inputs), expected)

    # Test with lists
    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j]")
    inputs = {
        "a": [0, 1, 2, 3],
        "b": [4, 5, 6],
    }
    expected = np.array(
        [
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
    )
    assert np.array_equal(expected_mask(mapspec, inputs), expected)


def test_array_mask():
    # Test with masked array
    arr = np.ma.array([1, 2, 3], mask=[False, True, False])
    assert np.array_equal(array_mask(arr), [False, True, False])

    # Test with unmasked array
    arr = np.array([1, 2, 3])
    assert np.array_equal(array_mask(arr), [False, False, False])

    # Test with list
    arr = [1, 2, 3]
    assert np.array_equal(array_mask(arr), [False, False, False])

    # Test with unsupported type
    with pytest.raises(TypeError, match="No array mask defined for type"):
        array_mask(42)


def test_array_shape():
    # Test with numpy array
    arr = np.array([[1, 2], [3, 4]])
    assert array_shape(arr) == (2, 2)

    # Test with list
    arr = [[1, 2], [3, 4]]
    assert array_shape(arr) == (2,)

    # Test with unsupported type
    with pytest.raises(TypeError, match="No array shape defined for type"):
        array_shape(42)


def test_mapspec_add_axes():
    spec = MapSpec.from_string("a[i], b[j] -> c[i, j]")
    new_spec = spec.add_axes("k")
    assert str(new_spec) == "a[i, k], b[j, k] -> c[i, j, k]"
    with pytest.raises(ValueError, match="Duplicate axes"):
        spec.add_axes("i")


def test_validate_consistent_axes():
    with pytest.raises(
        ValueError,
        match="All axes should have the same name at the same index",
    ):
        validate_consistent_axes(
            [
                MapSpec.from_string("a[i], b[i] -> f[i]"),
                MapSpec.from_string("f[k], g[k] -> h[k]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="All axes should have the same length",
    ):
        validate_consistent_axes(
            [
                MapSpec.from_string("a[i] -> f[i]"),
                MapSpec.from_string("a[i, j] -> g[i, j]"),
            ],
        )

    validate_consistent_axes(
        [
            MapSpec.from_string("a[i], b[j] -> f[i, j]"),
            MapSpec.from_string("f[i, j], c[k] -> g[i, j, k]"),
            MapSpec.from_string("g[i, j, k], d[l] -> h[i, j, k, l]"),
        ],
    )


def test_larger_output_then_input():
    mapspec = MapSpec.from_string("... -> b[j]")
    assert str(mapspec) == "... -> b[j]"
    assert mapspec.input_names == ()
    assert mapspec.output_names == ("b",)
    shape, mask = mapspec.shape(input_shapes={}, internal_shapes={"b": (3,)})
    assert shape == (3,)
    assert mask == (False,)

    mapspec = MapSpec.from_string("x[i, j], y[j, k] -> z[i, j, k, l]")
    assert mapspec.input_names == ("x", "y")
    assert mapspec.output_names == ("z",)
    input_shapes = {"x": (2, 3), "y": (3, 4)}
    shape, mask = mapspec.shape(input_shapes, internal_shapes={"z": (5,)})
    assert mask == (True, True, True, False)
    assert shape == (2, 3, 4, 5)

    mapspec = MapSpec.from_string("a[i] -> b[i, j]")
    shape, mask = mapspec.shape({"a": (3,)}, internal_shapes={"b": (4,)})
    assert mask == (True, False)
    assert shape == (3, 4)

    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j, k]")
    shape, mask = mapspec.shape(
        input_shapes={"a": (3,), "b": (4,)},
        internal_shapes={"c": (5,)},
    )
    assert mask == (True, True, False)
    assert shape == (3, 4, 5)

    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j, k, l]")
    input_shapes = {"a": (3,), "b": (4,)}
    shape, mask = mapspec.shape(input_shapes, internal_shapes={"c": (5, 6)})
    assert mask == (True, True, False, False)
    assert shape == (3, 4, 5, 6)

    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j]")
    shape, mask = mapspec.shape(
        input_shapes={"a": (3,), "b": (4,)},
        internal_shapes={"c": (...)},
    )
    assert mask == (True, True)
    assert shape == (3, 4)

    mapspec = MapSpec.from_string("a[i, j] -> b[i, j, k, l]")
    shape, mask = mapspec.shape(input_shapes={"a": (2, 3)}, internal_shapes={"b": (4, 5)})
    assert mask == (True, True, False, False)
    assert shape == (2, 3, 4, 5)

    mapspec = MapSpec.from_string("a[i, j, k] -> b[i, j, k, l, m]")
    input_shapes = {"a": (2, 3, 4)}
    shape, mask = mapspec.shape(input_shapes, internal_shapes={"b": (5, 6)})
    assert mask == (True, True, True, False, False)
    assert shape == (2, 3, 4, 5, 6)

    mapspec = MapSpec.from_string("a[j] -> b[i, j, k, l]")
    shape, mask = mapspec.shape(input_shapes={"a": (2,)}, internal_shapes={"b": (3, 4, 5)})
    assert mask == (False, True, False, False)
    assert shape == (3, 2, 4, 5)

    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j, k, l]")
    input_shapes = {"a": (3,), "b": (4,)}
    shape, mask = mapspec.shape(input_shapes, internal_shapes={"c": (5, 6)})
    assert mask == (True, True, False, False)
    assert shape == (3, 4, 5, 6)


def test_shape_exceptions():
    # Extra input arrays
    mapspec = MapSpec.from_string("a[i] -> b[i, j]")
    with pytest.raises(ValueError, match="Got extra array"):
        mapspec.shape({"a": (3,), "extra": (3,)}, internal_shapes={"b": (4,)})

    # Missing input arrays
    mapspec = MapSpec.from_string("a[i], b[j] -> c[i, j]")
    with pytest.raises(ValueError, match="Inputs expected by this map were not provided"):
        mapspec.shape({"a": (3,)}, internal_shapes={"c": (..., ...)})

    # Dimension mismatch
    mapspec = MapSpec.from_string("a[i, j], b[j, k] -> c[i, j, k]")
    with pytest.raises(ValueError, match="Dimension mismatch for arrays"):
        mapspec.shape({"a": (2, 3), "b": (4, 4)}, internal_shapes={"c": ()})

    # Missing output shapes for unshared axes
    mapspec = MapSpec.from_string("a[i, j] -> b[i, j, k]")
    with pytest.raises(ValueError, match="Internal shape for 'b' is too short."):
        mapspec.shape({"a": (2, 3)}, internal_shapes={"b": ()})

    # Output array not accepted by this map
    mapspec = MapSpec.from_string("a[i] -> b[i, j]")
    with pytest.raises(ValueError, match="Internal shape of `extra` is not accepted by this map."):
        mapspec.shape({"a": (3,)}, internal_shapes={"extra": (3, 4)})
