from __future__ import annotations

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
    shape_to_strides,
    trace_dependencies,
    validate_consistent_axes,
)


def test_shape_to_strides():
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

    with pytest.raises(ValueError, match="Array name 'a#.a' is not a valid Python identifier"):
        ArraySpec("a#.a", ("i", "i"))

    spec = ArraySpec("foo.x", ("i",))
    assert spec.name == "foo.x"
    assert spec.axes == ("i",)


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


def test_array_mask():
    # Test with masked array
    arr = np.ma.MaskedArray([1, 2, 3], mask=[False, True, False])
    assert np.array_equal(array_mask(arr), [False, True, False])

    # Test with unmasked array
    arr = np.array([1, 2, 3])
    assert np.array_equal(array_mask(arr), [False, False, False])

    # Test with list
    arr = [1, 2, 3]
    assert np.array_equal(array_mask(arr), [False, False, False])

    # Test with range
    arr = range(1, 4)
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
    with pytest.raises(TypeError, match="No array shape defined for"):
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


def test_larger_output_then_input():  # noqa: PLR0915
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

    with pytest.raises(
        TypeError,
        match="Internal shape for 'c' must be a tuple of integers or '?'",
    ):
        mapspec.shape(input_shapes, internal_shapes={"c": (object(),)})


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


def test_trace_dependencies():
    # Test 1: Single input to single output
    mapspecs_1 = [
        MapSpec.from_string("a[i] -> y[i]"),
    ]
    deps_1 = trace_dependencies(mapspecs_1)
    assert deps_1 == {"y": {"a": ("i",)}}

    # Test 2: Multiple inputs to single output
    mapspecs_2 = [
        MapSpec.from_string("a[i], b[i] -> y[i]"),
    ]
    deps_2 = trace_dependencies(mapspecs_2)
    assert deps_2 == {"y": {"a": ("i",), "b": ("i",)}}

    # Test 3: Multiple inputs to multiple outputs
    mapspecs_3 = [
        MapSpec.from_string("a[i], b[j] -> y[i, j]"),
        MapSpec.from_string("a[i], y[i, j] -> z[i, j]"),
    ]
    deps_3 = trace_dependencies(mapspecs_3)
    assert deps_3 == {
        "y": {"a": ("i",), "b": ("j",)},
        "z": {"a": ("i",), "b": ("j",)},
    }

    # Test 4: Nested dependencies
    mapspecs_4 = [
        MapSpec.from_string("a[i] -> x[i]"),
        MapSpec.from_string("x[i] -> y[i]"),
        MapSpec.from_string("y[i] -> z[i]"),
    ]
    deps_4 = trace_dependencies(mapspecs_4)
    assert deps_4 == {
        "x": {"a": ("i",)},
        "y": {"a": ("i",)},
        "z": {"a": ("i",)},
    }

    # Test 5: Multiple axes
    mapspecs_5 = [
        MapSpec.from_string("a[i], b[j] -> y[i, j]"),
        MapSpec.from_string("y[i, j], c[k] -> z[i, j, k]"),
    ]
    deps_5 = trace_dependencies(mapspecs_5)
    assert deps_5 == {
        "y": {"a": ("i",), "b": ("j",)},
        "z": {"a": ("i",), "b": ("j",), "c": ("k",)},
    }

    # Test 6: Mixed dependencies
    mapspecs_6 = [
        MapSpec.from_string("a[i], b[j] -> x[i, j]"),
        MapSpec.from_string("x[i, j], c[k] -> y[i, j, k]"),
        MapSpec.from_string("y[i, :, k] -> z[k, i]"),
    ]
    deps_6 = trace_dependencies(mapspecs_6)
    assert deps_6 == {
        "x": {"a": ("i",), "b": ("j",)},
        "y": {"a": ("i",), "b": ("j",), "c": ("k",)},
        "z": {"a": ("i",), "c": ("k",)},
    }

    # Test 7: Zipped in different MapSpecs
    mapspecs_7 = [
        MapSpec.from_string("a[i], b[i] -> x[i]"),
        MapSpec.from_string("x[i], c[i] -> y[i]"),
    ]
    deps_7 = trace_dependencies(mapspecs_7)
    assert deps_7 == {
        "x": {"a": ("i",), "b": ("i",)},
        "y": {"a": ("i",), "b": ("i",), "c": ("i",)},
    }

    # Test 8: Zipped in different MapSpecs multi output
    mapspecs_8 = [
        MapSpec.from_string("a[i], b[i] -> x[i], unused[i]"),
        MapSpec.from_string("x[i], c[i] -> y[i]"),
    ]
    deps_8 = trace_dependencies(mapspecs_8)
    assert deps_8 == {
        "x": {"a": ("i",), "b": ("i",)},
        "unused": {"a": ("i",), "b": ("i",)},
        "y": {"a": ("i",), "b": ("i",), "c": ("i",)},
    }

    # Test 9: Single mapspec
    mapspecs_9 = [
        MapSpec.from_string("x[i] -> y[i]"),
    ]
    deps_9 = trace_dependencies(mapspecs_9)
    assert deps_9 == {"y": {"x": ("i",)}}

    # Test 10: MapSpec from step
    mapspecs_10 = [
        MapSpec.from_string("... -> x[i]"),
        MapSpec.from_string("x[i] -> y[i]"),
    ]
    deps_10 = trace_dependencies(mapspecs_10)
    assert deps_10 == {"y": {"x": ("i",)}}

    # Test 11: Internal shapes to 2D to 1D
    mapspecs_11 = [
        MapSpec.from_string("n[j] -> x[i, j]"),
        MapSpec.from_string("x[i, j] -> y[i, j]"),
        MapSpec.from_string("y[:, j] -> sum[j]"),
    ]
    deps_11 = trace_dependencies(mapspecs_11)
    assert deps_11 == {
        "x": {"n": ("j",)},
        "y": {"n": ("j",)},
        "sum": {"n": ("j",)},
    }


def test_mapspec_from_string_with_scope() -> None:
    # Valid cases
    valid_cases = [
        ("foo.a[i] -> foo.c[i]", ("foo.a",), ("foo.c",)),
        ("a[i, j] -> b[i, j]", ("a",), ("b",)),
        ("foo.bar[i] -> baz.qux[i]", ("foo.bar",), ("baz.qux",)),
        ("x[i] -> y[i]", ("x",), ("y",)),
        ("simple_name[index1] -> another_name[index1]", ("simple_name",), ("another_name",)),
    ]

    for expr, expected_inputs, expected_outputs in valid_cases:
        spec = MapSpec.from_string(expr)
        assert spec.input_names == expected_inputs, f"Failed on input: {expr}"
        assert spec.output_names == expected_outputs, f"Failed on output: {expr}"
