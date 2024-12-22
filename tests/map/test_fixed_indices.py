"""Tests for the _mask_fixed_axes function."""

import numpy as np
import pytest

from pipefunc.map._mapspec import ArraySpec, MapSpec
from pipefunc.map._run import _mask_fixed_axes


def test_no_fixed_indices() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i",)),), (ArraySpec("b", ("i",)),))
    shape = (5,)
    shape_mask = (True,)
    result = _mask_fixed_axes(None, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert result is None


def test_fixed_index_resolved_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i",)),), (ArraySpec("b", ("i",)),))
    shape = (5,)
    shape_mask = (True,)
    fixed_indices = {"i": 2}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    assert list(result) == [False, False, True, False, False]


def test_fixed_slice_resolved_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i",)),), (ArraySpec("b", ("i",)),))
    shape = (5,)
    shape_mask = (True,)
    fixed_indices = {"i": slice(1, 4)}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    assert list(result) == [False, True, True, True, False]


def test_fixed_index_unresolved_shape_allowed() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, "?")
    shape_mask = (True, False)
    fixed_indices = {"i": 2}  # Fixed index on a resolved dimension
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    # Expecting a flat array of [False, False, True, False, False]
    # The exact output will depend on how you handle unresolved shapes later,
    # but the function should not raise an error.


def test_fixed_index_unresolved_shape_not_allowed() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, "?")
    shape_mask = (False, True)
    fixed_indices = {"j": 3}  # Fixed index on an unresolved dimension
    with pytest.raises(
        ValueError,
        match="Cannot mask fixed axes when unresolved dimensions are present in the external shape",
    ):
        _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]


def test_unresolved_shape_no_fixed_indices_allowed() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, "?")
    shape_mask = (True, False)
    result = _mask_fixed_axes(None, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert result is None  # Should be None when no fixed_indices


def test_multiple_fixed_indices_resolved_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, 10)
    shape_mask = (True, True)
    fixed_indices = {"i": 2, "j": 4}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    expected = [False] * 50
    expected[2 * 10 + 4] = True  # Index (2, 4) in a (5, 10) array
    assert list(result) == expected


def test_mixed_fixed_indices_and_slice_resolved_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, 10)
    shape_mask = (True, True)
    fixed_indices = {"i": slice(1, 3), "j": 4}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    expected = [False] * 50
    expected[1 * 10 + 4] = True  # Index (1, 4) in a (5, 10) array
    expected[2 * 10 + 4] = True  # Index (2, 4) in a (5, 10) array
    assert list(result) == expected


def test_fixed_index_with_internal_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", None)),), (ArraySpec("b", ("i",)),))
    shape = (5, "?")
    shape_mask = (True, False)
    fixed_indices = {"i": 2}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    assert list(result) == [False, False, True, False, False]


def test_fixed_index_none_slice_resolved_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", "j")),), (ArraySpec("b", ("i", "j")),))
    shape = (5, 10)
    shape_mask = (True, True)
    fixed_indices = {"i": 2, "j": slice(None)}  # Explicit slice(None)
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    expected = [False] * 50
    for j in range(10):
        expected[2 * 10 + j] = True  # All j indices are True when i is 2
    assert list(result) == expected


def test_fixed_index_with_2d_internal_shape() -> None:
    mapspec = MapSpec((ArraySpec("a", ("i", None, None)),), (ArraySpec("b", ("i",)),))
    shape = (5, "?", "?")
    shape_mask = (True, False, False)
    fixed_indices = {"i": 2}
    result = _mask_fixed_axes(fixed_indices, mapspec, shape, shape_mask)  # type: ignore[arg-type]
    assert isinstance(result, np.flatiter)
    assert list(result) == [False, False, True, False, False]
