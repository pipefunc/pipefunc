from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt

from pipefunc._typing import is_type_compatible


def test_are_types_compatible_standard():
    assert is_type_compatible(list[int], list[int])
    assert not is_type_compatible(list[int], list[float])
    assert not is_type_compatible(int | str, Union[str, float])  # noqa: UP007
    assert not is_type_compatible(int | str, str | float)
    assert not is_type_compatible(int, float)
    assert not is_type_compatible(Any, int)
    assert is_type_compatible(int, Any)


def test_are_types_compatible_union():
    assert not is_type_compatible(int | str, str)
    assert not is_type_compatible(int | str, int)
    assert is_type_compatible(int, int | str)
    assert is_type_compatible(str, int | str)
    assert not is_type_compatible(int | str, float)
    assert is_type_compatible(dict[int, str], dict[int, str | int])


def test_are_types_compatible_numpy():
    assert is_type_compatible(npt.NDArray[np.int64], npt.NDArray[np.float64])
    assert not is_type_compatible(npt.NDArray[np.float32], npt.NDArray[np.int64])
    assert is_type_compatible(np.int32, np.int64)
    assert not is_type_compatible(np.float64, np.int32)
    assert is_type_compatible(npt.NDArray, npt.NDArray)
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[Any])


def test_are_types_compatible_standard_edge_cases():
    # Test with nested lists
    assert is_type_compatible(list[list[int]], list[list[int]])
    assert not is_type_compatible(list[list[int]], list[list[float]])
    assert is_type_compatible(list[list[int]], list[list[Any]])

    # Test with generic containers and Any
    assert not is_type_compatible(list[Any], list[int])
    assert is_type_compatible(list[int], list[Any])
    assert is_type_compatible(dict[int, str], dict[Any, Any])
    assert is_type_compatible(dict[int, str], dict[Any, Any])
    assert not is_type_compatible(dict[int, str], dict[int, float])
    assert not is_type_compatible(dict[int, str], list[int])  # Different container types


def test_are_types_compatible_union_edge_cases():
    # Test with more complex unions
    # Same elements, different order
    assert is_type_compatible(int | str | float, str | float | int)
    # Subset of a larger union
    assert is_type_compatible(int | str, int | str | float)
    # Completely different sets
    assert not is_type_compatible(int | str, float | complex)

    # Test with deeply nested unions
    assert is_type_compatible(Union[int | str, float], Union[str, int, float])  # noqa: UP007
    assert not is_type_compatible(Union[int | str, float], Union[float, complex])  # noqa: UP007

    # Test union with Any
    assert is_type_compatible(int, int | complex)
    assert is_type_compatible(complex, int | complex)
    assert is_type_compatible(int | str, Any)


def test_are_types_compatible_numpy_edge_cases():
    # Test NDArray with different type parameters
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[np.int32])
    assert not is_type_compatible(npt.NDArray[np.float64], npt.NDArray[np.int32])
    # Any in NDArray should be compatible
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[Any])

    # Test with numpy.generic (base class for all NumPy scalars)
    assert is_type_compatible(np.int64, np.generic)
    assert is_type_compatible(np.float64, np.generic)

    # Test scalar types compatibility
    # Can cast between NumPy integer types
    assert is_type_compatible(np.int64, np.int_)
    # Cannot cast from float to integer directly
    assert not is_type_compatible(np.float64, np.int32)


def test_directionality_union():
    # Test directional compatibility
    # Broader incoming type is not compatible with narrower required type
    assert not is_type_compatible(int | str, str)
    # Narrower incoming type is compatible with broader required type
    assert is_type_compatible(str, int | str)

    # Test specific type vs. Union
    # Compatible
    assert is_type_compatible(int, int | str)
    # Not compatible, broader incoming type isn't allowed
    assert not is_type_compatible(int | str, int)
