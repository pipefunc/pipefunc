from typing import Any, Union

import numpy as np
import numpy.typing as npt

from pipefunc._typing import are_types_compatible


def test_are_types_compatible_standard():
    assert are_types_compatible(list[int], list[int])
    assert not are_types_compatible(list[int], list[float])
    assert are_types_compatible(Union[int, str], Union[str, float])
    assert are_types_compatible(Union[int, str], Union[str, float])
    assert not are_types_compatible(int, float)
    assert are_types_compatible(Any, int)
    assert are_types_compatible(int, Any)


def test_are_types_compatible_union():
    assert are_types_compatible(Union[int, str], str)
    assert are_types_compatible(int, Union[int, str])
    assert not are_types_compatible(Union[int, str], float)
    assert are_types_compatible(dict[int, str | int], dict[int, str])


def test_are_types_compatible_numpy():
    assert are_types_compatible(npt.NDArray[np.int64], npt.NDArray[np.float64])
    assert not are_types_compatible(npt.NDArray[np.float32], npt.NDArray[np.int64])
    assert are_types_compatible(np.int32, np.int64)
    assert not are_types_compatible(np.float64, np.int32)
    assert are_types_compatible(npt.NDArray, npt.NDArray)
    assert are_types_compatible(npt.NDArray[Any], npt.NDArray[np.int32])


def test_are_types_compatible_standard_edge_cases():
    # Test with nested lists
    assert are_types_compatible(list[list[int]], list[list[int]])
    assert not are_types_compatible(list[list[int]], list[list[float]])
    assert are_types_compatible(list[list[Any]], list[list[int]])

    # Test with generic containers and Any
    assert are_types_compatible(list[Any], list[int])
    assert are_types_compatible(list[int], list[Any])
    assert are_types_compatible(dict[Any, Any], dict[int, str])
    assert not are_types_compatible(dict[int, str], dict[int, float])
    assert not are_types_compatible(dict[int, str], list[int])  # Different container types


def test_are_types_compatible_union_edge_cases():
    # Test with more complex unions
    # Same elements, different order
    assert are_types_compatible(Union[int, str, float], Union[str, float, int])
    # Subset of a larger union
    assert are_types_compatible(Union[int, str], Union[int, str, float])
    # Completely different sets
    assert not are_types_compatible(Union[int, str], Union[float, complex])

    # Test with deeply nested unions
    assert are_types_compatible(Union[int | str, float], Union[str, int, float])
    assert not are_types_compatible(Union[int | str, float], Union[float, complex])

    # Test union with Any
    assert are_types_compatible(Union[int, Any], int)
    assert are_types_compatible(Union[int, Any], float)
    assert are_types_compatible(Any, Union[int, str])


def test_are_types_compatible_numpy_edge_cases():
    # Test NDArray with different type parameters
    assert are_types_compatible(npt.NDArray[np.int32], npt.NDArray[np.int32])
    assert not are_types_compatible(npt.NDArray[np.float64], npt.NDArray[np.int32])
    # Any in NDArray should be compatible
    assert are_types_compatible(npt.NDArray[np.int32], npt.NDArray[Any])

    # Test NDArray with ArrayLike
    assert are_types_compatible(npt.NDArray, npt.ArrayLike)
    assert are_types_compatible(npt.ArrayLike, npt.NDArray[np.int64])
    # ArrayLike should be compatible with itself
    assert are_types_compatible(npt.ArrayLike, npt.ArrayLike)

    # Test with numpy.generic (base class for all NumPy scalars)
    assert are_types_compatible(np.generic, np.int64)
    assert are_types_compatible(np.generic, np.float64)

    # Test scalar types compatibility
    # Can cast between NumPy integer types
    assert are_types_compatible(np.int64, np.int32)
    # Cannot cast from float to integer directly
    assert not are_types_compatible(np.float64, np.int32)
