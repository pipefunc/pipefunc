import array
import collections
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipefunc._cache import _Type, to_hashable


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({1: "a", 2: "b"}, (_Type("dict"), ((1, "a"), (2, "b")))),
        (
            collections.OrderedDict([(1, "a"), (2, "b")]),
            (_Type("OrderedDict"), ((1, "a"), (2, "b"))),
        ),
        (
            collections.defaultdict(int, {1: "a", 2: "b"}),  # type: ignore[arg-type]
            (_Type("defaultdict"), (int, ((1, "a"), (2, "b")))),
        ),
        (collections.Counter({1: 2, 3: 4}), (_Type("Counter"), ((1, 2), (3, 4)))),
        ({1, 2, 3}, (_Type("set"), (1, 2, 3))),
        (frozenset([1, 2, 3]), frozenset([1, 2, 3])),
        ([1, 2, 3], (_Type("list"), (1, 2, 3))),
        ((1, 2, 3), (1, 2, 3)),
        (collections.deque([1, 2, 3], maxlen=5), (_Type("deque"), (5, (1, 2, 3)))),
        (array.array("i", [1, 2, 3]), (_Type("array"), ("i", (1, 2, 3)))),
    ],
)
def test_to_hashable_basic_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_numpy_array() -> None:
    arr = np.array([[1, 2], [3, 4]])
    result = to_hashable(arr)
    assert isinstance(result, tuple)
    assert result[0] == _Type("ndarray")
    # (shape, dtype, flattened array)
    assert result[1][0] == (2, 2)  # type: ignore[index]
    assert result[1][1] == "<i8"  # type: ignore[index]
    assert result[1][2] == (1, 2, 3, 4)  # type: ignore[index]


def test_to_hashable_pandas_series() -> None:
    series = pd.Series([1, 2, 3], name="test")
    result = to_hashable(series)
    assert isinstance(result, tuple)
    assert result[0] == _Type("Series")
    assert result[1][0] == "test"  # type: ignore[index]
    assert result[1][1] == (_Type("dict"), ((0, 1), (1, 2), (2, 3)))  # type: ignore[index]


def test_to_hashable_pandas_dataframe() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = to_hashable(df)
    assert isinstance(result, tuple)
    assert result[0] == _Type("DataFrame")
    assert result[1] == (
        _Type("dict"),
        (
            ("A", (_Type("list"), (1, 2))),
            ("B", (_Type("list"), (3, 4))),
        ),
    )


def test_to_hashable_nested_structures() -> None:
    nested = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}}
    result = to_hashable(nested)
    expected = (
        _Type("dict"),
        (
            ("a", (_Type("list"), (1, 2, (_Type("dict"), (("b", (3, 4)),))))),
            ("c", (_Type("set"), (5, 6))),
        ),
    )
    assert result == expected


def test_to_hashable_unhashable_object() -> None:
    class Unhashable:
        def __hash__(self):
            msg = "unhashable type"
            raise TypeError(msg)

    obj = Unhashable()
    result = to_hashable(obj)
    assert result == (_Type("unhashable"), str(obj))


def test_to_hashable_unhashable_object_no_fallback() -> None:
    class Unhashable:
        def __hash__(self):
            msg = "unhashable type"
            raise TypeError(msg)

    obj = Unhashable()
    with pytest.raises(TypeError):
        to_hashable(obj, fallback_to_str=False)


def test_to_hashable_custom_hashable_object() -> None:
    class CustomHashable:
        def __hash__(self):
            return 42

    obj = CustomHashable()
    result = to_hashable(obj)
    assert result == obj


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        (42, 42),
        (3.14, 3.14),
        ("hello", "hello"),
        (complex(1, 2), complex(1, 2)),
        (b"bytes", b"bytes"),
        (bytearray(b"bytearray"), (_Type("bytearray"), tuple(bytearray(b"bytearray")))),
    ],
)
def test_to_hashable_builtin_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_recursive_structure() -> None:
    lst: list[Any] = [1, 2]
    lst.append(lst)
    with pytest.raises(RecursionError):
        to_hashable(lst)
