import array
from collections import Counter, OrderedDict, defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipefunc._cache import to_hashable


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({1: "a", 2: "b"}, (dict, ((1, "a"), (2, "b")))),
        (
            OrderedDict([(1, "a"), (2, "b")]),
            (OrderedDict, ((1, "a"), (2, "b"))),
        ),
        (
            defaultdict(int, {1: "a", 2: "b"}),  # type: ignore[arg-type]
            (defaultdict, (int, ((1, "a"), (2, "b")))),
        ),
        (Counter({1: 2, 3: 4}), (Counter, ((1, 2), (3, 4)))),
        ({1, 2, 3}, (set, (1, 2, 3))),
        (frozenset([1, 2, 3]), frozenset([1, 2, 3])),
        ([1, 2, 3], (list, (1, 2, 3))),
        ((1, 2, 3), (1, 2, 3)),
        (deque([1, 2, 3], maxlen=5), (deque, (5, (1, 2, 3)))),
        (array.array("i", [1, 2, 3]), (array.array, ("i", (1, 2, 3)))),
    ],
)
def test_to_hashable_basic_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_numpy_array() -> None:
    arr = np.array([[1, 2], [3, 4]])
    result = to_hashable(arr)
    assert isinstance(result, tuple)
    assert result[0] == np.ndarray
    # (shape, dtype, flattened array)
    assert result[1][0] == (2, 2)  # type: ignore[index]
    assert result[1][1] == "<i8"  # type: ignore[index]
    assert result[1][2] == (1, 2, 3, 4)  # type: ignore[index]


def test_to_hashable_pandas_series() -> None:
    series = pd.Series([1, 2, 3], name="test")
    result = to_hashable(series)
    assert isinstance(result, tuple)
    assert result[0] == pd.Series
    assert result[1][0] == "test"  # type: ignore[index]
    assert result[1][1] == (dict, ((0, 1), (1, 2), (2, 3)))  # type: ignore[index]


def test_to_hashable_pandas_dataframe() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = to_hashable(df)
    assert isinstance(result, tuple)
    assert result[0] == pd.DataFrame
    assert result[1] == (
        dict,
        (
            ("A", (list, (1, 2))),
            ("B", (list, (3, 4))),
        ),
    )


def test_to_hashable_nested_structures() -> None:
    nested = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}}
    result = to_hashable(nested)
    expected = (
        dict,
        (
            ("a", (list, (1, 2, (dict, (("b", (3, 4)),))))),
            ("c", (set, (5, 6))),
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
    assert result == (Unhashable, str(obj))


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
        (bytearray(b"bytearray"), (bytearray, tuple(bytearray(b"bytearray")))),
    ],
)
def test_to_hashable_builtin_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_recursive_structure() -> None:
    lst: list[Any] = [1, 2]
    lst.append(lst)
    with pytest.raises(RecursionError):
        to_hashable(lst)
