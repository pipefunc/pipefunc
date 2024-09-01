import array
import collections
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipefunc._cache import to_hashable


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({1: "a", 2: "b"}, ("dict", ((1, ("hashable", hash("a"))), (2, ("hashable", hash("b")))))),
        (
            collections.OrderedDict([(1, "a"), (2, "b")]),
            ("OrderedDict", ((1, ("hashable", hash("a"))), (2, ("hashable", hash("b"))))),
        ),
        (
            collections.defaultdict(int, {1: "a", 2: "b"}),
            (
                "defaultdict",
                (
                    ("hashable", hash(int)),
                    ((1, ("hashable", hash("a"))), (2, ("hashable", hash("b")))),
                ),
            ),
        ),
        (
            collections.Counter({1: 2, 3: 4}),
            ("Counter", ((1, 2), (3, 4))),
        ),
        (
            {1, 2, 3},
            ("set", (("hashable", hash(1)), ("hashable", hash(2)), ("hashable", hash(3)))),
        ),
        (
            frozenset([1, 2, 3]),
            ("frozenset", (("hashable", hash(1)), ("hashable", hash(2)), ("hashable", hash(3)))),
        ),
        (
            [1, 2, 3],
            ("list", (("hashable", hash(1)), ("hashable", hash(2)), ("hashable", hash(3)))),
        ),
        (
            (1, 2, 3),
            ("tuple", (("hashable", hash(1)), ("hashable", hash(2)), ("hashable", hash(3)))),
        ),
        (
            collections.deque([1, 2, 3], maxlen=5),
            ("deque", (5, (("hashable", hash(1)), ("hashable", hash(2)), ("hashable", hash(3))))),
        ),
        (array.array("i", [1, 2, 3]), ("array", ("i", (1, 2, 3)))),
    ],
)
def test_to_hashable_basic_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_numpy_array() -> None:
    arr = np.array([[1, 2], [3, 4]])
    result = to_hashable(arr)
    assert result[0] == "ndarray"
    assert result[1][0] == (2, 2)  # shape
    assert result[1][1] == "<i8"  # dtype
    assert result[1][2] == (1, 2, 3, 4)  # flattened array


def test_to_hashable_pandas_series() -> None:
    series = pd.Series([1, 2, 3], name="test")
    result = to_hashable(series)
    assert result[0] == "Series"
    assert result[1][0] == "test"  # name
    assert result[1][1] == (
        "dict",
        ((0, ("hashable", hash(1))), (1, ("hashable", hash(2))), (2, ("hashable", hash(3)))),
    )


def test_to_hashable_pandas_dataframe() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = to_hashable(df)
    assert result[0] == "DataFrame"
    assert result[1] == (
        "dict",
        (
            ("A", ("list", (("hashable", hash(1)), ("hashable", hash(2))))),
            ("B", ("list", (("hashable", hash(3)), ("hashable", hash(4))))),
        ),
    )


def test_to_hashable_nested_structures() -> None:
    nested = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}}
    result = to_hashable(nested)
    expected = (
        "dict",
        (
            (
                "a",
                (
                    "list",
                    (
                        ("hashable", hash(1)),
                        ("hashable", hash(2)),
                        (
                            "dict",
                            (("b", ("tuple", (("hashable", hash(3)), ("hashable", hash(4))))),),
                        ),
                    ),
                ),
            ),
            ("c", ("set", (("hashable", hash(5)), ("hashable", hash(6))))),
        ),
    )
    assert result == expected


def test_to_hashable_unhashable_object() -> None:
    class Unhashable:
        def __hash__(self):
            raise TypeError("unhashable type")

    obj = Unhashable()
    result = to_hashable(obj)
    assert result == ("unhashable", str(obj))


def test_to_hashable_unhashable_object_no_fallback() -> None:
    class Unhashable:
        def __hash__(self):
            raise TypeError("unhashable type")

    obj = Unhashable()
    with pytest.raises(TypeError):
        to_hashable(obj, fallback_to_str=False)


def test_to_hashable_custom_hashable_object() -> None:
    class CustomHashable:
        def __hash__(self):
            return 42

    obj = CustomHashable()
    result = to_hashable(obj)
    assert result == ("hashable", 42)


@pytest.mark.parametrize(
    "obj, expected",
    [
        (None, ("hashable", hash(None))),
        (True, ("hashable", hash(True))),  # noqa: FBT003
        (False, ("hashable", hash(False))),  # noqa: FBT003
        (42, ("hashable", hash(42))),
        (3.14, ("hashable", hash(3.14))),
        ("hello", ("hashable", hash("hello"))),
        (complex(1, 2), ("hashable", hash(complex(1, 2)))),
        (b"bytes", ("hashable", hash(b"bytes"))),
        (bytearray(b"bytearray"), ("bytearray", tuple(bytearray(b"bytearray")))),
    ],
)
def test_to_hashable_builtin_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_recursive_structure() -> None:
    lst = [1, 2]
    lst.append(lst)
    with pytest.raises(RecursionError):
        to_hashable(lst)
