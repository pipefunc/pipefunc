import array
import importlib.util
from collections import Counter, OrderedDict, defaultdict, deque
from typing import Any

import numpy as np
import pytest

from pipefunc.cache import _HASH_MARKER, UnhashableError, _cloudpickle_key, to_hashable

has_pandas = importlib.util.find_spec("pandas") is not None

M = _HASH_MARKER


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({1: "a", 2: "b"}, (M, dict, ((1, "a"), (2, "b")))),
        (OrderedDict([(1, "a"), (2, "b")]), (M, OrderedDict, ((1, "a"), (2, "b")))),
        (
            defaultdict(int, {1: "a", 2: "b"}),  # type: ignore[arg-type]
            (M, defaultdict, (int, ((1, "a"), (2, "b")))),
        ),
        (Counter({1: 2, 3: 4}), (M, Counter, ((1, 2), (3, 4)))),
        ({1, 2, 3}, (M, set, (1, 2, 3))),
        (frozenset([1, 2, 3]), frozenset([1, 2, 3])),
        ([1, 2, 3], (M, list, (1, 2, 3))),
        ((1, 2, 3), (1, 2, 3)),
        (deque([1, 2, 3], maxlen=5), (M, deque, (5, (1, 2, 3)))),
        (array.array("i", [1, 2, 3]), (M, array.array, ("i", (1, 2, 3)))),
    ],
)
def test_to_hashable_basic_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_numpy_array() -> None:
    arr = np.array([[1, 2], [3, 4]])
    result = to_hashable(arr)
    assert isinstance(result, tuple)
    assert result[1] == np.ndarray
    # (shape, dtype, flattened array)
    assert result[2][0] == (2, 2)  # type: ignore[index]
    assert result[2][1] == "<i8"  # type: ignore[index]
    assert result[2][2] == (1, 2, 3, 4)  # type: ignore[index]


@pytest.mark.skipif(not has_pandas, reason="pandas not installed")
def test_to_hashable_pandas_series() -> None:
    import pandas as pd

    series = pd.Series([1, 2, 3], name="test")
    result = to_hashable(series)
    assert isinstance(result, tuple)
    assert result[0] == M
    assert result[1] == pd.Series
    assert result[2][0] == "test"  # type: ignore[index]
    assert result[2][1] == (M, dict, ((0, 1), (1, 2), (2, 3)))  # type: ignore[index]


@pytest.mark.skipif(not has_pandas, reason="pandas not installed")
def test_to_hashable_pandas_dataframe() -> None:
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = to_hashable(df)
    assert isinstance(result, tuple)
    assert result[0] == M
    assert result[1] == pd.DataFrame
    assert result[2] == (M, dict, (("A", (M, list, (1, 2))), ("B", (M, list, (3, 4)))))


def test_to_hashable_nested_structures() -> None:
    nested = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}}
    result = to_hashable(nested)
    expected = (
        M,
        dict,
        (("a", (M, list, (1, 2, (M, dict, (("b", (3, 4)),))))), ("c", (M, set, (5, 6)))),
    )
    assert result == expected


def test_to_hashable_unhashable_object() -> None:
    class Unhashable:
        def __hash__(self):
            msg = "unhashable type"
            raise TypeError(msg)

    obj = Unhashable()
    result = to_hashable(obj, fallback_to_pickle=True)
    assert result == (M, Unhashable, _cloudpickle_key(obj))


def test_to_hashable_unhashable_object_no_fallback() -> None:
    class Unhashable:
        def __hash__(self):
            msg = "unhashable type"
            raise TypeError(msg)

    obj = Unhashable()
    with pytest.raises(TypeError):
        to_hashable(obj, fallback_to_pickle=False)


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
        (bytearray(b"bytearray"), (M, bytearray, tuple(bytearray(b"bytearray")))),
    ],
)
def test_to_hashable_builtin_types(obj: Any, expected: Any) -> None:
    assert to_hashable(obj) == expected


def test_to_hashable_recursive_structure() -> None:
    lst: list[Any] = [1, 2]
    lst.append(lst)
    with pytest.raises(RecursionError):
        to_hashable(lst)


def test_hash_duplicates():
    x1 = (list, (1,))
    h1 = to_hashable(x1)
    x2 = [1]
    h2 = to_hashable(x2)
    assert h1 != h2
    assert hash(h1) != hash(h2)


def test_unhashable_type():
    # Test the case where a hash(type(obj)) AND hash(obj) doesn't work
    class Meta(type):
        def __hash__(cls):
            msg = "Not implemented"
            raise NotImplementedError(msg)

    class Unhashable(metaclass=Meta):
        def __hash__(self):
            msg = "Not implemented"
            raise NotImplementedError(msg)

    with pytest.raises(NotImplementedError, match="Not implemented"):
        hash(Unhashable)
    with pytest.raises(NotImplementedError, match="Not implemented"):
        hash(Unhashable())

    x = Unhashable()
    with pytest.raises(
        UnhashableError,
        match="cannot be hashed using `pipefunc.cache.to_hashable`",
    ):
        to_hashable(x, fallback_to_pickle=True)

    class UnhashableWithMeta(metaclass=Meta):  # only hash(type(obj)) works
        def __hash__(self) -> int:
            return 1

    x = UnhashableWithMeta()
    assert to_hashable(x) == x
