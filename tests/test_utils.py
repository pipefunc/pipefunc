from __future__ import annotations

import cloudpickle
import numpy as np
import pytest

from pipefunc._utils import (
    _cached_load,
    _is_equal,
    equal_dicts,
    format_args,
    format_function_call,
    format_kwargs,
    load,
)


@pytest.fixture(autouse=True)  # Automatically use in all tests
def _clear_cache() -> None:
    _cached_load.cache_clear()  # Clear the cache before each test


def create_temp_file(tmp_path, data, filename="test.pickle"):
    file_path = tmp_path / filename
    with file_path.open("wb") as f:
        cloudpickle.dump(data, f)
    return file_path


def test_cached_load_hits_cache(tmp_path):
    assert _cached_load.cache_info().hits == 0
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # First load to populate the cache
    result1 = load(file_path, cache=True)
    # Second load should hit the cache
    result2 = load(file_path, cache=True)

    # Check that the function returns the correct data
    assert result1 == data
    assert result1 == result2
    # Ensure the file was not loaded again from disk
    assert _cached_load.cache_info().hits == 1


def test_cached_load_invalidates_on_file_change(tmp_path):
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # First load to populate the cache
    result1 = load(file_path, cache=True)
    assert _cached_load.cache_info().misses == 1

    # Modify the file
    with file_path.open("wb") as f:
        data_modified = {"key": "new value"}
        cloudpickle.dump(data_modified, f)

    # Second load should not hit the cache due to file modification
    result2 = load(file_path, cache=True)

    # Check that the results are correct
    assert result1 == {"key": "value"}
    assert result2 == {"key": "new value"}

    # Ensure the file was loaded again from disk
    assert _cached_load.cache_info().misses == 2
    assert _cached_load.cache_info().hits == 0


@pytest.mark.parametrize("modify_size", [True, False])
def test_cache_invalidation_on_file_size_change(tmp_path, modify_size):
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # Load once to cache
    result1 = load(file_path, cache=True)

    # Modify file and optionally change its size
    with file_path.open("wb") as f:
        if modify_size:
            data_modified = {"key": "new value", "extra": "data to increase size"}
        else:
            data_modified = {"key": "new value"}
        cloudpickle.dump(data_modified, f)

    # Second load should reload because of file size change or content change
    result2 = load(file_path, cache=True)

    # Ensure data is as expected
    assert result1 != result2
    # Check cache was invalidated (i.e., miss occurred)
    assert _cached_load.cache_info().misses == 2


def test_format_args_empty():
    assert format_args(()) == ""
    assert format_args((42,)) == "42"
    assert format_args((42, "hello", [1, 2, 3])) == "42, 'hello', [1, 2, 3]"
    assert format_kwargs({}) == ""
    assert format_kwargs({"a": 1}) == "a=1"
    assert format_kwargs({"a": 1, "b": "test", "c": [1, 2, 3]}) == "a=1, b='test', c=[1, 2, 3]"
    assert format_function_call("func", (), {}) == "func()"
    assert format_function_call("func", (1, 2), {}) == "func(1, 2)"
    assert format_function_call("func", (), {"x": 1, "y": "test"}) == "func(x=1, y='test')"
    assert (
        format_function_call("func", (1, 2), {"x": 1, "y": "test"}) == "func(1, 2, x=1, y='test')"
    )
    args = ("foo",)
    kwargs = {"func2": "func2(arg)"}
    assert format_function_call("func1", args, kwargs) == "func1('foo', func2='func2(arg)')"


class CustomObject:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, CustomObject):
            return self.value == other.value
        return False


def test_is_equal_dict():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 2}
    assert _is_equal(d1, d2)

    d3 = {"a": 1, "b": 3}
    assert not _is_equal(d1, d3)

    assert not equal_dicts({"a": 1}, {"a": 1, "b": 3}, verbose=True)
    assert not equal_dicts({"a": 1}, {"b": 1}, verbose=True)
    assert not equal_dicts({"a": [1]}, {"b": (1,)}, verbose=True)


def test_is_equal_numpy_array():
    a1 = np.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    assert _is_equal(a1, a2)

    a3 = np.array([1, 2, 4])
    assert not _is_equal(a1, a3)

    a4 = np.array([1, 2, np.nan])
    a5 = np.array([1, 2, np.nan])
    assert _is_equal(a4, a5)


def test_is_equal_set():
    s1 = {1, 2, 3}
    s2 = {1, 2, 3}
    assert _is_equal(s1, s2)

    s3 = {1, 2, 4}
    assert not _is_equal(s1, s3)


def test_is_equal_list_and_tuple():
    assert not _is_equal([1, 2, 3], (1, 2, 3))
    assert _is_equal([1, 2, 3], [1, 2, 3])
    assert not _is_equal([1, 2, 3], [1, 2, 4])
    assert _is_equal([np.array([1, 2, 3])], [np.array([1, 2, 3])])
    assert not _is_equal([1], [1, 2])


def test_is_equal_float():
    assert _is_equal(1.0, 1.0)
    assert _is_equal(1.0, 1.0000000001)
    assert not _is_equal(1.0, 1.1)


def test_is_equal_custom_object():
    obj1 = CustomObject(1)
    obj2 = CustomObject(1)
    assert _is_equal(obj1, obj2)

    obj3 = CustomObject(2)
    assert not _is_equal(obj1, obj3)


def test_is_equal_iterable():
    assert _is_equal([1, 2, 3], [1, 2, 3])
    assert not _is_equal([1, 2, 3], [1, 2, 4])
    assert _is_equal((1, 2, 3), (1, 2, 3))
    assert not _is_equal((1, 2, 3), (1, 2, 4))


def test_is_equal_other_types():
    assert _is_equal(1, 1)
    assert not _is_equal(1, 2)
    assert _is_equal("abc", "abc")
    assert not _is_equal("abc", "def")


def test_equal_dicts():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 2}
    assert equal_dicts(d1, d2, verbose=True)

    d3 = {"a": 1, "b": 3}
    assert not equal_dicts(d1, d3, verbose=True)

    d4 = {"a": 1, "b": {"c": 2}}
    d5 = {"a": 1, "b": {"c": 2}}
    assert equal_dicts(d4, d5, verbose=True)

    d6 = {"a": 1, "b": {"c": 3}}
    assert not equal_dicts(d4, d6, verbose=True)

    # test with numpy arrays
    d7 = {"a": np.array([1, 2, 3])}
    d8 = {"a": np.array([1, 2, 3])}
    assert equal_dicts(d7, d8, verbose=True)

    d9 = {"a": {"a": np.array([1, 2, 3])}}
    d10 = {"a": {"a": np.array([1, 2, 3])}}
    assert equal_dicts(d9, d10, verbose=True)

    d11 = {"a": {"a": np.array([1, 2, 3])}}
    d12 = {"a": {"a": np.array([1, 2, 4])}}
    assert not equal_dicts(d11, d12, verbose=True)

    class A:
        def __eq__(self, other):
            msg = "Error"
            raise RuntimeError(msg)

    with pytest.warns(Warning, match="Errors comparing keys"):
        assert equal_dicts({"a": A()}, {"a": A()}, verbose=True) is None
