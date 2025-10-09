from __future__ import annotations

import importlib.util
import sys
import types

import cloudpickle
import numpy as np
import pytest

from pipefunc._utils import (
    _cached_load,
    equal_dicts,
    format_args,
    format_function_call,
    format_kwargs,
    get_ncores,
    handle_error,
    infer_shape,
    is_classmethod,
    is_equal,
    is_installed,
    is_min_version,
    is_pydantic_base_model,
    load,
    requires,
)

has_pydantic = importlib.util.find_spec("pydantic") is not None
has_polars = importlib.util.find_spec("polars") is not None


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


def test_get_ncores_slurm_executor(monkeypatch):
    fake_module = types.ModuleType("adaptive_scheduler")

    class FakeSlurmExecutor:
        """Minimal stand-in for adaptive_scheduler.SlurmExecutor."""

    fake_module.SlurmExecutor = FakeSlurmExecutor
    monkeypatch.setitem(sys.modules, "adaptive_scheduler", fake_module)

    assert get_ncores(FakeSlurmExecutor()) == 1


def test_format_args_empty() -> None:
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


class CustomObject:  # noqa: PLW1641
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, CustomObject):
            return self.value == other.value
        return False


def test_is_equal_dict() -> None:
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 2}
    assert is_equal(d1, d2)

    d3 = {"a": 1, "b": 3}
    assert not is_equal(d1, d3)

    assert not equal_dicts({"a": 1}, {"a": 1, "b": 3}, verbose=True)
    assert not equal_dicts({"a": 1}, {"b": 1}, verbose=True)
    assert not equal_dicts({"a": [1]}, {"b": (1,)}, verbose=True)


def test_is_equal_numpy_array() -> None:
    a1 = np.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    assert is_equal(a1, a2)

    a3 = np.array([1, 2, 4])
    assert not is_equal(a1, a3)

    a4 = np.array([1, 2, np.nan])
    a5 = np.array([1, 2, np.nan])
    assert is_equal(a4, a5)


def test_is_equal_set() -> None:
    s1 = {1, 2, 3}
    s2 = {1, 2, 3}
    assert is_equal(s1, s2)

    s3 = {1, 2, 4}
    assert not is_equal(s1, s3)


def test_is_equal_list_and_tuple() -> None:
    assert not is_equal([1, 2, 3], (1, 2, 3))
    assert is_equal([1, 2, 3], [1, 2, 3])
    assert not is_equal([1, 2, 3], [1, 2, 4])
    assert is_equal([np.array([1, 2, 3])], [np.array([1, 2, 3])])
    assert not is_equal([1], [1, 2])


def test_is_equal_float() -> None:
    assert is_equal(1.0, 1.0)
    assert is_equal(1.0, 1.0000000001)
    assert not is_equal(1.0, 1.1)


def test_is_equal_custom_object() -> None:
    obj1 = CustomObject(1)
    obj2 = CustomObject(1)
    assert is_equal(obj1, obj2)

    obj3 = CustomObject(2)
    assert not is_equal(obj1, obj3)


def test_is_equal_iterable() -> None:
    assert is_equal([1, 2, 3], [1, 2, 3])
    assert not is_equal([1, 2, 3], [1, 2, 4])
    assert is_equal((1, 2, 3), (1, 2, 3))
    assert not is_equal((1, 2, 3), (1, 2, 4))


def test_is_equal_other_types() -> None:
    assert is_equal(1, 1)
    assert not is_equal(1, 2)
    assert is_equal("abc", "abc")
    assert not is_equal("abc", "def")


def test_is_equal_nested_error() -> None:
    class ErrorOnEq:  # noqa: PLW1641
        def __eq__(self, other):
            msg = "Comparison error"
            raise ValueError(msg)

    obj1 = ErrorOnEq()
    obj2 = ErrorOnEq()

    assert is_equal([obj1], [obj2], on_error="return_none") is None
    with pytest.raises(ValueError, match="Comparison error"):
        is_equal([obj1], [obj2], on_error="raise")


def test_equal_dicts_with_pandas() -> None:
    """Test equal_dicts with pandas DataFrames."""
    if not is_installed("pandas"):
        pytest.skip("pandas not installed")
    import pandas as pd

    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df3 = pd.DataFrame({"x": [1, 2], "y": [3, 5]})
    d1 = {"a": df1}
    d2 = {"a": df2}
    d3 = {"a": df3}
    assert equal_dicts(d1, d2)
    assert not equal_dicts(d1, d3)


@pytest.mark.skipif(not has_polars, reason="polars not installed")
def test_equal_dicts_with_polars() -> None:
    """Test equal_dicts with polars DataFrames."""
    import polars as pl

    df1 = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    df3 = pl.DataFrame({"x": [1, 2], "y": [3, 5]})
    d1 = {"a": df1}
    d2 = {"a": df2}
    d3 = {"a": df3}
    assert equal_dicts(d1, d2)
    assert not equal_dicts(d1, d3)


@pytest.mark.skipif(not has_polars, reason="polars not installed")
def test_equal_dicts_with_polars_nulls() -> None:
    """Test equal_dicts with polars DataFrames containing null values."""
    import polars as pl

    # Test DataFrames with nulls
    df1 = pl.DataFrame({"x": [1, None, 3], "y": [None, 4, 5]})
    df2 = pl.DataFrame({"x": [1, None, 3], "y": [None, 4, 5]})
    df3 = pl.DataFrame({"x": [1, 2, 3], "y": [None, 4, 5]})
    d1 = {"a": df1}
    d2 = {"a": df2}
    d3 = {"a": df3}
    assert equal_dicts(d1, d2)
    assert not equal_dicts(d1, d3)

    # Test Series with nulls
    s1 = pl.Series("test", [1, None, 3])
    s2 = pl.Series("test", [1, None, 3])
    s3 = pl.Series("test", [1, 2, 3])
    d1 = {"a": s1}
    d2 = {"a": s2}
    d3 = {"a": s3}
    assert equal_dicts(d1, d2)
    assert not equal_dicts(d1, d3)


@pytest.mark.skipif(not has_polars, reason="polars not installed")
def test_equal_dicts_with_polars_different_dtypes() -> None:
    """Test equal_dicts with polars DataFrames with different dtypes."""
    import polars as pl

    # Same values but different dtypes should not be equal
    df1 = pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int64})
    df2 = pl.DataFrame({"x": [1, 2, 3]}, schema={"x": pl.Int32})
    d1 = {"a": df1}
    d2 = {"a": df2}
    assert not equal_dicts(d1, d2)

    # Same for Series
    s1 = pl.Series("test", [1, 2, 3], dtype=pl.Int64)
    s2 = pl.Series("test", [1, 2, 3], dtype=pl.Int32)
    d1 = {"a": s1}
    d2 = {"a": s2}
    assert not equal_dicts(d1, d2)


def test_equal_dicts() -> None:
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

    class A:  # noqa: PLW1641
        def __eq__(self, other):
            msg = "Error"
            raise RuntimeError(msg)

    with pytest.warns(Warning, match="Errors comparing keys"):
        assert equal_dicts({"a": A()}, {"a": A()}, verbose=True) is None
    with pytest.raises(RuntimeError, match="Error"):
        equal_dicts({"a": A()}, {"a": A()}, on_error="raise")


def test_requires() -> None:
    with pytest.raises(ImportError, match="package is required for"):
        requires("package_name_missing_for_sure", reason="testing", extras="test")


def test_is_min_version() -> None:
    # Basic version checks
    assert is_min_version("numpy", "1.0.0")
    assert not is_min_version("pipefunc", "999.0.0")

    major, minor, patch = map(int, np.__version__.split("."))  # noqa: RUF048

    assert is_min_version("numpy", f"{major}.{minor}.{patch - 1}")
    assert not is_min_version("numpy", f"{major}.{minor}.{patch + 1}")

    assert is_min_version("numpy", f"{major}.{minor - 1}.{patch}")
    assert not is_min_version("numpy", f"{major}.{minor + 1}.{patch}")

    assert is_min_version("numpy", f"{major - 1}.{minor}.{patch}")
    assert not is_min_version("numpy", f"{major + 1}.{minor}.{patch}")


def function_that_raises_empty_args() -> None:
    # Some exceptions can be raised with no arguments
    raise ValueError


def test_handle_error_empty_args() -> None:
    # We use a context manager to ensure the exception is raised
    with pytest.raises(ValueError) as exc_info:  # noqa: PT011, PT012
        try:
            function_that_raises_empty_args()
        except Exception as e:  # noqa: BLE001
            handle_error(e, function_that_raises_empty_args, {})

    # Get the actual error message from the exception
    error_message = str(exc_info.value)
    # The message should contain our added text even if original exception was empty
    msg = "Error occurred while executing function"
    assert msg in error_message or msg in exc_info.value.__notes__[0]
    func_name = "function_that_raises_empty_args"
    assert func_name in error_message or func_name in exc_info.value.__notes__[0]


def test_handle_error_with_args() -> None:
    original_message = "Original error message"
    with pytest.raises(ValueError) as exc_info:  # noqa: PT011, PT012
        try:
            raise ValueError(original_message)  # noqa: TRY301
        except Exception as e:  # noqa: BLE001
            handle_error(e, function_that_raises_empty_args, {})

    error_message = str(exc_info.value)
    assert original_message in error_message
    msg = "Error occurred while executing function"
    assert msg in error_message or msg in exc_info.value.__notes__[0]


@pytest.mark.skipif(not has_pydantic, reason="pydantic not installed")
def test_is_pydantic_base_model() -> None:
    from pydantic import BaseModel

    class CustomModel(BaseModel):
        pass

    class Foo:
        pass

    assert is_pydantic_base_model(BaseModel)
    assert is_pydantic_base_model(CustomModel)
    assert not is_pydantic_base_model(Foo)
    assert not is_pydantic_base_model(lambda x: x)


def test_is_classmethod() -> None:
    class Foo:
        def method(self):
            pass

        @classmethod
        def classmethod(cls):
            pass

        @staticmethod
        def staticmethod():
            pass

        @property
        def property(self):
            pass

        def __init__(self):
            pass

        class Nested:
            @classmethod
            def nested_classmethod(cls):
                pass

    assert not is_classmethod(Foo.method)
    assert is_classmethod(Foo.classmethod)
    assert not is_classmethod(Foo.staticmethod)
    assert not is_classmethod(Foo.property)
    assert not is_classmethod(Foo.__init__)
    assert is_classmethod(Foo.Nested.nested_classmethod)
    assert not is_classmethod(lambda x: x)
    assert not is_classmethod(1)  # type: ignore[arg-type]
    assert not is_classmethod("method")  # type: ignore[arg-type]


def test_infer_shape() -> None:
    """Test the infer_shape function with various inputs."""
    # Test with a simple list
    assert infer_shape([1, 2, 3]) == (3,)

    # Test with a nested list
    assert infer_shape([[1, 2], [3, 4]]) == (2, 2)
    assert infer_shape([[1], [2], [3]]) == (3, 1)
    assert infer_shape([[[1]], [[2]], [[3]]]) == (3, 1, 1)

    # Test with a ragged lists
    assert infer_shape([[[1], [2], [3]], [4, 5, 6]]) == (2, 3)
    assert infer_shape([[1, 2, 3], [[4], [5], [6]]]) == (2, 3)
    assert infer_shape([[[1]], [[2]], [3]]) == (3, 1)
    assert infer_shape([[1], [2, 3]]) == (2,)

    # Test with an empty list
    assert infer_shape([]) == (0,)

    # Test with a list of empty lists
    assert infer_shape([[], []]) == (2, 0)

    # Test with a NumPy array
    assert infer_shape(np.array([[1, 2], [3, 4]])) == (2, 2)

    # Test with a range object
    assert infer_shape(range(5)) == (5,)

    # Test with a scalar value
    assert infer_shape(123) == ()

    # Test with a string
    assert infer_shape("hello") == ()

    # Test with mixed list/tuple
    assert infer_shape(([1, 2], (3, 4))) == (2, 2)
    assert infer_shape(((1, 2), [3, 4])) == (2, 2)

    # Test with different sub-shapes
    assert infer_shape([[[1]], [[2, 3]]]) == (2, 1)
