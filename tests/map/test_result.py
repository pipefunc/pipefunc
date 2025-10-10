import importlib.util
from typing import Any

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._result import ResultDict

has_xarray = importlib.util.find_spec("xarray") is not None


class LongRepresentation:
    def __repr__(self) -> str:
        return "a" * 50_000


def test_truncating_result_dict_repr() -> None:
    result = ResultDict()
    assert repr(result) == "{}"
    result["a"] = LongRepresentation()  # type: ignore[assignment]
    with pytest.warns(UserWarning, match="ResultDict is too large to display completely"):
        text = repr(result)
    assert text.startswith(
        "{'a': aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )


def test_xarray() -> None:
    result = ResultDict()
    with pytest.raises(ValueError, match="method can only be used when"):
        result.to_xarray()


@pytest.mark.parametrize(
    ("type_", "numpy_type"),
    [
        (int, np.int_),
        (float, np.float64),
        (bool, np.bool_),
        (np.int32, np.int32),
        (np.int64, np.int64),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (list, object),
    ],
)
def test_type_cast(type_: type, numpy_type: type) -> None:
    @pipefunc(output_name="y", mapspec="x[int] -> y[int]")
    def double(x: type_) -> type_:  # type: ignore[valid-type]
        return x

    @pipefunc(output_name="other")
    def other() -> int:
        return 1

    pipeline = Pipeline([double, other])
    x: Any
    if numpy_type is np.bool_:
        x = [True, False, True]
    elif numpy_type is list:
        x = [[1, 2], [3, 4], [5, 6]]
    else:
        x = np.arange(10, dtype=numpy_type)
    result = pipeline.map({"x": x}, parallel=False, storage="dict")
    assert result["y"].output.dtype == object
    new = result.type_cast(inplace=False)
    assert result["y"].output.dtype == object
    assert new["y"].output.dtype == numpy_type
    assert new._pipeline is result._pipeline
    if has_xarray:
        ds = result.to_xarray(type_cast=True)
        assert ds["y"].dtype == numpy_type
        ds = result.to_xarray(type_cast=False)
        assert ds["y"].dtype == object
        # Check that the original result is not modified
        assert result["y"].output.dtype == object


def test_type_cast_with_incorrect_annotation() -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        return (x,)  # type: ignore[return-value]

    pipeline = Pipeline([double])
    x = np.arange(10, dtype=np.int_)
    result = pipeline.map({"x": x}, parallel=False, storage="dict")
    assert result["y"].output.dtype == object
    with pytest.warns(UserWarning, match="Could not cast output 'y' to <class 'int'> due to error"):
        result = result.type_cast()
    assert result["y"].output.dtype == object
