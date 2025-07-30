"""Basic tests for error handling that can run without PropagatedErrorSnapshot."""

from __future__ import annotations

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot


# Test 1: Single function with error_handling="continue"
def test_single_function_error_continue():
    """Test that a single function can store errors when error_handling='continue'."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # Test with error_handling="continue"
    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue", parallel=False)

    y_output = result["y"].output
    assert isinstance(y_output, np.ndarray)
    assert y_output.dtype == object  # Should be object array to hold mixed types

    # Check individual elements
    assert y_output[0] == 2  # 1 * 2
    assert y_output[1] == 4  # 2 * 2
    assert isinstance(y_output[2], ErrorSnapshot)  # Failed on x=3
    assert "Cannot process x=3" in str(y_output[2].exception)
    assert y_output[3] == 8  # 4 * 2
    assert y_output[4] == 10  # 5 * 2


# Test 2: Error handling with parallel=False
def test_sequential_execution_error_continue():
    """Test error handling works in sequential mode."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # Test with parallel=False
    result = pipeline.map(
        {"x": [1, 2, 3, 4, 5]},
        error_handling="continue",
        parallel=False,
    )

    # Should still process all elements despite error
    y_output = result["y"].output
    assert y_output[0] == 2
    assert y_output[1] == 4
    assert isinstance(y_output[2], ErrorSnapshot)
    assert y_output[3] == 8
    assert y_output[4] == 10


# Test 3: Multiple errors in same mapspec function
def test_multiple_errors_single_function():
    """Test handling multiple errors in the same function."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail_multiple(x: int) -> int:
        if x in [2, 4, 7]:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x**2

    pipeline = Pipeline([may_fail_multiple])

    result = pipeline.map({"x": list(range(1, 9))}, error_handling="continue", parallel=False)

    # Check y has errors at correct positions
    y = result["y"].output
    assert y[0] == 1  # 1^2
    assert isinstance(y[1], ErrorSnapshot)  # x=2 failed
    assert y[2] == 9  # 3^2
    assert isinstance(y[3], ErrorSnapshot)  # x=4 failed
    assert y[4] == 25  # 5^2
    assert y[5] == 36  # 6^2
    assert isinstance(y[6], ErrorSnapshot)  # x=7 failed
    assert y[7] == 64  # 8^2


# Test 4: Verify error_handling="raise" still works (default behavior)
def test_error_handling_raise_default():
    """Test that error_handling='raise' maintains default behavior."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def will_fail(x: int) -> int:
        if x == 3:
            msg = "Expected error"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([will_fail])

    # Should raise exception with default error_handling
    with pytest.raises(ValueError, match="Expected error"):
        pipeline.map({"x": [1, 2, 3, 4, 5]}, parallel=False)  # error_handling="raise" is default

    # Explicit error_handling="raise" should also raise
    with pytest.raises(ValueError, match="Expected error"):
        pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="raise", parallel=False)


# Test 5: Test 2D mapspec with errors
def test_2d_mapspec_with_errors():
    """Test error handling in 2D mapspec operations."""

    @pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
    def compute(x: int, y: int) -> int:
        if x == 2 and y == 3:
            msg = f"Cannot compute x={x}, y={y}"
            raise ValueError(msg)
        return x * y

    pipeline = Pipeline([compute])

    result = pipeline.map(
        {"x": [1, 2, 3], "y": [2, 3, 4]},
        error_handling="continue",
        parallel=False,
    )

    # Check matrix
    matrix = result["matrix"].output
    assert matrix.shape == (3, 3)
    assert matrix.dtype == object

    # Check values
    assert matrix[0, 0] == 2  # 1 * 2
    assert matrix[0, 1] == 3  # 1 * 3
    assert matrix[0, 2] == 4  # 1 * 4
    assert matrix[1, 0] == 4  # 2 * 2
    assert isinstance(matrix[1, 1], ErrorSnapshot)  # 2 * 3 failed
    assert matrix[1, 2] == 8  # 2 * 4
    assert matrix[2, 0] == 6  # 3 * 2
    assert matrix[2, 1] == 9  # 3 * 3
    assert matrix[2, 2] == 12  # 3 * 4


# Test 6: Error in function without mapspec
def test_no_mapspec_single_error():
    """Test error handling in function without mapspec."""

    @pipefunc(output_name="b")
    def may_fail(a: int) -> int:
        if a == 5:
            msg = "Cannot process a=5"
            raise ValueError(msg)
        return a * 2

    pipeline = Pipeline([may_fail])

    # With single value that causes error
    result = pipeline.map({"a": 5}, error_handling="continue", parallel=False)

    # Should get ErrorSnapshot as output
    assert isinstance(result["b"].output, ErrorSnapshot)
    assert "Cannot process a=5" in str(result["b"].output.exception)

    # With value that doesn't cause error
    result_ok = pipeline.map({"a": 3}, error_handling="continue", parallel=False)
    assert result_ok["b"].output == 6


# Test 7: Test ErrorSnapshot has proper attributes
def test_error_snapshot_attributes():
    """Test that ErrorSnapshot objects have expected attributes."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_with_info(x: int) -> int:
        if x == 2:
            msg = f"Failed on x={x}"
            raise ValueError(msg)
        return x * 10

    pipeline = Pipeline([fail_with_info])
    result = pipeline.map({"x": [1, 2, 3]}, error_handling="continue", parallel=False)

    error = result["y"].output[1]
    assert isinstance(error, ErrorSnapshot)

    # Check ErrorSnapshot attributes
    assert hasattr(error, "function")
    assert hasattr(error, "exception")
    assert hasattr(error, "kwargs")
    assert hasattr(error, "traceback")
    assert hasattr(error, "timestamp")

    # Check content
    assert error.function.__name__ == "fail_with_info"
    assert isinstance(error.exception, ValueError)
    assert error.kwargs == {"x": 2}
    assert "Failed on x=2" in str(error.exception)


# Test 8: Test caching with errors
def test_caching_with_errors():
    """Test that errors are properly cached when cache is enabled."""

    call_count = 0

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", cache=True)
    def cached_may_fail(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([cached_may_fail], cache_type="simple")

    # First run
    result1 = pipeline.map({"x": [1, 2, 3, 4]}, error_handling="continue", parallel=False)
    first_call_count = call_count

    # Second run - should use cache
    result2 = pipeline.map({"x": [1, 2, 3, 4]}, error_handling="continue", parallel=False)

    # Function should not be called again
    assert call_count == first_call_count

    # Results should be the same
    assert result1["y"].output[0] == result2["y"].output[0] == 2
    assert result1["y"].output[1] == result2["y"].output[1] == 4
    assert isinstance(result1["y"].output[2], ErrorSnapshot)
    assert isinstance(result2["y"].output[2], ErrorSnapshot)
    assert result1["y"].output[3] == result2["y"].output[3] == 8
