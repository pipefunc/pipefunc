"""Tests for error handling with error_handling='continue' in map operations."""

from __future__ import annotations

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot

# PropagatedErrorSnapshot will be implemented as part of this feature
try:
    from pipefunc.exceptions import PropagatedErrorSnapshot
except ImportError:
    # Will be implemented
    PropagatedErrorSnapshot = None


# Skip tests that require unimplemented features
skip_if_no_propagated_error = pytest.mark.skipif(
    PropagatedErrorSnapshot is None,
    reason="PropagatedErrorSnapshot not yet implemented",
)


# Test 1: Simple pipeline with no mapspec - error propagation through pipeline
def test_simple_pipeline_no_mapspec_error_propagation():
    """Test error propagation in a simple 3-step pipeline without mapspec."""

    @pipefunc(output_name="b")
    def step1(a: int) -> int:
        return a * 2

    @pipefunc(output_name="c")
    def step2(b: int) -> int:
        if b == 4:
            msg = "Cannot process b=4"
            raise ValueError(msg)
        return b + 10

    @pipefunc(output_name="d")
    def step3(c: int) -> int:
        # This should not be called when c is an error
        return c * 3

    pipeline = Pipeline([step1, step2, step3])

    # Test with error_handling="continue"
    result = pipeline.map({"a": 2}, error_handling="continue")

    # b should be 4
    assert result["b"].output == 4

    # c should be an ErrorSnapshot
    assert isinstance(result["c"].output, ErrorSnapshot)
    assert "Cannot process b=4" in str(result["c"].output.exception)

    # d should be a PropagatedErrorSnapshot
    # (This will fail until PropagatedErrorSnapshot is implemented)
    if PropagatedErrorSnapshot is None:
        pytest.skip("PropagatedErrorSnapshot not yet implemented")
    assert isinstance(result["d"].output, PropagatedErrorSnapshot)
    assert result["d"].output.skipped_function.output_name == "d"
    assert "c" in result["d"].output.error_info


# Test 2: Single pipefunc with element-wise mapspec and errors
def test_single_pipefunc_element_wise_errors():
    """Test error handling in element-wise operations."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # Test with array input where one element causes error
    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue")

    y_output = result["y"].output
    assert isinstance(y_output, np.ndarray)
    assert y_output.dtype == object  # Should be object array to hold mixed types

    # Check individual elements
    assert y_output[0] == 2  # 1 * 2
    assert y_output[1] == 4  # 2 * 2
    assert isinstance(y_output[2], ErrorSnapshot)  # Failed on x=3
    assert y_output[3] == 8  # 4 * 2
    assert y_output[4] == 10  # 5 * 2


# Test 3: Pipeline with element-wise operations and error propagation
def test_pipeline_element_wise_propagation():
    """Test error propagation through element-wise operations."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def step1(x: int) -> int:
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def step2(y: int) -> int:
        # Should not be called for y[i] that is an ErrorSnapshot
        return y + 10

    pipeline = Pipeline([step1, step2])

    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue")

    # Check y array
    y_output = result["y"].output
    assert y_output[0] == 2
    assert y_output[1] == 4
    assert isinstance(y_output[2], ErrorSnapshot)
    assert y_output[3] == 8
    assert y_output[4] == 10

    # Check z array - propagated errors
    z_output = result["z"].output
    assert z_output[0] == 12  # 2 + 10
    assert z_output[1] == 14  # 4 + 10
    assert isinstance(z_output[2], PropagatedErrorSnapshot)  # Propagated from y[2]
    assert z_output[3] == 18  # 8 + 10
    assert z_output[4] == 20  # 10 + 10


# Test 4: 1D reduction with partial errors
def test_1d_reduction_partial_errors():
    """Test error handling in 1D reduction operations."""

    @pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
    def compute(x: int, y: int) -> int:
        if x == 2 and y == 3:
            msg = f"Cannot compute x={x}, y={y}"
            raise ValueError(msg)
        return x * y

    @pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
    def sum_rows(matrix: np.ndarray) -> int:
        # Should only be called for rows without errors
        return int(np.sum(matrix))

    pipeline = Pipeline([compute, sum_rows])

    result = pipeline.map({"x": [1, 2, 3], "y": [2, 3, 4]}, error_handling="continue")

    # Check matrix
    matrix = result["matrix"].output
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 2  # 1 * 2
    assert matrix[0, 1] == 3  # 1 * 3
    assert matrix[1, 0] == 4  # 2 * 2
    assert isinstance(matrix[1, 1], ErrorSnapshot)  # 2 * 3 failed
    assert matrix[1, 2] == 8  # 2 * 4

    # Check row sums
    row_sums = result["row_sums"].output
    assert row_sums[0] == 9  # sum([2, 3, 4])
    assert isinstance(row_sums[1], PropagatedErrorSnapshot)  # Row contains error
    assert row_sums[2] == 27  # sum([6, 9, 12])


# Test 5: Full array reduction with errors
def test_full_array_reduction_with_errors():
    """Test error handling when reducing entire array that contains errors."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        if x == 3:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x * 2

    @pipefunc(output_name="sum")  # No mapspec - receives entire array
    def sum_all(y: np.ndarray) -> int:
        # Should not be called if y contains any errors
        return int(np.sum(y))

    pipeline = Pipeline([double, sum_all])

    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue")

    # Check y array has mixed values and errors
    y_output = result["y"].output
    assert y_output[0] == 2
    assert isinstance(y_output[2], ErrorSnapshot)
    assert y_output[4] == 10

    # Check sum is PropagatedErrorSnapshot because y contains errors
    assert isinstance(result["sum"].output, PropagatedErrorSnapshot)
    assert result["sum"].output.reason == "array_contains_errors"


# Test 6: 2D reduction with errors in different dimensions
def test_2d_reduction_multi_dimension_errors():
    """Test error handling in 2D reductions across different axes."""

    @pipefunc(output_name="tensor", mapspec="x[i], y[j], z[k] -> tensor[i, j, k]")
    def compute_3d(x: int, y: int, z: int) -> int:
        if (x == 1 and y == 2 and z == 1) or (x == 2 and y == 1 and z == 0):
            msg = f"Cannot compute x={x}, y={y}, z={z}"
            raise ValueError(msg)
        return x + y + z

    @pipefunc(output_name="sum_z", mapspec="tensor[i, j, :] -> sum_z[i, j]")
    def sum_along_z(tensor: np.ndarray) -> int:
        # Should only be called for [i, j] slices without errors
        return int(np.sum(tensor))

    @pipefunc(output_name="sum_yz", mapspec="tensor[i, :, :] -> sum_yz[i]")
    def sum_along_yz(tensor: np.ndarray) -> int:
        # Should only be called for [i, :, :] slices without errors
        return int(np.sum(tensor))

    pipeline = Pipeline([compute_3d, sum_along_z, sum_along_yz])

    result = pipeline.map(
        {"x": [1, 2, 3], "y": [1, 2], "z": [0, 1]},
        error_handling="continue",
    )

    # Check tensor has errors at specific positions
    tensor = result["tensor"].output
    assert isinstance(tensor[0, 1, 1], ErrorSnapshot)  # x=1, y=2, z=1
    assert isinstance(tensor[1, 0, 0], ErrorSnapshot)  # x=2, y=1, z=0

    # Check sum_z - should have errors where tensor[i, j, :] contains errors
    sum_z = result["sum_z"].output
    assert isinstance(sum_z[0, 1], PropagatedErrorSnapshot)  # Has error in z=1
    assert isinstance(sum_z[1, 0], PropagatedErrorSnapshot)  # Has error in z=0
    assert isinstance(sum_z[0, 0], int)  # No errors
    assert isinstance(sum_z[2, 0], int)  # No errors

    # Check sum_yz - should have errors where tensor[i, :, :] contains errors
    sum_yz = result["sum_yz"].output
    assert isinstance(sum_yz[0], PropagatedErrorSnapshot)  # Has error at [0, 1, 1]
    assert isinstance(sum_yz[1], PropagatedErrorSnapshot)  # Has error at [1, 0, 0]
    assert isinstance(sum_yz[2], int)  # No errors in slice [2, :, :]


# Test 7: Complex multi-step pipeline with mixed mapspecs
def test_complex_pipeline_mixed_mapspecs():
    """Test error handling in complex pipeline with various mapspec patterns."""

    @pipefunc(output_name="doubled", mapspec="nums[i] -> doubled[i]")
    def double(nums: int) -> int:
        if nums == 5:
            msg = "Cannot double 5"
            raise ValueError(msg)
        return nums * 2

    @pipefunc(output_name="matrix", mapspec="doubled[i], factors[j] -> matrix[i, j]")
    def multiply(doubled: int, factors: int) -> int:
        if doubled == 8 and factors == 3:
            msg = "Cannot multiply 8 by 3"
            raise ValueError(msg)
        return doubled * factors

    @pipefunc(output_name="row_max", mapspec="matrix[i, :] -> row_max[i]")
    def max_per_row(matrix: np.ndarray) -> int:
        # Should skip rows with errors
        return int(np.max(matrix))

    @pipefunc(output_name="total")  # No mapspec - full reduction
    def sum_all_max(row_max: np.ndarray) -> int:
        # Should not run if row_max contains errors
        return int(np.sum(row_max))

    pipeline = Pipeline([double, multiply, max_per_row, sum_all_max])

    result = pipeline.map(
        {"nums": [1, 2, 3, 4, 5], "factors": [2, 3, 4]},
        error_handling="continue",
    )

    # Check doubled - should have error at index 4 (nums=5)
    doubled = result["doubled"].output
    assert doubled[0] == 2
    assert doubled[3] == 8
    assert isinstance(doubled[4], ErrorSnapshot)

    # Check matrix - should have errors in row 4 and at [3, 1]
    matrix = result["matrix"].output
    assert matrix[0, 0] == 4  # 2 * 2
    assert isinstance(matrix[3, 1], ErrorSnapshot)  # 8 * 3 failed
    assert all(isinstance(matrix[4, j], PropagatedErrorSnapshot) for j in range(3))

    # Check row_max - rows with any error should be PropagatedErrorSnapshot
    row_max = result["row_max"].output
    assert row_max[0] == 8  # max([4, 6, 8])
    assert row_max[1] == 16  # max([8, 12, 16])
    assert row_max[2] == 24  # max([12, 18, 24])
    assert isinstance(row_max[3], PropagatedErrorSnapshot)  # Row has error
    assert isinstance(row_max[4], PropagatedErrorSnapshot)  # Entire row is errors

    # Check total - should be PropagatedErrorSnapshot since row_max has errors
    assert isinstance(result["total"].output, PropagatedErrorSnapshot)
    assert result["total"].output.reason == "array_contains_errors"


# Test 8: Error handling with parallel=False (sequential execution)
def test_sequential_execution_error_handling():
    """Test that error handling works correctly in sequential mode."""

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


# Test 9: Multiple errors in same function
def test_multiple_errors_same_function():
    """Test handling multiple errors in the same function."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail_multiple(x: int) -> int:
        if x in [2, 4, 7]:
            msg = f"Cannot process x={x}"
            raise ValueError(msg)
        return x**2

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def add_ten(y: int) -> int:
        return y + 10

    pipeline = Pipeline([may_fail_multiple, add_ten])

    result = pipeline.map({"x": list(range(1, 9))}, error_handling="continue")

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

    # Check z propagates errors correctly
    z = result["z"].output
    assert z[0] == 11
    assert isinstance(z[1], PropagatedErrorSnapshot)
    assert z[2] == 19
    assert isinstance(z[3], PropagatedErrorSnapshot)
    assert isinstance(z[6], PropagatedErrorSnapshot)


# Test 10: Verify error_handling="raise" still works (default behavior)
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
        pipeline.map({"x": [1, 2, 3, 4, 5]})  # error_handling="raise" is default

    # Explicit error_handling="raise" should also raise
    with pytest.raises(ValueError, match="Expected error"):
        pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="raise")
