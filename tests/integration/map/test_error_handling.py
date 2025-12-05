"""Tests for error handling with error_handling='continue' in map operations."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import Any

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot
from pipefunc.resources import Resources


# Test 1: Simple pipeline with no mapspec - error propagation through pipeline
@pipefunc(output_name="b")
def test1_step1(a: int) -> int:
    return a * 2


@pipefunc(output_name="c")
def test1_step2(b: int) -> int:
    if b == 4:
        msg = "Cannot process b=4"
        raise ValueError(msg)
    return b + 10


@pipefunc(output_name="d")
def test1_step3(c: int) -> int:
    # This should not be called when c is an error
    return c * 3


@pytest.mark.parametrize("parallel", [False, True])
def test_simple_pipeline_no_mapspec_error_propagation(parallel):
    """Test error propagation in a simple 3-step pipeline without mapspec."""
    pipeline = Pipeline([test1_step1, test1_step2, test1_step3])

    # Test with error_handling="continue"
    result = pipeline.map({"a": 2}, error_handling="continue", parallel=parallel)

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
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test2_may_fail(x: int) -> int:
    if x == 3:
        msg = f"Cannot process x={x}"
        raise ValueError(msg)
    return x * 2


@pytest.mark.parametrize("parallel", [False, True])
def test_single_pipefunc_element_wise_errors(parallel):
    """Test error handling in element-wise operations."""
    pipeline = Pipeline([test2_may_fail])

    # Test with array input where one element causes error
    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue", parallel=parallel)

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
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test3_step1(x: int) -> int:
    if x == 3:
        msg = f"Cannot process x={x}"
        raise ValueError(msg)
    return x * 2


@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def test3_step2(y: int) -> int:
    # Should not be called for y[i] that is an ErrorSnapshot
    return y + 10


@pytest.mark.parametrize("parallel", [False, True])
def test_pipeline_element_wise_propagation(parallel):
    """Test error propagation through element-wise operations."""
    pipeline = Pipeline([test3_step1, test3_step2])

    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue", parallel=parallel)

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
@pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
def test4_compute(x: int, y: int) -> int:
    if x == 2 and y == 3:
        msg = f"Cannot compute x={x}, y={y}"
        raise ValueError(msg)
    return x * y


@pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
def test4_sum_rows(matrix: np.ndarray) -> int:
    # Should only be called for rows without errors
    return int(np.sum(matrix))


@pytest.mark.parametrize("parallel", [False, True])
def test_1d_reduction_partial_errors(parallel):
    """Test error handling in 1D reduction operations."""
    pipeline = Pipeline([test4_compute, test4_sum_rows])

    result = pipeline.map(
        {"x": [1, 2, 3], "y": [2, 3, 4]},
        error_handling="continue",
        parallel=parallel,
    )

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
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test5_double(x: int) -> int:
    if x == 3:
        msg = f"Cannot process x={x}"
        raise ValueError(msg)
    return x * 2


@pipefunc(output_name="sum")  # No mapspec - receives entire array
def test5_sum_all(y: np.ndarray) -> int:
    # Should not be called if y contains any errors
    return int(np.sum(y))


@pytest.mark.parametrize("parallel", [False, True])
def test_full_array_reduction_with_errors(parallel):
    """Test error handling when reducing entire array that contains errors."""
    pipeline = Pipeline([test5_double, test5_sum_all])

    result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue", parallel=parallel)

    # Check y array has mixed values and errors
    y_output = result["y"].output
    assert y_output[0] == 2
    assert isinstance(y_output[2], ErrorSnapshot)
    assert y_output[4] == 10

    # Check sum is PropagatedErrorSnapshot because y contains errors
    assert isinstance(result["sum"].output, PropagatedErrorSnapshot)
    assert result["sum"].output.reason == "array_contains_errors"


# Test 6: 2D reduction with errors in different dimensions
@pipefunc(output_name="tensor", mapspec="x[i], y[j], z[k] -> tensor[i, j, k]")
def test6_compute_3d(x: int, y: int, z: int) -> int:
    if (x == 1 and y == 2 and z == 1) or (x == 2 and y == 1 and z == 0):
        msg = f"Cannot compute x={x}, y={y}, z={z}"
        raise ValueError(msg)
    return x + y + z


@pipefunc(output_name="sum_z", mapspec="tensor[i, j, :] -> sum_z[i, j]")
def test6_sum_along_z(tensor: np.ndarray) -> int:
    # Should only be called for [i, j] slices without errors
    return int(np.sum(tensor))


@pipefunc(output_name="sum_yz", mapspec="tensor[i, :, :] -> sum_yz[i]")
def test6_sum_along_yz(tensor: np.ndarray) -> int:
    # Should only be called for [i, :, :] slices without errors
    return int(np.sum(tensor))


@pytest.mark.parametrize("parallel", [False, True])
def test_2d_reduction_multi_dimension_errors(parallel):
    """Test error handling in 2D reductions across different axes."""
    pipeline = Pipeline([test6_compute_3d, test6_sum_along_z, test6_sum_along_yz])

    result = pipeline.map(
        {"x": [1, 2, 3], "y": [1, 2], "z": [0, 1]},
        error_handling="continue",
        parallel=parallel,
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
@pipefunc(output_name="doubled", mapspec="nums[i] -> doubled[i]")
def test7_double(nums: int) -> int:
    if nums == 5:
        msg = "Cannot double 5"
        raise ValueError(msg)
    return nums * 2


@pipefunc(output_name="matrix", mapspec="doubled[i], factors[j] -> matrix[i, j]")
def test7_multiply(doubled: int, factors: int) -> int:
    if doubled == 8 and factors == 3:
        msg = "Cannot multiply 8 by 3"
        raise ValueError(msg)
    return doubled * factors


@pipefunc(output_name="row_max", mapspec="matrix[i, :] -> row_max[i]")
def test7_max_per_row(matrix: np.ndarray) -> int:
    # Should skip rows with errors
    return int(np.max(matrix))


@pipefunc(output_name="total")  # No mapspec - full reduction
def test7_sum_all_max(row_max: np.ndarray) -> int:
    # Should not run if row_max contains errors
    return int(np.sum(row_max))


@pytest.mark.parametrize("parallel", [False, True])
def test_complex_pipeline_mixed_mapspecs(parallel):
    """Test error handling in complex pipeline with various mapspec patterns."""
    pipeline = Pipeline([test7_double, test7_multiply, test7_max_per_row, test7_sum_all_max])

    result = pipeline.map(
        {"nums": [1, 2, 3, 4, 5], "factors": [2, 3, 4]},
        error_handling="continue",
        parallel=parallel,
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
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test8_may_fail(x: int) -> int:
    if x == 3:
        msg = f"Cannot process x={x}"
        raise ValueError(msg)
    return x * 2


@pytest.mark.parametrize("parallel", [False, True])
def test_sequential_execution_error_handling(parallel):
    """Test that error handling works correctly in sequential mode."""
    pipeline = Pipeline([test8_may_fail])

    # Test with parallel parameter
    result = pipeline.map(
        {"x": [1, 2, 3, 4, 5]},
        error_handling="continue",
        parallel=parallel,
    )

    # Should still process all elements despite error
    y_output = result["y"].output
    assert y_output[0] == 2
    assert y_output[1] == 4
    assert isinstance(y_output[2], ErrorSnapshot)
    assert y_output[3] == 8
    assert y_output[4] == 10


# Test 9: Multiple errors in same function
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test9_may_fail_multiple(x: int) -> int:
    if x in [2, 4, 7]:
        msg = f"Cannot process x={x}"
        raise ValueError(msg)
    return x**2


@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def test9_add_ten(y: int) -> int:
    return y + 10


@pytest.mark.parametrize("parallel", [False, True])
def test_multiple_errors_same_function(parallel):
    """Test handling multiple errors in the same function."""
    pipeline = Pipeline([test9_may_fail_multiple, test9_add_ten])

    result = pipeline.map({"x": list(range(1, 9))}, error_handling="continue", parallel=parallel)

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
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def test10_will_fail(x: int) -> int:
    if x == 3:
        msg = "Expected error"
        raise ValueError(msg)
    return x * 2


@pytest.mark.parametrize("parallel", [False, True])
def test_error_handling_raise_default(parallel):
    """Test that error_handling='raise' maintains default behavior."""
    pipeline = Pipeline([test10_will_fail])

    # Should raise exception with default error_handling
    with pytest.raises(ValueError, match="Expected error"):
        pipeline.map({"x": [1, 2, 3, 4, 5]}, parallel=parallel)  # error_handling="raise" is default

    # Explicit error_handling="raise" should also raise
    with pytest.raises(ValueError, match="Expected error"):
        pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="raise", parallel=parallel)


def test_tuple_output_with_error_handling():
    """Test that error handling works with tuple output names."""

    @pipefunc(output_name=("y1", "y2"), mapspec="x[i] -> y1[i], y2[i]")
    def process_tuple(x: int) -> tuple[int, int]:
        if x == 3:
            msg = f"Cannot process {x}"
            raise ValueError(msg)
        return x * 2, x * 3

    @pipefunc(output_name="z", mapspec="y1[i], y2[i] -> z[i]")
    def combine(y1: int, y2: int) -> int:
        return y1 + y2

    pipeline = Pipeline([process_tuple, combine])

    # Test with error_handling="continue"
    result = pipeline.map(
        {"x": [1, 2, 3, 4, 5]},
        error_handling="continue",
    )

    # Check y1 output
    y1 = result["y1"].output
    assert isinstance(y1, np.ndarray)
    assert y1.dtype == object
    assert list(y1[:2]) == [2, 4]
    assert isinstance(y1[2], ErrorSnapshot)
    assert list(y1[3:]) == [8, 10]

    # Check y2 output
    y2 = result["y2"].output
    assert isinstance(y2, np.ndarray)
    assert y2.dtype == object
    assert list(y2[:2]) == [3, 6]
    assert isinstance(y2[2], ErrorSnapshot)
    assert list(y2[3:]) == [12, 15]

    # Check z output - should have PropagatedErrorSnapshot at index 2
    z = result["z"].output
    assert isinstance(z, np.ndarray)
    assert z.dtype == object
    assert list(z[:2]) == [5, 10]  # 2+3=5, 4+6=10
    assert isinstance(z[2], PropagatedErrorSnapshot)
    assert list(z[3:]) == [20, 25]  # 8+12=20, 10+15=25


def test_tuple_output_with_mapspec_reduction():
    """Test tuple outputs with reduction mapspecs."""

    @pipefunc(
        output_name=("matrix1", "matrix2"),
        mapspec="x[i], y[j] -> matrix1[i, j], matrix2[i, j]",
    )
    def compute_matrices(x: int, y: int) -> tuple[int, int]:
        if x == 2 and y == 2:
            msg = f"Cannot compute for x={x}, y={y}"
            raise ValueError(msg)
        return x * y, x + y

    @pipefunc(output_name="row_sums1", mapspec="matrix1[i, :] -> row_sums1[i]")
    def sum_rows1(matrix1: np.ndarray) -> int:
        return np.sum(matrix1)

    @pipefunc(output_name="row_sums2", mapspec="matrix2[i, :] -> row_sums2[i]")
    def sum_rows2(matrix2: np.ndarray) -> int:
        return np.sum(matrix2)

    pipeline = Pipeline([compute_matrices, sum_rows1, sum_rows2])

    result = pipeline.map(
        {"x": [1, 2, 3], "y": [1, 2, 3]},
        error_handling="continue",
    )

    # Check matrix1
    matrix1 = result["matrix1"].output
    expected_matrix1 = np.array(
        [
            [1, 2, 3],
            [2, None, 6],  # Error at [1,1]
            [3, 6, 9],
        ],
        dtype=object,
    )

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                assert isinstance(matrix1[i, j], ErrorSnapshot)
            else:
                assert matrix1[i, j] == expected_matrix1[i, j]

    # Check matrix2
    matrix2 = result["matrix2"].output
    expected_matrix2 = np.array(
        [
            [2, 3, 4],
            [3, None, 5],  # Error at [1,1]
            [4, 5, 6],
        ],
        dtype=object,
    )

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                assert isinstance(matrix2[i, j], ErrorSnapshot)
            else:
                assert matrix2[i, j] == expected_matrix2[i, j]

    # Check row_sums1
    row_sums1 = result["row_sums1"].output
    assert row_sums1[0] == 6  # 1+2+3
    assert isinstance(row_sums1[1], PropagatedErrorSnapshot)  # Row contains error
    assert row_sums1[2] == 18  # 3+6+9

    # Check row_sums2
    row_sums2 = result["row_sums2"].output
    assert row_sums2[0] == 9  # 2+3+4
    assert isinstance(row_sums2[1], PropagatedErrorSnapshot)  # Row contains error
    assert row_sums2[2] == 15  # 4+5+6


def test_tuple_output_with_output_picker():
    """Test tuple outputs with custom output picker."""

    def custom_picker(result, key):
        """Custom output picker function."""
        # If result is an error, return it as-is for all outputs

        if isinstance(result, (ErrorSnapshot, PropagatedErrorSnapshot)):
            return result

        if key == "out1":
            return result[0]
        if key == "out2":
            return result[1]
        msg = f"Unknown output: {key}"
        raise KeyError(msg)

    @pipefunc(
        output_name=("out1", "out2"),
        mapspec="x[i] -> out1[i], out2[i]",
        output_picker=custom_picker,
    )
    def picker_func(x: int) -> tuple[int, int]:
        if x == 2:
            msg = f"Error at {x}"
            raise ValueError(msg)
        return (x * 10, x * 20)  # Return as tuple

    pipeline = Pipeline([picker_func])

    result = pipeline.map(
        {"x": [1, 2, 3]},
        error_handling="continue",
        parallel=False,  # Custom picker is not picklable
    )

    out1 = result["out1"].output
    out2 = result["out2"].output

    assert out1[0] == 10
    assert isinstance(out1[1], ErrorSnapshot)
    assert out1[2] == 30

    assert out2[0] == 20
    assert isinstance(out2[1], ErrorSnapshot)
    assert out2[2] == 60


def test_output_picker_skipped_for_error_snapshots():
    """Error outputs bypass pickers that expect a successful result."""

    @pipefunc(
        output_name=("a", "b"),
        mapspec="x[i] -> a[i], b[i]",
        output_picker=dict.__getitem__,
    )
    def picker_func(x: int) -> dict[str, int]:
        if x == 1:
            msg = f"Error at {x}"
            raise ValueError(msg)
        return {"a": x, "b": x + 1}

    pipeline = Pipeline([picker_func])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    a = result["a"].output
    b = result["b"].output

    assert a[0] == 0
    assert isinstance(a[1], ErrorSnapshot)
    assert a[2] == 2

    assert b[0] == 1
    assert isinstance(b[1], ErrorSnapshot)
    assert b[2] == 3


def test_output_picker_skipped_for_error_snapshots_single_task():
    """Non-mapped functions also bypass pickers when returning errors."""

    @pipefunc(output_name=("left", "right"), output_picker=dict.__getitem__)
    def picker_func(flag: bool) -> dict[str, int]:
        if flag:
            msg = "boom"
            raise ValueError(msg)
        return {"left": 1, "right": 2}

    pipeline = Pipeline([picker_func])

    result = pipeline.map({"flag": True}, error_handling="continue", parallel=False)

    assert isinstance(result["left"].output, ErrorSnapshot)
    assert isinstance(result["right"].output, ErrorSnapshot)


@pytest.mark.parametrize("parallel", [False, True])
def test_tuple_output_parallel_execution(parallel):
    """Test tuple outputs with error handling in parallel execution."""

    @pipefunc(output_name=("a", "b"), mapspec="x[i] -> a[i], b[i]")
    def parallel_func(x: int) -> tuple[int, int]:
        if x in [2, 4]:
            msg = f"Error at {x}"
            raise ValueError(msg)
        return x**2, x**3

    pipeline = Pipeline([parallel_func])

    # Test with parallel=True
    result = pipeline.map(
        {"x": list(range(6))},
        error_handling="continue",
        parallel=parallel,
    )

    a = result["a"].output
    b = result["b"].output

    expected_a = [0, 1, None, 9, None, 25]
    expected_b = [0, 1, None, 27, None, 125]

    for i in range(6):
        if i in [2, 4]:
            assert isinstance(a[i], ErrorSnapshot)
            assert isinstance(b[i], ErrorSnapshot)
        else:
            assert a[i] == expected_a[i]
            assert b[i] == expected_b[i]


def test_tuple_output_with_complex_mapspec():
    """Test complex mapspec patterns with tuple outputs."""

    @pipefunc(output_name=("y1", "y2", "y3"), mapspec="x[i], z[j] -> y1[i, j], y2[i, j], y3[i, j]")
    def complex_func(x: int, z: int) -> tuple[int, int, int]:
        if x == 1 and z == 2:
            msg = "Special error case"
            raise ValueError(msg)
        return x + z, x * z, x - z

    @pipefunc(output_name="sum_all")
    def sum_all_arrays(y1: np.ndarray, y2: np.ndarray, y3: np.ndarray) -> int:
        return np.sum(y1) + np.sum(y2) + np.sum(y3)

    pipeline = Pipeline([complex_func, sum_all_arrays])

    result = pipeline.map(
        {"x": [0, 1, 2], "z": [1, 2, 3]},
        error_handling="continue",
    )

    # Check individual outputs
    y1 = result["y1"].output
    y2 = result["y2"].output
    y3 = result["y3"].output

    # y1 should be x + z
    assert y1[0, 0] == 1  # 0+1
    assert y1[0, 1] == 2  # 0+2
    assert y1[0, 2] == 3  # 0+3
    assert y1[1, 0] == 2  # 1+1
    assert isinstance(y1[1, 1], ErrorSnapshot)  # Error case
    assert y1[1, 2] == 4  # 1+3
    assert y1[2, 0] == 3  # 2+1
    assert y1[2, 1] == 4  # 2+2
    assert y1[2, 2] == 5  # 2+3
    # y2 should be x * z
    assert y2[0, 0] == 0  # 0*1
    assert y2[0, 1] == 0  # 0*2
    assert y2[0, 2] == 0  # 0*3
    assert y2[1, 0] == 1  # 1*1
    assert isinstance(y2[1, 1], ErrorSnapshot)  # Error case
    assert y2[1, 2] == 3  # 1*3
    assert y2[2, 0] == 2  # 2*1
    assert y2[2, 1] == 4  # 2*2
    assert y2[2, 2] == 6  # 2*3
    # y3 should be x - z
    assert y3[0, 0] == -1  # 0-1
    assert y3[0, 1] == -2  # 0-2
    assert y3[0, 2] == -3  # 0-3
    assert y3[1, 0] == 0  # 1-1
    assert isinstance(y3[1, 1], ErrorSnapshot)  # Error case
    assert y3[1, 2] == -2  # 1-3
    assert y3[2, 0] == 1  # 2-1
    assert y3[2, 1] == 0  # 2-2
    assert y3[2, 2] == -1  # 2-3

    # sum_all should be PropagatedErrorSnapshot because y1, y2, y3 contain errors
    sum_all = result["sum_all"].output
    assert isinstance(sum_all, PropagatedErrorSnapshot)


def test_edge_case_single_error_in_tuple():
    """Test when only one output in a tuple fails (shouldn't happen with current impl)."""

    # In the current implementation, if a function fails, all outputs fail together
    # This test verifies that behavior

    @pipefunc(output_name=("out1", "out2"), mapspec="x[i] -> out1[i], out2[i]")
    def func(x: int) -> tuple[int, int]:
        if x == 1:
            msg = "Error"
            raise ValueError(msg)
        return x, x * 2

    pipeline = Pipeline([func])
    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue")

    out1 = result["out1"].output
    out2 = result["out2"].output

    # Both outputs should have errors at the same index
    assert out1[0] == 0
    assert isinstance(out1[1], ErrorSnapshot)
    assert out1[2] == 2

    assert out2[0] == 0
    assert isinstance(out2[1], ErrorSnapshot)  # Same error for both outputs
    assert out2[2] == 4


def test_resources_callback_not_evaluated_on_error_input() -> None:
    """Regression: resources callable must not run when upstream inputs are errors."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": kw["y"] + 1},
        resources_scope="element",
    )
    def downstream(y: int) -> int:
        return y * 2

    pipeline = Pipeline([fail_for_one, downstream])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    y_output = result["y"].output
    z_output = result["z"].output

    assert y_output[0] == 0
    assert isinstance(y_output[1], ErrorSnapshot)
    assert y_output[2] == 2

    assert z_output[0] == 0
    assert isinstance(z_output[1], PropagatedErrorSnapshot)
    assert z_output[2] == 4


def test_map_scope_resources_skipped_on_error_input() -> None:
    """Regression: map-scope resources must be skipped when inputs contain errors."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": int(np.sum(kw["y"].to_array()))},
        resources_scope="map",
    )
    def downstream(y: int) -> int:
        return y * 2

    pipeline = Pipeline([fail_for_one, downstream])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    y_output = result["y"].output
    z_output = result["z"].output

    assert y_output[0] == 0
    assert isinstance(y_output[1], ErrorSnapshot)
    assert y_output[2] == 2

    assert z_output[0] == 0
    assert isinstance(z_output[1], PropagatedErrorSnapshot)
    assert z_output[2] == 4


def test_continue_mode_preserves_map_scope_resources_for_clean_indices() -> None:
    """Continue mode should still provide map-scope resources to downstream functions."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x

    def map_scope_resources(kwargs: dict[str, Any]) -> Resources:
        values = kwargs["y"].to_array()
        successes = [
            v for v in values if not isinstance(v, (ErrorSnapshot, PropagatedErrorSnapshot))
        ]
        return Resources(extra_args={"success_sum": sum(successes)})

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=map_scope_resources,
        resources_scope="map",
        resources_variable="resources",
    )
    def downstream(y: int, resources: Resources) -> int:
        return y + resources.extra_args["success_sum"]

    pipeline = Pipeline([fail_for_one, downstream])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    y_output = result["y"].output
    z_output = result["z"].output

    assert y_output[0] == 0
    assert isinstance(y_output[1], ErrorSnapshot)
    assert y_output[2] == 2

    assert z_output[0] == 2
    assert isinstance(z_output[1], PropagatedErrorSnapshot)
    assert z_output[2] == 4


def test_map_scope_resource_failure_propagates_errors() -> None:
    """Map-scope resource errors should propagate when resources cannot be evaluated."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x

    def map_scope_resources(kwargs: dict[str, Any]) -> Resources:
        # Trigger a TypeError when encountering ErrorSnapshot entries
        total = sum(kwargs["y"].to_array())  # type: ignore[arg-type]
        return Resources(extra_args={"total": total})

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=map_scope_resources,
        resources_scope="map",
        resources_variable="resources",
    )
    def downstream(y: int, resources: Resources) -> int:
        return y + resources.extra_args["total"]

    pipeline = Pipeline([fail_for_one, downstream])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    z_output = result["z"].output

    assert isinstance(z_output[0], PropagatedErrorSnapshot)
    assert isinstance(z_output[1], PropagatedErrorSnapshot)
    assert isinstance(z_output[2], PropagatedErrorSnapshot)


def test_resource_failure_skips_executor_submission() -> None:
    """If resources fail, we should not submit work to external executors."""

    def failing_resources(_kwargs: dict[str, Any]) -> Resources:
        msg = "resource failure"
        raise RuntimeError(msg)

    @pipefunc(output_name="z", resources=failing_resources)
    def downstream(x: int) -> int:
        return x * 2

    pipeline = Pipeline([downstream])

    class DummyExecutor(Executor):
        def submit(self, *_args: Any, **_kwargs: Any) -> Future[Any]:  # type: ignore[override]
            message = "executor.submit should not be called when resources fail"
            raise AssertionError(message)

    result = pipeline.map(
        {"x": 1},
        error_handling="continue",
        parallel=True,
        executor={"z": DummyExecutor()},
    )

    assert isinstance(result["z"].output, PropagatedErrorSnapshot)


def test_resource_failure_raise_mode() -> None:
    """When error_handling='raise', resource failures surface immediately."""

    call_count = 0

    def failing_resources(_kwargs: dict[str, Any]) -> Resources:
        nonlocal call_count
        call_count += 1
        msg = "resource failure"
        raise RuntimeError(msg)

    @pipefunc(output_name="z", resources=failing_resources)
    def downstream(x: int) -> int:
        return x * 2

    pipeline = Pipeline([downstream])

    with pytest.raises(RuntimeError, match="resource failure"):
        pipeline.map({"x": 1}, error_handling="raise", parallel=False)

    assert call_count == 1


def test_parallel_error_snapshot_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots returned from parallel failures must stay per-thread."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    import pipefunc._pipefunc_utils as utils

    barrier = threading.Barrier(2)
    original = utils.handle_pipefunc_error

    def wrapped_handle(e, func, kwargs, error_handling="raise"):
        snapshot = original(e, func, kwargs, error_handling)
        if error_handling == "continue":
            barrier.wait()
            func.error_snapshot = utils.ErrorSnapshot(  # type: ignore[attr-defined]
                func.func,
                e,
                args=(),
                kwargs={"x": 999},
            )
        return snapshot

    monkeypatch.setattr(utils, "handle_pipefunc_error", wrapped_handle)

    @pipefunc(output_name="a", mapspec="x[i] -> a[i]")
    def boom(x: int) -> int:  # pragma: no cover - executed via pipeline
        msg = f"boom {x}"
        raise ValueError(msg)

    pipeline = Pipeline([boom])
    executor = ThreadPoolExecutor(max_workers=2)

    result = pipeline.map(
        {"x": [0, 1]},
        error_handling="continue",
        parallel=True,
        executor={"a": executor},
    )

    snapshots = result["a"].output
    assert snapshots[0].kwargs == {"x": 0}
    assert snapshots[1].kwargs == {"x": 1}


def test_non_mapspec_map_scope_resources_skipped_on_error_input() -> None:
    """Regression: map-scope resources on scalar functions must skip when inputs fail."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x

    @pipefunc(
        output_name="total",
        resources=lambda kw: {"cpus": int(np.sum(kw["y"]))},
        resources_scope="map",
    )
    def aggregate(y: np.ndarray) -> int:
        return int(np.sum(y))

    pipeline = Pipeline([fail_for_one, aggregate])

    result = pipeline.map({"x": [0, 1, 2]}, error_handling="continue", parallel=False)

    y_output = result["y"].output
    assert y_output[0] == 0
    assert isinstance(y_output[1], ErrorSnapshot)
    assert y_output[2] == 2

    total_output = result["total"].output
    assert isinstance(total_output, PropagatedErrorSnapshot)


def test_non_mapspec_resources_evaluated_correctly() -> None:
    """Regression: non-mapspec functions with callable resources must evaluate them.

    This is a regression test for a bug where _execute_single() (used for non-mapspec
    functions) was missing the _maybe_eval_resources() call, causing callable resources
    to never be evaluated.
    """
    resources_was_called = []

    def track_resources(kw: dict) -> dict:
        """Track that resources function was called and return resources."""
        resources_was_called.append(kw["x"])
        return {"cpus": kw["x"]}

    @pipefunc(
        output_name="y",
        resources=track_resources,
    )
    def compute(x: int) -> int:
        return x * 2

    pipeline = Pipeline([compute])

    # Test with error_handling="raise" (default)
    result = pipeline.map({"x": 5}, error_handling="raise", parallel=False)
    assert result["y"].output == 10
    assert resources_was_called == [5], "Resources callback should have been called"

    # Test with error_handling="continue"
    resources_was_called.clear()
    result = pipeline.map({"x": 3}, error_handling="continue", parallel=False)
    assert result["y"].output == 6
    assert resources_was_called == [3], "Resources callback should have been called"


def test_return_results_false_with_errors_uses_lightweight_markers(tmp_path) -> None:
    """Regression: return_results=False should use _ErrorMarker, not full ErrorSnapshot.

    When return_results=False, we discard heavy data to save memory. ErrorSnapshots
    can be huge (they store full kwargs which may contain large arrays). This test
    verifies that we replace ErrorSnapshots with lightweight _ErrorMarker objects.

    We verify this indirectly by:
    1. Running with return_results=False and errors
    2. Loading the data from storage
    3. Confirming the saved ErrorSnapshots are correct (proving lightweight markers worked)
    """
    from pipefunc.map import load_outputs

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_for_one(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def downstream(y: int) -> int:
        return y + 10

    pipeline = Pipeline([fail_for_one, downstream])

    # Use run_folder to enable persistence with return_results=False
    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        parallel=False,
        return_results=False,
        run_folder=tmp_path,
    )

    # With return_results=False, result dict should be empty
    assert result == {}

    # But data should be persisted to disk
    y_output = load_outputs("y", run_folder=tmp_path)
    z_output = load_outputs("z", run_folder=tmp_path)

    # y[0] = 0, y[1] = ErrorSnapshot, y[2] = 4
    assert y_output[0] == 0
    assert isinstance(y_output[1], ErrorSnapshot)
    assert y_output[2] == 4

    # z[0] = 10, z[1] = PropagatedErrorSnapshot, z[2] = 14
    assert z_output[0] == 10
    assert isinstance(z_output[1], PropagatedErrorSnapshot)
    assert z_output[2] == 14

    # The key test: this should have worked without keeping full ErrorSnapshots in memory
    # during execution (they were replaced with lightweight _ErrorMarker objects)


def test_empty_inputs_continue_mode():
    """Test that empty input arrays work correctly with error_handling='continue'."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        return x * 2

    pipeline = Pipeline([double])
    result = pipeline.map({"x": []}, error_handling="continue")

    y_output = result["y"].output
    assert len(y_output) == 0
    assert isinstance(y_output, np.ndarray)


def test_resume_rejects_different_error_handling_mode(tmp_path):
    """Test that resume=True rejects runs with different error_handling modes."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # First run with error_handling="continue"
    pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        run_folder=tmp_path,
    )

    # Attempt to resume with error_handling="raise" should fail
    with pytest.raises(ValueError, match="error_handling.*does not match"):
        pipeline.map(
            {"x": [0, 1, 2]},
            error_handling="raise",
            resume=True,
            run_folder=tmp_path,
        )

    # Resuming with same error_handling should work
    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        resume=True,
        run_folder=tmp_path,
    )
    assert result["y"].output[0] == 0
    assert isinstance(result["y"].output[1], ErrorSnapshot)
    assert result["y"].output[2] == 4


def test_output_picker_exception_in_continue_mode():
    """Test that output_picker exceptions produce ErrorSnapshot in continue mode."""

    def bad_picker(result, key):
        if key == "bad":
            msg = "picker boom"
            raise RuntimeError(msg)
        return result[key]

    @pipefunc(
        output_name=("good", "bad"),
        mapspec="x[i] -> good[i], bad[i]",
        output_picker=bad_picker,
    )
    def my_func(x: int) -> dict:
        return {"good": x * 2, "bad": x * 3}

    pipeline = Pipeline([my_func])

    # With error_handling="continue", picker failure should produce ErrorSnapshot
    result = pipeline.map(
        {"x": [1, 2, 3]},
        error_handling="continue",
        parallel=False,
    )

    # "good" output should work fine
    assert result["good"].output[0] == 2
    assert result["good"].output[1] == 4
    assert result["good"].output[2] == 6

    # "bad" output should have ErrorSnapshots due to picker failure
    assert isinstance(result["bad"].output[0], ErrorSnapshot)
    assert isinstance(result["bad"].output[1], ErrorSnapshot)
    assert isinstance(result["bad"].output[2], ErrorSnapshot)
    assert "picker boom" in str(result["bad"].output[0].exception)


def test_output_picker_exception_raises_in_raise_mode():
    """Test that output_picker exceptions raise in error_handling='raise' mode."""

    def bad_picker(result, key):
        msg = "picker boom"
        raise RuntimeError(msg)

    @pipefunc(
        output_name=("a", "b"),
        mapspec="x[i] -> a[i], b[i]",
        output_picker=bad_picker,
    )
    def my_func(x: int) -> dict:
        return {"a": x, "b": x}

    pipeline = Pipeline([my_func])

    # With error_handling="raise", picker failure should raise
    with pytest.raises(RuntimeError, match="picker boom"):
        pipeline.map({"x": [1]}, error_handling="raise", parallel=False)


def test_output_picker_exception_with_file_storage(tmp_path):
    """Test output_picker exceptions with file storage for mapped functions."""

    def bad_picker(result, key):
        if key == "bad":
            msg = "picker boom"
            raise RuntimeError(msg)
        return result[key]

    @pipefunc(
        output_name=("good", "bad"),
        mapspec="x[i] -> good[i], bad[i]",
        output_picker=bad_picker,
    )
    def my_func(x: int) -> dict:
        return {"good": x * 2, "bad": x * 3}

    pipeline = Pipeline([my_func])

    # With file storage, this exercises _pick_single_output error handling
    result = pipeline.map(
        {"x": [1, 2]},
        error_handling="continue",
        run_folder=tmp_path,
        parallel=False,
    )

    # "good" output should work fine
    assert result["good"].output[0] == 2
    assert result["good"].output[1] == 4

    # "bad" output should have ErrorSnapshots
    assert isinstance(result["bad"].output[0], ErrorSnapshot)
    assert isinstance(result["bad"].output[1], ErrorSnapshot)


def test_output_picker_exception_raises_with_file_storage(tmp_path):
    """Test output_picker exceptions raise in raise mode with file storage."""

    def bad_picker(result, key):
        msg = "picker boom"
        raise RuntimeError(msg)

    @pipefunc(
        output_name=("a", "b"),
        mapspec="x[i] -> a[i], b[i]",
        output_picker=bad_picker,
    )
    def my_func(x: int) -> dict:
        return {"a": x, "b": x}

    pipeline = Pipeline([my_func])

    with pytest.raises(RuntimeError, match="picker boom"):
        pipeline.map(
            {"x": [1]},
            error_handling="raise",
            run_folder=tmp_path,
            parallel=False,
        )


def test_output_picker_exception_non_mapped_with_file_storage(tmp_path):
    """Test output_picker exceptions for non-mapped functions with file storage.

    This covers _dump_single_output error handling path (lines 649-659).
    """

    def bad_picker(result, key):
        if key == "bad":
            msg = "picker boom in single"
            raise RuntimeError(msg)
        return result[key]

    @pipefunc(
        output_name=("good", "bad"),
        output_picker=bad_picker,
    )
    def single_func(x: np.ndarray) -> dict:
        return {"good": int(x.sum()) * 2, "bad": int(x.sum()) * 3}

    # Need a mapped function to provide input
    @pipefunc(output_name="x", mapspec="i[j] -> x[j]")
    def make_x(i: int) -> int:
        return i

    pipeline = Pipeline([make_x, single_func])

    result = pipeline.map(
        {"i": [5]},
        error_handling="continue",
        run_folder=tmp_path,
        parallel=False,
    )

    # "good" output should work (scalar, not array)
    assert result["good"].output == 10

    # "bad" output should be ErrorSnapshot
    assert isinstance(result["bad"].output, ErrorSnapshot)
    assert "picker boom in single" in str(result["bad"].output.exception)


def test_output_picker_exception_non_mapped_raises(tmp_path):
    """Test output_picker exceptions raise for non-mapped functions in raise mode."""

    def bad_picker(result, key):
        msg = "picker boom in single"
        raise RuntimeError(msg)

    @pipefunc(
        output_name=("a", "b"),
        output_picker=bad_picker,
    )
    def single_func(x: np.ndarray) -> dict:
        return {"a": x.sum(), "b": x.sum()}

    @pipefunc(output_name="x", mapspec="i[j] -> x[j]")
    def make_x(i: int) -> int:
        return i

    pipeline = Pipeline([make_x, single_func])

    with pytest.raises(RuntimeError, match="picker boom in single"):
        pipeline.map(
            {"i": [5]},
            error_handling="raise",
            run_folder=tmp_path,
            parallel=False,
        )


def test_all_indices_propagate_errors_with_executor(tmp_path):
    """Test that all-error indices are run locally instead of submitted to executor.

    When all indices would propagate errors, they should be processed locally
    rather than submitted to the executor (covers _all_indices_propagate_errors path).
    """
    from concurrent.futures import ThreadPoolExecutor

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def always_fail(x: int) -> int:
        msg = "always fails"
        raise ValueError(msg)

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def downstream(y: int) -> int:
        return y + 1

    pipeline = Pipeline([always_fail, downstream])

    with ThreadPoolExecutor(max_workers=2) as ex:
        result = pipeline.map(
            {"x": [1, 2, 3]},
            error_handling="continue",
            run_folder=tmp_path,
            executor={"y": ex, "z": ex},
        )

    # All upstream failed, so all downstream should be PropagatedErrorSnapshot
    for i in range(3):
        assert isinstance(result["y"].output[i], ErrorSnapshot)
        assert isinstance(result["z"].output[i], PropagatedErrorSnapshot)
