"""Test for irregular array handling with sum operation."""

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array


def test_irregular_array_sum():
    """Test that irregular arrays are properly handled when summing.

    This test demonstrates the issue that when irregular arrays (with masked values)
    are passed to functions that don't understand masked arrays, the behavior is incorrect.
    """

    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i*] -> y[i*]")
    def double_it(x: int) -> int:
        """Double the input integer."""
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        """Sum a list of integers.

        NOTE: This uses the built-in sum() which doesn't handle masked arrays correctly.
        It will fail when encountering np.ma.masked sentinels.
        """
        return sum(y)

    pipeline_sum = Pipeline([generate_ints, double_it, take_sum])
    pipeline_sum.add_mapspec_axis("n", axis="j")

    inputs = {"n": [4, 3]}
    results = pipeline_sum.map(
        inputs,
        internal_shapes={"x": ("?",)},
        cleanup=True,
        parallel=False,
    )

    # Expected behavior:
    # For n=4: x=[0,1,2,3], y=[0,2,4,6], sum=12
    # For n=3: x=[0,1,2], y=[0,2,4], sum=6

    # Actual behavior:
    # The y array is correctly masked: [[0,2,4,6], [0,2,4,masked]]
    # But sum() doesn't know how to handle masked arrays, so:
    # - sum([0,2,4,6]) = 12 (correct)
    # - sum([0,2,4,masked]) fails and returns masked
    sum_output = results["sum"].output
    assert len(sum_output) == 2
    assert sum_output[0] == 12
    # The second value is masked because sum() can't handle the masked sentinel
    assert np.ma.is_masked(sum_output[1])


def test_irregular_array_sum_with_proper_handling():
    """Test the correct way to handle irregular arrays when summing."""

    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i*] -> y[i*]")
    def double_it(x: int) -> int:
        """Double the input integer."""
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        """Sum a list of integers, properly handling masked arrays."""
        # Correct way: check if it's a masked array and use compressed()
        if hasattr(y, "compressed"):
            return sum(y.compressed())
        return sum(y)

    pipeline_sum = Pipeline([generate_ints, double_it, take_sum])
    pipeline_sum.add_mapspec_axis("n", axis="j")

    inputs = {"n": [4, 3]}
    results = pipeline_sum.map(
        inputs,
        internal_shapes={"x": ("?",)},
        cleanup=True,
        parallel=False,
    )

    sum_output = results["sum"].output
    assert len(sum_output) == 2
    assert sum_output[0] == 12
    assert sum_output[1] == 6  # Now correctly sums [0,2,4] = 6


def test_irregular_array_mask_is_set_correctly():
    """Test that the mask is properly set for irregular arrays."""

    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i*] -> y[i*]")
    def double_it(x: int) -> int:
        """Double the input integer."""
        return 2 * x

    pipeline = Pipeline([generate_ints, double_it])
    pipeline.add_mapspec_axis("n", axis="j")

    inputs = {"n": [4, 3]}
    results = pipeline.map(
        inputs,
        internal_shapes={"x": ("?",)},
        cleanup=True,
        parallel=False,
    )

    y_output = results["y"].output
    assert isinstance(y_output, np.ma.MaskedArray)
    # Shape is (4, 2) because max length is 4 and we have 2 values of n
    assert y_output.shape == (4, 2)

    # Check the first column (n=4): should have no masked values
    assert not np.any(y_output.mask[:, 0])
    assert list(y_output[:, 0]) == [0, 2, 4, 6]

    # Check the second column (n=3): should have last element masked
    assert y_output.mask[3, 1]
    # compressed() correctly removes masked values
    assert list(y_output[:, 1].compressed()) == [0, 2, 4]
