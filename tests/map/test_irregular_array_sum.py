"""Test for irregular array handling with sum operation."""

from typing import Literal

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array


@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_irregular_array_sum_regression(
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
    """Regression test ensuring ragged sums skip masked padding."""

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
        scheduling_strategy=scheduling_strategy,
    )

    # For n=4: x=[0,1,2,3], y=[0,2,4,6], sum=12
    # For n=3: x=[0,1,2], y=[0,2,4], sum=6 (padding trimmed before calling take_sum)
    sum_output = results["sum"].output
    assert len(sum_output) == 2
    assert sum_output[0] == 12
    assert sum_output[1] == 6
    assert not np.ma.is_masked(sum_output[0])
    assert not np.ma.is_masked(sum_output[1])


@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_irregular_array_sum_with_proper_handling(
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
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
        scheduling_strategy=scheduling_strategy,
    )

    sum_output = results["sum"].output
    assert len(sum_output) == 2
    assert sum_output[0] == 12
    assert sum_output[1] == 6  # Now correctly sums [0,2,4] = 6


@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_irregular_array_mask_is_set_correctly(
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
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
        scheduling_strategy=scheduling_strategy,
    )

    y_output = results["y"].output
    assert isinstance(y_output, np.ma.MaskedArray)
    # Shape is (4, 2) because max length is 4 and we have 2 values of n
    assert y_output.shape == (4, 2)

    # Check the first column (n=4): should have no masked values
    assert not np.any(y_output.mask[:, 0])  # type: ignore[index]
    assert list(y_output[:, 0]) == [0, 2, 4, 6]

    # Check the second column (n=3): should have last element masked
    assert y_output.mask[3, 1]  # type: ignore[index]
    # compressed() correctly removes masked values
    assert list(y_output[:, 1].compressed()) == [0, 2, 4]


@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_irregular_pipeline_without_inputs(
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
    arr = np.empty(5, dtype=object)
    arr[0] = [0, 1]
    arr[1] = [0, 1, 2]
    arr[2] = [0]
    arr[3] = [0, 1, 2, 3]
    arr[4] = []

    @pipefunc(output_name="x")
    def produce() -> np.ndarray:
        return arr

    @pipefunc(output_name="y", mapspec="x[i] -> y[i, j*]")
    def expand(x: Array[int]) -> list[int]:
        return [value * 2 for value in x]

    @pipefunc(output_name="z", mapspec="y[:, j*] -> z[j*]")
    def sums(y: Array[int]) -> int:
        return sum(y)

    pipeline = Pipeline([produce, expand, sums])

    results = pipeline.map(
        {},
        storage="dict",
        internal_shapes={"x": (5,), "y": (4,), "z": (4,)},
        cleanup=True,
        parallel=False,
        scheduling_strategy=scheduling_strategy,
    )

    output = results["z"].output
    assert list(output) == [0, 6, 8, 6]  # Sums per ragged column


@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_irregular_multi_axis_reductions(
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
    samples = 4
    max_channels = samples  # longest channel list we will emit
    max_times = samples + 1  # longest time list we will emit

    @pipefunc(output_name="samples")
    def sample_indices() -> list[int]:
        return list(range(samples))

    @pipefunc(output_name="channels", mapspec="samples[i] -> channels[i, j*]")
    def build_channels(samples: int) -> list[int]:
        return list(range(samples + 1))

    @pipefunc(output_name="values", mapspec="channels[i, j*] -> values[i, j*, k*]")
    def build_values(channels: int) -> list[int]:
        return list(range(channels + 1))

    @pipefunc(output_name="channel_sums", mapspec="values[i, j*, :] -> channel_sums[i, j*]")
    def channel_sum(values: Array[int]) -> int:
        arr = np.ma.masked_equal(np.array(values, dtype=object), None)
        total = np.ma.sum(arr)
        if np.ma.is_masked(total):
            return np.ma.masked  # type: ignore[return-value]
        return int(total)

    @pipefunc(output_name="time_sums", mapspec="values[i, :, k*] -> time_sums[i, k*]")
    def time_sum(values: Array[int]) -> int:
        arr = np.ma.MaskedArray(values, copy=False)
        total = np.ma.sum(arr)
        if np.ma.is_masked(total):
            return np.ma.masked  # type: ignore[return-value]
        return int(total)

    @pipefunc(output_name="sample_totals", mapspec="values[i, :, :] -> sample_totals[i]")
    def sample_total(values: Array[int]) -> int:
        arr = np.ma.MaskedArray(values, copy=False)
        total = np.ma.sum(arr)
        if np.ma.is_masked(total):
            return np.ma.masked  # type: ignore[return-value]
        return int(total)

    pipeline = Pipeline(
        [sample_indices, build_channels, build_values, channel_sum, time_sum, sample_total],
    )

    results = pipeline.map(
        inputs={},
        storage="dict",
        internal_shapes={
            "channels": (max_channels,),
            "values": (max_channels, max_times),
            "channel_sums": (max_channels,),
            "time_sums": (max_times,),
        },
        cleanup=True,
        parallel=False,
        scheduling_strategy=scheduling_strategy,
    )

    for sample_index in range(samples):
        expected_channel_sums = [c * (c + 1) // 2 for c in range(sample_index + 1)]
        masked_channels = results["channel_sums"].output[sample_index]
        assert masked_channels.count() == len(expected_channel_sums)
        assert list(masked_channels.compressed()) == expected_channel_sums

        longest = sample_index + 1
        expected_time_sums = [k * (sample_index - k + 1) for k in range(longest)]
        masked_times = results["time_sums"].output[sample_index]
        assert masked_times.count() == longest
        assert list(masked_times.compressed()) == expected_time_sums

        expected_sample_total = sum(expected_channel_sums)
        assert results["sample_totals"].output[sample_index] == expected_sample_total

    total_expected = sum(
        sum(range(channel + 1)) for sample in range(samples) for channel in range(sample + 1)
    )
    assert sum(results["sample_totals"].output) == total_expected
