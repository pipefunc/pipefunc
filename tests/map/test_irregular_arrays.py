"""Tests for irregular/jagged arrays support."""

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map._run import _set_output
from pipefunc.typing import Array


def test_basic_irregular_dimension():
    """Test basic irregular dimension with varying lengths using explicit mapspec."""

    @pipefunc(output_name="x", mapspec="n[i] -> x[i, j*]")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j*] -> y[i, j*]")
    def double_it(x: int) -> int:
        """Double the input integer."""
        return 2 * x

    @pipefunc(output_name="sum", mapspec="y[i, :] -> sum[i]")
    def take_sum(y: Array[int]) -> int:
        """Sum a list of integers."""
        # y might be a masked array or an object array with masked sentinels
        if hasattr(y, "compressed"):
            return sum(y.compressed())
        # Handle object array with np.ma.masked sentinels
        total = 0
        for val in y:
            if val is not np.ma.masked:
                total += val
        return total

    pipeline = Pipeline([generate_ints, double_it, take_sum])

    # Different lengths for each n
    inputs = {"n": [1, 2, 3, 4, 5]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"x": (5,), "y": (5,)},  # Maximum size
        parallel=False,
        storage="dict",
    )

    # Check that x is properly stored with masking
    x_array = results["x"].output
    assert isinstance(x_array, np.ma.MaskedArray)
    # First row should have [0] and rest masked
    assert x_array[0, 0] == 0
    assert all(x_array.mask[0, 1:])
    # Second row should have [0, 1] and rest masked
    assert x_array[1, 0] == 0
    assert x_array[1, 1] == 1
    assert all(x_array.mask[1, 2:])
    # Fifth row should have [0, 1, 2, 3, 4] with no masking
    assert x_array[4, 0] == 0
    assert x_array[4, 4] == 4
    assert not any(x_array.mask[4, :])

    # Check y values (doubled)
    y_array = results["y"].output
    assert isinstance(y_array, np.ma.MaskedArray)
    assert y_array[0, 0] == 0  # 2 * 0
    assert y_array[1, 1] == 2  # 2 * 1
    assert y_array[4, 4] == 8  # 2 * 4

    # Check sum (sum of all valid doubled values per row)
    sum_array = results["sum"].output
    assert sum_array[0] == 0  # sum([0])
    assert sum_array[1] == 2  # sum([0, 2])
    assert sum_array[2] == 6  # sum([0, 2, 4])
    assert sum_array[3] == 12  # sum([0, 2, 4, 6])
    assert sum_array[4] == 20  # sum([0, 2, 4, 6, 8])


def test_irregular_with_different_storage(tmp_path):
    """Test irregular arrays with file storage."""

    @pipefunc(output_name="data", mapspec="size[i] -> data[i, j*]")
    def generate_data(size: int) -> list[float]:
        """Generate data with variable length."""
        return [float(i) for i in range(size)]

    pipeline = Pipeline([generate_data])

    inputs = {"size": [2, 3, 1]}
    results = pipeline.map(
        inputs=inputs,
        run_folder=tmp_path,
        internal_shapes={"data": (3,)},  # Max size
        parallel=False,
        storage="file_array",
    )

    data = results["data"].output
    assert isinstance(data, np.ma.MaskedArray)
    assert data.shape == (3, 3)

    # Check values and masking
    assert data[0, 0] == 0.0
    assert data[0, 1] == 1.0
    assert data.mask[0, 2]  # Should be masked

    assert data[1, 0] == 0.0
    assert data[1, 1] == 1.0
    assert data[1, 2] == 2.0
    assert not any(data.mask[1, :])  # No masking in row 1

    assert data[2, 0] == 0.0
    assert all(data.mask[2, 1:])  # Rest should be masked

    # Test loading from disk
    loaded = load_outputs("data", run_folder=tmp_path)
    assert isinstance(loaded, np.ma.MaskedArray)
    # Compare masks
    np.testing.assert_array_equal(loaded.mask, data.mask)
    # Compare only non-masked values (masked values might differ)
    assert loaded[~loaded.mask].tolist() == data[~data.mask].tolist()


def test_irregular_string_arrays():
    """Test irregular arrays with string content."""

    @pipefunc(output_name="content", mapspec="file_path[i] -> content[i]")
    def read_file(file_path: str, suffix: str) -> str:
        """Simulate reading file content."""
        return file_path + suffix

    @pipefunc(output_name="chars", mapspec="content[i] -> chars[i, j*]")
    def get_chars(content: str) -> list[str]:
        """Convert string to list of characters."""
        return list(content)

    @pipefunc(output_name="char_count", mapspec="chars[i, :] -> char_count[i]")
    def count_chars(chars: Array) -> int:
        """Count non-masked characters."""
        if hasattr(chars, "compressed"):
            return len(chars.compressed())
        # Handle object array with np.ma.masked sentinels
        count = 0
        for val in chars:
            if val is not np.ma.masked:
                count += 1
        return count

    pipeline = Pipeline([read_file, get_chars, count_chars])

    # Different length strings
    inputs = {
        "file_path": ["a.txt", "test.py", "x"],
        "suffix": "_suf",
    }

    # Maximum expected length is "test.py_suf" = 11 chars
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"chars": (11,)},
        parallel=False,
        storage="dict",
    )

    content = results["content"].output
    assert content[0] == "a.txt_suf"  # 9 chars
    assert content[1] == "test.py_suf"  # 11 chars
    assert content[2] == "x_suf"  # 5 chars

    chars = results["chars"].output
    assert isinstance(chars, np.ma.MaskedArray)
    assert chars.shape == (3, 11)

    # Check character arrays
    assert "".join(chars[0].compressed()) == "a.txt_suf"
    assert "".join(chars[1].compressed()) == "test.py_suf"
    assert "".join(chars[2].compressed()) == "x_suf"

    # Check counts
    counts = results["char_count"].output
    assert counts[0] == 9
    assert counts[1] == 11
    assert counts[2] == 5


def test_multiple_irregular_dimensions():
    """Test multiple irregular dimensions in the same pipeline."""

    @pipefunc(output_name="array1", mapspec="size1[i] -> array1[i, j*]")
    def make_array1(size1: int) -> list[int]:
        return list(range(size1))

    @pipefunc(output_name="array2", mapspec="size2[k] -> array2[k, l*]")
    def make_array2(size2: int) -> list[int]:
        return list(range(size2 * 2))

    pipeline = Pipeline([make_array1, make_array2])

    inputs = {
        "size1": [1, 3, 2],
        "size2": [2, 1],
    }

    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"array1": (3,), "array2": (4,)},
        parallel=False,
        storage="dict",
    )

    array1 = results["array1"].output
    assert array1.shape == (3, 3)
    assert not array1.mask[0, 0]
    assert all(array1.mask[0, 1:])  # size1[0]=1, so only 1 element

    array2 = results["array2"].output
    assert array2.shape == (2, 4)
    assert not any(array2.mask[0, :])  # size2[0]=2 -> 4 elements
    assert not any(array2.mask[1, :2])  # size2[1]=1 -> 2 elements
    assert all(array2.mask[1, 2:])


def test_irregular_with_downstream_operations():
    """Test that downstream operations handle masked arrays correctly."""

    @pipefunc(output_name="lists", mapspec="n[i] -> lists[i, j*]")
    def make_lists(n: int) -> list[int]:
        return [x * n for x in range(n)]

    @pipefunc(output_name="doubled", mapspec="lists[i, j*] -> doubled[i, j*]")
    def double(lists: int) -> int:
        # This will receive individual elements, including masked ones
        if lists is np.ma.masked:
            return np.ma.masked  # type: ignore[return-value]
        return lists * 2

    @pipefunc(output_name="row_sums", mapspec="doubled[i, :] -> row_sums[i]")
    def sum_row(doubled: Array[int]) -> int:
        # Should handle masked array or object array with sentinels
        if hasattr(doubled, "compressed"):
            return sum(doubled.compressed())
        # Handle object array with np.ma.masked sentinels
        total = 0
        for val in doubled:
            if val is not np.ma.masked:
                total += val
        return total

    pipeline = Pipeline([make_lists, double, sum_row])

    inputs = {"n": [1, 2, 3]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"lists": (3,)},
        parallel=False,
        storage="dict",
    )

    # Check lists
    lists = results["lists"].output
    assert lists[0, 0] == 0  # [0]
    assert lists[1, 0] == 0  # [0, 2]
    assert lists[1, 1] == 2
    assert lists[2, 0] == 0  # [0, 3, 6]
    assert lists[2, 1] == 3
    assert lists[2, 2] == 6

    # Check doubled
    doubled = results["doubled"].output
    assert doubled[0, 0] == 0
    assert doubled[1, 1] == 4
    assert doubled[2, 2] == 12

    # Check row sums
    row_sums = results["row_sums"].output
    assert row_sums[0] == 0  # sum([0])
    assert row_sums[1] == 4  # sum([0, 4])
    assert row_sums[2] == 18  # sum([0, 6, 12])


def test_irregular_dimension_errors():
    """Test error cases for irregular dimensions."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def make_data(n: int) -> list[int]:
        return list(range(n))

    pipeline = Pipeline([make_data])

    # Without internal_shape, the first result fixes the capacity. Larger
    # subsequent outputs must now raise instead of being truncated.
    with pytest.raises(ValueError, match="exceeds the configured internal shape"):
        pipeline.map(
            inputs={"n": [2, 3]},
            parallel=False,
            storage="dict",
        )

    # Test when actual data exceeds internal_shape
    @pipefunc(output_name="data2", mapspec="n[i] -> data2[i, j*]")
    def make_too_much_data(n: int) -> list[int]:
        return list(range(n * 2))  # Will exceed max for larger n

    pipeline2 = Pipeline([make_too_much_data])

    # This should work but data will be truncated
    with pytest.raises(ValueError, match="exceeds the configured internal shape"):
        pipeline2.map(
            inputs={"n": [2, 3]},
            internal_shapes={"data2": (3,)},  # Max 3, but n=3 will produce 6 elements
            parallel=False,
            storage="dict",
        )


def test_irregular_with_parallel_execution():
    """Test irregular arrays with parallel execution."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def generate(n: int) -> list[int]:
        return list(range(n))

    pipeline = Pipeline([generate])

    inputs = {"n": list(range(1, 11))}  # 1 to 10
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"data": (10,)},
        parallel=True,
        storage="shared_memory_dict",
    )

    data = results["data"].output
    assert isinstance(data, np.ma.MaskedArray)
    assert data.shape == (10, 10)

    # Check first and last rows
    assert data[0, 0] == 0
    assert all(data.mask[0, 1:])

    assert not any(data.mask[9, :])  # Last row should have all 10 elements
    assert data[9, 9] == 9


def test_irregular_syntax_detection():
    """Test that irregular dimensions are correctly detected from mapspec."""

    @pipefunc(output_name="regular", mapspec="x[i] -> regular[i, j]")
    def regular_func(x: int) -> list[int]:
        return [x] * 3

    @pipefunc(output_name="irregular", mapspec="x[i] -> irregular[i, j*]")
    def irregular_func(x: int) -> list[int]:
        return [x] * x

    # Check that the functions correctly identify irregular outputs
    assert not regular_func._irregular_output
    assert irregular_func._irregular_output

    # Check pipeline behavior
    pipeline_regular = Pipeline([regular_func])
    pipeline_irregular = Pipeline([irregular_func])

    # Regular pipeline auto-infers shape correctly when all outputs are the same size
    results_regular = pipeline_regular.map(
        inputs={"x": [1, 2]},
        parallel=False,
        storage="dict",
    )
    assert results_regular["regular"].output.shape == (2, 3)  # All outputs have 3 elements

    # Irregular requires internal_shape for max size
    results = pipeline_irregular.map(
        inputs={"x": [1, 2, 3]},
        internal_shapes={"irregular": (3,)},
        parallel=False,
        storage="dict",
    )

    assert results["irregular"].output.shape == (3, 3)
    assert results["irregular"].output[0, 0] == 1  # [1]
    assert results["irregular"].output[1, 0] == 2  # [2, 2]
    assert results["irregular"].output[2, 2] == 3  # [3, 3, 3]


def test_pipefunc_without_mapspec_is_not_irregular():
    @pipefunc(output_name="plain")
    def plain(n: int) -> int:
        return n

    assert not plain._irregular_output


def test_empty_irregular_arrays():
    """Test irregular arrays with empty entries."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def maybe_empty(n: int) -> list[int]:
        if n == 0:
            return []
        return list(range(n))

    pipeline = Pipeline([maybe_empty])

    inputs = {"n": [0, 2, 0, 3]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"data": (3,)},
        parallel=False,
        storage="dict",
    )

    data = results["data"].output
    assert all(data.mask[0, :])  # First row all masked (empty)
    assert not data.mask[1, 0]
    assert not data.mask[1, 1]
    assert data.mask[1, 2]
    assert all(data.mask[2, :])  # Third row all masked (empty)
    assert not any(data.mask[3, :])  # Fourth row not masked


def test_regular_output_index_error_propagates():
    class RaisingArray(np.ndarray):
        def __getitem__(self, key):
            msg = "intentional index error"
            raise IndexError(msg)

    @pipefunc(output_name="values", mapspec="n[i] -> values[i, j]")
    def produce_raising_array(n: int) -> np.ndarray:
        base = np.zeros(2)
        return base.view(RaisingArray)

    func = produce_raising_array
    arr = np.empty(4, dtype=object)
    output = produce_raising_array(0)
    shape = (2, 2)
    shape_mask = (True, False)

    with pytest.raises(IndexError, match="intentional index error"):
        _set_output(arr, output, 0, shape, shape_mask, func)
