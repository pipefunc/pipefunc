"""Tests for irregular/jagged arrays support."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._utils import is_installed
from pipefunc.map import load_outputs
from pipefunc.map._run import _coerce_irregular_output, _set_output
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.typing import Array

has_xarray = is_installed("xarray")


def _dummy_pipefunc() -> PipeFunc[Any]:
    return cast("PipeFunc[Any]", SimpleNamespace(__name__="dummy", output_name="out"))


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
    x_mask = np.asarray(x_array.mask)
    # First row should have [0] and rest masked
    assert x_array[0, 0] == 0
    assert all(x_mask[0, 1:])
    # Second row should have [0, 1] and rest masked
    assert x_array[1, 0] == 0
    assert x_array[1, 1] == 1
    assert all(x_mask[1, 2:])
    # Fifth row should have [0, 1, 2, 3, 4] with no masking
    assert x_array[4, 0] == 0
    assert x_array[4, 4] == 4
    assert not any(x_mask[4, :])

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


def test_irregular_with_different_storage(tmp_path: Path) -> None:
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
    data_mask = np.asarray(data.mask)
    assert data_mask[0, 2]  # Should be masked

    assert data[1, 0] == 0.0
    assert data[1, 1] == 1.0
    assert data[1, 2] == 2.0
    assert not any(data_mask[1, :])  # No masking in row 1

    assert data[2, 0] == 0.0
    assert all(data_mask[2, 1:])  # Rest should be masked

    # Test loading from disk
    loaded = load_outputs("data", run_folder=tmp_path)
    assert isinstance(loaded, np.ma.MaskedArray)
    # Compare masks
    loaded_mask = np.asarray(loaded.mask)
    np.testing.assert_array_equal(loaded_mask, data_mask)
    # Compare only non-masked values (masked values might differ)
    assert loaded[~loaded_mask].tolist() == data[~data_mask].tolist()


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
    array1_mask = np.asarray(array1.mask)
    assert not array1_mask[0, 0]
    assert all(array1_mask[0, 1:])  # size1[0]=1, so only 1 element

    array2 = results["array2"].output
    assert array2.shape == (2, 4)
    array2_mask = np.asarray(array2.mask)
    assert not any(array2_mask[0, :])  # size2[0]=2 -> 4 elements
    assert not any(array2_mask[1, :2])  # size2[1]=1 -> 2 elements
    assert all(array2_mask[1, 2:])


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
        assert isinstance(doubled, np.ndarray)
        return int(np.sum(doubled)) if doubled.size else 0

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


def test_irregular_dict_storage_slice_reducer() -> None:
    """Reducers over dict storage receive trimmed 1D arrays."""

    seen: list[type[Any]] = []

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def create(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="totals", mapspec="data[i, :] -> totals[i]")
    def totals(data: Array[int]) -> int:
        assert isinstance(data, np.ndarray)
        seen.append(type(data))
        return int(np.sum(data)) if data.size else 0

    pipeline = Pipeline([create, totals])
    results = pipeline.map(
        inputs={"n": [0, 3]},
        internal_shapes={"data": (4,)},
        storage="dict",
        parallel=False,
    )

    np.testing.assert_array_equal(results["totals"].output, [0, 3])
    assert seen == [np.ndarray, np.ndarray]


def test_irregular_dimension_errors():
    """Test error cases for irregular dimensions."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def make_data(n: int) -> list[int]:
        return list(range(n))

    pipeline = Pipeline([make_data])

    # Without internal_shape, the first result fixes the capacity. Larger
    # subsequent outputs must now raise instead of being truncated.
    with pytest.raises(ValueError, match="exceeds internal_shape at axis"):
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
    with pytest.raises(ValueError, match="exceeds internal_shape at axis"):
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
    data_mask = np.asarray(data.mask)

    # Check first and last rows
    assert data[0, 0] == 0
    assert all(data_mask[0, 1:])

    assert not any(data_mask[9, :])  # Last row should have all 10 elements
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
    data_mask = np.asarray(data.mask)
    assert all(data_mask[0, :])  # First row all masked (empty)
    assert not data_mask[1, 0]
    assert not data_mask[1, 1]
    assert data_mask[1, 2]
    assert all(data_mask[2, :])  # Third row all masked (empty)
    assert not any(data_mask[3, :])  # Fourth row not masked


def test_irregular_masks_skip_function_calls() -> None:
    """Element-wise maps should not execute for padded masked entries."""

    call_order: list[Any] = []

    @pipefunc(output_name="x", mapspec="n[i] -> x[i, j*]")
    def generate_values(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j*] -> y[i, j*]")
    def double_entry(x: int) -> Any:
        call_order.append(x)
        if x is np.ma.masked:
            return np.ma.masked  # type: ignore[return-value]
        return 2 * x

    pipeline = Pipeline([generate_values, double_entry])

    inputs = {"n": [1, 3, 0]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"x": (5,), "y": (5,)},
        parallel=False,
        storage="dict",
    )

    expected_inputs: list[int] = [0, 0, 1, 2]
    assert call_order == expected_inputs

    y_array = results["y"].output
    assert isinstance(y_array, np.ma.MaskedArray)
    y_mask = np.asarray(y_array.mask)
    # First column (full length) is unmasked, ragged column ends with masked sentinel
    # Each row corresponds to an input element; mask tails past realised length
    np.testing.assert_array_equal(y_array[0].compressed(), [0])
    assert y_mask[0, 1:].all()
    np.testing.assert_array_equal(y_array[1].compressed(), [0, 2, 4])
    assert y_mask[1, 3:].all()
    assert y_array[2].mask.all()

    x_store = results["x"].store
    assert isinstance(x_store, StorageBase)
    assert x_store.irregular_extent((0,)) == (1,)
    assert x_store.irregular_extent((0,)) == (1,)  # cached result
    assert x_store.is_element_masked((0, 1))
    assert not x_store.is_element_masked((0, 0))
    assert not x_store.is_element_masked((0, slice(None)))


def test_irregular_masks_skip_function_calls_file_storage(tmp_path: Path) -> None:
    """Skip behaviour also applies to file-backed irregular storage."""

    call_order: list[Any] = []

    @pipefunc(output_name="seq", mapspec="n[i] -> seq[i, j*]")
    def sequence(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="scaled", mapspec="seq[i, j*] -> scaled[i, j*]")
    def scale(seq: int) -> Any:
        call_order.append(seq)
        if seq is np.ma.masked:
            return np.ma.masked  # type: ignore[return-value]
        return seq + 1

    pipeline = Pipeline([sequence, scale])

    inputs = {"n": [2, 1]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"seq": (4,), "scaled": (4,)},
        run_folder=tmp_path,
        parallel=False,
        storage="file_array",
    )

    assert call_order == [0, 1, 0]

    seq_store = results["seq"].store
    assert isinstance(seq_store, StorageBase)
    assert seq_store.irregular_extent((0,)) == (2,)
    assert seq_store.irregular_extent((0,)) == (2,)
    assert seq_store.is_element_masked((0, 3))
    assert not seq_store.is_element_masked((0, 0))
    assert not seq_store.is_element_masked((0, slice(None)))

    scaled_array = results["scaled"].output
    assert isinstance(scaled_array, np.ma.MaskedArray)
    scaled_mask = np.asarray(scaled_array.mask)
    np.testing.assert_array_equal(scaled_array[0].compressed(), [1, 2])
    assert scaled_mask[0, 2:].all()
    np.testing.assert_array_equal(scaled_array[1].compressed(), [1])
    assert scaled_mask[1, 1:].all()


def test_irregular_file_storage_slice_reducer(tmp_path: Path) -> None:
    """Reproducer: reduction over irregular FileArray should not crash."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def generate_data(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="totals", mapspec="data[i, :] -> totals[i]")
    def sum_row(data: Array[int]) -> int:
        assert isinstance(data, np.ndarray)
        return int(np.sum(data)) if data.size else 0

    pipeline = Pipeline([generate_data, sum_row])

    results = pipeline.map(
        inputs={"n": [1, 3, 0]},
        internal_shapes={"data": (4,)},
        run_folder=tmp_path,
        storage="file_array",
        parallel=False,
    )

    data = results["data"].output
    assert isinstance(data, np.ma.MaskedArray)
    assert data.shape == (3, 4)
    np.testing.assert_array_equal(data[0].compressed(), [0])
    data_mask = np.asarray(data.mask)
    assert data_mask[0, 1:].all()
    np.testing.assert_array_equal(data[1].compressed(), [0, 1, 2])
    assert data_mask[1, 3:].all()
    assert data_mask[2].all()

    totals = results["totals"].output
    np.testing.assert_array_equal(totals, [0, 3, 0])


def test_irregular_dataframe_drops_padding() -> None:
    @pipefunc(output_name="words", mapspec="text[i] -> words[i, j*]")
    def split_text(text: str) -> list[str]:
        return text.split()

    @pipefunc(output_name="lengths", mapspec="words[i, j*] -> lengths[i, j*]")
    def word_lengths(words: str) -> int:
        return len(words)

    pipeline = Pipeline([split_text, word_lengths])
    inputs = {"text": ["Hello world", "Python is great", "A"]}
    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"words": (3,), "lengths": (3,)},
        storage="dict",
        parallel=False,
        show_progress=False,
    )
    if has_xarray:
        from pipefunc.map.xarray import xarray_dataset_from_results, xarray_dataset_to_dataframe

        ds = xarray_dataset_from_results(inputs, results, pipeline, load_intermediate=False)
        df = xarray_dataset_to_dataframe(ds)
        # Only six real entries should remain (2 + 3 + 1)
        assert len(df) == 6
        assert not df["lengths"].isna().any()


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


def test_irregular_slice_retains_multidimensional_mask() -> None:
    captured: list[np.ma.MaskedArray] = []

    @pipefunc(output_name="tensor", mapspec="count[i] -> tensor[i, j*, k]")
    def make_tensor(count: int) -> list[list[float]]:
        return [[float(row), float(row + 10)] for row in range(count)]

    @pipefunc(output_name="shapes", mapspec="tensor[i, :, :] -> shapes[i]")
    def collect(tensor: np.ndarray) -> tuple[int, ...]:
        assert isinstance(tensor, np.ma.MaskedArray)
        captured.append(tensor)
        return tensor.shape

    pipeline = Pipeline([make_tensor, collect])
    results = pipeline.map(
        inputs={"count": [0, 2, 3]},
        internal_shapes={"tensor": (4, 2)},
        storage="dict",
        parallel=False,
    )

    assert [arr.ndim for arr in captured] == [2, 2, 2]
    assert all(isinstance(arr, np.ma.MaskedArray) for arr in captured)
    assert all(arr.shape == (4, 2) for arr in captured)
    assert list(results["shapes"].output) == [(4, 2), (4, 2), (4, 2)]


@pytest.mark.parametrize("storage", ["dict", "file_array"])
def test_multi_irregular_axes_invoke_padded_elements(tmp_path: Path, storage: str) -> None:
    calls: list[int] = []

    @pipefunc(output_name="values", mapspec="n[i] -> values[i, j*, k*]")
    def make_values(n: int) -> np.ma.MaskedArray:
        data = np.ma.masked_all((4, 4), dtype=int)
        for j in range(n):
            for k in range(j + 1):
                data[j, k] = (j + 1) * 100 + (k + 1)
        return data

    @pipefunc(output_name="recorded", mapspec="values[i, j*, k*] -> recorded[i, j*, k*]")
    def record(values: int) -> int:
        calls.append(int(values))
        return values

    inputs = {"n": [0, 2, 3]}
    pipeline = Pipeline([make_values, record])
    map_kwargs: dict[str, Any] = {
        "inputs": inputs,
        "internal_shapes": {"values": (4, 4)},
        "storage": storage,
        "parallel": False,
    }
    if storage == "file_array":
        map_kwargs["run_folder"] = tmp_path
    pipeline.map(**map_kwargs)

    expected_real = sum(n * (n + 1) // 2 for n in inputs["n"])
    assert expected_real == 9  # safeguard for the scenario under test
    assert len(calls) == expected_real
    assert all(value != 0 for value in calls)


def test_multi_axis_irregular_python_lists(tmp_path: Path) -> None:
    """Test that raw nested Python lists are normalized correctly."""

    @pipefunc(output_name="values", mapspec="n[i] -> values[i, j*, k*]")
    def make_values(n: int) -> list[list[int]]:
        # Return raw nested Python lists (not wrapped in np.array)
        return [[10 * j + k for k in range(j + 1)] for j in range(n)]

    pipeline = Pipeline([make_values])

    result = pipeline.map(
        {"n": [3]},
        internal_shapes={"values": (4, 4)},
        storage="dict",
        parallel=False,
        run_folder=tmp_path,
    )
    arr = result["values"].output
    assert isinstance(arr, np.ma.MaskedArray)
    assert arr.shape == (1, 4, 4)
    assert arr[0, 0, 0] == 0
    assert arr[0, 1, 1] == 11
    assert arr[0, 2, 2] == 22
    assert arr[0, 3].mask.all()


def test_multi_axis_irregular_object_arrays(tmp_path: Path) -> None:
    """Test that object-dtype arrays with ragged structure are normalized correctly."""

    @pipefunc(output_name="values", mapspec="n[i] -> values[i, j*, k*]")
    def make_values(n: int) -> np.ndarray:
        # Return np.array with dtype=object containing ragged structure
        return np.array([[10 * j + k for k in range(j + 1)] for j in range(n)], dtype=object)

    pipeline = Pipeline([make_values])

    result = pipeline.map(
        {"n": [3]},
        internal_shapes={"values": (4, 4)},
        storage="dict",
        parallel=False,
        run_folder=tmp_path,
    )
    arr = result["values"].output
    assert isinstance(arr, np.ma.MaskedArray)
    assert arr.shape == (1, 4, 4)
    assert arr[0, 0, 0] == 0
    assert arr[0, 1, 1] == 11
    assert arr[0, 2, 2] == 22
    assert arr[0, 3].mask.all()


def test_irregular_with_masked_sentinels() -> None:
    """Test that masked sentinels in irregular data are preserved."""

    @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
    def make_data_with_masked(n: int) -> list:
        # Return lists with masked sentinels
        if n == 0:
            return [np.ma.masked]
        if n == 1:
            return [np.ma.masked, 1]
        return [0, 1, np.ma.masked]

    pipeline = Pipeline([make_data_with_masked])

    result = pipeline.map(
        {"n": [0, 1, 2]},
        internal_shapes={"data": (3,)},
        storage="dict",
        parallel=False,
    )
    arr = result["data"].output
    assert isinstance(arr, np.ma.MaskedArray)
    assert arr.shape == (3, 3)
    mask = np.ma.getmaskarray(arr)

    # First row: [masked, masked, masked] (n=0 returns [np.ma.masked])
    assert mask[0, 0]  # First element is masked
    assert mask[0, 1]  # Padding is masked
    assert mask[0, 2]  # Padding is masked

    # Second row: [masked, 1, masked] (n=1 returns [np.ma.masked, 1])
    assert mask[1, 0]  # First element is masked
    assert arr[1, 1] == 1  # Second element is 1
    assert not mask[1, 1]  # Second element is not masked
    assert mask[1, 2]  # Padding is masked

    # Third row: [0, 1, masked] (n=2 returns [0, 1, np.ma.masked])
    assert arr[2, 0] == 0
    assert not mask[2, 0]
    assert arr[2, 1] == 1
    assert not mask[2, 1]
    assert mask[2, 2]  # Third element is masked sentinel


def test_coerce_irregular_requires_internal_shape() -> None:
    with pytest.raises(ValueError, match="requires a non-empty internal_shape"):
        _coerce_irregular_output([[1]], (), func=_dummy_pipefunc())


def test_coerce_irregular_numpy_scalar() -> None:
    result = _coerce_irregular_output([np.array(5)], (1,), func=_dummy_pipefunc())
    assert isinstance(result, np.ma.MaskedArray)
    assert result.shape == (1,)
    assert result[0] == 5


def test_coerce_irregular_exceeds_internal_shape() -> None:
    with pytest.raises(ValueError, match="exceeds internal_shape at axis 1"):
        _coerce_irregular_output([[0, [1, 2]]], (2, 1), func=_dummy_pipefunc())


def test_coerce_irregular_masked_array_branch() -> None:
    masked_row = np.ma.array([1, np.ma.masked, 3], dtype=object)
    result = _coerce_irregular_output([masked_row], (1, 3), func=_dummy_pipefunc())
    assert isinstance(result, np.ma.MaskedArray)
    mask = np.ma.getmaskarray(result)
    assert result[0, 0] == 1
    assert result[0, 2] == 3
    assert mask[0, 1]


def test_coerce_irregular_exceeds_at_first_axis() -> None:
    """Test that exceeding internal_shape at axis 0 raises clear error."""
    with pytest.raises(ValueError, match="exceeds internal_shape at axis 0"):
        # Two elements at root level, but internal_shape[0] = 1
        _coerce_irregular_output([1, 2], (1, 2), func=_dummy_pipefunc())
