from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc._utils import dump, load
from pipefunc.map._storage_array._chunked_file import FILENAME_TEMPLATE, ChunkedFileArray

if TYPE_CHECKING:
    from pipefunc.map._types import ShapeTuple


@pytest.mark.parametrize(
    ("shape", "chunk_size", "expected_num_chunks", "last_chunk_len_expected"),
    [
        ((10,), 3, 4, 1),
        ((10,), 5, 2, 5),
        ((10,), 10, 1, 10),
        ((10,), 11, 1, 10),
        ((0,), 5, 0, 0),
        ((5,), 5, 1, 5),
        ((6,), 2, 3, 2),
        ((7, 3), 5, 5, 1),
    ],
)
def test_chunk_properties(
    tmp_path: Path,
    shape: ShapeTuple,
    chunk_size: int,
    expected_num_chunks: int,
    last_chunk_len_expected: int,
):
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)
    assert len(list(arr._files())) == expected_num_chunks
    if expected_num_chunks > 0:
        assert arr._get_chunk_len(expected_num_chunks - 1) == last_chunk_len_expected
        if expected_num_chunks > 1:
            assert arr._get_chunk_len(0) == chunk_size
    elif shape == (0,):  # Special case for completely empty array
        assert arr.size == 0
        assert (
            arr._get_chunk_len(0) == 0
        )  # Will raise IndexError if num_chunks is 0, which is correct


def test_dump_and_get_single_elements_chunked(tmp_path: Path):
    shape = (12,)
    chunk_size = 5
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)

    arr.dump((0,), "val_0")
    arr.dump((4,), "val_4")
    arr.dump((5,), "val_5")
    arr.dump((9,), "val_9")
    arr.dump((10,), "val_10")
    arr.dump((11,), "val_11")

    assert arr.get_from_index(0) == "val_0"
    assert arr[0] == "val_0"
    assert arr.get_from_index(4) == "val_4"
    assert arr[4] == "val_4"
    assert arr.get_from_index(5) == "val_5"
    assert arr[5] == "val_5"
    assert arr.get_from_index(9) == "val_9"
    assert arr[9] == "val_9"
    assert arr.get_from_index(10) == "val_10"
    assert arr[10] == "val_10"
    assert arr.get_from_index(11) == "val_11"
    assert arr[11] == "val_11"

    assert arr.get_from_index(1) is np.ma.masked
    assert arr[1] is np.ma.masked
    assert arr.has_index(0)
    assert not arr.has_index(1)
    assert arr.has_index(11)

    chunk0 = load(arr._get_chunk_path(0))
    assert chunk0 == ["val_0", np.ma.masked, np.ma.masked, np.ma.masked, "val_4"]
    chunk1 = load(arr._get_chunk_path(1))
    assert chunk1 == ["val_5", np.ma.masked, np.ma.masked, np.ma.masked, "val_9"]
    chunk2 = load(arr._get_chunk_path(2))
    assert chunk2 == ["val_10", "val_11"]


def test_dump_slice_spanning_chunks(tmp_path: Path):
    shape = (10,)
    chunk_size = 3
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)
    arr.dump((slice(1, 5)), "multi-val")

    chunk0_data = load(arr._get_chunk_path(0))
    assert chunk0_data == [np.ma.masked, "multi-val", "multi-val"]
    chunk1_data = load(arr._get_chunk_path(1))
    assert chunk1_data == ["multi-val", "multi-val", np.ma.masked]


def test_getitem_slice_spanning_chunks(tmp_path: Path):
    shape = (7,)
    chunk_size = 3
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)

    dump(["a", "b", "c"], arr._get_chunk_path(0))
    dump(["d", "e", "f"], arr._get_chunk_path(1))
    dump(["g"], arr._get_chunk_path(2))

    result = arr[slice(1, 5)]
    expected = np.ma.array(["b", "c", "d", "e"], dtype=object)
    assert np.ma.allequal(result, expected)

    result_in_chunk = arr[slice(3, 5)]
    expected_in_chunk = np.ma.array(["d", "e"], dtype=object)
    assert np.ma.allequal(result_in_chunk, expected_in_chunk)

    result_last_chunk = arr[slice(5, 7)]
    expected_last_chunk = np.ma.array(["f", "g"], dtype=object)
    assert np.ma.allequal(result_last_chunk, expected_last_chunk)


def test_to_array_chunked(tmp_path: Path):
    shape = (8,)
    chunk_size = 3
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)

    dump([0, 1, np.ma.masked], arr._get_chunk_path(0))
    dump([np.ma.masked, 4, 5], arr._get_chunk_path(1))
    dump([6, np.ma.masked], arr._get_chunk_path(2))

    expected_data = np.array(
        [0, 1, np.ma.masked, np.ma.masked, 4, 5, 6, np.ma.masked],
        dtype=object,
    )
    expected_mask = [False, False, True, True, False, False, False, True]

    result_array = arr.to_array(splat_internal=False)
    assert result_array.shape == shape
    # Use np.ma.allequal for comparing masked arrays
    expected_masked_array = np.ma.array(expected_data, mask=expected_mask, dtype=object)
    assert np.ma.allequal(result_array, expected_masked_array)


def test_mask_linear_chunked(tmp_path: Path):
    shape = (5,)
    chunk_size = 2
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)
    dump(["val0", np.ma.masked], arr._get_chunk_path(0))

    expected_mask = [False, True, True, True, True]
    assert arr.mask_linear() == expected_mask

    dump([np.ma.masked], arr._get_chunk_path(2))
    # linear external indices: 0,1 (chunk 0), 2,3 (chunk 1), 4 (chunk 2)
    # chunk 0: ["val0", masked] -> mask [F, T]
    # chunk 1: no file -> mask [T, T]
    # chunk 2: [masked] -> mask [T]
    expected_mask_updated = [False, True, True, True, True]
    assert arr.mask_linear() == expected_mask_updated


def test_from_data_chunked(tmp_path: Path):
    data = np.arange(10)
    chunk_size = 4

    arr = ChunkedFileArray.from_data(data, tmp_path, chunk_size=chunk_size)
    assert arr.shape == (10,)
    assert arr.chunk_size == chunk_size
    assert len(list(arr._files())) == 3

    chunk0 = load(arr._get_chunk_path(0))
    assert chunk0 == [0, 1, 2, 3]
    chunk1 = load(arr._get_chunk_path(1))
    assert chunk1 == [4, 5, 6, 7]
    chunk2 = load(arr._get_chunk_path(2))
    assert chunk2 == [8, 9]

    assert np.array_equal(arr.to_array(), data)


def test_chunk_size_equals_array_size(tmp_path: Path):
    chunk_size = 5
    data_to_dump = list(range(5))
    arr = ChunkedFileArray.from_data(data_to_dump, tmp_path, chunk_size=chunk_size)
    assert len(list(arr._files())) == 1
    loaded_chunk = load(arr._get_chunk_path(0))
    assert loaded_chunk == data_to_dump  # The chunk itself should contain the list
    assert np.array_equal(arr.to_array(), np.array(data_to_dump))


def test_chunk_size_greater_than_array_size(tmp_path: Path):
    shape = (3,)
    chunk_size = 5
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)
    assert len(list(arr._files())) == 1

    data_to_dump = list(range(3))
    for i, v in enumerate(data_to_dump):
        arr.dump((i,), v)

    loaded_chunk = load(arr._get_chunk_path(0))
    assert loaded_chunk == data_to_dump
    assert np.array_equal(arr.to_array(), np.array(data_to_dump))


def test_empty_array_chunked(tmp_path: Path):
    shape = (0,)
    chunk_size = 5
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)
    assert arr.size == 0
    assert len(list(arr._files())) == 0
    assert arr.to_array().shape == (0,)
    assert arr.mask_linear() == []

    shape_2d_empty = (0, 3)
    arr_2d = ChunkedFileArray(tmp_path / "2d", shape_2d_empty, chunk_size=chunk_size)
    assert arr_2d.size == 0
    assert len(list(arr_2d._files())) == 0
    assert arr_2d.to_array().shape == (0, 3)


def test_chunked_with_internal_shape(tmp_path: Path):
    external_shape = (2,)
    internal_shape = (2, 2)
    shape_mask = (True, False, False)
    chunk_size = 1

    arr = ChunkedFileArray(
        tmp_path,
        external_shape,
        internal_shape=internal_shape,
        shape_mask=shape_mask,
        chunk_size=chunk_size,
    )

    data0 = np.array([[0, 1], [2, 3]])
    data1 = np.array([[4, 5], [6, 7]])

    arr.dump((0,), data0)
    arr.dump((1,), data1)

    assert np.array_equal(arr.get_from_index(0), data0)
    assert np.array_equal(arr.get_from_index(1), data1)

    assert arr[0, 0, 0] == 0
    assert arr[0, 1, 1] == 3
    assert arr[1, 0, 1] == 5
    assert arr[1, 1, 0] == 6

    assert np.array_equal(arr[0, :, 0], np.array([0, 2]))

    arr_no_splat = arr.to_array(splat_internal=False)
    assert arr_no_splat.shape == external_shape
    assert np.array_equal(arr_no_splat[0], data0)
    assert np.array_equal(arr_no_splat[1], data1)

    arr_splat = arr.to_array(splat_internal=True)
    assert arr_splat.shape == (2, 2, 2)
    assert np.array_equal(arr_splat[0], data0)
    assert np.array_equal(arr_splat[1], data1)


def test_dump_slice_value_not_matching_internal_shape(tmp_path: Path):
    arr = ChunkedFileArray(
        tmp_path,
        shape=(2,),
        internal_shape=(2,),
        shape_mask=(True, False),
        chunk_size=1,
    )
    # This test expects the ValueError to be raised if a value of incompatible shape is dumped.
    # The regex needs to match the exact error message, including the shapes.
    with pytest.raises(ValueError, match=re.escape("Value shape (3,) != internal_shape (2,)")):
        arr.dump((0,), np.array([1, 2, 3]))

    arr_no_internal = ChunkedFileArray(tmp_path / "no_int", shape=(2,), chunk_size=1)
    arr_no_internal.dump((slice(None),), 5)  # This implies broadcasting 5 to all elements
    assert arr_no_internal[0] == 5
    assert arr_no_internal[1] == 5


def test_dump_and_get_with_multidimensional_chunks(tmp_path: Path):
    shape = (4, 4)
    chunk_size = 5
    arr = ChunkedFileArray(tmp_path, shape, chunk_size=chunk_size)

    for i in range(4):
        for j in range(4):
            arr.dump((i, j), f"val_{i}_{j}")

    for i in range(4):
        for j in range(4):
            assert arr[i, j] == f"val_{i}_{j}"
            assert arr.get_from_index(np.ravel_multi_index((i, j), shape)) == f"val_{i}_{j}"
            assert arr.has_index(np.ravel_multi_index((i, j), shape))

    slice1 = arr[0, slice(1, 3)]
    assert slice1.tolist() == ["val_0_1", "val_0_2"]

    slice2 = arr[slice(1, 3), slice(2, None)]
    expected2 = np.array([["val_1_2", "val_1_3"], ["val_2_2", "val_2_3"]], dtype=object)
    assert np.array_equal(slice2, expected2)

    expected_full_array = np.array(
        [[f"val_{i}_{j}" for j in range(4)] for i in range(4)],
        dtype=object,
    )
    assert np.array_equal(arr.to_array(), expected_full_array)


def test_load_and_dump(tmp_path):
    obj = {"a": 1, "b": [2, 3]}
    file_path = tmp_path / "test.pickle"
    dump(obj, file_path)
    loaded_obj = load(file_path)
    assert loaded_obj == obj


def test_chunked_file_array_init(tmp_path: Path):  # Renamed for clarity
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = ChunkedFileArray(folder, shape)
    assert arr.folder == folder
    assert arr.shape == shape
    # Strides still refer to the external shape for ravel_multi_index
    assert arr.strides == (12, 4, 1)
    assert arr.filename_template == FILENAME_TEMPLATE


def test_chunked_file_array_normalize_key(tmp_path: Path):  # Renamed for clarity
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = ChunkedFileArray(folder, shape)
    # Test with external shape for dump
    assert arr._normalize_key((1, 2, 3), for_dump=True) == (1, 2, 3)
    assert arr._normalize_key((slice(None), 1, 2), for_dump=True) == (slice(None), 1, 2)
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 3, 4), for_dump=True)  # Too many indices for external shape
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 10), for_dump=True)  # Index out of bounds for external shape

    # Test with full shape for getitem
    assert arr._normalize_key((1, 2, 3), for_dump=False) == (1, 2, 3)


def test_chunked_file_array_get_chunk_path(tmp_path: Path):  # Renamed
    folder = Path(tmp_path)
    shape = (2, 3, 4)  # 24 elements
    arr_cs1 = ChunkedFileArray(folder / "cs1", shape, chunk_size=1)
    assert arr_cs1._get_chunk_path(0) == folder / "cs1" / "__0__.pickle"
    assert arr_cs1._get_chunk_path(23) == folder / "cs1" / "__23__.pickle"

    arr_cs5 = ChunkedFileArray(folder / "cs5", shape, chunk_size=5)
    assert (
        arr_cs5._get_chunk_path(0) == folder / "cs5" / "__0__.pickle"
    )  # Chunk for linear_chunk_idx 0
    assert (
        arr_cs5._get_chunk_path(4) == folder / "cs5" / "__4__.pickle"
    )  # Chunk for linear_chunk_idx 4 (elements 20-23)


def test_chunked_file_array_files(tmp_path: Path):  # Renamed
    folder = Path(tmp_path)
    shape = (2, 3)  # 6 elements
    arr_cs2 = ChunkedFileArray(folder, shape, chunk_size=2)  # 3 chunk files: __0__, __1__, __2__
    files = list(arr_cs2._files())
    assert len(files) == 3
    assert files[0] == folder / "__0__.pickle"
    assert files[-1] == folder / "__2__.pickle"

    arr_cs1 = ChunkedFileArray(folder / "cs1", shape, chunk_size=1)  # 6 chunk files
    files_cs1 = list(arr_cs1._files())
    assert len(files_cs1) == 6

    arr_cs6 = ChunkedFileArray(folder / "cs6", shape, chunk_size=6)  # 1 chunk file
    files_cs6 = list(arr_cs6._files())
    assert len(files_cs6) == 1
    assert files_cs6[0] == folder / "cs6" / "__0__.pickle"

    arr_empty = ChunkedFileArray(folder / "empty", (0, 3), chunk_size=2)
    assert list(arr_empty._files()) == []


def test_chunked_file_array_dump(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3)  # 6 elements
    arr = ChunkedFileArray(folder, shape, chunk_size=2)

    # Dump (0, 0) -> linear external 0. Chunk file 0, index in chunk 0
    arr.dump((0, 0), {"a": 1})
    chunk0_path = arr._get_chunk_path(0)
    chunk0_data = load(chunk0_path)
    assert len(chunk0_data) == 2  # Chunk size
    assert chunk0_data[0] == {"a": 1}
    assert chunk0_data[1] is np.ma.masked

    # Dump (1, 2) -> linear external 5. Chunk file 2 (5 // 2), index in chunk 1 (5 % 2)
    arr.dump((1, 2), {"b": 2})
    chunk2_path = arr._get_chunk_path(2)
    chunk2_data = load(chunk2_path)
    assert len(chunk2_data) == 2  # last chunk is full
    assert chunk2_data[0] is np.ma.masked
    assert chunk2_data[1] == {"b": 2}

    # Overwrite (0,0) by dumping to a slice that only covers it.
    # dump key is relative to external shape.
    arr.dump((slice(0, 1), 0), {"c": 3})  # This refers to external element (0,0)
    chunk0_data_updated = load(chunk0_path)
    assert chunk0_data_updated[0] == {"c": 3}
    assert chunk0_data_updated[1] is np.ma.masked

    # Dump another element in chunk 0: (0,1) -> linear 1. Chunk 0, idx_in_chunk 1
    arr.dump((0, 1), {"d": 4})
    chunk0_data_final = load(chunk0_path)
    assert chunk0_data_final[0] == {"c": 3}
    assert chunk0_data_final[1] == {"d": 4}


# def test_load_all(tmp_path):
# This test was for an internal helper _load_all which might not be relevant
# for ChunkedFileArray in the same way, or its logic is embedded elsewhere.
# Commenting out for now.
# file1 = tmp_path / "file1.pickle"
# file2 = tmp_path / "file2.pickle"
# file3 = tmp_path / "file3.pickle"  # Non-existent file
# dump({"a": 1}, file1)
# dump({"b": 2}, file2)
# result = _load_all([file1, file2, file3]) # _load_all is not defined
# assert result == [{"a": 1}, {"b": 2}, None]


def test_chunked_file_array_dump_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)  # 24 elements
    arr = ChunkedFileArray(folder, shape, chunk_size=5)

    # Test dumping with slicing along a single axis
    # (0, slice(None), 0) means external coords (0,0,0), (0,1,0), (0,2,0)
    # Linear external indices: 0, 4, 8
    # Chunk 0: elements for linear_external_idx 0, 4 (idx_in_chunk 0, 4)
    # Chunk 1: element for linear_external_idx 8 (idx_in_chunk 3)
    arr.dump((0, slice(None), 0), {"a": 1})

    chunk0_path = arr._get_chunk_path(0)  # Elements 0-4
    chunk0_data = load(chunk0_path)
    assert chunk0_data[0] == {"a": 1}  # Linear external 0
    assert chunk0_data[1] is np.ma.masked  # Linear external 1
    assert chunk0_data[2] is np.ma.masked  # Linear external 2
    assert chunk0_data[3] is np.ma.masked  # Linear external 3
    assert chunk0_data[4] == {"a": 1}  # Linear external 4

    chunk1_path = arr._get_chunk_path(1)  # Elements 5-9
    chunk1_data = load(chunk1_path)
    assert chunk1_data[3] == {"a": 1}  # Linear external 8 (idx_in_chunk 8 % 5 = 3)
    for i in [0, 1, 2, 4]:
        assert chunk1_data[i] is np.ma.masked

    # Test dumping with slicing along multiple axes
    # (slice(None), 0, 0) means external (0,0,0), (1,0,0)
    # Linear external indices: 0, 12
    # Chunk 0: element for linear_external_idx 0 (idx_in_chunk 0)
    # Chunk 2: element for linear_external_idx 12 (idx_in_chunk 12 % 5 = 2)
    arr.dump((slice(None), 0, 0), {"b": 2})
    chunk0_data = load(chunk0_path)
    assert chunk0_data[0] == {"b": 2}  # Linear external 0 (overwrites previous)

    chunk2_path = arr._get_chunk_path(2)  # Elements 10-14
    chunk2_data = load(chunk2_path)
    assert chunk2_data[2] == {"b": 2}  # Linear external 12

    # Test dumping with step
    # (0, slice(None, None, 2), 0) means external (0,0,0), (0,2,0)
    # Linear external indices: 0, 8
    arr.dump((0, slice(None, None, 2), 0), {"c": 3})
    chunk0_data = load(chunk0_path)
    assert chunk0_data[0] == {"c": 3}  # Linear external 0 (overwrites again)

    chunk1_data = load(chunk1_path)  # (Linear external 8 is in chunk 1)
    assert chunk1_data[3] == {"c": 3}  # Linear external 8 (overwrites)


def test_chunked_file_array_slice_indices(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = ChunkedFileArray(folder, shape, chunk_size=1)

    key = (0, slice(None), slice(1, 3))
    indices = arr._slice_indices(key, base_shape=arr.resolved_shape)
    assert indices == [range(1), range(3), range(1, 3)]

    key2 = (slice(None), 1, slice(None, None, 2))
    indices = arr._slice_indices(key2, base_shape=arr.resolved_shape)
    assert indices == [range(2), range(1, 2), range(0, 4, 2)]


def test_chunked_file_array_normalize_key_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = ChunkedFileArray(folder, shape, chunk_size=1)

    key = (0, slice(None), 1)
    normalized_key = arr._normalize_key(key, for_dump=True)  # for dump, key is for external_shape
    assert normalized_key == (0, slice(None, None, None), 1)

    # for getitem, key is for full_shape
    normalized_key_get = arr._normalize_key(key, for_dump=False)
    assert normalized_key_get == (0, slice(None, None, None), 1)

    key2 = (slice(None), -1, slice(1, None, 2))
    normalized_key2_dump = arr._normalize_key(key2, for_dump=True)
    assert normalized_key2_dump == (slice(None, None, None), 2, slice(1, None, 2))

    normalized_key2_get = arr._normalize_key(key2, for_dump=False)
    assert normalized_key2_get == (slice(None, None, None), 2, slice(1, None, 2))

    with pytest.raises(IndexError):
        arr._normalize_key((0, slice(None), 10), for_dump=True)


def test_chunked_slice_indices_with_step_size(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (5, 6, 7)
    arr = ChunkedFileArray(folder, shape, chunk_size=1)

    key1 = (slice(0, 5, 2), 1, 2)
    indices1 = arr._slice_indices(key1, base_shape=arr.resolved_shape)
    assert indices1 == [range(0, 5, 2), range(1, 2), range(2, 3)]

    key2 = (slice(1, 4, 2), slice(2, 5, 2), slice(3, 7, 3))
    indices2 = arr._slice_indices(key2, base_shape=arr.resolved_shape)
    assert indices2 == [range(1, 4, 2), range(2, 5, 2), range(3, 7, 3)]

    key3 = (slice(4, 1, -2), slice(5, 2, -2), slice(6, 1, -3))
    indices3 = arr._slice_indices(key3, base_shape=arr.resolved_shape)
    assert indices3 == [range(4, 1, -2), range(5, 2, -2), range(6, 1, -3)]


def test_chunked_file_array_with_internal_arrays_full_array_different_order_simple(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (1,)  # external shape
    internal_shape = (2,)
    shape_mask = (True, False)
    arr = ChunkedFileArray(
        folder,
        shape,
        shape_mask=shape_mask,
        internal_shape=internal_shape,
        chunk_size=1,
    )
    full_shape = arr.full_shape  # This is (1, 2)
    assert full_shape == (1, 2)

    data1 = np.array([42, 69])
    arr.dump((0,), data1)  # Dumping external index (0,) with internal value data1

    r_no_splat = arr.to_array(splat_internal=False)
    assert r_no_splat.shape == (1,)
    assert np.array_equal(r_no_splat[0], data1)

    r_splat = arr.to_array(splat_internal=True)
    assert r_splat.shape == full_shape
    assert np.array_equal(r_splat[0, :], data1)

    # Check _slice_indices when used for __getitem__ like slicing (operates on full_shape)
    expected_slice_indices = [range(1), range(2)]  # Corresponds to full_shape (1,2)
    slice_indices = arr._slice_indices((slice(None), slice(None)), base_shape=arr.full_shape)
    assert slice_indices == expected_slice_indices

    # Check _normalize_key (this is always against full_shape for __getitem__)
    assert arr._normalize_key((slice(None), slice(None)), for_dump=False) == (
        slice(None),
        slice(None),
    )
    assert arr._normalize_key((0, 0), for_dump=False) == (0, 0)

    # Test __getitem__
    result_0_0 = arr[(0, 0)]  # full_key (0,0) -> external_key (0), internal_key (0)
    assert result_0_0 == 42

    result_0_slice = arr[(0, slice(None))]  # full_key (0, :) -> external (0), internal (:)
    assert np.array_equal(result_0_slice, np.array([42, 69]))

    result_slice_0 = arr[(slice(None), 0)]  # full_key (:, 0) -> external (:), internal (0)
    assert np.array_equal(result_slice_0, np.array([42]))

    result_slice_slice = arr[(slice(None), slice(None))]  # full_key (:,:)
    assert result_slice_slice.shape == (1, 2)
    assert np.array_equal(result_slice_slice, np.array([[42, 69]]))
