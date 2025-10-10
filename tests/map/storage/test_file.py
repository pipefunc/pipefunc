from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipefunc.map._storage_array._file import FileArray, _load_all, dump, load, select_by_mask

# The tests for all file array types are in `test_base_filearray.py`!
# Here are only the tests that are specific to the `FileArray` class.


def test_load_and_dump(tmp_path):
    obj = {"a": 1, "b": [2, 3]}
    file_path = tmp_path / "test.pickle"
    dump(obj, file_path)
    loaded_obj = load(file_path)
    assert loaded_obj == obj


def test_file_based_object_array_init(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    assert arr.folder == folder
    assert arr.shape == shape
    assert arr.strides == (12, 4, 1)
    assert arr.filename_template == "__{:d}__.pickle"


def test_file_based_object_array_normalize_key(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    assert arr._normalize_key((1, 2, 3)) == (1, 2, 3)
    assert arr._normalize_key((slice(None), 1, 2)) == (slice(None), 1, 2)
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 3, 4))
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 10))


def test_file_based_object_array_index_to_file(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    assert arr._index_to_file(0) == folder / "__0__.pickle"
    assert arr._index_to_file(23) == folder / "__23__.pickle"


def test_file_based_object_array_key_to_file(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    assert arr._key_to_file((0, 0, 0)) == folder / "__0__.pickle"
    assert arr._key_to_file((1, 2, 3)) == folder / "__23__.pickle"


def test_file_based_object_array_files(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3)
    arr = FileArray(folder, shape)
    files = list(arr._files())
    assert len(files) == 6
    assert files[0] == folder / "__0__.pickle"
    assert files[-1] == folder / "__5__.pickle"


def test_file_based_object_array_dump(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3)
    arr = FileArray(folder, shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert load(arr._key_to_file((0, 0))) == {"a": 1}
    assert load(arr._key_to_file((1, 2))) == {"b": 2}
    arr.dump((slice(0, 1), 0), {"c": 3})
    assert load(arr._key_to_file((0, 0))) == {"c": 3}


def test_load_all(tmp_path):
    file1 = tmp_path / "file1.pickle"
    file2 = tmp_path / "file2.pickle"
    file3 = tmp_path / "file3.pickle"  # Non-existent file
    dump({"a": 1}, file1)
    dump({"b": 2}, file2)
    result = _load_all([file1, file2, file3])
    assert result == [{"a": 1}, {"b": 2}, None]


def test_file_array_dump_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)

    # Test dumping with slicing along a single axis
    arr.dump((0, slice(None), 0), {"a": 1})
    assert load(arr._key_to_file((0, 0, 0))) == {"a": 1}
    assert load(arr._key_to_file((0, 1, 0))) == {"a": 1}
    assert load(arr._key_to_file((0, 2, 0))) == {"a": 1}

    # Test dumping with slicing along multiple axes
    arr.dump((slice(None), 0, 0), {"b": 2})
    assert load(arr._key_to_file((0, 0, 0))) == {"b": 2}
    assert load(arr._key_to_file((1, 0, 0))) == {"b": 2}

    # Test dumping with step
    arr.dump((0, slice(None, None, 2), 0), {"c": 3})
    assert load(arr._key_to_file((0, 0, 0))) == {"c": 3}
    assert load(arr._key_to_file((0, 2, 0))) == {"c": 3}


def test_file_array_slice_indices(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)

    key = (0, slice(None), slice(1, 3))
    indices = arr._slice_indices(key)
    assert indices == [range(1), range(3), range(1, 3)]

    key2 = (slice(None), 1, slice(None, None, 2))
    indices = arr._slice_indices(key2)
    assert indices == [range(2), range(1, 2), range(0, 4, 2)]


def test_file_array_normalize_key_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)

    key = (0, slice(None), 1)
    normalized_key = arr._normalize_key(key)
    assert normalized_key == (0, slice(None, None, None), 1)

    key2 = (slice(None), -1, slice(1, None, 2))
    normalized_key2 = arr._normalize_key(key2)
    assert normalized_key2 == (slice(None, None, None), 2, slice(1, None, 2))

    with pytest.raises(IndexError):
        arr._normalize_key((0, slice(None), 10))


def test_slice_indices_with_step_size(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (5, 6, 7)
    arr = FileArray(folder, shape)

    # Test case 1: Slice indices with step size along one axis
    key1 = (slice(0, 5, 2), 1, 2)
    indices1 = arr._slice_indices(key1)
    assert indices1 == [range(0, 5, 2), range(1, 2), range(2, 3)]

    # Test case 2: Slice indices with step size along multiple axes
    key2 = (slice(1, 4, 2), slice(2, 5, 2), slice(3, 7, 3))
    indices2 = arr._slice_indices(key2)
    assert indices2 == [range(1, 4, 2), range(2, 5, 2), range(3, 7, 3)]

    # Test case 3: Slice indices with step size and negative indices
    key3 = (slice(4, 1, -2), slice(5, 2, -2), slice(6, 1, -3))
    indices3 = arr._slice_indices(key3)
    assert indices3 == [range(4, 1, -2), range(5, 2, -2), range(6, 1, -3)]


def test_key_to_file(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (3, 4, 5)
    arr = FileArray(folder, shape)

    key = (1, 2, 3)
    file_path = arr._key_to_file(key)
    assert file_path == (folder / "__33__.pickle")

    key = (0, 1, 2)
    file_path = arr._key_to_file(key)
    assert file_path == (folder / "__7__.pickle")


def test_file_array_with_internal_arrays_full_array_different_order_simple(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (1,)
    internal_shape = (2,)
    shape_mask = (True, False)  # means shape is (1, 2)
    full_shape = select_by_mask(shape_mask, shape, internal_shape)
    assert full_shape == (1, 2)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.array([42, 69])
    expected_full = np.ma.MaskedArray(data1, mask=False, dtype=object).reshape(full_shape)

    arr.dump((0,), data1)

    r = arr.to_array(splat_internal=False)
    assert r.shape == (1,)
    r = arr.to_array(splat_internal=True)
    assert r.shape == full_shape

    # Check _slice_indices
    expected_slice_indices = [range(1), range(2)]
    slice_indices = arr._slice_indices((slice(None), slice(None)))
    assert slice_indices == expected_slice_indices

    # Check _normalize_key
    assert arr._normalize_key((slice(None), slice(None))) == (slice(None), slice(None))
    assert arr._normalize_key((0, 0)) == (0, 0)

    assert expected_full.shape == (1, 2)
    assert expected_full[:, 0].shape == (1,)

    result = arr[:, 0]
    assert result.shape == (1,)

    result = arr[:, :]
    assert result.shape == full_shape
    expected1 = data1.reshape(1, 2)
    expected1 = np.ma.MaskedArray(expected1, mask=False, dtype=object)
    assert np.array_equal(result, expected1)

    result = arr[0, :]
    assert result.shape == (2,), result.shape
    expected2 = data1
    expected2 = np.ma.MaskedArray(expected2, mask=False, dtype=object)
    assert np.array_equal(result, expected2)
