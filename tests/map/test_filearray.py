from pathlib import Path

import numpy as np
import pytest

from pipefunc._utils import prod
from pipefunc.map._filearray import FileArray, _load_all, _select_by_mask, dump, load


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


def test_file_based_object_array_properties(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    assert arr.size == 24
    assert arr.rank == 3


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


def test_file_based_object_array_getitem(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3)
    arr = FileArray(folder, shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr[0, 0] == {"a": 1}
    assert arr[1, 2] == {"b": 2}
    assert arr[0, 1] is np.ma.masked
    assert arr[0:1, 0] == {"a": 1}


def test_file_based_object_array_to_array(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3)
    arr = FileArray(folder, shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    result = arr.to_array()
    assert result.shape == (2, 3)
    assert result.dtype == object
    assert result[0, 0] == {"a": 1}
    assert result[1, 2] == {"b": 2}
    assert result[0, 1] is np.ma.masked
    assert result[1, 0] is np.ma.masked


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


def test_file_array_getitem_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4)
    arr = FileArray(folder, shape)
    arr.dump((0, 0, 0), {"a": 1})
    arr.dump((0, 1, 0), {"b": 2})
    arr.dump((1, 0, 0), {"c": 3})
    arr.dump((1, 1, 0), {"d": 4})

    # Test slicing along a single axis
    result = arr[0, :, 0]
    assert result.shape == (3,)
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] is np.ma.masked

    # Test slicing along multiple axes
    result = arr[1, 0:2, 0]
    assert result.shape == (2,)
    assert result[0] == {"c": 3}
    assert result[1] == {"d": 4}

    # Test slicing with step
    result = arr[0, ::2, 0]
    assert result.shape == (2,)
    assert result[0] == {"a": 1}
    assert result[1] is np.ma.masked


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


def test_high_dim_with_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 3, 4, 5)
    arr = FileArray(folder, shape)
    np_arr: np.ndarray = np.zeros(shape, dtype=object)
    np_arr[:] = np.ma.masked
    keys = [
        (0, 0, slice(None), 0),
        (0, 1, slice(None), 0),
        (1, 0, slice(None), 0),
        (1, 1, slice(None), 0),
    ]
    values = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    for k, v in zip(keys, values):
        np_arr[k] = v
        arr.dump(k, v)

    assert (arr.to_array() == np_arr).all()

    # Test slicing along a single axis
    result = arr[0, :, :, 0]
    assert result.shape == (3, 4), result.shape
    assert result[0, 0] == {"a": 1}
    assert result[1, 0] == {"b": 2}
    assert result[2, 0] is np.ma.masked

    # Test slicing along multiple axes
    result = arr[1, 0:2, :, 0]
    assert result.shape == (2, 4)
    assert result[0, 0] == {"c": 3}
    assert result[1, 0] == {"d": 4}

    # Test slicing with step
    result = arr[0, ::2, :, 0]
    result_arr = np_arr[0, ::2, :, 0]
    assert result.shape == (2, 4)
    assert result[0, 0] == result_arr[0, 0]
    assert result[1, 0] is np.ma.masked
    assert result[0, 1] == result_arr[0, 1]
    assert result[1, 1] is np.ma.masked


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


def test_sliced_arange(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (3, 4, 5)
    arr = FileArray(folder, shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[:, :, :] == np_arr[:, :, :]).all()
    assert (arr[:, ::2, :] == np_arr[:, ::2, :]).all()
    assert (arr[:, ::3, :] == np_arr[:, ::3, :]).all()
    assert (arr[:, ::-1, :] == np_arr[:, ::-1, :]).all()
    assert (arr[:, ::-1, ::2] == np_arr[:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::-1] == np_arr[1:, ::-1, ::-1]).all()
    assert (arr[1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()
    assert (arr[2, ::-1, -1] == np_arr[2, ::-1, -1]).all()
    assert (arr[:1, :1, -1] == np_arr[:1, :1, -1]).all()
    assert (arr[:1, :1, 5:1:-1] == np_arr[:1, :1, 5:1:-1]).all()


def test_sliced_arange_minimal(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (1, 2)
    arr = FileArray(folder, shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[:, 1] == np_arr[:, 1]).all()
    assert (arr[0, -1] == np_arr[0, -1]).all()
    assert (arr[:, -1] == np_arr[:, -1]).all()


def test_sliced_arange_minimal2(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2, 4)
    arr = FileArray(folder, shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[0, :, 1] == np_arr[0, :, 1]).all()
    assert (arr[0, 0, -1] == np_arr[0, 0, -1]).all()
    assert (arr[0, :, -1] == np_arr[0, :, -1]).all()
    assert (arr[0, ::-1, 0] == np_arr[0, ::-1, 0]).all()
    assert (arr[0, ::-1, -1] == np_arr[0, ::-1, -1]).all()
    assert (arr[:, ::-1, -1] == np_arr[:, ::-1, -1]).all()
    assert (arr[1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()


def test_file_array_with_internal_arrays(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)
    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)
    full_shape = (2, 2, 3, 3, 4)
    assert _select_by_mask(shape_mask, shape, internal_shape) == full_shape
    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test indexing into the internal arrays
    assert np.array_equal(arr[0, 0, 1, 1, 1], data1[1, 1, 1])
    assert np.array_equal(arr[1, 1, 2, 2, 2], data2[2, 2, 2])

    # Test slicing that includes internal array dimensions
    result = arr[0, :, 1, 1, 1]
    expected: np.ma.masked_array = np.ma.masked_array(
        [data1[1, 1, 1], np.ma.masked],
        mask=[False, True],
        dtype=object,
    )
    assert np.ma.allequal(result, expected)

    result = arr[1, :, 2, 2, 2]
    expected = np.ma.masked_array(
        [np.ma.masked, data2[2, 2, 2]],
        mask=[True, False],
        dtype=object,
    )
    assert np.ma.allequal(result, expected)


def test_file_array_with_internal_arrays_slicing(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test full slicing including internal array dimensions
    result = arr[:, :, 1, 1, 1]
    expected: np.ma.masked_array = np.ma.masked_array(
        [[data1[1, 1, 1], np.ma.masked], [np.ma.masked, data2[1, 1, 1]]],
        mask=[[False, True], [True, False]],
        dtype=object,
    )
    assert np.ma.allequal(result, expected)

    result = arr[:, :, 2, 2, 2]
    expected = np.ma.masked_array(
        [
            [np.ma.masked, np.ma.masked],
            [np.ma.masked, data2[2, 2, 2]],
        ],
        mask=[
            [True, True],
            [True, False],
        ],
        dtype=object,
    )
    assert np.ma.allequal(result, expected)


def test_file_array_with_internal_arrays_full_array(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array
    result = arr.to_array(splat_internal=False)
    assert result.shape == (2, 2)
    assert np.array_equal(result[0, 0], data1)
    assert result[0, 1] is np.ma.masked
    assert result[1, 0] is np.ma.masked
    assert np.array_equal(result[1, 1], data2)


def test_file_array_with_internal_arrays_splat(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array with splat_internal=True
    result = arr.to_array(splat_internal=True)
    expected_shape = (2, 2, *internal_shape)
    assert result.shape == expected_shape
    assert np.array_equal(result[0, 0], data1)
    assert np.ma.is_masked(result[0, 1])
    assert np.ma.is_masked(result[1, 0])
    assert np.array_equal(result[1, 1], data2)


def test_file_array_with_internal_arrays_splat_different_order(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array with splat_internal=True
    result = arr.to_array(splat_internal=True)
    expected_shape = (3, 2, 2, 3, 4)
    assert result.shape == expected_shape


def test_file_array_with_internal_arrays_splat_1(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array with splat_internal=True
    result = arr.to_array(splat_internal=True)
    expected_shape = (3, 2, 2, 3, 4)
    assert result.shape == expected_shape


def test_file_array_with_internal_arrays_full_array_different_order(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    assert arr.to_array(splat_internal=False).shape == (2, 2)
    full_shape = _select_by_mask(shape_mask, shape, internal_shape)
    assert arr.to_array(splat_internal=True).shape == full_shape

    # Test slicing
    result = arr[:, 0, 0, :, :]
    expected = np.ma.array(data1, mask=False, dtype=object)
    assert np.array_equal(result, expected)


def test_file_array_with_internal_arrays_full_array_different_order_simple(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (1,)
    internal_shape = (2,)
    shape_mask = (True, False)  # means shape is (1, 2)
    full_shape = _select_by_mask(shape_mask, shape, internal_shape)
    assert full_shape == (1, 2)

    arr = FileArray(folder, shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.array([42, 69])
    expected_full = np.ma.array(data1, mask=False, dtype=object).reshape(full_shape)

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
    expected = data1.reshape(1, 2)
    expected = np.ma.array(expected, mask=False, dtype=object)
    assert np.array_equal(result, expected)

    result = arr[0, :]
    assert result.shape == (2,), result.shape
    expected = data1
    expected = np.ma.array(expected, mask=False, dtype=object)
    assert np.array_equal(result, expected)


def test_sliced_arange_splat(tmp_path: Path):
    folder = Path(tmp_path)
    shape = (1,)
    internal_shape = (3, 4, 5)
    arr = FileArray(
        folder,
        shape,
        internal_shape=internal_shape,
        shape_mask=(True, False, False, False),
    )
    np_arr = np.arange(prod(internal_shape)).reshape(internal_shape)
    arr.dump((0,), np_arr)

    assert (arr[0, :, :, :] == np_arr[:, :, :]).all()
    assert (arr[0, :, ::2, :] == np_arr[:, ::2, :]).all()
    assert (arr[0, :, ::3, :] == np_arr[:, ::3, :]).all()
    assert (arr[0, :, ::-1, :] == np_arr[:, ::-1, :]).all()
    assert (arr[0, :, ::-1, ::2] == np_arr[:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::-1] == np_arr[1:, ::-1, ::-1]).all()
    assert (arr[0, 1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()
    assert (arr[0, 2, ::-1, -1] == np_arr[2, ::-1, -1]).all()
    assert (arr[0, :1, :1, -1] == np_arr[:1, :1, -1]).all()
    assert (arr[0, :1, :1, 5:1:-1] == np_arr[:1, :1, 5:1:-1]).all()


def test_exceptions(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="shape_mask must be provided if internal_shape is provided",
    ):
        FileArray(tmp_path, shape=(1, 2), internal_shape=(2, 3))
    with pytest.raises(
        ValueError,
        match="shape_mask must have the same length",
    ):
        FileArray(tmp_path, shape=(1, 2), internal_shape=(2, 3), shape_mask=(True, True, False))
    arr = FileArray(tmp_path, shape=(2,))
    arr.dump((0,), np.array([1, 2]))
    with pytest.raises(
        ValueError,
        match="internal_shape must be provided if splat_internal is True",
    ):
        arr.to_array(splat_internal=True)


@pytest.mark.parametrize("typ", [list, np.array])
def test_internal_shape_list(typ: type, tmp_path: Path) -> None:
    arr = FileArray(tmp_path, shape=(2,), internal_shape=(2,), shape_mask=(True, False))
    arr.dump((0,), typ([1, 2]))
    arr.dump((1,), typ([3, 4]))
    if typ == list:
        assert arr[0, :].tolist() == [1, 2]
        assert arr[1, :].tolist() == [3, 4]
    else:
        assert arr[0, :].tolist() == [1, 2]
        assert arr[1, :].tolist() == [3, 4]
    assert arr.to_array().tolist() == [[1, 2], [3, 4]]
    assert arr[:, :].shape == (2, 2)


def test_internal_nparray_with_dicts(tmp_path: Path) -> None:
    arr = FileArray(tmp_path, shape=(2,), internal_shape=(2,), shape_mask=(True, False))
    arr.dump((0,), np.array([{"a": 1}, {"b": 2}], dtype=object))
    arr.dump((1,), np.array([{"c": 1}, {"d": 2}], dtype=object))
    assert arr.to_array().tolist() == [[{"a": 1}, {"b": 2}], [{"c": 1}, {"d": 2}]]
    assert arr[0, 0] == {"a": 1}
    assert arr[0, 1] == {"b": 2}
    assert arr[1, 0] == {"c": 1}
    assert arr[1, 1] == {"d": 2}
    assert arr[:, 0].tolist() == [{"a": 1}, {"c": 1}]
