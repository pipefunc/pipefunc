from pathlib import Path

import numpy as np
import pytest
import zarr

from pipefunc._utils import prod
from pipefunc.map._filearray import FileArray, _select_by_mask
from pipefunc.map.zarr import ZarrArray


@pytest.fixture(params=["file_array", "zarr_array"])
def array_type(request, tmp_path: Path):
    if request.param == "file_array":

        def _array_type(shape, internal_shape=None, shape_mask=None):
            return FileArray(tmp_path, shape, internal_shape, shape_mask)
    elif request.param == "zarr_array":

        def _array_type(shape, internal_shape=None, shape_mask=None):
            store = zarr.MemoryStore()
            return ZarrArray(store, shape, internal_shape, shape_mask)

    return _array_type


def test_file_based_object_array_getitem(array_type):
    shape = (2, 3)
    arr = array_type(shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr[0, 0] == {"a": 1}
    assert arr[1, 2] == {"b": 2}
    assert arr[0, 1] is np.ma.masked
    assert arr[0:1, 0] == {"a": 1}


def test_file_based_object_array_properties(array_type):
    shape = (2, 3, 4)
    arr = array_type(shape)
    assert arr.size == 24
    assert arr.rank == 3


def test_file_based_object_array_to_array(array_type):
    shape = (2, 3)
    arr = array_type(shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    result = arr.to_array()
    assert result.shape == (2, 3)
    assert result.dtype == object
    assert result[0, 0] == {"a": 1}
    assert result[1, 2] == {"b": 2}
    assert result[0, 1] is np.ma.masked
    assert result[1, 0] is np.ma.masked


def test_file_array_getitem_with_slicing(array_type):
    shape = (2, 3, 4)
    arr = array_type(shape)
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


def test_high_dim_with_slicing(array_type):
    shape = (2, 3, 4, 5)
    arr = array_type(shape)
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


def test_sliced_arange(array_type):
    shape = (3, 4, 5)
    arr = array_type(shape)
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


def test_sliced_arange_minimal(array_type):
    shape = (1, 2)
    arr = array_type(shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[:, 1] == np_arr[:, 1]).all()
    assert (arr[0, -1] == np_arr[0, -1]).all()
    assert (arr[:, -1] == np_arr[:, -1]).all()


def test_sliced_arange_minimal2(array_type):
    shape = (2, 2, 4)
    arr = array_type(shape)
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


def test_file_array_with_internal_arrays(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)
    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)
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


def test_file_array_with_internal_arrays_slicing(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

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


def test_file_array_with_internal_arrays_full_array(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

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


def test_file_array_with_internal_arrays_splat(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

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


def test_file_array_with_internal_arrays_splat_different_order(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array with splat_internal=True
    result = arr.to_array(splat_internal=True)
    expected_shape = (3, 2, 2, 3, 4)
    assert result.shape == expected_shape


def test_file_array_with_internal_arrays_splat_1(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array with splat_internal=True
    result = arr.to_array(splat_internal=True)
    expected_shape = (3, 2, 2, 3, 4)
    assert result.shape == expected_shape


def test_file_array_with_internal_arrays_full_array_different_order(array_type):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

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


def test_sliced_arange_splat(array_type):
    shape = (1,)
    internal_shape = (3, 4, 5)
    arr = array_type(
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


def test_exceptions(array_type) -> None:
    with pytest.raises(
        ValueError,
        match="shape_mask must be provided if internal_shape is provided",
    ):
        array_type(shape=(1, 2), internal_shape=(2, 3))
    with pytest.raises(
        ValueError,
        match="shape_mask must have the same length",
    ):
        array_type(shape=(1, 2), internal_shape=(2, 3), shape_mask=(True, True, False))
    arr = array_type(shape=(2,))
    arr.dump((0,), np.array([1, 2]))
    with pytest.raises(
        ValueError,
        match="internal_shape must be provided if splat_internal is True",
    ):
        arr.to_array(splat_internal=True)


@pytest.mark.parametrize("typ", [list, np.array])
def test_internal_shape_list(typ: type, array_type) -> None:
    arr = array_type(shape=(2,), internal_shape=(2,), shape_mask=(True, False))
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


def test_internal_nparray_with_dicts(array_type) -> None:
    arr = array_type(shape=(2,), internal_shape=(2,), shape_mask=(True, False))
    arr.dump((0,), np.array([{"a": 1}, {"b": 2}], dtype=object))
    arr.dump((1,), np.array([{"c": 1}, {"d": 2}], dtype=object))
    assert arr.to_array().tolist() == [[{"a": 1}, {"b": 2}], [{"c": 1}, {"d": 2}]]
    assert arr[0, 0] == {"a": 1}
    assert arr[0, 1] == {"b": 2}
    assert arr[1, 0] == {"c": 1}
    assert arr[1, 1] == {"d": 2}
    assert arr[:, 0].tolist() == [{"a": 1}, {"c": 1}]


def test_list_or_arrays(array_type) -> None:
    shape = (1, 1, 1)
    mask = (True, True, True)
    arr = array_type(shape=shape, shape_mask=mask, internal_shape=())
    # list of 2 (4,4) arrays
    value = [np.random.rand(4, 4) for _ in range(2)]  # noqa: NPY002
    arr.dump((0, 0, 0), value)
    assert isinstance(arr[0, 0, 0], list)
    for x in arr[0, 0, 0]:
        assert isinstance(x, np.ndarray)
        assert x.shape == (4, 4)
    r = arr[:, :, :]
    assert r.shape == (1, 1, 1)
    assert r.dtype == object
    assert isinstance(r[0, 0, 0], list)
    assert np.array_equal(r[0, 0, 0], value)

    r = arr[:, 0, 0]
    assert r.shape == (1,)
    assert r.dtype == object
    assert isinstance(r[0], list)
    assert np.array_equal(r[0], value)
