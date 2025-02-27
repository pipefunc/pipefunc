from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc._utils import prod
from pipefunc.map._storage_array._base import (
    StorageBase,
    get_storage_class,
    iterate_shape_indices,
    select_by_mask,
)
from pipefunc.map._storage_array._dict import DictArray
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

has_zarr = importlib.util.find_spec("zarr") is not None


@pytest.fixture(params=["file_array", "zarr_array", "dict"])
def array_type(request, tmp_path: Path):
    if request.param == "file_array":

        def _array_type(shape, internal_shape=None, shape_mask=None):
            return FileArray(tmp_path, shape, internal_shape, shape_mask)
    elif request.param == "zarr_array":
        if not has_zarr:
            pytest.skip("zarr not installed")

        def _array_type(shape, internal_shape=None, shape_mask=None):
            import zarr

            from pipefunc.map import ZarrFileArray

            store = zarr.MemoryStore()
            return ZarrFileArray(None, shape, internal_shape, shape_mask, store=store)
    elif request.param == "dict":

        def _array_type(shape, internal_shape=None, shape_mask=None):
            return DictArray(None, shape, internal_shape, shape_mask)

    return _array_type


def test_file_based_object_array_getitem(array_type: Callable[..., StorageBase]):
    shape = (2, 3)
    arr = array_type(shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr[0, 0] == {"a": 1}
    assert arr[1, 2] == {"b": 2}
    assert arr[0, 1] is np.ma.masked
    assert arr[0:1, 0] == {"a": 1}


def test_file_based_object_array_properties(array_type: Callable[..., StorageBase]):
    shape = (2, 3, 4)
    arr = array_type(shape)
    assert arr.size == 24
    assert arr.rank == 3


def test_file_based_object_array_to_array(array_type: Callable[..., StorageBase]):
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


def test_file_array_getitem_with_slicing(array_type: Callable[..., StorageBase]):
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


def test_high_dim_with_slicing(array_type: Callable[..., StorageBase]):
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


def test_sliced_arange(array_type: Callable[..., StorageBase]):
    shape = (3, 4, 5)
    arr = array_type(shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[:, :, :] == np_arr[:, :, :]).all()
    assert (arr[:, ::2, :] == np_arr[:, ::2, :]).all()
    assert (arr[:, ::3, :] == np_arr[:, ::3, :]).all()
    if arr.storage_id == "zarr_file_array":
        return  # ZarrFileArray does not support negative step
    assert (arr[:, ::-1, :] == np_arr[:, ::-1, :]).all()
    assert (arr[:, ::-1, ::2] == np_arr[:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[1:, ::-1, ::-1] == np_arr[1:, ::-1, ::-1]).all()
    assert (arr[1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()
    assert (arr[2, ::-1, -1] == np_arr[2, ::-1, -1]).all()
    assert (arr[:1, :1, -1] == np_arr[:1, :1, -1]).all()
    assert (arr[:1, :1, 5:1:-1] == np_arr[:1, :1, 5:1:-1]).all()


def test_sliced_arange_minimal(array_type: Callable[..., StorageBase]):
    shape = (1, 2)
    arr = array_type(shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[:, 1] == np_arr[:, 1]).all()
    assert (arr[0, -1] == np_arr[0, -1]).all()
    assert (arr[:, -1] == np_arr[:, -1]).all()
    if arr.storage_id != "zarr_file_array":
        assert (arr[:, ::-1] == np_arr[:, ::-1]).all()


def test_sliced_arange_minimal2(array_type: Callable[..., StorageBase]):
    shape = (2, 2, 4)
    arr = array_type(shape)
    np_arr = np.arange(prod(shape)).reshape(shape)
    for key in np.ndindex(shape):
        arr.dump(key, np_arr[key])

    assert (arr[0, :, 1] == np_arr[0, :, 1]).all()
    assert (arr[0, 0, -1] == np_arr[0, 0, -1]).all()
    assert (arr[0, :, -1] == np_arr[0, :, -1]).all()
    if arr.storage_id == "zarr_file_array":
        return  # ZarrFileArray does not support negative step
    assert (arr[0, ::-1, 0] == np_arr[0, ::-1, 0]).all()
    assert (arr[0, ::-1, -1] == np_arr[0, ::-1, -1]).all()
    assert (arr[:, ::-1, -1] == np_arr[:, ::-1, -1]).all()
    assert (arr[1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()


def test_file_array_with_internal_arrays(array_type: Callable[..., StorageBase]):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)
    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)
    full_shape = (2, 2, 3, 3, 4)
    assert select_by_mask(shape_mask, shape, internal_shape) == full_shape
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


def test_file_array_with_internal_arrays_slicing(array_type: Callable[..., StorageBase]):
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


def test_file_array_with_internal_arrays_full_array(array_type: Callable[..., StorageBase]):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test retrieving the entire array
    if arr.storage_id == "zarr_file_array":
        with pytest.raises(NotImplementedError):
            arr.to_array(splat_internal=False)
        return
    result = arr.to_array(splat_internal=False)
    assert result.shape == (2, 2)
    assert np.array_equal(result[0, 0], data1)
    assert result[0, 1] is np.ma.masked
    assert result[1, 0] is np.ma.masked
    assert np.array_equal(result[1, 1], data2)


def test_file_array_with_internal_arrays_splat(array_type: Callable[..., StorageBase]):
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


def test_file_array_with_internal_arrays_splat_different_order(
    array_type: Callable[..., StorageBase],
):
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


def test_file_array_with_internal_arrays_splat_1(array_type: Callable[..., StorageBase]):
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


def test_file_array_with_internal_arrays_full_array_different_order(
    array_type: Callable[..., StorageBase],
):
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (False, True, True, False, False)

    arr = array_type(shape, shape_mask=shape_mask, internal_shape=internal_shape)

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    if arr.storage_id == "zarr_file_array":
        with pytest.raises(NotImplementedError):
            arr.to_array(splat_internal=False)
    else:
        assert arr.to_array(splat_internal=False).shape == (2, 2)
    full_shape = select_by_mask(shape_mask, shape, internal_shape)
    assert arr.to_array(splat_internal=True).shape == full_shape

    # Test slicing
    result = arr[:, 0, 0, :, :]
    expected = np.ma.array(data1, mask=False, dtype=object)
    assert np.array_equal(result, expected)


def test_sliced_arange_splat(array_type: Callable[..., StorageBase]):
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
    assert (arr[0, :, ::2, :] == np_arr[:, ::2, :]).all()
    assert (arr[0, :, ::2, ::2] == np_arr[:, ::2, ::2]).all()
    assert (arr[0, 1:, ::2, ::2] == np_arr[1:, ::2, ::2]).all()
    assert (arr[0, 1:, ::2, ::2] == np_arr[1:, ::2, ::2]).all()
    assert (arr[0, 1:, ::2, ::2] == np_arr[1:, ::2, ::2]).all()
    assert (arr[0, 1:, ::2, -1] == np_arr[1:, ::2, -1]).all()
    assert (arr[0, 2, ::2, -1] == np_arr[2, ::2, -1]).all()
    assert (arr[0, :1, :1, -1] == np_arr[:1, :1, -1]).all()
    if arr.storage_id == "zarr_file_array":
        return  # ZarrFileArray does not support negative step
    assert (arr[0, :, ::-1, :] == np_arr[:, ::-1, :]).all()
    assert (arr[0, :, ::-1, ::2] == np_arr[:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
    assert (arr[0, 1:, ::-1, ::-1] == np_arr[1:, ::-1, ::-1]).all()
    assert (arr[0, 1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()
    assert (arr[0, 2, ::-1, -1] == np_arr[2, ::-1, -1]).all()
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
    if typ is list:
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
    if arr.storage_id != "zarr_file_array":
        assert isinstance(arr[0, 0, 0], list)
    else:
        assert isinstance(arr[0, 0, 0], np.ndarray)
        assert arr[0, 0, 0].shape == (2, 4, 4)
    for x in arr[0, 0, 0]:  # type: ignore[attr-defined]
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
    assert arr.has_index(0)


def test_with_internal_shape_list(array_type) -> None:
    shape = (1, 1)
    internal_shape = (2,)
    mask = (True, True, False)
    arr = array_type(shape=shape, internal_shape=internal_shape, shape_mask=mask)
    value = [1, 2]
    arr.dump((0, slice(None)), value)
    assert arr[0, 0, 0] == 1
    assert arr[0, 0, 1] == 2


@pytest.mark.skipif(not has_zarr, reason="zarr not installed")
def test_compare_equal(tmp_path: Path) -> None:
    from pipefunc.map import ZarrFileArray

    external_shape = (2, 3)
    internal_shape = (4, 5)
    z_arr = ZarrFileArray(
        tmp_path / "zarr",
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    f_arr = FileArray(
        tmp_path / "filearray",
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    d_arr = DictArray(
        None,
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    arrs = [f_arr, z_arr, d_arr]
    for index in iterate_shape_indices(external_shape):
        x = np.random.rand(*internal_shape)  # noqa: NPY002
        for arr in arrs:
            arr.dump(key=index, value=x)
    base_arr = arrs[0]
    for arr in arrs[1:]:
        assert np.array_equal(base_arr.to_array(), arr.to_array())
        assert np.array_equal(base_arr[:, :, :, :], arr[:, :, :, :])
        assert np.array_equal(base_arr[0, :, :, :], arr[0, :, :, :])
        assert np.array_equal(base_arr[1, :, :, :], arr[1, :, :, :])
        assert np.array_equal(base_arr[0, 0, :, :], arr[0, 0, :, :])
        assert np.array_equal(base_arr[1, 0, :, :], arr[1, 0, :, :])
        assert np.array_equal(base_arr[0, -1, :, :], arr[0, -1, :, :])
        assert base_arr.size == arr.size
        assert base_arr.rank == arr.rank
        assert base_arr.shape == arr.shape
        assert base_arr.internal_shape == arr.internal_shape
        assert base_arr.shape_mask == arr.shape_mask
        assert np.array_equal(base_arr.get_from_index(0), arr.get_from_index(0))
        assert np.array_equal(base_arr.get_from_index(5), arr.get_from_index(5))
        assert base_arr.has_index(0), arr.has_index(0)
        assert base_arr.full_shape == arr.full_shape
        assert base_arr.strides == arr.strides
        assert np.array_equal(base_arr.mask[0, 0], arr.mask[0, 0])
        assert np.array_equal(base_arr.mask, arr.mask)
        assert np.array_equal(base_arr.mask_linear(), arr.mask_linear())

    with pytest.raises(ValueError, match="is out of bounds"):
        z_arr.get_from_index(1_000_000)
    with pytest.raises(ValueError, match="is out of bounds"):
        d_arr.get_from_index(1_000_000)
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        f_arr.get_from_index(1_000_000)

    # Now with a partially filled array and compare masks
    z_arr = ZarrFileArray(
        tmp_path / "zarr2",
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    f_arr = FileArray(
        tmp_path / "filearray2",
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    dict_arr = DictArray(
        None,
        external_shape,
        internal_shape,
        shape_mask=(True, False, True, False),
    )
    arrs = [f_arr, z_arr, dict_arr]
    for index in iterate_shape_indices(external_shape):
        if np.random.rand() < 0.5:  # noqa: NPY002
            continue
        x = np.random.rand(*internal_shape)  # noqa: NPY002
        for arr in arrs:
            arr.dump(key=index, value=x)
    base_arr = arrs[0]
    for arr in arrs[1:]:
        assert np.ma.allequal(base_arr.mask, arr.mask), arr
        assert np.array_equal(base_arr.mask_linear(), arr.mask_linear()), arr
        assert np.ma.allequal(base_arr.to_array(), arr.to_array()), arr


def test_repr(array_type: Callable[..., StorageBase]):
    shape = (2, 3)
    arr = array_type(shape)
    repr(arr)


@pytest.mark.parametrize("storage_id", ["file_array", "shared_memory_dict", "dict"])
def test_persist(storage_id, tmp_path: Path) -> None:
    shape = (1,)
    internal_shape = (2,)
    shape_mask = (False, True)
    array_class = get_storage_class(storage_id)
    arr = array_class(tmp_path, shape, internal_shape=internal_shape, shape_mask=shape_mask)
    x = [0, 1]
    arr.dump((0,), x)
    arr.persist()
    y_original = arr.to_array()

    arr_new = array_class(tmp_path, shape, internal_shape=internal_shape, shape_mask=shape_mask)
    y_new = arr_new.to_array()
    np.testing.assert_almost_equal(y_original, y_new)


@pytest.mark.parametrize("storage_id", ["file_array", "shared_memory_dict", "dict"])
def test_size_one_with_internal_shape(storage_id, tmp_path: Path) -> None:
    shape = (1,)
    internal_shape = (2,)
    shape_mask = (False, True)
    array_class = get_storage_class(storage_id)
    arr = array_class(tmp_path, shape, internal_shape=internal_shape, shape_mask=shape_mask)
    x = np.arange(0, 6).reshape((2, 3))
    arr.dump((0,), x)
    arr.persist()
    y_original = arr.to_array()

    arr_new = array_class(tmp_path, shape, internal_shape=internal_shape, shape_mask=shape_mask)
    y_new = arr_new.to_array()
    for i in range(*internal_shape):
        for j in range(*shape):
            assert np.array_equal(y_original[i, j], y_new[i, j])
