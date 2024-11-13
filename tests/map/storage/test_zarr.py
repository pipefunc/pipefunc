from __future__ import annotations

import numpy as np
import zarr

from pipefunc.map._storage_array._zarr import CloudPickleCodec, ZarrFileArray, select_by_mask


def test_zarr_array_init():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    assert arr.shape == shape
    assert arr.strides == (12, 4, 1)


def test_zarr_array_properties():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    assert arr.size == 24
    assert arr.rank == 3
    assert str(arr.array.filters[0]) == "CloudPickleCodec(protocol=5)"


def test_zarr_array_getitem():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr[0, 0] == {"a": 1}
    assert arr[1, 2] == {"b": 2}
    assert arr[0, 1] is np.ma.masked
    assert arr[0:1, 0] == {"a": 1}
    assert arr.has_index(0)
    assert not arr.has_index(3)
    assert arr.dump_in_subprocess


def test_zarr_array_to_array():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    result = arr.to_array()
    assert result.shape == (2, 3)
    assert result.dtype == object
    assert result[0, 0] == {"a": 1}
    assert result[1, 2] == {"b": 2}
    assert result[0, 1] is np.ma.masked
    assert result[1, 0] is np.ma.masked


def test_zarr_array_dump():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr.get_from_index(0) == {"a": 1}
    assert arr.get_from_index(5) == {"b": 2}
    arr.dump((slice(0, 1), 0), {"c": 3})
    assert arr.get_from_index(0) == {"c": 3}


def test_zarr_array_getitem_with_slicing():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
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


def test_zarr_array_with_internal_arrays():
    store = zarr.MemoryStore()
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)
    arr = ZarrFileArray(
        folder=None,
        store=store,
        shape=shape,
        shape_mask=shape_mask,
        internal_shape=internal_shape,
    )
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
    expected = np.ma.masked_array(
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


def test_zarr_array_with_internal_arrays_slicing():
    store = zarr.MemoryStore()
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)

    arr = ZarrFileArray(
        folder=None,
        store=store,
        shape=shape,
        shape_mask=shape_mask,
        internal_shape=internal_shape,
    )

    data1 = np.arange(np.prod(internal_shape)).reshape(internal_shape)
    data2 = np.ones(internal_shape)

    arr.dump((0, 0), data1)
    arr.dump((1, 1), data2)

    # Test full slicing including internal array dimensions
    result = arr[:, :, 1, 1, 1]
    expected = np.ma.masked_array(
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


def test_zarr_array_set_and_get_single_item():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)

    arr.dump((0, 0), {"a": 1})
    assert arr[0, 0] == {"a": 1}

    arr.dump((1, 2), {"b": 2})
    assert arr[1, 2] == {"b": 2}


def test_zarr_array_set_and_get_single_item_with_internal_shape():
    shape = (2, 2)
    internal_shape = (3, 3)
    shape_mask = (True, True, False, False)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(
        folder=None,
        store=store,
        shape=shape,
        internal_shape=internal_shape,
        shape_mask=shape_mask,
    )

    data1 = np.arange(9).reshape(3, 3)
    arr.dump((0, 0), data1)
    assert np.array_equal(arr[0, 0], data1)

    data2 = np.ones((3, 3))
    arr.dump((1, 1), data2)
    assert np.array_equal(arr[1, 1], data2)


def test_zarr_array_set_and_get_single_item_with_internal_shape_and_indexing():
    shape = (2, 2)
    internal_shape = (3, 3)
    shape_mask = (True, True, False, False)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(
        folder=None,
        store=store,
        shape=shape,
        internal_shape=internal_shape,
        shape_mask=shape_mask,
    )

    data1 = np.arange(9).reshape(3, 3)
    arr.dump((0, 0), data1)
    assert arr[0, 0, 1, 1] == data1[1, 1]

    data2 = np.ones((3, 3))
    arr.dump((1, 1), data2)
    assert arr[1, 1, 2, 2] == data2[2, 2]


def test_zarr_array_set_and_get_slice():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)

    data1 = 42
    arr.dump((slice(None), slice(None), slice(None)), data1)
    assert np.all(arr[:, :, :] == data1)

    data2 = 69
    arr.dump((slice(1, 2), slice(None), slice(1, 3)), data2)
    assert np.all(arr[1, :, 1] == data2)


def test_zarr_array_set_and_get_slice_with_internal_shape():
    shape = (2, 2)
    internal_shape = (3, 3, 4)
    shape_mask = (True, True, False, False, False)
    store = zarr.MemoryStore()
    arr = ZarrFileArray(
        folder=None,
        store=store,
        shape=shape,
        internal_shape=internal_shape,
        shape_mask=shape_mask,
    )

    data1 = np.arange(36).reshape(*internal_shape)
    arr.dump((0, 0), data1)
    assert np.array_equal(arr[0, 0], data1)

    data2 = np.ones(internal_shape)
    arr.dump((1, slice(1, 2)), data2)
    assert np.array_equal(arr[1, 1], data2)


def test_cloudpickle_codec():
    codec = CloudPickleCodec()
    data = {"a": 1, "b": 2}
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert data == decoded

    data = np.array(["foo", "bar", "baz"], dtype="object")
    encoded_data = codec.encode(data)
    out = np.empty(data.shape, dtype=data.dtype)
    decoded_data = codec.decode(encoded_data, out=out)
    assert decoded_data is out
    assert np.array_equal(decoded_data, data)
