from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from pipefunc.map import ZarrFileArray, ZarrMemoryArray
from pipefunc.map._storage_array._zarr import CloudPickleCodec, select_by_mask

if TYPE_CHECKING:
    from pathlib import Path


def test_zarr_array_init():
    shape = (2, 3, 4)
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    assert arr.shape == shape
    assert arr.strides == (12, 4, 1)


def test_zarr_array_properties():
    shape = (2, 3, 4)
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    assert arr.size == 24
    assert arr.rank == 3
    assert repr(arr.object_codec) == "CloudPickleCodec(protocol=5)"


def test_zarr_array_getitem():
    shape = (2, 3)
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr.get_from_index(0) == {"a": 1}
    assert arr.get_from_index(5) == {"b": 2}
    arr.dump((slice(0, 1), 0), {"c": 3})
    assert arr.get_from_index(0) == {"c": 3}


def test_zarr_array_getitem_with_slicing():
    shape = (2, 3, 4)
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(folder=None, store=store, shape=shape)

    arr.dump((0, 0), {"a": 1})
    assert arr[0, 0] == {"a": 1}

    arr.dump((1, 2), {"b": 2})
    assert arr[1, 2] == {"b": 2}


def test_zarr_array_set_and_get_single_item_with_internal_shape():
    shape = (2, 2)
    internal_shape = (3, 3)
    shape_mask = (True, True, False, False)
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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
    store = zarr.storage.MemoryStore()
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


def test_shape_mismatch_on_reopen(tmp_path: Path):
    """Test error when reopening array with different shape."""
    # Create an array with shape (2, 3)
    arr1 = ZarrFileArray(tmp_path, shape=(2, 3))
    arr1.dump((0, 0), {"test": 1})
    del arr1

    # Try to reopen with different shape - should raise ValueError
    with pytest.raises(ValueError, match="Existing array 'array' has unexpected shape"):
        ZarrFileArray(tmp_path, shape=(3, 3))


def test_encode_numpy_scalar():
    """Test encoding numpy scalar arrays."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(None, shape=(2,), store=store)

    # Dump a numpy scalar (0-dimensional array)
    scalar = np.array(42)
    assert scalar.shape == ()  # 0-dimensional
    arr.dump((0,), scalar)

    result = arr[0]
    assert result == 42


def test_decode_memoryview():
    """Test decoding memoryview objects."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(None, shape=(2,), store=store)

    # Dump a memoryview
    data = bytearray(b"hello world")
    mv = memoryview(data)
    arr.dump((0,), mv)

    result = arr[0]
    assert result == mv


def test_decode_non_bytes():
    """Test decoding values that are not bytes/bytearray."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(None, shape=(2,), store=store)

    # Dump various non-bytes objects
    arr.dump((0,), {"key": "value"})
    arr.dump((1,), [1, 2, 3])

    assert arr[0] == {"key": "value"}
    # Lists are normalized to arrays
    assert isinstance(arr[1], np.ndarray)
    assert list(arr[1]) == [1, 2, 3]


def test_no_store_or_folder():
    """Test error when neither store nor folder is provided."""
    with pytest.raises(ValueError, match="Either a `store` or `folder` must be provided"):
        ZarrFileArray(folder=None, shape=(2, 2), store=None)


def test_get_from_index_masked():
    """Test get_from_index returns masked for missing values."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(None, shape=(3, 3), store=store)

    # Dump only one value
    arr.dump((1, 1), {"data": 1})

    # Get from linear index - unmapped indices should be masked
    linear_index = np.ravel_multi_index((0, 0), (3, 3))
    result = arr.get_from_index(linear_index)
    assert result is np.ma.masked


def test_get_from_index_with_internal_shape():
    """Test get_from_index with internal_shape."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(
        None,
        shape=(2, 2),
        internal_shape=(3, 3),
        shape_mask=(True, True, False, False),
        store=store,
    )

    # Dump array data
    data = np.arange(9).reshape(3, 3)
    arr.dump((0, 0), data)

    # Get from linear index
    result = arr.get_from_index(0)
    np.testing.assert_array_equal(result, data)


def test_dump_slice_wrong_internal_shape():
    """Test error when dumping slice with wrong internal_shape."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(
        None,
        shape=(2, 2),
        internal_shape=(3, 3),
        shape_mask=(True, True, False, False),
        store=store,
    )

    # Try to dump with wrong internal shape
    wrong_data = np.arange(6).reshape(2, 3)  # Should be (3, 3)
    with pytest.raises(ValueError, match="Value has incorrect internal_shape"):
        arr.dump((slice(None), 0), wrong_data)


def test_dump_wrong_internal_shape():
    """Test error when dumping with wrong internal_shape."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(
        None,
        shape=(2, 2),
        internal_shape=(3, 3),
        shape_mask=(True, True, False, False),
        store=store,
    )

    # Try to dump with wrong internal shape
    wrong_data = np.arange(6).reshape(2, 3)  # Should be (3, 3)
    with pytest.raises(ValueError, match="Value has incorrect internal_shape"):
        arr.dump((0, 0), wrong_data)


def test_cloudpickle_codec_get_config():
    """Test CloudPickleCodec.get_config method."""
    from pipefunc.map._storage_array._zarr import CloudPickleCodec

    codec = CloudPickleCodec(protocol=4)
    config = codec.get_config()

    assert config == {"id": "cloudpickle", "protocol": 4}


# Additional edge case: has_index method
def test_has_index():
    """Test has_index method."""
    store = zarr.storage.MemoryStore()
    arr = ZarrFileArray(None, shape=(3, 3), store=store)

    # Initially nothing is dumped
    assert not arr.has_index(0)

    # Dump something
    arr.dump((0, 0), {"test": 1})

    # Now index 0 should exist
    assert arr.has_index(0)

    # Other indices should not exist
    assert not arr.has_index(1)


# Test get_from_index with scalar values
def test_get_from_index_scalar():
    """Test get_from_index with scalar values."""
    store = zarr.storage.MemoryStore()

    # Use 1D array to test scalar retrieval
    arr = ZarrFileArray(None, shape=(5,), store=store)

    # Dump various types of scalar values
    arr.dump((0,), 42)
    arr.dump((1,), "hello")
    arr.dump((2,), {"key": "value"})
    arr.dump((3,), [1, 2, 3])

    # Get via linear index
    result = arr.get_from_index(0)
    assert result == 42

    result = arr.get_from_index(1)
    assert result == "hello"

    result = arr.get_from_index(2)
    assert result == {"key": "value"}


def test_shared_dict_store_default():
    """Test _SharedDictStore creates new dict when none provided."""
    from pipefunc.map import ZarrSharedMemoryArray

    # Don't provide a store - should create default
    arr = ZarrSharedMemoryArray(None, shape=(2, 2))
    arr.dump((0, 0), {"test": 1})

    result = arr[0, 0]
    assert result == {"test": 1}


def test_zarr_memory_array_persist_no_folder():
    """Test ZarrMemoryArray.persist when folder is None."""
    # Create array without folder
    arr = ZarrMemoryArray(None, shape=(2, 2))
    arr.dump((0, 0), {"test": 1})

    # persist() should return early without error
    arr.persist()  # No-op when folder is None


def test_zarr_memory_array_dump_in_subprocess():
    """Test ZarrMemoryArray.dump_in_subprocess returns False."""
    arr = ZarrMemoryArray(None, shape=(2, 2))
    assert arr.dump_in_subprocess is False


def test_zarr_shared_memory_array_default_store():
    """Test ZarrSharedMemoryArray creates default store."""
    from pipefunc.map import ZarrSharedMemoryArray

    # Don't provide store - should create _SharedDictStore
    arr = ZarrSharedMemoryArray(None, shape=(2, 2), store=None)
    arr.dump((0, 0), {"data": 123})

    assert arr[0, 0] == {"data": 123}


def test_zarr_shared_memory_array_dump_in_subprocess():
    """Test ZarrSharedMemoryArray.dump_in_subprocess returns True."""
    from pipefunc.map import ZarrSharedMemoryArray

    arr = ZarrSharedMemoryArray(None, shape=(2, 2))
    assert arr.dump_in_subprocess is True
