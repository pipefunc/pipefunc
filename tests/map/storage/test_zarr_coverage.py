"""Tests to achieve 100% coverage for pipefunc/map/_storage_array/_zarr.py using public API only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from pipefunc.map import ZarrFileArray, ZarrMemoryArray

if TYPE_CHECKING:
    from pathlib import Path


# Line 69-70: Shape mismatch when opening existing array
def test_shape_mismatch_on_reopen(tmp_path: Path):
    """Test error when reopening array with different shape."""
    # Create an array with shape (2, 3)
    arr1 = ZarrFileArray(tmp_path, shape=(2, 3))
    arr1.dump((0, 0), {"test": 1})
    del arr1

    # Try to reopen with different shape - should raise ValueError
    with pytest.raises(ValueError, match="Existing array 'array' has unexpected shape"):
        ZarrFileArray(tmp_path, shape=(3, 3))


# Line 76: Encode scalar with numpy scalar array
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


# Line 89: Decode memoryview
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


# Line 97: Decode non-bytes value (return value directly)
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


# Lines 126-127: Invalid if_exists parameter
def test_copy_store_invalid_if_exists(tmp_path: Path):
    """Test _copy_store with invalid if_exists parameter."""
    from zarr.storage import LocalStore

    from pipefunc.map._storage_array._zarr import _copy_store

    source = LocalStore(str(tmp_path / "source"))
    dest = LocalStore(str(tmp_path / "dest"))

    with pytest.raises(ValueError, match="if_exists must be 'raise', 'replace', or 'skip'"):
        _copy_store(source, dest, if_exists="invalid")


# Lines 137, 139-140: Test if_exists modes in _copy_store
def test_copy_store_if_exists_modes(tmp_path: Path):
    """Test different if_exists modes in persist/load."""
    # Create a ZarrMemoryArray with folder
    arr = ZarrMemoryArray(tmp_path, shape=(2, 2))
    arr.dump((0, 0), {"initial": 1})
    arr.persist()

    # Modify in-memory
    arr.dump((1, 1), {"modified": 2})

    # Create another array and load - tests skip and replace modes
    arr2 = ZarrMemoryArray(tmp_path, shape=(2, 2))
    assert arr2[0, 0] == {"initial": 1}

    # Test if_exists="raise" by directly calling _copy_store
    from zarr.storage import LocalStore, MemoryStore

    from pipefunc.map._storage_array._zarr import _copy_store

    source = LocalStore(str(tmp_path))
    dest = MemoryStore()

    # First copy succeeds
    _copy_store(source, dest, if_exists="replace")

    # Second copy with if_exists="raise" should fail
    with pytest.raises(ValueError, match="already exists in destination store"):
        _copy_store(source, dest, if_exists="raise")

    # if_exists="skip" should not raise
    _copy_store(source, dest, if_exists="skip")


# Lines 181-182: Neither store nor folder provided
def test_no_store_or_folder():
    """Test error when neither store nor folder is provided."""
    with pytest.raises(ValueError, match="Either a `store` or `folder` must be provided"):
        ZarrFileArray(folder=None, shape=(2, 2), store=None)


# Line 242: Return masked value from get_from_index
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


# Line 249: Return normalized decoded value from get_from_index
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


# Lines 336-337: Wrong internal_shape during slice dump
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


# Lines 354-355: Wrong internal_shape during non-slice dump
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


# Line 592: CloudPickleCodec.get_config
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


# Lines 411-413: _SharedDictStore without providing shared_dict
def test_shared_dict_store_default():
    """Test _SharedDictStore creates new dict when none provided."""
    from pipefunc.map import ZarrSharedMemoryArray

    # Don't provide a store - should create default
    arr = ZarrSharedMemoryArray(None, shape=(2, 2))
    arr.dump((0, 0), {"test": 1})

    result = arr[0, 0]
    assert result == {"test": 1}


# Line 458: ZarrMemoryArray.persist with no folder
def test_zarr_memory_array_persist_no_folder():
    """Test ZarrMemoryArray.persist when folder is None."""
    # Create array without folder
    arr = ZarrMemoryArray(None, shape=(2, 2))
    arr.dump((0, 0), {"test": 1})

    # persist() should return early without error
    arr.persist()  # No-op when folder is None


# Line 474: ZarrMemoryArray.dump_in_subprocess property
def test_zarr_memory_array_dump_in_subprocess():
    """Test ZarrMemoryArray.dump_in_subprocess returns False."""
    arr = ZarrMemoryArray(None, shape=(2, 2))
    assert arr.dump_in_subprocess is False


# Lines 496-498: ZarrSharedMemoryArray without store
def test_zarr_shared_memory_array_default_store():
    """Test ZarrSharedMemoryArray creates default store."""
    from pipefunc.map import ZarrSharedMemoryArray

    # Don't provide store - should create _SharedDictStore
    arr = ZarrSharedMemoryArray(None, shape=(2, 2), store=None)
    arr.dump((0, 0), {"data": 123})

    assert arr[0, 0] == {"data": 123}


# Line 509: ZarrSharedMemoryArray.dump_in_subprocess property
def test_zarr_shared_memory_array_dump_in_subprocess():
    """Test ZarrSharedMemoryArray.dump_in_subprocess returns True."""
    from pipefunc.map import ZarrSharedMemoryArray

    arr = ZarrSharedMemoryArray(None, shape=(2, 2))
    assert arr.dump_in_subprocess is True


# Lines 89, 97: Test _decode_scalar edge cases by manipulating internal data
def test_decode_scalar_edge_cases():
    """Test _decode_scalar with memoryview and non-bytes values."""
    from pipefunc.map._storage_array._zarr import CloudPickleCodec, _decode_scalar

    codec = CloudPickleCodec()

    # Line 89: memoryview case
    # When Zarr reads data, it might return memoryview in some cases
    encoded = codec.encode({"test": "value"})
    mv = memoryview(encoded)
    result = _decode_scalar(codec, mv)
    assert result == {"test": "value"}

    # Line 97: non-bytes value case (pass through values that aren't bytes)
    # This happens when the value is already decoded or is not in bytes format
    plain_value = 123
    result = _decode_scalar(codec, plain_value)
    assert result == 123
