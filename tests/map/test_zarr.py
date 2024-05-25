# test_zarr_array.py

# test_zarr_array.py
import numpy as np
import pytest
import zarr

from pipefunc.map.zarr import ZarrArray


def test_zarr_array_init():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
    assert arr.shape == shape
    assert arr.strides == (12, 4, 1)


def test_zarr_array_properties():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
    assert arr.size == 24
    assert arr.rank == 3


def test_zarr_array_normalize_key():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
    assert arr._normalize_key((1, 2, 3)) == (1, 2, 3)
    assert arr._normalize_key((slice(None), 1, 2)) == (slice(None), 1, 2)
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 3, 4))
    with pytest.raises(IndexError):
        arr._normalize_key((1, 2, 10))


def test_zarr_array_getitem():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr[0, 0] == {"a": 1}
    assert arr[1, 2] == {"b": 2}
    assert arr[0, 1] is np.ma.masked
    assert arr[0:1, 0] == {"a": 1}


def test_zarr_array_to_array():
    shape = (2, 3)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
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
    arr = ZarrArray(store, shape)
    arr.dump((0, 0), {"a": 1})
    arr.dump((1, 2), {"b": 2})
    assert arr.get_from_index(0) == {"a": 1}
    assert arr.get_from_index(5) == {"b": 2}
    arr.dump((slice(0, 1), 0), {"c": 3})
    assert arr.get_from_index(0) == {"c": 3}


def test_zarr_array_getitem_with_slicing():
    shape = (2, 3, 4)
    store = zarr.MemoryStore()
    arr = ZarrArray(store, shape)
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
