from __future__ import annotations

import pickle
import time
from typing import TYPE_CHECKING

import pytest

from pipefunc._cache import DiskCache, HybridCache, LRUCache, SimpleCache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_init(shared):
    cache = HybridCache(shared=shared)
    assert cache.max_size == 128
    assert cache.access_weight == 0.5
    assert cache.duration_weight == 0.5


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_put_and_get(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    assert cache.get("key1") == "value1"
    assert isinstance(cache.cache, dict)
    assert "key1" in cache
    assert cache.get("not_exist") is None


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_contains(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    assert "key1" in cache
    assert "key2" not in cache


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_expire(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 1.0)
    cache.put("key2", "value2", 2.0)
    cache.put("key3", "value3", 3.0)
    cache.put("key4", "value4", 4.0)
    assert "key1" not in cache
    assert "key4" in cache
    assert len(cache) == 3
    assert len(cache.cache) == 3


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_str(shared):
    cache = HybridCache(max_size=3, shared=shared, allow_cloudpickle=False)
    cache.put("key1", "value1", 2.0)
    assert "Cache: {'key1': 'value1'}" in str(cache)


@pytest.mark.parametrize("shared", [True, False])
def test_access_count(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    cache.get("key1")
    cache.get("key1")
    assert cache.access_counts["key1"] == 3
    assert len(cache) == 1


@pytest.mark.parametrize("shared", [True, False])
def test_computation_duration(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    cache.get("key1")
    assert cache.computation_durations["key1"] == 2.0


@pytest.mark.parametrize("shared", [True, False])
def test_cache_initialization(shared):
    cache = LRUCache(max_size=2, shared=shared, allow_cloudpickle=False)
    assert cache.max_size == 2
    assert cache._allow_cloudpickle is False
    assert len(cache._cache_queue) == 0
    assert len(cache._cache_dict) == 0

    cache_with_cp = LRUCache(max_size=2, allow_cloudpickle=True, shared=shared)
    assert cache_with_cp._allow_cloudpickle is True


@pytest.mark.parametrize("shared", [True, False])
def test_put_and_get(shared):
    cache = LRUCache(max_size=2, shared=shared)
    cache.put("test", "value")
    assert len(cache._cache_queue) == 1
    assert len(cache._cache_dict) == 1

    value = cache.get("test")
    assert value == "value"

    cache.put("test2", "value2")
    assert len(cache._cache_queue) == 2
    assert len(cache._cache_dict) == 2

    cache.put("test3", "value3")
    assert len(cache._cache_queue) == 2
    assert len(cache._cache_dict) == 2
    assert "test" not in cache._cache_dict
    assert len(cache) == 2


@pytest.mark.parametrize("shared", [True, False])
def test_get_nonexistent_key(shared):
    cache = LRUCache(max_size=2, shared=shared)
    value = cache.get("nonexistent")
    assert value is None


@pytest.mark.parametrize("shared", [True, False])
def test_put_and_get_none(shared):
    cache = LRUCache(max_size=2, shared=shared)
    cache.put("test", None)
    value = cache.get("test")
    assert value is None


@pytest.mark.parametrize("shared", [True, False])
def test_put_and_get_with_cloudpickle(shared):
    cache = LRUCache(max_size=2, allow_cloudpickle=True, shared=shared)
    cache.put("test", "value")
    assert len(cache._cache_queue) == 1
    assert len(cache._cache_dict) == 1

    value = cache.get("test")
    assert value == "value"


@pytest.mark.parametrize("shared", [True, False])
def test_contains(shared):
    cache = LRUCache(max_size=2, shared=shared)
    cache.put("test", "value")
    assert "test" in cache


@pytest.mark.parametrize("shared", [True, False])
def test_cache_property(shared):
    cache = LRUCache(max_size=2, shared=shared, allow_cloudpickle=True)
    cache.put("test", "value")
    cache_dict = cache.cache
    assert cache_dict == {"test": "value"}


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


def test_file_cache_init(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    assert cache.cache_dir == cache_dir
    assert cache.max_size is None
    assert cache.use_cloudpickle is True
    assert cache_dir.exists()


def test_file_cache_put_and_get(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert "key1" in cache


def test_file_cache_get_nonexistent_key(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    assert cache.get("nonexistent") is None


def test_file_cache_evict_if_needed(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), max_size=2, lru_cache_size=2)
    cache.put("key1", "value1")
    time.sleep(0.2)
    cache.put("key2", "value2")
    time.sleep(0.2)
    cache.put("key3", "value3")
    assert len(list(cache_dir.glob("*.pkl"))) == 2
    assert len(cache.lru_cache) == 2
    assert "key1" not in cache.lru_cache
    assert "key1" not in cache


def test_file_cache_clear(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache) == 2
    assert len(cache.lru_cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert len(cache.lru_cache) == 0


def test_file_cache_contains(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    cache.put("key1", "value1")
    assert "key1" in cache
    assert "key2" not in cache


def test_file_cache_len(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir))
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache) == 2


def test_file_cache_without_cloudpickle(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), use_cloudpickle=False)
    cache.put("key1", b"value1")
    assert cache.get("key1") == b"value1"


def test_file_cache_with_custom_max_size(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), max_size=10)
    assert cache.max_size == 10


def test_file_cache_with_lru_cache(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=True, lru_cache_size=2)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert cache.get("key1") == "value1"
    assert "key1" in cache.lru_cache
    assert "key2" in cache.lru_cache


def test_file_cache_lru_cache_eviction(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=True, lru_cache_size=2)
    cache.put("key1", "value1")
    time.sleep(0.01)
    cache.put("key2", "value2")
    time.sleep(0.01)
    cache.put("key3", "value3")
    assert "key1" not in cache.lru_cache
    assert "key2" in cache.lru_cache
    assert "key3" in cache.lru_cache


def test_file_cache_contains_with_lru_cache(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=True)
    cache.put("key1", "value1")
    assert "key1" in cache
    assert "key1" in cache.lru_cache


def test_file_cache_clear_with_lru_cache(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=True, lru_shared=True)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache) == 2
    assert len(cache.lru_cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert len(cache.lru_cache) == 0


@pytest.mark.parametrize("shared", [True, False])
def test_file_cache_put_and_get_none(cache_dir, shared: bool):  # noqa: FBT001
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=True, lru_shared=shared)
    cache.put("key1", None)
    assert cache.get("key1") is None
    assert "key1" in cache
    assert "key1" in cache.lru_cache
    assert cache.cache["key1"] is None


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_clear(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    cache.put("key2", "value2", 3.0)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert "key1" not in cache
    assert "key2" not in cache

    cache.put("key3", "value3", 4.0)
    assert cache.get("key3") == "value3"
    assert len(cache) == 1


@pytest.mark.parametrize("shared", [True, False])
def test_lru_cache_clear(shared):
    cache = LRUCache(max_size=2, shared=shared)
    cache.put("test", "value")
    cache.put("test2", "value2")
    assert len(cache._cache_queue) == 2
    assert len(cache._cache_dict) == 2

    cache.clear()
    assert len(cache._cache_queue) == 0
    assert len(cache._cache_dict) == 0
    assert len(cache) == 0

    cache.put("test3", "value3")
    assert cache.get("test3") == "value3"
    assert len(cache) == 1


def test_disk_cache_clear(cache_dir):
    cache = DiskCache(cache_dir=str(cache_dir), with_lru_cache=False)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0

    cache.put("key3", "value3")
    assert cache.get("key3") == "value3"
    assert len(cache) == 1


@pytest.mark.parametrize("shared", [True, False])
def test_disk_cache_clear_with_lru_cache(cache_dir: Path, shared: bool):  # noqa: FBT001
    cache = DiskCache(
        cache_dir=str(cache_dir),
        with_lru_cache=True,
        lru_shared=shared,
    )
    assert len(cache) == 0
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    assert len(cache) == 2, cache._all_files()
    assert len(cache.lru_cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert len(cache.lru_cache) == 0

    cache.put("key3", "value3")
    assert cache.get("key3") == "value3"
    assert len(cache) == 1
    assert len(cache.lru_cache) == 1

    cache2 = DiskCache(
        cache_dir=str(cache_dir),
        with_lru_cache=True,
        lru_shared=shared,
    )
    assert len(cache2) == 1
    assert cache2.get("key3") == "value3"


@pytest.mark.parametrize("cache_cls", [HybridCache, LRUCache, DiskCache])
@pytest.mark.parametrize("shared", [True, False])
def test_cache_pickling(cache_cls, shared, tmp_path):
    if cache_cls == DiskCache:
        cache = cache_cls(cache_dir=str(tmp_path), lru_shared=shared)
    else:
        cache = cache_cls(shared=shared)

    duration1 = (1.0,) if cache_cls == HybridCache else ()
    duration2 = (2.0,) if cache_cls == HybridCache else ()
    cache.put("key1", "value1", *duration1)
    cache.put("key2", "value2", *duration2)

    if not shared:
        with pytest.raises(
            RuntimeError,
            match="Cannot pickle non-shared cache instances",
        ):
            pickle.dumps(cache)
        return

    pickled_cache = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

    assert unpickled_cache.get("key1") == "value1"
    assert unpickled_cache.get("key2") == "value2"
    assert len(unpickled_cache) == 2

    assert shared
    assert unpickled_cache.shared == shared
    # Test that the unpickled cache is still shared
    duration3 = (3.0,) if cache_cls == HybridCache else ()
    unpickled_cache.put("key3", "value3", *duration3)
    assert cache.get("key3") == "value3"


def test_simple_cache():
    cache = SimpleCache()
    cache.put("key1", "value1")
    assert "key1" in cache
    assert "key2" not in cache
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    assert isinstance(cache.cache, dict)
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
