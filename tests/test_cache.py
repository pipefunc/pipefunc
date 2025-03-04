from __future__ import annotations

import copy
import pickle
import tempfile
import time
from pathlib import Path

import pytest

from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache


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

    pickled_cache = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

    assert unpickled_cache.get("key1") == "value1"
    assert unpickled_cache.get("key2") == "value2"
    assert len(unpickled_cache) == 2

    assert unpickled_cache.shared == shared
    # Test that the unpickled cache is still shared
    duration3 = (3.0,) if cache_cls == HybridCache else ()
    unpickled_cache.put("key3", "value3", *duration3)
    if cache_cls == DiskCache:
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


def test_disk_cache_pickling(tmp_path: Path) -> None:
    cache = DiskCache(cache_dir=tmp_path, with_lru_cache=False)
    cache.put("key1", "value1")
    cache2 = pickle.loads(pickle.dumps(cache))  # noqa: S301
    assert cache2.get("key1") == "value1"


@pytest.mark.parametrize("permissions", [0o600, 0o660, 0o777, 0o644, None])
def test_disk_cache_permissions(cache_dir: Path, permissions: int | None) -> None:
    """Test that DiskCache sets file permissions correctly."""
    cache = DiskCache(cache_dir=str(cache_dir), permissions=permissions)
    cache.put("key1", "value1")
    file_path = cache._get_file_path("key1")
    assert file_path.exists()

    if permissions is not None:
        # stat().st_mode returns the full mode, including file type bits.
        # We only want the permission bits, so we mask with 0o777.
        actual_permissions = file_path.stat().st_mode & 0o777
        assert actual_permissions == permissions


@pytest.mark.parametrize("shared", [True, False])
def test_pickling_and_deepcopy_hybrid(shared: bool) -> None:  # noqa: FBT001
    # Create and populate cache
    cache = HybridCache(max_size=5, shared=shared)
    cache.put("key1", "value1", 0.1)
    cache.put("key2", "value2", 0.2)

    # Test pickling
    pickled_cache = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

    assert "key1" in unpickled_cache
    assert "key2" in unpickled_cache
    assert unpickled_cache.get("key1") == "value1"
    assert unpickled_cache.get("key2") == "value2"
    assert unpickled_cache.shared == shared

    # Test deep copying
    copied_cache = copy.deepcopy(cache)

    assert "key1" in copied_cache
    assert "key2" in copied_cache
    assert copied_cache.get("key1") == "value1"
    assert copied_cache.get("key2") == "value2"
    assert copied_cache.shared == shared

    # Verify that changes to original don't affect copy
    cache.put("key3", "value3", 0.3)
    assert "key3" in cache
    assert "key3" not in copied_cache


@pytest.mark.parametrize("shared", [True, False])
def test_pickling_and_deepcopy_lru(shared: bool) -> None:  # noqa: FBT001
    # Create and populate cache
    cache = LRUCache(max_size=5, shared=shared)
    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Test pickling
    pickled_cache = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

    assert "key1" in unpickled_cache
    assert "key2" in unpickled_cache
    assert unpickled_cache.get("key1") == "value1"
    assert unpickled_cache.get("key2") == "value2"
    assert unpickled_cache.shared == shared

    # Test deep copying
    copied_cache = copy.deepcopy(cache)

    assert "key1" in copied_cache
    assert "key2" in copied_cache
    assert copied_cache.get("key1") == "value1"
    assert copied_cache.get("key2") == "value2"
    assert copied_cache.shared == shared

    # Verify that changes to original don't affect copy
    cache.put("key3", "value3")
    assert "key3" in cache
    assert "key3" not in copied_cache


def test_pickling_and_deepcopy_simple() -> None:
    # Create and populate cache
    cache = SimpleCache()
    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Test pickling
    pickled_cache = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

    assert "key1" in unpickled_cache
    assert "key2" in unpickled_cache
    assert unpickled_cache.get("key1") == "value1"
    assert unpickled_cache.get("key2") == "value2"

    # Test deep copying
    copied_cache = copy.deepcopy(cache)

    assert "key1" in copied_cache
    assert "key2" in copied_cache
    assert copied_cache.get("key1") == "value1"
    assert copied_cache.get("key2") == "value2"

    # Verify that changes to original don't affect copy
    cache.put("key3", "value3")
    assert "key3" in cache
    assert "key3" not in copied_cache


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("with_lru", [True, False])
def test_pickling_and_deepcopy_disk(shared: bool, with_lru: bool) -> None:  # noqa: FBT001
    # Create a temporary directory for the cache
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache"

        # Create and populate cache
        cache = DiskCache(
            cache_dir=cache_dir,
            max_size=5,
            with_lru_cache=with_lru,
            lru_shared=shared,
        )
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Test pickling
        pickled_cache = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_cache)  # noqa: S301

        assert "key1" in unpickled_cache
        assert "key2" in unpickled_cache
        assert unpickled_cache.get("key1") == "value1"
        assert unpickled_cache.get("key2") == "value2"
        if with_lru:
            assert unpickled_cache.lru_cache.shared == shared

        # Test deep copying
        copied_cache = copy.deepcopy(cache)

        assert "key1" in copied_cache
        assert "key2" in copied_cache
        assert copied_cache.get("key1") == "value1"
        assert copied_cache.get("key2") == "value2"
        if with_lru:
            assert copied_cache.lru_cache.shared == shared

        # Verify that disk cache is shared
        cache.put("key3", "value3")
        assert "key3" in cache
        assert "key3" in copied_cache
