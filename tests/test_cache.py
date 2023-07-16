import pytest

from pipefunc._cache import HybridCache, LRUCache


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
    assert "key1" not in cache, cache.cache
    assert "key4" in cache, cache.cache


@pytest.mark.parametrize("shared", [True, False])
def test_hybrid_cache_str(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    assert "Cache: {'key1': 'value1'}" in str(cache)


@pytest.mark.parametrize("shared", [True, False])
def test_access_count(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    cache.get("key1")
    cache.get("key1")
    assert cache.access_counts["key1"] == 3


@pytest.mark.parametrize("shared", [True, False])
def test_computation_duration(shared):
    cache = HybridCache(max_size=3, shared=shared)
    cache.put("key1", "value1", 2.0)
    cache.get("key1")
    assert cache.computation_durations["key1"] == 2.0


@pytest.mark.parametrize("shared", [True, False])
def test_cache_initialization(shared):
    cache = LRUCache(max_size=2, shared=shared)
    assert cache.max_size == 2
    assert cache._with_cloudpickle is False
    assert len(cache._cache_queue) == 0
    assert len(cache._cache_dict) == 0

    cache_with_cp = LRUCache(max_size=2, with_cloudpickle=True, shared=shared)
    assert cache_with_cp._with_cloudpickle is True


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
    cache = LRUCache(max_size=2, with_cloudpickle=True, shared=shared)
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
    cache = LRUCache(max_size=2, shared=shared)
    cache.put("test", "value")
    cache_dict = cache.cache
    assert cache_dict == {"test": "value"}
