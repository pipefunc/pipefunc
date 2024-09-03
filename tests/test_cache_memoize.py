import pytest

from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache, memoize


@pytest.mark.parametrize("cache_type", [SimpleCache, LRUCache, HybridCache])
def test_memoize_with_different_caches(cache_type):
    cache = cache_type()
    calls = {"count": 0}

    @memoize(cache=cache)
    def add(x, y):
        calls["count"] += 1
        return x + y

    assert add(2, 3) == 5
    assert calls["count"] == 1
    assert add(2, 3) == 5
    assert calls["count"] == 1  # Should not increase, result from cache
    assert add(3, 4) == 7
    assert calls["count"] == 2  # New arguments, should increase


def test_memoize_with_disk_cache(tmp_path):
    cache = DiskCache(str(tmp_path))
    calls = {"count": 0}

    @memoize(cache=cache)
    def multiply(x, y):
        calls["count"] += 1
        return x * y

    assert multiply(2, 3) == 6
    assert calls["count"] == 1
    assert multiply(2, 3) == 6
    assert calls["count"] == 1  # Should not increase, result from cache
    assert multiply(3, 4) == 12
    assert calls["count"] == 2  # New arguments, should increase


def test_memoize_with_custom_key_function():
    cache = SimpleCache()
    calls = {"count": 0}

    def custom_key(*args, **kwargs):
        return args[0]  # Only use the first argument as the key

    @memoize(cache=cache, key_func=custom_key)
    def greet(name, greeting="Hello"):
        calls["count"] += 1
        return f"{greeting}, {name}!"

    assert greet("Alice") == "Hello, Alice!"
    assert calls["count"] == 1
    assert greet("Alice", greeting="Hi") == "Hello, Alice!"  # Should use cached value
    assert calls["count"] == 1
    assert greet("Bob") == "Hello, Bob!"
    assert calls["count"] == 2


def test_memoize_with_unhashable_arguments():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache)
    def process_list(lst):
        calls["count"] += 1
        return sum(lst)

    assert process_list([1, 2, 3]) == 6
    assert calls["count"] == 1
    assert process_list([1, 2, 3]) == 6
    assert calls["count"] == 1  # Should not increase, result from cache
    assert process_list([4, 5, 6]) == 15
    assert calls["count"] == 2  # New arguments, should increase


def test_memoize_with_keyword_arguments():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache)
    def calculate(x, y, operation="add"):
        calls["count"] += 1
        if operation == "add":
            return x + y
        if operation == "multiply":
            return x * y
        msg = f"Unknown operation: {operation}"
        raise ValueError(msg)

    assert calculate(2, 3, operation="add") == 5
    assert calls["count"] == 1
    assert calculate(2, 3, operation="add") == 5
    assert calls["count"] == 1  # Should not increase, result from cache
    assert calculate(2, 3, operation="multiply") == 6
    assert calls["count"] == 2  # New operation, should increase


def test_memoize_with_none_result():
    cache = None  # uses SimpleCache by default
    calls = {"count": 0}

    @memoize(cache=cache)
    def return_none(x):
        calls["count"] += 1

    assert return_none(1) is None
    assert calls["count"] == 1
    assert return_none(1) is None
    assert calls["count"] == 1  # Should not increase, result from cache


def test_memoize_clear_cache():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache)
    def add(x, y):
        calls["count"] += 1
        return x + y

    assert add(2, 3) == 5
    assert calls["count"] == 1
    assert add(2, 3) == 5
    assert calls["count"] == 1  # Should not increase, result from cache

    cache.clear()

    assert add(2, 3) == 5
    assert calls["count"] == 2  # Should increase after cache clear
