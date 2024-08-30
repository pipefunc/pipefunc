from collections.abc import Callable

import pytest

from pipefunc._cache import DiskCache, HybridCache, LRUCache, SimpleCache, memoize


# Helper function to count function calls
def call_counter(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


@pytest.mark.parametrize("cache_type", [SimpleCache, LRUCache, HybridCache])
def test_memoize_with_different_caches(cache_type):
    cache = cache_type()

    @memoize(cache=cache)
    @call_counter
    def add(x, y):
        return x + y

    assert add(2, 3) == 5
    assert add.calls == 1
    assert add(2, 3) == 5
    assert add.calls == 1  # Should not increase, result from cache
    assert add(3, 4) == 7
    assert add.calls == 2  # New arguments, should increase


def test_memoize_with_disk_cache(tmp_path):
    cache = DiskCache(str(tmp_path))

    @memoize(cache=cache)
    @call_counter
    def multiply(x, y):
        return x * y

    assert multiply(2, 3) == 6
    assert multiply.calls == 1
    assert multiply(2, 3) == 6
    assert multiply.calls == 1  # Should not increase, result from cache
    assert multiply(3, 4) == 12
    assert multiply.calls == 2  # New arguments, should increase


def test_memoize_with_custom_key_function():
    cache = SimpleCache()

    def custom_key(*args, **kwargs):
        return args[0]  # Only use the first argument as the key

    @memoize(cache=cache, key_func=custom_key)
    @call_counter
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    assert greet("Alice") == "Hello, Alice!"
    assert greet.calls == 1
    assert greet("Alice", greeting="Hi") == "Hello, Alice!"  # Should use cached value
    assert greet.calls == 1
    assert greet("Bob") == "Hello, Bob!"
    assert greet.calls == 2


def test_memoize_with_unhashable_arguments():
    cache = SimpleCache()

    @memoize(cache=cache)
    @call_counter
    def process_list(lst):
        return sum(lst)

    assert process_list([1, 2, 3]) == 6
    assert process_list.calls == 1
    assert process_list([1, 2, 3]) == 6
    assert process_list.calls == 1  # Should not increase, result from cache
    assert process_list([4, 5, 6]) == 15
    assert process_list.calls == 2  # New arguments, should increase


def test_memoize_with_keyword_arguments():
    cache = SimpleCache()

    @memoize(cache=cache)
    @call_counter
    def calculate(x, y, operation="add"):
        if operation == "add":
            return x + y
        if operation == "multiply":
            return x * y

    assert calculate(2, 3) == 5
    assert calculate.calls == 1
    assert calculate(2, 3, operation="add") == 5
    assert calculate.calls == 1  # Should not increase, result from cache
    assert calculate(2, 3, operation="multiply") == 6
    assert calculate.calls == 2  # New operation, should increase


def test_memoize_with_none_result():
    cache = SimpleCache()

    @memoize(cache=cache)
    @call_counter
    def return_none(x):
        return None

    assert return_none(1) is None
    assert return_none.calls == 1
    assert return_none(1) is None
    assert return_none.calls == 1  # Should not increase, result from cache


def test_memoize_clear_cache():
    cache = SimpleCache()

    @memoize(cache=cache)
    @call_counter
    def add(x, y):
        return x + y

    assert add(2, 3) == 5
    assert add.calls == 1
    assert add(2, 3) == 5
    assert add.calls == 1  # Should not increase, result from cache

    cache.clear()

    assert add(2, 3) == 5
    assert add.calls == 2  # Should increase after cache clear
