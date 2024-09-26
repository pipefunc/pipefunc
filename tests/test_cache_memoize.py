import pytest

from pipefunc.cache import (
    DiskCache,
    HybridCache,
    LRUCache,
    SimpleCache,
    UnhashableError,
    memoize,
)


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


class UnhashableMeta(type):
    def __hash__(cls):
        msg = "Not implemented"
        raise NotImplementedError(msg)


class TrulyUnhashable(metaclass=UnhashableMeta):
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        msg = "I am truly unhashable!"
        raise TypeError(msg)

    def __eq__(self, other):
        if isinstance(other, TrulyUnhashable):
            return self.value == other.value
        return False

    def __str__(self):
        return f"TrulyUnhashable({self.value})"


class Unhashable:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        msg = "I am truly unhashable!"
        raise TypeError(msg)

    def __str__(self):
        return f"Unhashable({self.value})"


def test_memoize_with_unhashable_arguments_fallback_to_str():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache, fallback_to_pickle=True)
    def process_unhashable(obj):
        calls["count"] += 1
        return str(obj)

    # Using a truly unhashable object
    unhashable = Unhashable(42)
    result1 = process_unhashable(unhashable)
    assert result1 == "Unhashable(42)"
    assert calls["count"] == 1

    # Call again with the same unhashable object
    result2 = process_unhashable(unhashable)
    assert result2 == "Unhashable(42)"
    assert calls["count"] == 1  # Should not increase, result from cache

    # Call with a different unhashable object
    result3 = process_unhashable(Unhashable(43))
    assert result3 == "Unhashable(43)"
    assert calls["count"] == 2  # New arguments, should increase


def test_memoize_unhashable_action_error():
    cache = SimpleCache()

    @memoize(cache=cache, unhashable_action="error")
    def process_unhashable(obj):
        return str(obj)

    obj = TrulyUnhashable(42)
    # Using a truly unhashable object should raise an UnhashableError
    with pytest.raises(UnhashableError):
        process_unhashable(obj)


def test_memoize_unhashable_action_warning():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache, unhashable_action="warning")
    def process_unhashable(obj):
        calls["count"] += 1
        return str(obj)

    # Using a truly unhashable object should issue a warning and not cache
    with pytest.warns(
        UserWarning,
        match="Unhashable arguments in 'process_unhashable'. Skipping cache.",
    ):
        result1 = process_unhashable(TrulyUnhashable(42))

    assert result1 == "TrulyUnhashable(42)"
    assert calls["count"] == 1

    # Call again with the same unhashable object
    with pytest.warns(
        UserWarning,
        match="Unhashable arguments in 'process_unhashable'. Skipping cache.",
    ):
        result2 = process_unhashable(TrulyUnhashable(42))

    assert result2 == "TrulyUnhashable(42)"
    assert calls["count"] == 2  # Should increase, not cached


def test_memoize_unhashable_action_ignore():
    cache = SimpleCache()
    calls = {"count": 0}

    @memoize(cache=cache, unhashable_action="ignore")
    def process_unhashable(obj):
        calls["count"] += 1
        return str(obj)

    # Using a truly unhashable object should silently skip caching
    result1 = process_unhashable(TrulyUnhashable(42))
    assert result1 == "TrulyUnhashable(42)"
    assert calls["count"] == 1

    # Call again with the same unhashable object
    result2 = process_unhashable(TrulyUnhashable(42))
    assert result2 == "TrulyUnhashable(42)"
    assert calls["count"] == 2  # Should increase, not cached


def test_memoize_with_custom_key_function_for_unhashable():
    cache = SimpleCache()
    calls = {"count": 0}

    def custom_key(*args, **kwargs):
        # Convert all arguments to strings for the key
        return tuple(str(arg) for arg in args) + tuple(f"{k}:{v}" for k, v in kwargs.items())

    @memoize(cache=cache, key_func=custom_key)
    def process_unhashable(obj, extra=None):
        calls["count"] += 1
        return f"{obj} - {extra}"

    # Using a truly unhashable object with custom key function
    result1 = process_unhashable(TrulyUnhashable(42), extra="test")
    assert result1 == "TrulyUnhashable(42) - test"
    assert calls["count"] == 1

    # Call again with the same arguments
    result2 = process_unhashable(TrulyUnhashable(42), extra="test")
    assert result2 == "TrulyUnhashable(42) - test"
    assert calls["count"] == 1  # Should not increase, result from cache

    # Call with different arguments
    result3 = process_unhashable(TrulyUnhashable(43), extra="other")
    assert result3 == "TrulyUnhashable(43) - other"
    assert calls["count"] == 2  # New arguments, should increase
