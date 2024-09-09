"""Provides `pipefunc.cache` module with cache classes for memoization and caching."""

from __future__ import annotations

import abc
import array
import collections
import functools
import hashlib
import pickle
import sys
import time
from contextlib import nullcontext, suppress
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable


class _CacheBase(abc.ABC):
    @abc.abstractmethod
    def get(self, key: Hashable) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, key: Hashable, value: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __contains__(self, key: Hashable) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    def __getstate__(self) -> object:
        if hasattr(self, "shared") and self.shared:
            return self.__dict__
        msg = "Cannot pickle non-shared cache instances, use `shared=True`."
        raise RuntimeError(msg)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class HybridCache(_CacheBase):
    """A hybrid cache implementation.

    This uses a combination of Least Frequently Used (LFU) and
    Least Computationally Expensive (LCE) strategies for invalidating cache entries.

    The cache invalidation strategy calculates a score for each entry based on its
    access frequency and computation duration. The entry with the lowest score will
    be invalidated when the cache reaches its maximum size.

    Attributes
    ----------
    max_size
        The maximum number of entries the cache can store.
    access_weight
        The weight given to the access frequency in the score calculation.
    duration_weight
        The weight given to the computation duration in the score calculation.
    allow_cloudpickle
        Use cloudpickle for storing the data in memory if using shared memory.
    shared
        Whether the cache should be shared between multiple processes.

    """

    def __init__(
        self,
        max_size: int = 128,
        access_weight: float = 0.5,
        duration_weight: float = 0.5,
        *,
        allow_cloudpickle: bool = True,
        shared: bool = True,
    ) -> None:
        """Initialize the HybridCache instance."""
        if shared:
            manager = Manager()
            self._cache_dict = manager.dict()
            self._access_counts = manager.dict()
            self._computation_durations = manager.dict()
            self._cache_lock = manager.Lock()
        else:
            self._cache_dict = {}  # type: ignore[assignment]
            self._access_counts = {}  # type: ignore[assignment]
            self._computation_durations = {}  # type: ignore[assignment]
            self._cache_lock = nullcontext()  # type: ignore[assignment]
        self.max_size: int = max_size
        self.access_weight: float = access_weight
        self.duration_weight: float = duration_weight
        self.shared: bool = shared
        self._allow_cloudpickle: bool = allow_cloudpickle

    @property
    def cache(self) -> dict[Hashable, Any]:
        """Return the cache entries."""
        if not self.shared:
            assert isinstance(self._cache_dict, dict)
            return self._cache_dict
        with self._cache_lock:
            return {k: _maybe_load(v, self._allow_cloudpickle) for k, v in self._cache_dict.items()}

    @property
    def access_counts(self) -> dict[Hashable, int]:
        """Return the access counts of the cache entries."""
        if not self.shared:
            assert isinstance(self._access_counts, dict)
            return self._access_counts
        with self._cache_lock:
            return dict(self._access_counts.items())

    @property
    def computation_durations(self) -> dict[Hashable, float]:
        """Return the computation durations of the cache entries."""
        if not self.shared:
            assert isinstance(self._computation_durations, dict)
            return self._computation_durations
        with self._cache_lock:
            return dict(self._computation_durations.items())

    def get(self, key: Hashable) -> Any | None:
        """Retrieve a value from the cache by its key.

        If the key is present in the cache, its access count is incremented.

        Parameters
        ----------
        key
            The key associated with the value in the cache.

        Returns
        -------
            The value associated with the key if the key is present in the cache,
            otherwise None.

        """
        if key not in self._cache_dict:
            return None
        with self._cache_lock:
            self._access_counts[key] += 1
        value = self._cache_dict[key]
        if self._allow_cloudpickle and self.shared:
            value = cloudpickle.loads(value)
        return value

    def put(self, key: Hashable, value: Any, duration: float) -> None:  # type: ignore[override]
        """Add a value to the cache with its associated key and computation duration.

        If the cache is full, the entry with the lowest score based on the access
        frequency and computation duration will be invalidated.

        Parameters
        ----------
        key
            The key associated with the value.
        value
            The value to store in the cache.
        duration
            The duration of the computation that generated the value.

        """
        if self._allow_cloudpickle and self.shared:
            value = cloudpickle.dumps(value)
        with self._cache_lock:
            if len(self._cache_dict) >= self.max_size:
                self._expire()
            self._cache_dict[key] = value
            self._access_counts[key] = 1
            self._computation_durations[key] = duration

    def _expire(self) -> None:
        """Invalidate the entry with the lowest score based on the access frequency."""
        # Calculate normalized access frequencies and computation durations
        total_access_count = sum(self._access_counts.values())
        total_duration = sum(self._computation_durations.values())
        normalized_access_counts = {
            k: v / total_access_count for k, v in self._access_counts.items()
        }
        normalized_durations = {
            k: v / total_duration for k, v in self._computation_durations.items()
        }

        # Calculate scores using a weighted sum
        scores = {
            k: self.access_weight * normalized_access_counts[k]
            + self.duration_weight * normalized_durations[k]
            for k in self._access_counts
        }

        # Find the key with the lowest score
        lowest_score_key = min(scores, key=lambda k: scores[k])
        del self._cache_dict[lowest_score_key]
        del self._access_counts[lowest_score_key]
        del self._computation_durations[lowest_score_key]

    def clear(self) -> None:
        """Clear the cache."""
        with self._cache_lock:
            self._cache_dict.clear()
            self._access_counts.clear()
            self._computation_durations.clear()

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache.

        Parameters
        ----------
        key
            The key to check for in the cache.

        Returns
        -------
            True if the key is present in the cache, otherwise False.

        """
        return key in self._cache_dict

    def __str__(self) -> str:
        """Return a string representation of the HybridCache.

        The string representation includes information about the cache, access counts,
        and computation durations for each key.

        Returns
        -------
            A string representation of the HybridCache.

        """
        cache_str = f"Cache: {self._cache_dict}\n"
        access_counts_str = f"Access Counts: {self._access_counts}\n"
        computation_durations_str = f"Computation Durations: {self._computation_durations}\n"
        return cache_str + access_counts_str + computation_durations_str

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache_dict)


def _maybe_load(value: bytes | str, allow_cloudpickle: bool) -> Any:  # noqa: FBT001
    return cloudpickle.loads(value) if allow_cloudpickle else value


class LRUCache(_CacheBase):
    """A shared memory LRU cache implementation.

    Parameters
    ----------
    max_size
        Cache size of the LRU cache, by default 128.
    allow_cloudpickle
        Use cloudpickle for storing the data in memory if using shared memory.
    shared
        Whether the cache should be shared between multiple processes.

    """

    def __init__(
        self,
        *,
        max_size: int = 128,
        allow_cloudpickle: bool = True,
        shared: bool = True,
    ) -> None:
        """Initialize the cache."""
        self.max_size = max_size
        self.shared = shared
        self._allow_cloudpickle = allow_cloudpickle
        if max_size == 0:  # pragma: no cover
            msg = "max_size must be greater than 0"
            raise ValueError(msg)
        if shared:
            manager = Manager()
            self._cache_dict = manager.dict()
            self._cache_queue = manager.list()
            self._cache_lock = manager.Lock()
        else:
            self._cache_dict = {}  # type: ignore[assignment]
            self._cache_queue = []  # type: ignore[assignment]
            self._cache_lock = nullcontext()  # type: ignore[assignment]

    def get(self, key: Hashable) -> Any:
        """Get a value from the cache by key."""
        if key not in self._cache_dict:
            return None
        with self._cache_lock:
            value = self._cache_dict[key]
            # Move key to back of queue
            self._cache_queue.remove(key)
            self._cache_queue.append(key)
        if self._allow_cloudpickle and self.shared:
            return cloudpickle.loads(value)
        return value

    def put(self, key: Hashable, value: Any) -> None:
        """Insert a key value pair into the cache."""
        if self._allow_cloudpickle and self.shared:
            value = cloudpickle.dumps(value)
        with self._cache_lock:
            self._cache_dict[key] = value
            cache_size = len(self._cache_queue)
            if cache_size < self.max_size:
                self._cache_queue.append(key)
            else:
                key_to_evict = self._cache_queue.pop(0)
                self._cache_dict.pop(key_to_evict)
                self._cache_queue.append(key)

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache."""
        return key in self._cache_dict

    @property
    def cache(self) -> dict:
        """Returns a copy of the cache."""
        if not self.shared:
            assert isinstance(self._cache_dict, dict)
            return self._cache_dict
        with self._cache_lock:
            return {k: _maybe_load(v, self._allow_cloudpickle) for k, v in self._cache_dict.items()}

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache_dict)

    def clear(self) -> None:
        """Clear the cache."""
        with self._cache_lock:
            keys = list(self._cache_dict.keys())
            for key in keys:
                del self._cache_dict[key]
            del self._cache_queue[:]


class SimpleCache(_CacheBase):
    """A simple cache without any eviction strategy."""

    def __init__(self) -> None:
        """Initialize the cache."""
        self._cache_dict: dict[Hashable, Any] = {}

    def get(self, key: Hashable) -> Any:
        """Get a value from the cache by key."""
        return self._cache_dict.get(key)

    def put(self, key: Hashable, value: Any) -> None:
        """Insert a key value pair into the cache."""
        self._cache_dict[key] = value

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache."""
        return key in self._cache_dict

    @property
    def cache(self) -> dict:
        """Returns a copy of the cache."""
        return self._cache_dict

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache_dict)

    def clear(self) -> None:
        """Clear the cache."""
        keys = list(self._cache_dict.keys())
        for key in keys:
            del self._cache_dict[key]


class DiskCache(_CacheBase):
    """Disk cache implementation using pickle or cloudpickle for serialization.

    Parameters
    ----------
    cache_dir
        The directory where the cache files are stored.
    max_size
        The maximum number of cache files to store. If None, no limit is set.
    use_cloudpickle
        Use cloudpickle for storing the data in memory.
    with_lru_cache
        Use an in-memory LRU cache to prevent reading from disk too often.
    lru_cache_size
        The maximum size of the in-memory LRU cache. Only used if with_lru_cache is True.
    lru_shared
        Whether the in-memory LRU cache should be shared between multiple processes.

    """

    def __init__(
        self,
        cache_dir: str,
        max_size: int | None = None,
        *,
        use_cloudpickle: bool = True,
        with_lru_cache: bool = True,
        lru_cache_size: int = 128,
        lru_shared: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.use_cloudpickle = use_cloudpickle
        self.with_lru_cache = with_lru_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.with_lru_cache:
            self.lru_cache = LRUCache(
                max_size=lru_cache_size,
                allow_cloudpickle=use_cloudpickle,
                shared=lru_shared,
            )

    def _get_file_path(self, key: Hashable) -> Path:
        data = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.md5(data).hexdigest()  # noqa: S324
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: Hashable) -> Any:
        """Get a value from the cache by key."""
        if self.with_lru_cache and key in self.lru_cache:
            return self.lru_cache.get(key)

        file_path = self._get_file_path(key)
        if file_path.exists():
            with file_path.open("rb") as f:
                value = (
                    cloudpickle.load(f) if self.use_cloudpickle else pickle.load(f)  # noqa: S301
                )
            if self.with_lru_cache:
                self.lru_cache.put(key, value)
            return value
        return None

    def put(self, key: Hashable, value: Any) -> None:
        """Insert a key value pair into the cache."""
        file_path = self._get_file_path(key)
        with file_path.open("wb") as f:
            if self.use_cloudpickle:
                cloudpickle.dump(value, f)
            else:
                pickle.dump(value, f)
        if self.with_lru_cache:
            self.lru_cache.put(key, value)
        self._evict_if_needed()

    def _all_files(self) -> list[Path]:
        return list(self.cache_dir.glob("*.pkl"))

    def _evict_if_needed(self) -> None:
        if self.max_size is not None:
            files = self._all_files()
            for _ in range(len(files) - self.max_size):
                oldest_file = min(files, key=lambda f: f.stat().st_ctime_ns)
                oldest_file.unlink()

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache."""
        if self.with_lru_cache and key in self.lru_cache:
            return True
        file_path = self._get_file_path(key)
        return file_path.exists()

    def __len__(self) -> int:
        """Return the number of cache files."""
        files = self._all_files()
        return len(files)

    def clear(self) -> None:
        """Clear the cache by deleting all cache files."""
        for file_path in self._all_files():
            with suppress(Exception):
                file_path.unlink()
        if self.with_lru_cache:
            self.lru_cache.clear()

    @property
    def cache(self) -> dict:
        """Returns a copy of the cache, but only if with_lru_cache is True."""
        if not self.with_lru_cache:  # pragma: no cover
            msg = "LRU cache is not enabled."
            raise AttributeError(msg)
        return self.lru_cache.cache

    @property
    def shared(self) -> bool:
        """Return whether the cache is shared."""
        return self.lru_cache.shared


def memoize(
    cache: HybridCache | LRUCache | SimpleCache | DiskCache | None = None,
    key_func: Callable[..., Hashable] | None = None,
) -> Callable:
    """A flexible memoization decorator that works with different cache types.

    Parameters
    ----------
    cache
        An instance of a cache class (_CacheBase). If None, a SimpleCache is used.
    key_func
        A function to generate cache keys. If None, the default key generation which
        attempts to make all arguments hashable.

    Returns
    -------
    Decorated function with memoization.

    """
    if cache is None:
        cache = SimpleCache()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func:  # noqa: SIM108
                key = key_func(*args, **kwargs)
            else:
                key = _generate_cache_key(args, kwargs)

            if key in cache:
                return cache.get(key)

            if isinstance(cache, HybridCache):
                t_start = time.monotonic()
            result = func(*args, **kwargs)
            if isinstance(cache, HybridCache):
                # For HybridCache, we need to provide a duration
                # Here, we're using a default duration of 1.0
                cache.put(key, result, time.monotonic() - t_start)
            else:
                cache.put(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _hashable_iterable(
    iterable: Iterable,
    *,
    fallback_to_str: bool = True,
    sort: bool = False,
) -> tuple:
    items = sorted(iterable) if sort else iterable
    return tuple(to_hashable(item, fallback_to_str) for item in items)


def _hashable_mapping(
    mapping: dict,
    *,
    fallback_to_str: bool = True,
    sort: bool = False,
) -> tuple:
    items = sorted(mapping.items()) if sort else mapping.items()
    return tuple((k, to_hashable(v, fallback_to_str)) for k, v in items)


# Unique string added to hashable representations to avoid hash collisions
_HASH_MARKER = "__CONVERTED__"


def to_hashable(obj: Any, fallback_to_str: bool = True) -> Any:  # noqa: FBT001, FBT002, PLR0911, PLR0912
    """Convert any object to a hashable representation if not hashable yet.

    Parameters
    ----------
    obj
        The object to convert.
    fallback_to_str
        If True, unhashable objects will be converted to strings as a last resort.
        If False, an exception will be raised for unhashable objects.

    Returns
    -------
    A hashable representation of the input object.

    Raises
    ------
    TypeError
        If the object cannot be made hashable and fallback_to_str is False.

    Notes
    -----
    This function attempts to create a hashable representation of any input object.
    It handles most built-in Python types and some common third-party types like
    numpy arrays and pandas Series/DataFrames.

    """
    try:
        hash(obj)
    except Exception:  # noqa: BLE001, S110
        pass
    else:
        return obj

    tp: type | str = type(obj)
    try:
        hash(tp)
    except Exception:  # noqa: BLE001
        tp = tp.__name__  # type: ignore[union-attr]

    m = _HASH_MARKER
    if isinstance(obj, collections.OrderedDict):
        return (m, tp, _hashable_mapping(obj, fallback_to_str=fallback_to_str))
    if isinstance(obj, collections.defaultdict):
        data = (
            to_hashable(obj.default_factory, fallback_to_str),
            _hashable_mapping(obj, sort=True, fallback_to_str=fallback_to_str),
        )
        return (m, tp, data)
    if isinstance(obj, collections.Counter):
        return (m, tp, tuple(sorted(obj.items())))
    if isinstance(obj, dict):
        return (m, tp, _hashable_mapping(obj, sort=True, fallback_to_str=fallback_to_str))
    if isinstance(obj, set | frozenset):
        return (m, tp, _hashable_iterable(obj, sort=True, fallback_to_str=fallback_to_str))
    if isinstance(obj, list | tuple):
        return (m, tp, _hashable_iterable(obj, fallback_to_str=fallback_to_str))
    if isinstance(obj, collections.deque):
        return (m, tp, (obj.maxlen, _hashable_iterable(obj, fallback_to_str=fallback_to_str)))
    if isinstance(obj, bytearray):
        return (m, tp, tuple(obj))
    if isinstance(obj, array.array):
        return (m, tp, (obj.typecode, tuple(obj)))

    # Handle numpy arrays
    if "numpy" in sys.modules and isinstance(obj, sys.modules["numpy"].ndarray):
        return (m, tp, (obj.shape, obj.dtype.str, tuple(obj.flatten())))

    # Handle pandas Series and DataFrames
    if "pandas" in sys.modules:
        if isinstance(obj, sys.modules["pandas"].Series):
            return (m, tp, (obj.name, to_hashable(obj.to_dict(), fallback_to_str)))
        if isinstance(obj, sys.modules["pandas"].DataFrame):
            return (m, tp, to_hashable(obj.to_dict("list"), fallback_to_str))

    if fallback_to_str:
        return (m, tp, str(obj))
    msg = f"Object of type {type(obj)} cannot be hashed"
    raise TypeError(msg)


def _generate_cache_key(args: tuple, kwargs: dict, *, fallback_to_str: bool = True) -> Hashable:
    """Generate a hashable key from function arguments.

    Parameters
    ----------
    args
        Positional arguments to be included in the key.
    kwargs
        Keyword arguments to be included in the key.
    fallback_to_str
        If True, unhashable objects will be converted to strings as a last resort.
        If False, an exception will be raised for unhashable objects.

    Returns
    -------
    A hash value that can be used as a cache key.

    Notes
    -----
    This function creates a hashable representation of both positional and keyword
    arguments, allowing for effective caching of function calls with various
    argument types.

    """
    return to_hashable((args, kwargs), fallback_to_str)
