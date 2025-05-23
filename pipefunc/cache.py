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
import warnings
from contextlib import nullcontext, suppress
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable
    from multiprocessing.managers import DictProxy, ListProxy, SyncManager


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

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the object for pickling."""
        state = self.__dict__.copy()

        if self.shared:
            # Convert shared structures to regular ones
            state["_cache_dict"] = _dict_to_regular(self._cache_dict)
            state["_access_counts"] = _dict_to_regular(self._access_counts)
            state["_computation_durations"] = _dict_to_regular(self._computation_durations)
            # Remove unpicklable lock
            state.pop("_cache_lock", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the object after unpickling."""
        self.__dict__.update(state)
        if not self.shared:
            return
        manager = Manager()
        self._cache_dict = _create_shared_dict(manager, self._cache_dict)  # type: ignore[arg-type]
        self._access_counts = _create_shared_dict(manager, self._access_counts)  # type: ignore[arg-type]
        self._computation_durations = _create_shared_dict(manager, self._computation_durations)  # type: ignore[arg-type]
        self._cache_lock = manager.Lock()


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

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the object for pickling."""
        state = self.__dict__.copy()

        if self.shared:
            # Convert shared structures to regular ones
            state["_cache_dict"] = _dict_to_regular(self._cache_dict)
            state["_cache_queue"] = _list_to_regular(self._cache_queue)
            # Remove unpicklable lock
            state.pop("_cache_lock", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the object after unpickling."""
        self.__dict__.update(state)
        if not self.shared:
            return
        manager = Manager()
        self._cache_dict = _create_shared_dict(manager, self._cache_dict)  # type: ignore[arg-type]
        self._cache_queue = _create_shared_list(manager, self._cache_queue)  # type: ignore[arg-type]
        self._cache_lock = manager.Lock()


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
    permissions
        The file permissions to set for the cache files.
        If None, the default permissions are used.
        Some examples:

            - 0o660 (read/write for owner and group, no access for others)
            - 0o644 (read/write for owner, read-only for group and others)
            - 0o777 (read/write/execute for everyone - generally not recommended)
            - 0o600 (read/write for owner, no access for group and others)
            - None (use the system's default umask)

    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_size: int | None = None,
        *,
        use_cloudpickle: bool = True,
        with_lru_cache: bool = True,
        lru_cache_size: int = 128,
        lru_shared: bool = True,
        permissions: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.use_cloudpickle = use_cloudpickle
        self.with_lru_cache = with_lru_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.permissions = permissions

        if self.with_lru_cache:
            self.lru_cache = LRUCache(
                max_size=lru_cache_size,
                allow_cloudpickle=use_cloudpickle,
                shared=lru_shared,
            )

    def _get_file_path(self, key: Hashable) -> Path:
        key_hash = _pickle_key(key)
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
        if self.permissions is not None:
            file_path.chmod(self.permissions)  # Set permissions after writing

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
                try:
                    oldest_file.unlink()
                except PermissionError:  # pragma: no cover
                    warnings.warn(
                        f"Permission denied when trying to delete {oldest_file}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

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
            with suppress(PermissionError, FileNotFoundError):
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
        return self.lru_cache.shared if self.with_lru_cache else True


def _pickle_key(obj: Any) -> str:
    # Based on the implementation of `diskcache` although that also
    # does pickle_tools.optimize which we don't need here
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(data).hexdigest()  # noqa: S324


def _cloudpickle_key(obj: Any) -> str:
    data = cloudpickle.dumps(obj)
    return hashlib.md5(data).hexdigest()  # noqa: S324


def memoize(
    cache: HybridCache | LRUCache | SimpleCache | DiskCache | None = None,
    key_func: Callable[..., Hashable] | None = None,
    *,
    fallback_to_pickle: bool = True,
    unhashable_action: Literal["error", "warning", "ignore"] = "error",
) -> Callable:
    """A flexible memoization decorator that works with different cache types.

    Parameters
    ----------
    cache
        An instance of a cache class (_CacheBase). If None, a SimpleCache is used.
    key_func
        A function to generate cache keys. If None, the default key generation which
        attempts to make all arguments hashable.
    fallback_to_pickle
        If ``True``, unhashable objects will be pickled to bytes using `cloudpickle` as a last resort.
        If ``False``, an exception will be raised for unhashable objects.
        Only used if ``key_func`` is None.
    unhashable_action
        Determines the behavior when encountering unhashable objects:
        - "error": Raise an UnhashableError (default).
        - "warning": Log a warning and skip caching for that call.
        - "ignore": Silently skip caching for that call.
        Only used if ``key_func`` is None.

    Returns
    -------
    Decorated function with memoization.

    Raises
    ------
    UnhashableError
        If the object cannot be made hashable and ``fallback_to_pickle`` is ``False``.

    Notes
    -----
    This function creates a hashable representation of both positional and keyword
    arguments, allowing for effective caching of function calls with various
    argument types.

    """
    if cache is None:
        cache = SimpleCache()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = try_to_hashable(  # type: ignore[assignment]
                    (args, kwargs),
                    fallback_to_pickle,
                    unhashable_action,
                    func.__name__,
                )
                if key is UnhashableError:
                    return func(*args, **kwargs)

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


def try_to_hashable(
    obj: Any,
    fallback_to_pickle: bool = True,  # noqa: FBT001, FBT002
    unhashable_action: Literal["error", "warning", "ignore"] = "error",
    where: str = "function",
) -> Hashable | type[UnhashableError]:
    """Try to convert an object to a hashable representation.

    Wrapper around ``to_hashable`` that allows for different actions when encountering
    unhashable objects.

    Parameters
    ----------
    obj
        The object to convert.
    fallback_to_pickle
        If ``True``, unhashable objects will be pickled to bytes using `cloudpickle` as a last resort.
        If ``False``, an exception will be raised for unhashable objects.
    unhashable_action
        Determines the behavior when encountering unhashable objects:
        - ``"error"``: Raise an `UnhashableError` (default).
        - ``"warning"``: Log a warning and skip caching for that call.
        - ``"ignore"``: Silently skip caching for that call. Returns `UnhashableError`.
    where
        The location where the unhashable object was encountered.
        Used for warning or error messages.

    Returns
    -------
        A hashable representation of the input object.

    Raises
    ------
    UnhashableError
        If the object cannot be made hashable and ``fallback_to_pickle`` is ``False``.

    Notes
    -----
    This function attempts to create a hashable representation of any input object.
    It handles most built-in Python types and some common third-party types like
    numpy arrays and pandas Series/DataFrames.

    """
    try:
        return to_hashable(obj, fallback_to_pickle=fallback_to_pickle)
    except UnhashableError:
        if unhashable_action == "error":
            raise
        if unhashable_action == "warning":
            warnings.warn(
                f"Unhashable arguments in '{where}'. Skipping cache.",
                UserWarning,
                stacklevel=3,
            )
        return UnhashableError


def _hashable_iterable(
    iterable: Iterable,
    fallback_to_pickle: bool,  # noqa: FBT001
    *,
    sort: bool = False,
) -> tuple:
    items = sorted(iterable) if sort else iterable
    return tuple(to_hashable(item, fallback_to_pickle) for item in items)


def _hashable_mapping(
    mapping: dict,
    fallback_to_pickle: bool,  # noqa: FBT001
    *,
    sort: bool = False,
) -> tuple:
    items = sorted(mapping.items()) if sort else mapping.items()
    return tuple((k, to_hashable(v, fallback_to_pickle)) for k, v in items)


# Unique string added to hashable representations to avoid hash collisions
_HASH_MARKER = "__CONVERTED__"


def to_hashable(  # noqa: C901, PLR0911, PLR0912
    obj: Any,
    fallback_to_pickle: bool = True,  # noqa: FBT001, FBT002
) -> Any:
    """Convert any object to a hashable representation if not hashable yet.

    Parameters
    ----------
    obj
        The object to convert.
    fallback_to_pickle
        If ``True``, unhashable objects will be pickled to bytes using `cloudpickle` as a last resort.
        If ``False``, an exception will be raised for unhashable objects.

    Returns
    -------
    A hashable representation of the input object.

    Raises
    ------
    UnhashableError
        If the object cannot be made hashable and fallback_to_pickle is False.

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
        return (m, tp, _hashable_mapping(obj, fallback_to_pickle))
    if isinstance(obj, collections.defaultdict):
        data = (
            to_hashable(obj.default_factory, fallback_to_pickle),
            _hashable_mapping(obj, fallback_to_pickle, sort=True),
        )
        return (m, tp, data)
    if isinstance(obj, collections.Counter):
        return (m, tp, tuple(sorted(obj.items())))
    if isinstance(obj, dict):
        return (m, tp, _hashable_mapping(obj, fallback_to_pickle, sort=True))
    if isinstance(obj, set | frozenset):
        return (m, tp, _hashable_iterable(obj, fallback_to_pickle, sort=True))
    if isinstance(obj, list | tuple):
        return (m, tp, _hashable_iterable(obj, fallback_to_pickle))
    if isinstance(obj, collections.deque):
        return (m, tp, (obj.maxlen, _hashable_iterable(obj, fallback_to_pickle)))
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
            return (m, tp, (obj.name, to_hashable(obj.to_dict(), fallback_to_pickle)))
        if isinstance(obj, sys.modules["pandas"].DataFrame):
            return (m, tp, to_hashable(obj.to_dict("list"), fallback_to_pickle))

    if fallback_to_pickle:
        try:
            return (m, tp, _cloudpickle_key(obj))
        except Exception as e:
            raise UnhashableError(obj) from e
    raise UnhashableError(obj)


class UnhashableError(TypeError):
    """Exception raised for objects that cannot be made hashable."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.message = (
            f"Object of type {type(obj)} cannot be hashed using `pipefunc.cache.to_hashable`."
        )
        super().__init__(self.message)


# Helper functions for pickling
def _dict_to_regular(shared_dict: DictProxy) -> dict:
    """Convert a shared dictionary to a regular dictionary."""
    return dict(shared_dict.items())


def _list_to_regular(shared_list: ListProxy) -> list:
    """Convert a shared list to a regular list."""
    return list(shared_list)


def _create_shared_dict(manager: SyncManager, regular_dict: dict) -> DictProxy:
    """Create a shared dictionary and populate it."""
    shared_dict = manager.dict()
    shared_dict.update(regular_dict)
    return shared_dict


def _create_shared_list(manager: SyncManager, regular_list: list) -> ListProxy:
    """Create a shared list and populate it."""
    shared_list = manager.list()
    shared_list.extend(regular_list)
    return shared_list
