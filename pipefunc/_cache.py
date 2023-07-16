from __future__ import annotations

from contextlib import nullcontext
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any

import cloudpickle

if TYPE_CHECKING:
    from collections.abc import Hashable
    from multiprocessing.managers import ListProxy

_NONE_RETURN_STR = "__ReturnsNone__"


class HybridCache:
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
    """

    def __init__(
        self,
        max_size: int = 128,
        access_weight: float = 0.5,
        duration_weight: float = 0.5,
        *,
        shared: bool = True,
    ) -> None:
        """Initialize the HybridCache instance."""
        if shared:
            manager = Manager()
            self._cache = manager.dict()
            self._access_counts = manager.dict()
            self._computation_durations = manager.dict()
            self.lock = manager.Lock()
        else:
            self._cache = {}  # type: ignore[assignment]
            self._access_counts = {}  # type: ignore[assignment]
            self._computation_durations = {}  # type: ignore[assignment]
            self.lock = nullcontext()  # type: ignore[assignment]
        self.max_size: int = max_size
        self.access_weight: float = access_weight
        self.duration_weight: float = duration_weight
        self.shared: bool = shared

    @property
    def cache(self) -> dict[Hashable, Any]:
        """Return the cache entries."""
        if not self.shared:
            assert isinstance(self._cache, dict)
            return self._cache
        with self.lock:
            return dict(self._cache.items())

    @property
    def access_counts(self) -> dict[Hashable, int]:
        """Return the access counts of the cache entries."""
        if not self.shared:
            assert isinstance(self._access_counts, dict)
            return self._access_counts
        with self.lock:
            return dict(self._access_counts.items())

    @property
    def computation_durations(self) -> dict[Hashable, float]:
        """Return the computation durations of the cache entries."""
        if not self.shared:
            assert isinstance(self._computation_durations, dict)
            return self._computation_durations
        with self.lock:
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
        Any
            The value associated with the key if the key is present in the cache,
            otherwise None.
        """
        with self.lock:
            if key in self._cache:
                self._access_counts[key] += 1
                return self._cache[key]
        return None  # pragma: no cover

    def put(self, key: Hashable, value: Any, duration: float) -> None:
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
        with self.lock:
            if len(self._cache) >= self.max_size:
                self._expire()
            self._cache[key] = value
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
        del self._cache[lowest_score_key]
        del self._access_counts[lowest_score_key]
        del self._computation_durations[lowest_score_key]

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache.

        Parameters
        ----------
        key
            The key to check for in the cache.

        Returns
        -------
        bool
            True if the key is present in the cache, otherwise False.
        """
        return key in self._cache

    def __str__(self) -> str:
        """Return a string representation of the HybridCache.

        The string representation includes information about the cache, access counts,
        and computation durations for each key.

        Returns
        -------
        str
            A string representation of the HybridCache.
        """
        cache_str = f"Cache: {self._cache}\n"
        access_counts_str = f"Access Counts: {self._access_counts}\n"
        computation_durations_str = (
            f"Computation Durations: {self._computation_durations}\n"
        )
        return cache_str + access_counts_str + computation_durations_str


class LRUCache:
    """A shared memory LRU cache implementation.

    Parameters
    ----------
    max_size
        Cache size of the LRU cache, by default 128.
    with_cloudpickle
        Use cloudpickle for storing the data in memory.
    shared
        Whether the cache should be shared between multiple processes.
    """

    def __init__(
        self,
        *,
        max_size: int = 128,
        with_cloudpickle: bool = False,
        shared: bool = True,
    ) -> None:
        """Initialize the cache."""
        self.max_size = max_size
        self._with_cloudpickle = with_cloudpickle
        self.shared = shared
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

    def get(self, key: Hashable) -> tuple[bool, Any]:
        """Get a value from the cache by key."""
        with self._cache_lock:
            value = self._cache_dict.get(key)
            if value is not None:  # Move key to back of queue
                self._cache_queue.remove(key)
                self._cache_queue.append(key)
        if value is not None:
            if value == _NONE_RETURN_STR:
                value = None
            elif self._with_cloudpickle:
                value = cloudpickle.loads(value)
        return value

    def put(self, key: Hashable, value: Any) -> ListProxy[Any]:
        """Insert a key value pair into the cache."""
        if value is None:
            value = _NONE_RETURN_STR
        elif self._with_cloudpickle:
            value = cloudpickle.dumps(value)
        with self._cache_lock:
            cache_size = len(self._cache_queue)
            self._cache_dict[key] = value
            if cache_size < self.max_size:
                self._cache_queue.append(key)
            else:
                key_to_evict = self._cache_queue.pop(0)
                self._cache_dict.pop(key_to_evict)
                self._cache_queue.append(key)
            return self._cache_queue

    def __contains__(self, key: Hashable) -> bool:
        """Check if a key is present in the cache."""
        return key in self._cache_dict

    @property
    def cache(self) -> dict:
        """Returns a copy of the cache."""
        return dict(self._cache_dict.items())
