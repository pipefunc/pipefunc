"""Implements the base class and helpers for file/memory-based arrays."""

from __future__ import annotations

import abc
import functools
import itertools
import os
from typing import TYPE_CHECKING, Any

from pipefunc._utils import prod
from pipefunc.map._mapspec import shape_to_strides
from pipefunc.map._shapes import shape_is_resolved

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

    from pipefunc.map._types import ShapeTuple

_IS_PYTEST = os.getenv("PYTEST_CURRENT_TEST")

storage_registry = {}


def iterate_shape_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Iterate over all indices of a given shape."""
    return itertools.product(*map(range, shape))


def select_by_mask(
    mask: tuple[bool, ...],
    tuple1: tuple[Any, ...],
    tuple2: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Select elements from two tuples based on a mask."""
    result = []
    index1, index2 = 0, 0
    for m in mask:
        if m:
            result.append(tuple1[index1])
            index1 += 1
        else:
            result.append(tuple2[index2])
            index2 += 1
    return tuple(result)


class StorageBase(abc.ABC):
    """Base class for file-based arrays."""

    shape: ShapeTuple
    internal_shape: ShapeTuple
    shape_mask: tuple[bool, ...]
    storage_id: str
    requires_serialization: bool

    @abc.abstractmethod
    def __init__(
        self,
        folder: str | Path | None,
        shape: ShapeTuple,
        internal_shape: ShapeTuple | None = None,
        shape_mask: tuple[bool, ...] | None = None,
    ) -> None: ...

    @property
    def resolved_shape(self) -> tuple[int, ...]:
        """Return the resolved shape of the array."""
        if TYPE_CHECKING or _IS_PYTEST:
            assert shape_is_resolved(self.shape)
        return self.shape

    @property
    def resolved_internal_shape(self) -> tuple[int, ...]:
        if TYPE_CHECKING or _IS_PYTEST:
            assert shape_is_resolved(self.internal_shape)
        return self.internal_shape

    def set_shape(
        self,
        shape: ShapeTuple | None = None,
        internal_shape: ShapeTuple | None = None,
    ) -> None:
        """Set the shape and internal shape of the array."""
        if shape is not None:
            self.shape = shape
        if internal_shape is not None:
            self.internal_shape = internal_shape

    @property
    def full_shape_is_resolved(self) -> bool:
        """Return whether the shape is resolved."""
        return all(isinstance(s, int) for s in self.shape + self.internal_shape)

    @abc.abstractmethod
    def get_from_index(self, index: int) -> Any: ...

    @abc.abstractmethod
    def has_index(self, index: int) -> bool: ...

    @abc.abstractmethod
    def __getitem__(self, key: tuple[int | slice, ...]) -> Any: ...

    @abc.abstractmethod
    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray: ...

    @property
    @abc.abstractmethod
    def mask(self) -> np.ma.core.MaskedArray: ...

    @abc.abstractmethod
    def mask_linear(self) -> list[bool]: ...

    @abc.abstractmethod
    def dump(self, key: tuple[int | slice, ...], value: Any) -> None: ...

    @property
    @abc.abstractmethod
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        assert shape_is_resolved(self.shape)
        return prod(self.shape)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        assert shape_is_resolved(self.shape)
        return len(self.shape)

    @functools.cached_property
    def full_shape(self) -> tuple[int, ...]:
        """Return the full shape of the array."""
        assert shape_is_resolved(self.shape)
        assert shape_is_resolved(self.internal_shape)
        return select_by_mask(self.shape_mask, self.shape, self.internal_shape)

    @functools.cached_property
    def strides(self) -> tuple[int, ...]:
        """Return the strides of the array."""
        assert shape_is_resolved(self.shape)
        return shape_to_strides(self.shape)

    def persist(self) -> None:  # noqa: B027
        """Save a memory-based storage to disk."""


def register_storage(cls: type[StorageBase], storage_id: str | None = None) -> None:
    """Register a StorageBase class.

    Parameters
    ----------
    cls
        Storage class that should be registered.
    storage_id
        Storage identifier, defaults to the `storage_id` attribute of the class.

    Notes
    -----
    This function maintains a mapping from storage identifiers to storage
    classes. When a storage class is registered, it will replace any class
    previously registered under the same storage identifier, if present.

    """
    if storage_id is None:
        storage_id = cls.storage_id
    storage_registry[storage_id] = cls


def normalize_key(
    key: tuple[int | slice, ...],
    shape: tuple[int, ...],
    internal_shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    *,
    for_dump: bool = False,
) -> tuple[int | slice, ...]:
    if not isinstance(key, tuple):
        key = (key,)

    expected_rank = sum(shape_mask) if for_dump else len(shape_mask)

    if len(key) != expected_rank:
        msg = (
            f"Too many indices for array: array is {expected_rank}-dimensional, "
            f"but {len(key)} were indexed"
        )
        raise IndexError(msg)

    normalized_key: list[int | slice] = []
    shape_index = 0
    internal_shape_index = 0

    for axis, (mask, k) in enumerate(zip(shape_mask, key)):
        if mask:
            axis_size = shape[shape_index]
            shape_index += 1
        else:
            axis_size = internal_shape[internal_shape_index]
            internal_shape_index += 1

        if isinstance(k, slice):
            normalized_key.append(k)
        else:
            normalized_k = k if k >= 0 else (k + axis_size)
            if not (0 <= normalized_k < axis_size):
                msg = f"Index {k} is out of bounds for axis {axis} with size {axis_size}"
                raise IndexError(msg)
            normalized_key.append(normalized_k)

    return tuple(normalized_key)


def get_storage_class(storage: str) -> type[StorageBase]:
    """Get the storage class by its identifier.

    See `pipefunc.map.storage_registry` for available storage classes.

    Parameters
    ----------
    storage
        The storage class identifier.

    Returns
    -------
    The storage class.

    Raises
    ------
    ValueError
        If the storage class is not found.

    """
    if storage not in storage_registry:
        available = ", ".join(storage_registry.keys())
        msg = f"Storage class `{storage}` not found, only `{available}` available."
        raise ValueError(msg)
    return storage_registry[storage]
