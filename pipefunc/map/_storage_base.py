# This file is part of the pipefunc package.
# Originally, it is based on code from the `aiida-dynamic-workflows` package.
# Its license can be found in the LICENSE file in this folder.
# See `git diff 98a1736 pipefunc/map/_filearray.py` for the changes made.

from __future__ import annotations

import abc
import functools
import itertools
from typing import TYPE_CHECKING, Any

from pipefunc.map._mapspec import shape_to_strides

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

storage_registry = {}


def _iterate_shape_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*map(range, shape))


def _select_by_mask(
    mask: tuple[bool, ...],
    tuple1: tuple[Any, ...],
    tuple2: tuple[Any, ...],
) -> tuple[Any, ...]:
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

    shape: tuple[int, ...]
    internal_shape: tuple[int, ...]
    shape_mask: tuple[bool, ...]
    storage_id: str

    @abc.abstractmethod
    def __init__(
        self,
        folder: str | Path,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
    ) -> None: ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def rank(self) -> int: ...

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

    @functools.cached_property
    def full_shape(self) -> tuple[int, ...]:
        """Return the full shape of the array."""
        return _select_by_mask(self.shape_mask, self.shape, self.internal_shape)

    @functools.cached_property
    def strides(self) -> tuple[int, ...]:
        """Return the strides of the array."""
        return shape_to_strides(self.shape)


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
