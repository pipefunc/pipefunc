"""Implements the base class and helpers for file/memory-based arrays."""

from __future__ import annotations

import abc
import functools
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from pipefunc._utils import create_mask_for_masked_values, prod
from pipefunc.map._mapspec import shape_to_strides
from pipefunc.map._shapes import shape_is_resolved

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pipefunc.map._types import ShapeTuple


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


def infer_irregular_length(value: Any) -> int:
    """Return the realised length for a single irregular axis."""
    if value is None or value is np.ma.masked:
        result = 0
    elif isinstance(value, np.ma.MaskedArray):
        if value.ndim == 0:
            result = 0 if np.ma.is_masked(value) else 1
        elif value.mask is np.ma.nomask:
            result = int(value.shape[0])
        else:
            mask = np.atleast_1d(value.mask)
            valid = np.nonzero(~mask)[0]
            result = 0 if valid.size == 0 else int(valid[-1]) + 1
    else:
        try:
            result = len(value)  # type: ignore[arg-type]
        except TypeError:
            result = 1
    return result


class StorageBase(abc.ABC):
    """Base class for file-based arrays."""

    shape: ShapeTuple
    internal_shape: ShapeTuple
    shape_mask: tuple[bool, ...]
    irregular: bool
    storage_id: str
    requires_serialization: bool
    _is_resolved: bool = False

    @abc.abstractmethod
    def __init__(
        self,
        folder: str | Path | None,
        shape: ShapeTuple,
        internal_shape: ShapeTuple | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        irregular: bool = False,  # noqa: FBT002
    ) -> None: ...  # pragma: no cover

    @functools.cached_property
    def resolved_shape(self) -> tuple[int, ...]:
        """Return the resolved shape of the array."""
        # This cached property (and resolved_internal_shape) only exist to help mypy.
        # For performance reasons, we assume this is only called once the shape is resolved.
        assert shape_is_resolved(self.shape)
        return self.shape

    @functools.cached_property
    def resolved_internal_shape(self) -> tuple[int, ...]:
        # See comment in `resolved_shapes`.
        assert shape_is_resolved(self.internal_shape)
        return self.internal_shape

    def full_shape_is_resolved(self) -> bool:
        """Return whether the shape is resolved."""
        # This function is called many times, so we cache the result
        if self._is_resolved:
            return True
        self._is_resolved = all(isinstance(s, int) for s in self.shape + self.internal_shape)
        return self._is_resolved

    @abc.abstractmethod
    def get_from_index(self, index: int) -> Any: ...

    @abc.abstractmethod
    def has_index(self, index: int) -> bool: ...

    @abc.abstractmethod
    def __getitem__(self, key: tuple[int | slice, ...]) -> Any: ...

    @abc.abstractmethod
    def to_array(
        self,
        *,
        splat_internal: bool | None = None,
    ) -> np.ma.core.MaskedArray: ...

    @property
    @abc.abstractmethod
    def mask(self) -> np.ma.core.MaskedArray: ...

    @abc.abstractmethod
    def mask_linear(self) -> list[bool]: ...

    @abc.abstractmethod
    def dump(self, key: tuple[int | slice, ...], value: Any) -> None: ...

    @property
    @abc.abstractmethod
    def dump_in_subprocess(self) -> bool:  # pragma: no cover
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""

    # ---- Irregular array helpers -------------------------------------------------

    def irregular_extent(self, external_index: tuple[int, ...]) -> tuple[int, ...] | None:
        """Return the realised extent along irregular axes for ``external_index``."""
        if not self.irregular or not self.internal_shape:
            return None
        return self._compute_irregular_extent(external_index)

    def _compute_irregular_extent(
        self,
        external_index: tuple[int, ...],  # noqa: ARG002
    ) -> tuple[int, ...] | None:  # pragma: no cover
        """Compute the realised size for irregular axes.

        Sub-classes should override this and return ``None`` when the extent
        cannot be determined cheaply (e.g. multi-dimensional ragged data).
        """
        return None

    def is_element_masked(
        self,
        key: tuple[int | slice, ...],
    ) -> bool:
        """Return ``True`` if the element requested by ``key`` is masked."""
        if not (self.irregular and self.internal_shape and len(self.internal_shape) == 1):
            return False

        normalized = normalize_key(
            key,
            self.resolved_shape,
            self.resolved_internal_shape,
            self.shape_mask,
        )

        internal_components = tuple(x for x, m in zip(normalized, self.shape_mask) if not m)
        if not internal_components:
            return False

        external_components = tuple(x for x, m in zip(normalized, self.shape_mask) if m)

        if any(isinstance(x, slice) for x in (*internal_components, *external_components)):
            return False

        internal_list = [x for x in internal_components if isinstance(x, int)]
        external_list = [x for x in external_components if isinstance(x, int)]
        if len(internal_list) != len(internal_components) or len(external_list) != len(
            external_components,
        ):
            return False
        internal_index = tuple(internal_list)
        external_index = tuple(external_list)

        extent = self.irregular_extent(external_index)
        if extent is None:
            return False

        return any(idx >= size for idx, size in zip(internal_index, extent))

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        return prod(self.resolved_shape)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        return len(self.resolved_shape)

    def _ensure_masked_array_for_irregular(self, data: Any) -> Any:
        """Convert arrays with masked sentinels to MaskedArrays if irregular=True.

        This should be called at the end of __getitem__ implementations
        to ensure consistent behavior across all storage types.
        """
        if not self.irregular:
            return data

        # Only process numpy arrays
        if not isinstance(data, np.ndarray):
            return data

        # Check if data contains any masked sentinels
        mask = create_mask_for_masked_values(data)

        if np.any(mask):
            # Contains masked values - return MaskedArray
            return np.ma.MaskedArray(data, mask=mask, dtype=object)

        return data

    @functools.cached_property
    def full_shape(self) -> tuple[int, ...]:
        """Return the full shape of the array."""
        full_shape = select_by_mask(
            self.shape_mask,
            self.resolved_shape,
            self.resolved_internal_shape,
        )
        assert shape_is_resolved(full_shape)
        return full_shape

    @functools.cached_property
    def strides(self) -> tuple[int, ...]:
        """Return the strides of the array."""
        return shape_to_strides(self.resolved_shape)

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

    if for_dump:  # internal_shape is not involved when dumping
        shape_mask = (True,) * len(key)

    for axis, (mask, k) in enumerate(zip(shape_mask, key, strict=True)):
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
