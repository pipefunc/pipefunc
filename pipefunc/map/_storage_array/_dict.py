"""Implements a `dict` based `StorageBase` class."""

from __future__ import annotations

import itertools
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pipefunc._utils import dump, load

from ._base import (
    StorageBase,
    iterate_shape_indices,
    normalize_key,
    register_storage,
    select_by_mask,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from multiprocessing.managers import DictProxy

    from pipefunc.map._types import ShapeTuple


class DictArray(StorageBase):
    """A `numpy.ndarray` backed by a `dict` with internal structure."""

    storage_id = "dict"
    requires_serialization = False

    def __init__(
        self,
        folder: str | Path | None,
        shape: ShapeTuple,
        internal_shape: ShapeTuple | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        mapping: MutableMapping[tuple[int, ...], Any] | None = None,
    ) -> None:
        """Create a `numpy.ndarray` backed by a `dict`."""
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        self.folder = Path(folder) if folder is not None else folder
        self.shape = tuple(shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()
        if mapping is None:
            mapping = {}
        self._dict: dict[tuple[int, ...], Any] = mapping  # type: ignore[assignment]
        self.load()

    def __repr__(self) -> str:
        folder = f"'{self.folder}'" if self.folder is not None else self.folder
        return (
            f"DictArray(folder={folder}, "
            f"shape={self.shape}, "
            f"internal_shape={self.internal_shape}, "
            f"shape_mask={self.shape_mask}, "
            f"mapping={self._dict})"
        )

    def get_from_index(self, index: int) -> Any:
        """Return the data associated with the given linear index."""
        np_index = np.unravel_index(index, self.resolved_shape)
        return self._dict[np_index]  # type: ignore[index]

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        np_index = np.unravel_index(index, self.resolved_shape)
        return np_index in self._dict

    def _internal_mask(self) -> np.ma.MaskedArray:
        if self.internal_shape:
            return np.ma.empty(self.internal_shape, dtype=object)
        return np.ma.masked

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        """Return the data associated with the given key."""
        key = normalize_key(key, self.resolved_shape, self.resolved_internal_shape, self.shape_mask)
        assert len(key) == len(self.full_shape)
        if any(isinstance(k, slice) for k in key):
            shape = tuple(
                len(range(*k.indices(s))) if isinstance(k, slice) else 1
                for s, k in zip(self.full_shape, key)
            )
            data: np.ndarray = np.empty(shape, dtype=object)
            for i, index in enumerate(
                itertools.product(*self._slice_indices(key, self.full_shape)),
            ):
                external_key = tuple(x for x, m in zip(index, self.shape_mask) if m)
                if self.internal_shape:
                    internal_key = tuple(x for x, m in zip(index, self.shape_mask) if not m)
                    if external_key in self._dict:
                        arr = np.asarray(self._dict[external_key])
                        value = arr[internal_key]
                    else:
                        value = self._internal_mask()[internal_key]
                else:  # noqa: PLR5501
                    if external_key in self._dict:
                        value = self._dict[external_key]
                    else:
                        value = self._internal_mask()
                j = np.unravel_index(i, shape)
                data[j] = value
            new_shape = tuple(
                len(range(*k.indices(s)))
                for s, k in zip(self.full_shape, key)
                if isinstance(k, slice)
            )
            return data.reshape(new_shape)

        external_key = tuple(x for x, m in zip(key, self.shape_mask) if m)  # type: ignore[misc]
        internal_key = tuple(x for x, m in zip(key, self.shape_mask) if not m)  # type: ignore[misc]

        if external_key in self._dict:
            data = self._dict[external_key]
        else:
            return self._internal_mask()
        if internal_key:
            arr = np.asarray(data)
            return arr[internal_key]
        return data

    def _slice_indices(self, key: tuple[int | slice, ...], shape: tuple[int, ...]) -> list[range]:
        assert len(key) == len(shape)
        slice_indices = []
        for size, k in zip(shape, key):
            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(size)))
            else:
                slice_indices.append(range(k, k + 1))
        return slice_indices

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Return the array as a NumPy masked array."""
        if splat_internal is None:
            splat_internal = bool(self.resolved_internal_shape)
        if not splat_internal:
            data: np.ndarray = _masked_empty(self.resolved_shape)
            mask: np.ndarray = np.full(self.resolved_shape, fill_value=True, dtype=bool)
            for external_index, value in self._dict.items():
                data[external_index] = value
                mask[external_index] = False
            return np.ma.MaskedArray(data, mask=mask, dtype=object)
        if not self.resolved_internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)

        data = _masked_empty(self.full_shape)
        mask = np.full(self.full_shape, fill_value=True, dtype=bool)
        for external_index, value in self._dict.items():
            value_array = np.asarray(value)

            if value_array.shape == self.resolved_internal_shape:
                # Normal case - shapes match
                full_index = select_by_mask(
                    self.shape_mask,
                    external_index,
                    (slice(None),) * len(self.resolved_internal_shape),
                )
                data[full_index] = value_array
                mask[full_index] = False
            else:
                for internal_index in iterate_shape_indices(self.resolved_internal_shape):
                    full_index = select_by_mask(self.shape_mask, external_index, internal_index)
                    data[full_index] = value_array[internal_index]
                    mask[full_index] = False
        return np.ma.MaskedArray(data, mask=mask, dtype=object)

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return the mask associated with the array."""
        mask: np.ndarray = np.full(self.resolved_shape, fill_value=True, dtype=bool)
        for external_index in self._dict:
            mask[external_index] = False
        return np.ma.MaskedArray(mask, mask=mask, dtype=bool)

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        return list(self.mask.data[:].flat)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = DictArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        key = normalize_key(
            key,
            self.resolved_shape,
            self.resolved_internal_shape,
            self.shape_mask,
            for_dump=True,
        )
        if any(isinstance(k, slice) for k in key):
            for external_index in itertools.product(*self._slice_indices(key, self.resolved_shape)):
                if self.resolved_internal_shape:
                    value = np.asarray(value)  # in case it's a list
                    assert value.shape == self.resolved_internal_shape
                    self._dict[external_index] = value
                else:
                    self._dict[external_index] = value
            return

        self._dict[key] = value  # type: ignore[index]

    def _path(self) -> Path:
        assert self.folder is not None
        return self.folder / "dict_array.cloudpickle"

    def persist(self) -> None:
        """Persist the dict storage to disk."""
        if self.folder is None:  # pragma: no cover
            return
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        dct = self._dict if isinstance(self._dict, dict) else dict(self._dict)
        dump(dct, path)

    def load(self) -> None:
        """Load the dict storage from disk."""
        if self.folder is None:  # pragma: no cover
            return
        path = self._path()
        if not path.exists():
            return
        self._dict = load(path)

    @property
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
        return False


def _masked_empty(shape: tuple[int, ...]) -> np.ndarray:
    # This is a workaround for the fact that setting `x[:] = np.ma.masked`
    # sets the elements to 0.0.
    x: np.ndarray = np.empty((1,), dtype=object)
    x[0] = np.ma.masked
    return np.tile(x, shape)


class SharedMemoryDictArray(DictArray):
    """Array interface to a shared memory dict store."""

    storage_id = "shared_memory_dict"
    requires_serialization = True

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        mapping: DictProxy[tuple[int, ...], Any] | None = None,
    ) -> None:
        """Initialize the SharedMemoryDictArray."""
        if mapping is None:
            manager = multiprocessing.Manager()
            mapping = manager.dict()
        super().__init__(
            folder=folder,
            shape=shape,
            internal_shape=internal_shape,
            shape_mask=shape_mask,
            mapping=mapping,
        )

    @property
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
        return True


register_storage(DictArray)
register_storage(SharedMemoryDictArray)
