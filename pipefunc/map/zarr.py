"""Provides `zarr` integration for `pipefunc`."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from numcodecs import Pickle

from pipefunc._utils import prod
from pipefunc.map._filearray import (
    FileArrayBase,
    _iterate_shape_indices,
    _select_by_mask,
)
from pipefunc.map._mapspec import shape_to_strides

if TYPE_CHECKING:
    from pathlib import Path


ARRAY_NAME = "data"


class ZarrArray(FileArrayBase):
    """Array interface to a Zarr store."""

    def __init__(
        self,
        store: zarr.storage.Store | str | Path,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrArray."""
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        if not isinstance(store, zarr.storage.Store):
            store = zarr.DirectoryStore(str(store))
        self.store = store
        self.shape = tuple(shape)
        self.strides = shape_to_strides(self.shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()
        self.full_shape = _select_by_mask(self.shape_mask, self.shape, self.internal_shape)

        if object_codec is None:
            object_codec = Pickle()

        self.array = zarr.open(
            self.store,
            mode="a",
            path="/array",
            shape=self.full_shape,
            dtype=object,
            object_codec=object_codec,
            chunks=internal_shape or 1,
        )
        self._mask = zarr.open(
            self.store,
            mode="a",
            path="/mask",
            shape=self.full_shape,
            dtype=bool,
            fill_value=True,
            object_codec=object_codec,
            chunks=internal_shape or 1,
        )

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        return prod(self.shape)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        return len(self.shape)

    def get_from_index(self, index: int) -> Any:
        """Return the data associated with the given linear index."""
        np_index = np.unravel_index(index, self.full_shape)
        return self.array[np_index]

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        np_index = np.unravel_index(index, self.full_shape)
        return not self._mask[np_index]

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        """Return the data associated with the given key."""
        data = self.array[key]
        mask = self._mask[key]
        item: np.ma.MaskedArray = np.ma.masked_array(data, mask=mask, dtype=object)
        if item.shape == ():
            if item.mask:
                return np.ma.masked
            return item.item()
        return item

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Return the array as a NumPy masked array."""
        if splat_internal and not self.internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)
        if splat_internal is None:
            splat_internal = True
        if not splat_internal:
            msg = "splat_internal must be True"
            raise NotImplementedError(msg)
        return np.ma.array(self.array[:], mask=self._mask[:], dtype=object)

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return the mask associated with the array."""
        return np.ma.array(self._mask[:], dtype=bool)

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        return list(self.mask.flat)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = ZarrArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        if any(isinstance(k, slice) for k in key):
            for external_index in itertools.product(*self._slice_indices(key)):
                if self.internal_shape:
                    value = np.asarray(value)  # in case it's a list
                    for internal_index in _iterate_shape_indices(self.internal_shape):
                        full_index = _select_by_mask(
                            self.shape_mask,
                            external_index,
                            internal_index,
                        )  # type: ignore[arg-type]
                        sub_array = value[internal_index]
                        self.array[full_index] = sub_array
                        self._mask[full_index] = False
                else:
                    self.array[external_index] = value
                    self._mask[external_index] = False
            return

        if self.internal_shape:
            value = np.asarray(value)  # in case it's a list
            for internal_index in _iterate_shape_indices(self.internal_shape):
                full_index = _select_by_mask(self.shape_mask, key, internal_index)  # type: ignore[arg-type]
                sub_array = value[internal_index]
                self.array[full_index] = sub_array
                self._mask[full_index] = False
        else:
            self.array[key] = value
            self._mask[key] = False

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        slice_indices = []
        for size, k in zip(self.shape, key):
            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(size)))
            else:
                slice_indices.append(range(k, k + 1))
        return slice_indices
