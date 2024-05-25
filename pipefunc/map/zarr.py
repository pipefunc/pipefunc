"""Provides `zarr` integration for `pipefunc`."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import zarr
from numcodecs import Pickle

from pipefunc._utils import prod
from pipefunc.map._filearray import (
    _FileArrayBase,
    _iterate_shape_indices,
    _select_by_mask,
)
from pipefunc.map._mapspec import shape_to_strides
from pipefunc.map._run import (
    _internal_shape,
)

ARRAY_NAME = "data"


class ZarrArray(_FileArrayBase):
    """Array interface to a Zarr store."""

    def __init__(
        self,
        store: zarr.Store,
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
        )
        self._mask = zarr.open(
            self.store,
            mode="a",
            path="/mask",
            shape=self.full_shape,
            dtype=bool,
            fill_value=True,
            object_codec=object_codec,
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
        np_index = np.unravel_index(index, self.shape)
        return self.array[np_index]

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        return index < self.size

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        """Return the data associated with the given key."""
        data = self.array[key]
        mask = self._mask[key]
        return np.ma.masked_array(data, mask=mask, dtype=object)

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Return the array as a NumPy masked array."""
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

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = ZarrArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        shape = self.full_shape
        shape_mask = self.shape_mask
        internal_shape = _internal_shape(shape, shape_mask)
        print(f"{key=}, {self.full_shape=}")
        if any(isinstance(k, slice) for k in key):
            external_indices = itertools.product(*self._slice_indices(key))
        else:
            external_indices = [key]

        for external_index in external_indices:
            for internal_index in _iterate_shape_indices(internal_shape):
                full_index = _select_by_mask(shape_mask, external_index, internal_index)  # type: ignore[arg-type]
                print(f"full_index: {full_index}, internal_index: {internal_index}")
                sub_array = value[internal_index] if self.internal_shape else value
                self.array[full_index] = sub_array
                self._mask[full_index] = False

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        slice_indices = []
        shape_index = 0
        print(f"{key=}")
        for k in key:
            shape = self.shape
            index = shape_index
            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(shape[index])))
            else:
                slice_indices.append(range(k, k + 1))
        print(f"{slice_indices=}")
        return slice_indices
