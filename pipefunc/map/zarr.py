# zarr_array.py

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from numcodecs import JSON

from pipefunc._utils import prod
from pipefunc.map._mapspec import shape_to_strides

if TYPE_CHECKING:
    from collections.abc import Iterator


ARRAY_NAME = "data"


class ZarrArray:
    """Array interface to a Zarr store."""

    def __init__(
        self,
        store: zarr.Store,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        strides: tuple[int, ...] | None = None,
        object_codec: Any = None,
    ) -> None:
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        self.store = store
        self.shape = tuple(shape)
        self.strides = shape_to_strides(self.shape) if strides is None else tuple(strides)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

        # Use JSON object codec if not provided
        if object_codec is None:
            object_codec = JSON()

        self.array = zarr.open(
            self.store,
            mode="a",
            shape=self.shape,
            dtype=object,
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

    def _normalize_key(
        self,
        key: tuple[int | slice, ...],
        *,
        for_dump: bool = False,
    ) -> tuple[int | slice, ...]:
        if not isinstance(key, tuple):
            key = (key,)

        expected_rank = sum(self.shape_mask) if for_dump else len(self.shape_mask)

        if len(key) != expected_rank:
            msg = (
                f"Too many indices for array: array is {expected_rank}-dimensional, "
                f"but {len(key)} were indexed"
            )
            raise IndexError(msg)

        normalized_key: list[int | slice] = []
        shape_index = 0
        internal_shape_index = 0

        for axis, (mask, k) in enumerate(zip(self.shape_mask, key)):
            if mask:
                axis_size = self.shape[shape_index]
                shape_index += 1
            else:
                axis_size = self.internal_shape[internal_shape_index]
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

    def _key_to_index(self, key: tuple[int, ...]) -> int:
        """Return the linear index associated with the given key."""
        return sum(k * s for k, s in zip(key, self.strides))

    def get_from_index(self, index: int) -> Any:
        """Return the data associated with the given linear index."""
        np_index = np.unravel_index(index, self.shape)
        return self.array[np_index]

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        return index < self.size

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        slice_indices = []
        shape_index = 0
        internal_shape_index = 0
        normalized_key = self._normalize_key(key)
        for k, m in zip(normalized_key, self.shape_mask):
            shape = self.shape if m else self.internal_shape
            index = shape_index if m else internal_shape_index

            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(shape[index])))
            else:
                slice_indices.append(range(k, k + 1))

            if m:
                shape_index += 1
            else:
                internal_shape_index += 1

        return slice_indices

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        normalized_key = self._normalize_key(key)

        if any(isinstance(k, slice) for k in normalized_key):
            slice_indices = self._slice_indices(key)
            sliced_data = []
            sliced_mask = []

            for index in itertools.product(*slice_indices):
                file_key = tuple(i for i, m in zip(index, self.shape_mask) if m)
                linear_index = self._key_to_index(file_key)
                if self.has_index(linear_index):
                    sub_array = self.get_from_index(linear_index)
                    internal_index = tuple(i for i, m in zip(index, self.shape_mask) if not m)
                    if internal_index:
                        sub_array = np.asarray(sub_array)  # could be a list
                        sliced_sub_array = sub_array[internal_index]
                        sliced_data.append(sliced_sub_array)
                    else:
                        sliced_data.append(sub_array)
                    sliced_mask.append(False)
                else:
                    sliced_data.append(0)  # Placeholder for missing value
                    sliced_mask.append(True)

            sliced_array: np.ndarray = np.empty(len(sliced_data), dtype=object)
            sliced_array[:] = sliced_data
            mask: np.ndarray = np.array(sliced_mask, dtype=bool)
            sliced_array = np.ma.masked_array(sliced_array, mask=mask)

            new_shape = tuple(
                len(range_)
                for k, range_ in zip(normalized_key, slice_indices)
                if isinstance(k, slice)
            )
            return sliced_array.reshape(new_shape)

        external_indices = tuple(i for i, m in zip(normalized_key, self.shape_mask) if m)
        linear_index = self._key_to_index(external_indices)  # type: ignore[arg-type]

        if not self.has_index(linear_index):
            return np.ma.masked

        sub_array = self.get_from_index(linear_index)
        internal_indices = tuple(i for i, m in zip(normalized_key, self.shape_mask) if not m)
        if internal_indices:
            sub_array = np.asarray(sub_array)
            return sub_array[internal_indices]
        return sub_array

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing all the data.

        The returned numpy array has dtype "object" and a mask for
        masking out missing data.

        Parameters
        ----------
        splat_internal : bool
            If True, the internal array dimensions will be splatted out.
            If None, it will happen if and only if `internal_shape` is provided.

        Returns
        -------
        np.ma.core.MaskedArray
            The array containing all the data.

        """
        if splat_internal is None:
            splat_internal = bool(self.internal_shape)

        if not splat_internal:
            mask = not self.array.astype(bool)
            return np.ma.array(self.array, mask=mask)

        if not self.internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)

        full_shape = _select_by_mask(self.shape_mask, self.shape, self.internal_shape)
        arr = np.empty(full_shape, dtype=object)  # type: ignore[var-annotated]
        full_mask = np.zeros(full_shape, dtype=bool)  # type: ignore[var-annotated]

        for external_index in _iterate_shape_indices(self.shape):
            linear_index = self._key_to_index(external_index)
            if self.has_index(linear_index):
                sub_array = self.get_from_index(linear_index)
                sub_array = np.asarray(sub_array)  # could be a list
                shape_mask = self.shape_mask
                for internal_index in _iterate_shape_indices(self.internal_shape):
                    full_index = _select_by_mask(shape_mask, external_index, internal_index)
                    arr[full_index] = sub_array[internal_index]
                    full_mask[full_index] = False

        return np.ma.array(arr, mask=full_mask, dtype=object)

    def _mask_list(self) -> list[bool]:
        return [not self.has_index(i) for i in range(self.size)]

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing the mask.

        The returned numpy array has dtype "bool" and a mask for
        masking out missing data.
        """
        mask = self._mask_list()
        return np.ma.array(mask, mask=mask, dtype=bool).reshape(self.shape)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = ZarrArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        key = self._normalize_key(key, for_dump=True)
        if not any(isinstance(k, slice) for k in key):
            self.array[key] = value
            return

        for index in itertools.product(*self._slice_indices(key)):
            self.array[index] = value


def _iterate_shape_indices(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*map(range, shape))


def _select_by_mask(
    mask: tuple[bool, ...],
    tuple1: tuple[int, ...],
    tuple2: tuple[int, ...],
) -> tuple[int, ...]:
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
