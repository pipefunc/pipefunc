"""Sparse array implementation for irregular/jagged arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator


class SparseIrregularArray:
    """Sparse array-like object for irregular arrays.

    Instead of allocating a full dense array with masked values,
    this class provides an array-like interface while storing only
    the actual data values.

    This is especially efficient when array sizes vary dramatically
    (e.g., most arrays have size 1 but one has size 1 billion).
    """

    def __init__(
        self,
        data_dict: dict[tuple[int, ...], Any],
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
    ) -> None:
        """Initialize sparse array.

        Parameters
        ----------
        data_dict
            Dictionary mapping indices to actual data values.
            For irregular arrays, values are lists/arrays of varying length.
        shape
            External shape of the array.
        internal_shape
            Maximum internal shape for irregular dimensions.
        shape_mask
            Mask indicating which dimensions are external (True) vs internal (False).

        """
        self._data = data_dict
        self.shape = shape
        self.internal_shape = internal_shape
        self.shape_mask = shape_mask or (True,) * len(shape)

        # Compute full shape if we have internal dimensions
        if internal_shape:
            self.full_shape = (
                tuple(
                    s
                    for s, m in zip(
                        list(shape) + list(internal_shape),
                        list(shape_mask or []) + [False] * len(internal_shape),
                    )
                    if m
                )
                + internal_shape
            )
        else:
            self.full_shape = shape

        # Precompute actual lengths for each irregular array
        self._lengths: dict[tuple[int, ...], int] = {}
        if internal_shape:
            for key, value in data_dict.items():
                self._lengths[key] = len(value) if hasattr(value, "__len__") else 1

    @property
    def dtype(self) -> npt.DTypeLike:
        """Return dtype (always object for sparse irregular arrays)."""
        return np.dtype("O")

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.full_shape)

    @property
    def size(self) -> int:
        """Return total number of elements (including masked)."""
        return int(np.prod(self.full_shape))

    @property
    def nbytes(self) -> int:
        """Return approximate memory usage (only actual data, not masked)."""
        total = 0
        for value in self._data.values():
            if hasattr(value, "nbytes"):
                total += value.nbytes
            elif hasattr(value, "__len__"):
                # Rough estimate for lists
                total += len(value) * 8  # 8 bytes per reference
        return total

    def __getitem__(self, key: int | slice | tuple) -> Any:
        """Get item or slice from the sparse array."""
        # Normalize the key to a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Handle full slices - return a view
        if all(isinstance(k, slice) for k in key):
            return self._get_slice(key)

        # Handle mixed indices and slices
        if any(isinstance(k, slice) for k in key):
            return self._get_mixed(key)

        # All indices - return single element
        return self._get_single(key)

    def _get_single(self, key: tuple[int, ...]) -> Any:
        """Get a single element."""
        if len(key) != len(self.full_shape):
            msg = f"Expected {len(self.full_shape)} indices, got {len(key)}"
            raise IndexError(msg)

        # Split into external and internal indices
        if self.internal_shape:
            external_key = key[: len(self.shape)]
            internal_idx = key[len(self.shape) :]

            if external_key in self._data:
                data = self._data[external_key]
                # Check if internal index is within bounds
                if internal_idx[0] < len(data):
                    return data[internal_idx[0]]

        elif key in self._data:
            return self._data[key]

        # Return masked value for missing data
        return np.ma.masked

    def _get_slice(self, key: tuple[slice, ...]) -> SparseIrregularArray:
        """Get a slice, returning a new sparse array."""
        # For now, convert slices to dense for compatibility
        # This could be optimized to return another sparse array
        return self.to_dense_masked()[key]

    def _get_mixed(self, key: tuple) -> Any:
        """Handle mixed indices and slices."""
        # For mixed access, convert to dense
        # This could be optimized for specific patterns
        return self.to_dense_masked()[key]

    def __iter__(self) -> Iterator:
        """Iterate over the first dimension."""
        if len(self.full_shape) == 1:
            # 1D array
            for i in range(self.full_shape[0]):
                yield self[i]
        else:
            # Multi-dimensional - return slices
            for i in range(self.full_shape[0]):
                yield self._get_row(i)

    def _get_row(self, row_idx: int) -> np.ma.MaskedArray:
        """Get a row as a masked array."""
        if self.internal_shape:
            # Create arrays for data and mask
            row_data = np.empty(self.internal_shape[0], dtype=object)
            row_mask = np.ones(self.internal_shape[0], dtype=bool)

            # Fill with masked values by default
            row_data.fill(np.ma.masked)

            # Return the irregular array for this row
            if (row_idx,) in self._data:
                data = self._data[(row_idx,)]
                for i, val in enumerate(data):
                    if i < self.internal_shape[0]:
                        row_data[i] = val
                        row_mask[i] = False

            return np.ma.MaskedArray(row_data, mask=row_mask)
        return self.to_dense_masked()[row_idx]

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        """Convert to NumPy array (triggers densification)."""
        return np.asarray(self.to_dense_masked(), dtype=dtype)

    def to_dense_masked(self) -> np.ma.MaskedArray:
        """Convert to a dense masked array.

        Warning: This allocates the full array! Only use when necessary.
        """
        data = np.empty(self.full_shape, dtype=object)
        mask = np.ones(self.full_shape, dtype=bool)

        # Fill with np.ma.masked
        data.fill(np.ma.masked)

        if self.internal_shape:
            # Irregular array case
            for external_key, values in self._data.items():
                values_array = np.asarray(values)
                for i, val in enumerate(values_array):
                    if i < self.internal_shape[0]:
                        full_key = (*external_key, i)
                        data[full_key] = val
                        mask[full_key] = False
        else:
            # Regular array case
            for key, value in self._data.items():
                data[key] = value
                mask[key] = False

        return np.ma.MaskedArray(data, mask=mask)

    def compressed(self) -> np.ndarray:
        """Return only non-masked values as a 1D array."""
        values = []
        if self.internal_shape:
            for vals in self._data.values():
                values.extend(vals)
        else:
            values = list(self._data.values())
        return np.array(values, dtype=object)

    def count(self) -> int:
        """Count non-masked elements."""
        if self.internal_shape:
            return sum(self._lengths.values())
        return len(self._data)

    @property
    def mask(self) -> SparseMask:
        """Return a sparse mask object."""
        return SparseMask(self)

    def __repr__(self) -> str:
        """String representation."""
        n_stored = len(self._data)
        n_total = self.size
        density = n_stored / n_total if n_total > 0 else 0

        return (
            f"SparseIrregularArray(shape={self.full_shape}, "
            f"stored={n_stored}/{n_total} ({density:.1%}), "
            f"memory={self.nbytes / 1024:.1f}KB)"
        )


class SparseMask:
    """Sparse mask for SparseIrregularArray."""

    def __init__(self, parent: SparseIrregularArray) -> None:
        """Initialize with parent array."""
        self.parent = parent

    def __getitem__(self, key: Any) -> Any:
        """Check if element(s) are masked."""
        # Single element
        if not any(isinstance(k, slice) for k in (key if isinstance(key, tuple) else (key,))):
            return self.parent[key] is np.ma.masked

        # Slice - need to return mask array
        # For now, use dense conversion
        mask = self.parent.to_dense_masked().mask
        return mask[key]  # type: ignore[index, no-any-return]

    @property
    def shape(self) -> tuple[int, ...]:
        """Mask shape."""
        return self.parent.full_shape
