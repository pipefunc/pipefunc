from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._base import StorageBase, register_storage

if TYPE_CHECKING:
    from pathlib import Path


class IrregularArray(StorageBase):
    """Dictionary-based storage for irregular arrays.

    Stores values with tuple indices to handle arrays with different shapes.
    """

    storage_id = "irregular_dict"
    requires_serialization = False
    dump_in_subprocess = False

    def __init__(
        self,
        folder: Path | None,
        shape: tuple[int | str, ...],
        internal_shape: tuple[int | str, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
    ) -> None:
        self.folder = folder
        self.shape = shape
        self.internal_shape = internal_shape or ()
        self.shape_mask = shape_mask or (True,) * len(shape)
        self._data: dict[tuple[int, ...], Any] = {}
        self._mask: dict[tuple[int, ...], bool] = {}

    def get_from_index(self, index: tuple[int, ...]) -> Any:
        """Return the data associated with the given index tuple."""
        if index in self._data:
            return self._data[index]
        return np.ma.masked

    def has_index(self, index: tuple[int, ...]) -> bool:
        """Return whether the given index tuple exists."""
        return index in self._data

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        """Get item by key, handling both tuple indices and slices."""
        if any(isinstance(k, slice) for k in key):
            raise NotImplementedError
        return self.get_from_index(key)

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Convert to masked array, with np.ma.masked for missing values."""
        raise NotImplementedError

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return a mask indicating which elements are missing."""
        raise NotImplementedError

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        raise NotImplementedError

    def dump(self, key: tuple[int, ...], value: Any) -> None:
        """Store value at the given index tuple."""
        self._data[key] = value
        self._mask[key] = False


register_storage(IrregularArray)
