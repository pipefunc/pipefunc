from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pipefunc._utils import dump, load
from pipefunc.map._storage_array._base import (
    StorageBase,
    iterate_shape_indices,
    normalize_key,
    register_storage,
    select_by_mask,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pipefunc.map._types import ShapeTuple

FILENAME_TEMPLATE = "__{:d}__.pickle"


class ChunkedFileArray(StorageBase):
    """Array interface to a folder of files on disk, supporting chunking.

    Conceptual array elements are grouped into chunks, each stored in a separate file.
    """

    storage_id = "chunked_file_array"
    requires_serialization = True

    def __init__(
        self,
        folder: str | Path,
        shape: ShapeTuple,  # External shape
        internal_shape: ShapeTuple | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        filename_template: str = FILENAME_TEMPLATE,
        chunk_size: int = 10,
    ) -> None:
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if (
            internal_shape is not None
            and shape_mask is not None
            and len(shape_mask) != len(shape) + len(internal_shape)
        ):
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        if chunk_size <= 0:
            msg = "chunk_size must be a positive integer."
            raise ValueError(msg)

        self.folder = Path(folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.filename_template = str(filename_template)
        self.chunk_size = chunk_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(folder='{self.folder}', "
            f"shape={self.shape}, "
            f"internal_shape={self.internal_shape}, "
            f"shape_mask={self.shape_mask}, "
            f"filename_template={self.filename_template!r}, "
            f"chunk_size={self.chunk_size})"
        )

    def _normalize_key(
        self,
        key: tuple[int | slice, ...],
        *,
        for_dump: bool = False,
    ) -> tuple[int | slice, ...]:
        return normalize_key(
            key,
            self.resolved_shape,
            self.resolved_internal_shape,
            self.shape_mask,
            for_dump=for_dump,
        )

    def _get_chunk_path(self, linear_chunk_index: int) -> Path:
        """Returns the file path for a given linear chunk index."""
        return self.folder / self.filename_template.format(linear_chunk_index)

    def _get_chunk_len(self, linear_chunk_index: int) -> int:
        """Returns the number of conceptual elements in a specific chunk."""
        num_total_elements = self.size
        if num_total_elements == 0:
            return 0
        num_chunks = math.ceil(num_total_elements / self.chunk_size)
        if linear_chunk_index < 0 or linear_chunk_index >= num_chunks:
            msg = "linear_chunk_index out of bounds for _get_chunk_len"
            raise IndexError(msg)
        if linear_chunk_index < num_chunks - 1:
            return self.chunk_size
        # Last chunk
        last_chunk_len = num_total_elements % self.chunk_size
        return last_chunk_len if last_chunk_len != 0 else self.chunk_size

    def _get_element_chunk_location(self, linear_external_index: int) -> tuple[int, int]:
        """Returns (chunk_file_index, index_within_chunk) for a linear external index."""
        if not (0 <= linear_external_index < self.size):
            msg = "linear_external_index out of bounds for _get_element_chunk_location"
            raise IndexError(msg)
        if self.chunk_size == 0:  # Should not happen with __init__ check, but for safety
            msg = "chunk_size cannot be zero."
            raise ValueError(msg)
        chunk_file_idx = linear_external_index // self.chunk_size
        idx_in_chunk = linear_external_index % self.chunk_size
        return chunk_file_idx, idx_in_chunk

    def _load_chunk_data(self, chunk_file_idx: int) -> list[Any] | None:
        """Loads a chunk file. Returns None if file doesn't exist or is unreadable."""
        chunk_path = self._get_chunk_path(chunk_file_idx)
        if not chunk_path.is_file():
            return None
        try:
            return load(chunk_path)
        except Exception:  # Catches EOFError, pickle errors etc.  # noqa: BLE001
            chunk_path.unlink(missing_ok=True)  # Remove the file if unreadable
            return None

    def _save_chunk_data(self, chunk_file_idx: int, chunk_data: list[Any]) -> None:
        """Saves a list of conceptual elements to a chunk file."""
        chunk_path = self._get_chunk_path(chunk_file_idx)
        dump(chunk_data, chunk_path)

    def _get_conceptual_element(self, linear_external_index: int) -> Any:
        """Retrieves a single conceptual element, returns np.ma.masked if not found."""
        chunk_file_idx, idx_in_chunk = self._get_element_chunk_location(linear_external_index)
        chunk_data = self._load_chunk_data(chunk_file_idx)

        if chunk_data is None or idx_in_chunk >= len(chunk_data):
            return np.ma.masked

        return chunk_data[idx_in_chunk]

    def get_from_index(self, linear_external_index: int) -> Any:
        element = self._get_conceptual_element(linear_external_index)
        # Ensure that if the stored value was None (and not np.ma.masked), it's returned as None
        return element if element is not np.ma.masked else np.ma.masked

    def has_index(self, linear_external_index: int) -> bool:
        if not (0 <= linear_external_index < self.size):
            return False
        return self._get_conceptual_element(linear_external_index) is not np.ma.masked

    def _files(self) -> Iterator[Path]:
        if self.size == 0:
            return
        num_chunks = math.ceil(self.size / self.chunk_size)
        for i in range(num_chunks):
            yield self._get_chunk_path(i)

    def _slice_indices(
        self,
        key: tuple[int | slice, ...],
        base_shape: tuple[int, ...],
    ) -> list[range]:
        if len(key) != len(base_shape):
            msg = f"Key length {len(key)} != base_shape length {len(base_shape)}"
            raise ValueError(msg)

        slice_indices = []
        for size, k_part in zip(base_shape, key):
            if isinstance(k_part, slice):
                slice_indices.append(range(*k_part.indices(size)))
            else:
                slice_indices.append(range(k_part, k_part + 1))
        return slice_indices

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        normalized_key = self._normalize_key(key, for_dump=False)

        if not any(isinstance(k, slice) for k in normalized_key):  # Single element access
            external_coords = tuple(k for k, m in zip(normalized_key, self.shape_mask) if m)
            internal_coords = tuple(k for k, m in zip(normalized_key, self.shape_mask) if not m)
            linear_external_idx: int = np.ravel_multi_index(external_coords, self.resolved_shape)

            element = self._get_conceptual_element(linear_external_idx)

            if element is np.ma.masked:
                return np.ma.masked
            return np.asarray(element)[internal_coords] if internal_coords else element

        # Slice access
        slice_indices = self._slice_indices(normalized_key, self.full_shape)
        new_shape = tuple(
            len(range_) for k, range_ in zip(normalized_key, slice_indices) if isinstance(k, slice)
        )

        sliced_data_flat = []
        for full_coord in itertools.product(*slice_indices):
            external_coords = tuple(full_coord[i] for i, m in enumerate(self.shape_mask) if m)
            internal_coords = tuple(full_coord[i] for i, m in enumerate(self.shape_mask) if not m)
            linear_external_idx = np.ravel_multi_index(external_coords, self.resolved_shape)

            element = self._get_conceptual_element(linear_external_idx)

            if element is np.ma.masked:
                sliced_data_flat.append(np.ma.masked)
            else:
                final_val = np.asarray(element)[internal_coords] if internal_coords else element
                sliced_data_flat.append(final_val)

        mask_flat = [x is np.ma.masked for x in sliced_data_flat]
        data_array: np.ndarray = np.empty(len(sliced_data_flat), dtype=object)
        data_array[:] = sliced_data_flat

        reshaped_array = np.ma.masked_array(data_array, mask=mask_flat)
        reshaped_array = reshaped_array.reshape(new_shape)

        return (
            reshaped_array.item()
            if reshaped_array.shape == () and not reshaped_array.mask
            else reshaped_array
        )

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        if splat_internal is None:
            splat_internal = bool(self.resolved_internal_shape)

        if not splat_internal:
            items = [self.get_from_index(i) for i in range(self.size)]
            if not items and self.size == 0:  # Handle empty array
                return np.ma.MaskedArray(np.empty(self.resolved_shape, dtype=object), mask=False)
            data = np.empty(self.size, dtype=object)
            data[:] = items
            mask = [item is np.ma.masked for item in items]
            return np.ma.MaskedArray(data, mask=mask, dtype=object).reshape(self.resolved_shape)

        if not self.resolved_internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)

        arr_data = np.empty(self.full_shape, dtype=object)
        arr_mask = np.ones(self.full_shape, dtype=bool)

        for linear_external_idx in range(self.size):
            conceptual_element = self._get_conceptual_element(linear_external_idx)
            external_coords = np.unravel_index(linear_external_idx, self.resolved_shape)

            if conceptual_element is not np.ma.masked:
                element_array = np.asarray(conceptual_element)
                for internal_coords in iterate_shape_indices(self.resolved_internal_shape):
                    full_coord = select_by_mask(self.shape_mask, external_coords, internal_coords)
                    arr_data[full_coord] = element_array[internal_coords]
                    arr_mask[full_coord] = False
            else:
                for internal_coords in iterate_shape_indices(self.resolved_internal_shape):
                    full_coord = select_by_mask(self.shape_mask, external_coords, internal_coords)
                    arr_data[full_coord] = np.ma.masked
                    # arr_mask is already True

        return np.ma.MaskedArray(arr_data, mask=arr_mask, dtype=object)

    def mask_linear(self) -> list[bool]:
        return [self._get_conceptual_element(i) is np.ma.masked for i in range(self.size)]

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        if self.size == 0:
            return np.ma.MaskedArray(np.empty(self.shape, dtype=bool), mask=False)
        mask_data_linear = self.mask_linear()
        mask_array = np.array(mask_data_linear, dtype=bool).reshape(self.shape)
        return np.ma.MaskedArray(mask_array, mask=mask_array, dtype=bool)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        normalized_key_for_dump = self._normalize_key(key, for_dump=True)
        target_external_coords = list(
            itertools.product(*self._slice_indices(normalized_key_for_dump, self.resolved_shape)),
        )

        chunk_updates: defaultdict[int, dict[int, Any]] = defaultdict(dict)
        for external_coords_tuple in target_external_coords:
            linear_external_idx = np.ravel_multi_index(external_coords_tuple, self.resolved_shape)
            chunk_file_idx, idx_in_chunk = self._get_element_chunk_location(linear_external_idx)
            chunk_updates[chunk_file_idx][idx_in_chunk] = value

        for chunk_file_idx, updates_in_chunk in chunk_updates.items():
            current_chunk_len = self._get_chunk_len(chunk_file_idx)
            chunk_data = self._load_chunk_data(chunk_file_idx)
            if chunk_data is None:
                chunk_data = [np.ma.masked] * current_chunk_len

            if len(chunk_data) < current_chunk_len:  # Ensure list is long enough
                chunk_data.extend([np.ma.masked] * (current_chunk_len - len(chunk_data)))

            for idx_in_chunk, val_to_set in updates_in_chunk.items():
                if self.internal_shape:
                    val_to_set_arr = np.asarray(val_to_set)
                    if val_to_set_arr.shape != self.resolved_internal_shape:
                        msg = (
                            f"Value shape {val_to_set_arr.shape} != "
                            f"internal_shape {self.resolved_internal_shape}"
                        )
                        raise ValueError(msg)
                chunk_data[idx_in_chunk] = val_to_set
            self._save_chunk_data(chunk_file_idx, chunk_data)

    @property
    def dump_in_subprocess(self) -> bool:
        return True

    @classmethod
    def from_data(
        cls,
        data: list[Any] | np.ndarray,
        folder: str | Path,
        chunk_size: int = 1,
    ) -> ChunkedFileArray:
        shape = np.shape(data)
        instance = cls(folder, shape, chunk_size=chunk_size)

        data_for_chunks: defaultdict[int, list[tuple[int, Any]]] = defaultdict(list)
        for external_coords, value_element in np.ndenumerate(data):
            linear_idx = np.ravel_multi_index(external_coords, shape)
            chunk_idx, idx_in_chunk = instance._get_element_chunk_location(linear_idx)
            data_for_chunks[chunk_idx].append((idx_in_chunk, value_element))

        num_chunks = math.ceil(instance.size / instance.chunk_size)
        for chunk_idx in range(num_chunks):
            chunk_len = instance._get_chunk_len(chunk_idx)
            current_chunk_data = [None] * chunk_len
            for idx_in_chunk, value_for_item in data_for_chunks[chunk_idx]:
                current_chunk_data[idx_in_chunk] = value_for_item
            instance._save_chunk_data(chunk_idx, current_chunk_data)

        return instance


register_storage(ChunkedFileArray, "chunked_file_array")
