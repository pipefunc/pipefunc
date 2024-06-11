# This file is part of the pipefunc package.
# Originally, it is based on code from the `aiida-dynamic-workflows` package.
# Its license can be found in the LICENSE file in this folder.
# See `git diff 98a1736 pipefunc/map/_filearray.py` for the changes made.

from __future__ import annotations

import concurrent.futures
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

from pipefunc._utils import dump, load
from pipefunc.map._storage_base import (
    StorageBase,
    _iterate_shape_indices,
    _normalize_key,
    _select_by_mask,
    register_storage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

storage_registry: dict[str, type[StorageBase]] = {}


def read(name: str | Path) -> bytes:
    """Load file contents as a bytestring."""
    with open(name, "rb") as f:  # noqa: PTH123
        return f.read()


FILENAME_TEMPLATE = "__{:d}__.pickle"


class FileArray(StorageBase):
    """Array interface to a folder of files on disk.

    __getitem__ returns "np.ma.masked" for non-existent files.
    """

    storage_id = "file_array"

    def __init__(
        self,
        folder: str | Path,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        filename_template: str = FILENAME_TEMPLATE,
    ) -> None:
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        self.folder = Path(folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(shape)
        self.filename_template = str(filename_template)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

    def _normalize_key(
        self,
        key: tuple[int | slice, ...],
        *,
        for_dump: bool = False,
    ) -> tuple[int | slice, ...]:
        return _normalize_key(
            key,
            self.shape,
            self.internal_shape,
            self.shape_mask,
            for_dump=for_dump,
        )

    def _index_to_file(self, index: int) -> Path:
        """Return the filename associated with the given index."""
        return self.folder / self.filename_template.format(index)

    def _key_to_file(self, key: tuple[int, ...]) -> Path:
        """Return the filename associated with the given key."""
        index = sum(k * s for k, s in zip(key, self.strides))
        return self._index_to_file(index)

    def get_from_index(self, index: int) -> Any:
        """Return the data associated with the given linear index."""
        return load(self._index_to_file(index))

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        return self._index_to_file(index).is_file()

    def _files(self) -> Iterator[Path]:
        """Yield all the filenames that constitute the data in this array."""
        return (self._key_to_file(x) for x in _iterate_shape_indices(self.shape))

    def _slice_indices(
        self,
        key: tuple[int | slice, ...],
        *,
        for_dump: bool = False,
    ) -> list[range]:
        slice_indices = []
        shape_index = 0
        internal_shape_index = 0
        normalized_key = self._normalize_key(key, for_dump=for_dump)
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
                file = self._key_to_file(file_key)
                if file.is_file():
                    sub_array = load(file)
                    internal_index = tuple(i for i, m in zip(index, self.shape_mask) if not m)
                    if internal_index:
                        sub_array = np.asarray(sub_array)  # could be a list
                        sliced_sub_array = sub_array[internal_index]
                        sliced_data.append(sliced_sub_array)
                    else:
                        sliced_data.append(sub_array)
                    sliced_mask.append(False)
                else:
                    sliced_data.append(np.ma.masked)
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
        internal_indices = tuple(i for i, m in zip(normalized_key, self.shape_mask) if not m)

        file = self._key_to_file(external_indices)  # type: ignore[arg-type]
        if not file.is_file():
            return np.ma.masked

        sub_array = load(file)
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
        splat_internal
            If True, the internal array dimensions will be splatted out.
            If None, it will happen if and only if `internal_shape` is provided.

        Returns
        -------
            The array containing all the data.

        """
        if splat_internal is None:
            splat_internal = bool(self.internal_shape)

        items = _load_all(map(self._index_to_file, range(self.size)))

        if not splat_internal:
            arr = np.empty(self.size, dtype=object)  # type: ignore[var-annotated]
            arr[:] = items
            mask = self.mask_linear()
            return np.ma.MaskedArray(arr, mask=mask, dtype=object).reshape(self.shape)

        if not self.internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)

        arr = np.empty(self.full_shape, dtype=object)  # type: ignore[var-annotated]
        full_mask = np.empty(self.full_shape, dtype=bool)  # type: ignore[var-annotated]

        for external_index in _iterate_shape_indices(self.shape):
            file = self._key_to_file(external_index)

            if file.is_file():
                sub_array = load(file)
                sub_array = np.asarray(sub_array)  # could be a list
                for internal_index in _iterate_shape_indices(self.internal_shape):
                    full_index = _select_by_mask(self.shape_mask, external_index, internal_index)
                    arr[full_index] = sub_array[internal_index]
                    full_mask[full_index] = False
            else:
                for internal_index in _iterate_shape_indices(self.internal_shape):
                    full_index = _select_by_mask(self.shape_mask, external_index, internal_index)
                    arr[full_index] = np.ma.masked
                    full_mask[full_index] = True
        return np.ma.MaskedArray(arr, mask=full_mask, dtype=object)

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        return [not self._index_to_file(i).is_file() for i in range(self.size)]

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing the mask.

        The returned numpy array has dtype "bool" and a mask for
        masking out missing data.
        """
        mask = self.mask_linear()
        return np.ma.MaskedArray(mask, mask=mask, dtype=bool).reshape(self.shape)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the file associated with 'key'.

        Examples
        --------
        >>> arr = FileArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        key = self._normalize_key(key, for_dump=True)
        if not any(isinstance(k, slice) for k in key):
            dump(value, self._key_to_file(key))  # type: ignore[arg-type]
            return

        for index in itertools.product(*self._slice_indices(key, for_dump=True)):
            file = self._key_to_file(index)
            dump(value, file)

    @property
    def parallelizable(self) -> bool:
        """Return whether the storage is parallelizable."""
        return True


def _load_all(filenames: Iterator[Path]) -> list[Any]:
    def maybe_read(f: Path) -> Any | None:
        return read(f) if f.is_file() else None

    def maybe_load(x: str | None) -> Any | None:
        return cloudpickle.loads(x) if x is not None else None

    # Delegate file reading to the threadpool but deserialize sequentially,
    # as this is pure Python and CPU bound
    with concurrent.futures.ThreadPoolExecutor() as tex:
        return [maybe_load(x) for x in tex.map(maybe_read, filenames)]


register_storage(FileArray)
