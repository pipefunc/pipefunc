# This file is part of the pipefunc package.
# Originally, it is based on code from the `aiida-dynamic-workflows` package.
# Its license can be found in the LICENSE file in this folder.

from __future__ import annotations

import concurrent.futures
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cloudpickle
import numpy as np

from pipefunc._utils import dump, load, prod
from pipefunc.map._mapspec import shape_to_strides

if TYPE_CHECKING:
    from collections.abc import Iterator


def read(name: str | Path) -> bytes:
    """Load file contents as a bytestring."""
    with open(name, "rb") as f:  # noqa: PTH123
        return f.read()


FILENAME_TEMPLATE = "__{:d}__.pickle"


def _full_shape(
    shape: tuple[int, ...],
    internal_shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
) -> tuple[int, ...]:
    full_shape = []
    shape_index = 0
    internal_shape_index = 0
    for mask in shape_mask:
        if mask:
            full_shape.append(shape[shape_index])
            shape_index += 1
        else:
            full_shape.append(internal_shape[internal_shape_index])
            internal_shape_index += 1
    return tuple(full_shape)


class FileArray:
    """Array interface to a folder of files on disk.

    __getitem__ returns "np.ma.masked" for non-existent files.
    """

    def __init__(
        self,
        folder: str | Path,
        shape: Sequence[int],
        strides: Sequence[int] | None = None,
        filename_template: str = FILENAME_TEMPLATE,
        shape_mask: Sequence[bool] | None = None,
        internal_shape: Sequence[int] | None = None,
    ) -> None:
        if (shape_mask is None) ^ (internal_shape is None):
            msg = "internal_shape must be provided if shape_mask is provided"
            raise ValueError(msg)
        if shape_mask is not None and len(shape_mask) != len(shape) + len(internal_shape):
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        self.folder = Path(folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(shape)
        self.strides = shape_to_strides(self.shape) if strides is None else tuple(strides)
        self.filename_template = str(filename_template)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

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
        print(f"_normalize_key called with key: {key}, for_dump: {for_dump}")
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

        print(f"Normalized key result: {normalized_key}")
        return tuple(normalized_key)

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
        return (self._key_to_file(x) for x in itertools.product(*map(range, self.shape)))

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        print(f"_slice_indices called with key: {key}")
        slice_indices = []
        shape_index = 0
        internal_shape_index = 0
        normalized_key = self._normalize_key(key)  # Use the normalized key
        for k, m in zip(normalized_key, self.shape_mask):  # Use the normalized key
            if m:
                if isinstance(k, slice):
                    slice_indices.append(
                        range(*k.indices(self.shape[shape_index])),
                    )
                else:
                    slice_indices.append(range(k, k + 1))
                shape_index += 1
            else:
                if isinstance(k, slice):
                    slice_indices.append(
                        range(*k.indices(self.internal_shape[internal_shape_index])),
                    )
                else:
                    slice_indices.append(range(k, k + 1))
                internal_shape_index += 1
        print(f"Slice indices result: {slice_indices}")
        return slice_indices

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        print(f"__getitem__ called with key: {key}")
        normalized_key = self._normalize_key(key)
        print(f"Normalized key: {normalized_key}")
        external_indices = tuple(i for i, m in zip(normalized_key, self.shape_mask) if m)
        internal_indices = tuple(i for i, m in zip(normalized_key, self.shape_mask) if not m)
        print(f"External indices: {external_indices}")
        print(f"Internal indices: {internal_indices}")

        if any(isinstance(k, slice) for k in normalized_key):
            slice_indices = self._slice_indices(key)
            print(f"Slice indices: {slice_indices}")
            sliced_data = []
            sliced_mask = []

            for index in itertools.product(*slice_indices):
                file_key = tuple(i for i, m in zip(index, self.shape_mask) if m)
                file = self._key_to_file(file_key)
                print(f"File key: {file_key}")
                print(f"File path: {file}")
                if file.is_file():
                    sub_array = load(file)
                    print(f"Loaded sub-array: {sub_array}")
                    internal_index = tuple(i for i, m in zip(index, self.shape_mask) if not m)
                    if internal_index:
                        sliced_sub_array = sub_array[internal_index]
                        sliced_data.append(sliced_sub_array)
                    else:
                        sliced_data.append(sub_array)
                    sliced_mask.append(False)
                else:
                    print(f"File not found: {file}")
                    sliced_data.append(np.ma.masked)
                    sliced_mask.append(True)

            sliced_array: np.ndarray = np.array(sliced_data, dtype=object)
            mask: np.ndarray = np.array(sliced_mask, dtype=bool)
            sliced_array = np.ma.masked_array(sliced_array, mask=mask)

            new_shape = tuple(
                len(range_)
                for k, range_ in zip(normalized_key, slice_indices)
                if isinstance(k, slice)
            )
            print(f"New shape: {new_shape}")
            return sliced_array.reshape(new_shape)  # .squeeze()

        file = self._key_to_file(external_indices)
        if not file.is_file():
            return np.ma.masked

        sub_array = load(file)
        print(f"Loaded sub-array: {sub_array}")
        if internal_indices:
            result = sub_array[internal_indices]
            print(f"Result after internal indexing: {result}")
            return result
        print("Returning sub-array directly")
        return sub_array

    def to_array(self, *, splat_internal: bool = False) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing all the data.

        The returned numpy array has dtype "object" and a mask for
        masking out missing data.

        Parameters
        ----------
        splat_internal : bool
            If True, the internal array dimensions will be splatted out.

        Returns
        -------
        np.ma.core.MaskedArray
            The array containing all the data.

        """
        items = _load_all(map(self._index_to_file, range(self.size)))

        if not splat_internal:
            arr = np.empty(self.size, dtype=object)  # type: ignore[var-annotated]
            arr[:] = items
            mask = self._mask_list()
            return np.ma.array(arr, mask=mask, dtype=object).reshape(self.shape)

        if not self.internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)

        full_shape = _full_shape(self.shape, self.internal_shape, self.shape_mask)
        arr = np.empty(full_shape, dtype=object)  # type: ignore[var-annotated]
        full_mask = np.empty(full_shape, dtype=bool)  # type: ignore[var-annotated]

        for external_index in itertools.product(*map(range, self.shape)):
            file = self._key_to_file(external_index)

            if file.is_file():
                print(f"Loading file: {file}")
                sub_array = load(file)
                for internal_index in itertools.product(*map(range, self.internal_shape)):
                    full_index = []
                    external_idx = 0
                    internal_idx = 0
                    for m in self.shape_mask:
                        if m:
                            print(f"external_index[external_idx]: {external_index[external_idx]}")
                            full_index.append(external_index[external_idx])
                            external_idx += 1
                        else:
                            print(f"internal_index[internal_idx]: {internal_index[internal_idx]}")
                            full_index.append(internal_index[internal_idx])
                            internal_idx += 1
                    full_index = tuple(full_index)  # type: ignore[assignment]
                    arr[full_index] = sub_array[internal_index]
                    full_mask[full_index] = False
            else:
                print(f"File not found: {file}")
                for internal_index in itertools.product(*map(range, self.internal_shape)):
                    full_index = []
                    external_idx = 0
                    internal_idx = 0
                    for m in self.shape_mask:
                        if m:
                            print(f"external_index[external_idx]: {external_index[external_idx]}")
                            full_index.append(external_index[external_idx])
                            external_idx += 1
                        else:
                            print(f"internal_index[internal_idx]: {internal_index[internal_idx]}")
                            full_index.append(internal_index[internal_idx])
                            internal_idx += 1
                    full_index = tuple(full_index)  # type: ignore[assignment]
                    arr[full_index] = np.ma.masked
                    full_mask[full_index] = True
        return np.ma.array(arr, mask=full_mask, dtype=object)

    def _mask_list(self) -> list[bool]:
        return [not self._index_to_file(i).is_file() for i in range(self.size)]

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing the mask.

        The returned numpy array has dtype "bool" and a mask for
        masking out missing data.
        """
        mask = self._mask_list()
        return np.ma.array(mask, mask=mask, dtype=bool).reshape(self.shape)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the file associated with 'key'.

        Examples
        --------
        >>> arr = FileArray(...)
        >>> arr.dump((2, 1, 5), np.array([1, 2, 3]))

        """
        key = self._normalize_key(key, for_dump=True)
        if not any(isinstance(k, slice) for k in key):
            dump(value, self._key_to_file(key))  # type: ignore[arg-type]
            return

        for index in itertools.product(*self._slice_indices(key)):
            file = self._key_to_file(index)
            dump(value, file)


def _load_all(filenames: Iterator[Path]) -> list[Any]:
    def maybe_read(f: Path) -> Any | None:
        return read(f) if f.is_file() else None

    def maybe_load(x: str | None) -> Any | None:
        return cloudpickle.loads(x) if x is not None else None

    # Delegate file reading to the threadpool but deserialize sequentially,
    # as this is pure Python and CPU bound
    with concurrent.futures.ThreadPoolExecutor() as tex:
        return [maybe_load(x) for x in tex.map(maybe_read, filenames)]
