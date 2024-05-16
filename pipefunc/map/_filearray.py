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
    ) -> None:
        self.folder = Path(folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(shape)
        self.strides = shape_to_strides(self.shape) if strides is None else tuple(strides)
        self.filename_template = str(filename_template)

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        return prod(self.shape)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        return len(self.shape)

    def _normalize_key(self, key: tuple[int | slice, ...]) -> tuple[int | slice, ...]:
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) != self.rank:
            msg = (
                f"Too many indices for array: array is {self.rank}-dimensional, "
                f"but {len(key)} were indexed"
            )
            raise IndexError(msg)

        normalized_key: list[int | slice] = []
        for axis, (axis_size, k) in enumerate(zip(self.shape, key)):
            if isinstance(k, slice):
                normalized_key.append(k)
            else:
                normalized_k = k if k >= 0 else (k + axis_size)
                if not (0 <= normalized_k < axis_size):
                    msg = f"Index {k} is out of bounds for axis {axis} with size {axis_size}"
                    raise IndexError(msg)
                normalized_key.append(normalized_k)

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
        return [
            range(*k.indices(self.shape[i])) if isinstance(k, slice) else range(k, k + 1)
            for i, k in enumerate(key)
        ]

    def __getitem__(self, key: tuple[int | slice, ...]) -> np.ma.core.MaskedArray:
        key = self._normalize_key(key)

        if any(isinstance(k, slice) for k in key):
            slice_indices = self._slice_indices(key)
            sliced_data = []

            for index in itertools.product(*slice_indices):
                file = self._key_to_file(index)
                if file.is_file():
                    sliced_data.append(load(file))
                else:
                    sliced_data.append(np.ma.masked)

            sliced_array = np.ma.array(sliced_data, dtype=object)

            # Determine the new shape based on the sliced dimensions
            new_shape = tuple(
                len(range_) if isinstance(k, slice) else None
                for k, range_ in zip(key, slice_indices)
            )
            new_shape = tuple(filter(None, new_shape))

            return sliced_array.reshape(new_shape)

        file = self._key_to_file(key)  # type: ignore[arg-type]
        if not file.is_file():
            return np.ma.masked
        return load(file)

    def to_array(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing all the data.

        The returned numpy array has dtype "object" and a mask for
        masking out missing data.
        """
        items = _load_all(map(self._index_to_file, range(self.size)))
        mask = self._mask_list()
        arr = np.empty(self.size, dtype=object)  # type: ignore[var-annotated]
        arr[:] = items
        return np.ma.array(arr, mask=mask, dtype=object).reshape(self.shape)

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
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        key = self._normalize_key(key)
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
