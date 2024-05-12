from __future__ import annotations

import concurrent.futures
import functools
import itertools
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cloudpickle
import numpy as np

from pipefunc._mapspec import _shape_to_strides

if TYPE_CHECKING:
    from collections.abc import Iterator


def read(name: str | Path) -> bytes:
    """Load file contents as a bytestring."""
    with open(name, "rb") as f:  # noqa: PTH123
        return f.read()


def load(name: str | Path) -> Any:
    """Load a cloudpickled object from the named file."""
    with open(name, "rb") as f:  # noqa: PTH123
        return cloudpickle.load(f)


def dump(obj: Any, name: str | Path) -> None:
    """Dump an object to the named file using cloudpickle."""
    with open(name, "wb") as f:  # noqa: PTH123
        cloudpickle.dump(obj, f)


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
        self.strides = (
            _shape_to_strides(self.shape) if strides is None else tuple(strides)
        )
        self.filename_template = str(filename_template)

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        return len(self.shape)

    def _normalize_key(self, key: tuple[int, ...]) -> tuple[int, ...]:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.rank:
            msg = (
                f"Too many indices for array: array is {self.rank}-dimensional, "
                f"but {len(key)} were indexed"
            )
            raise IndexError(msg)

        if any(isinstance(k, slice) for k in key):
            msg = "Cannot yet slice subarrays"
            raise NotImplementedError(msg)

        normalized_key = []
        for axis, k in enumerate(key):
            axis_size = self.shape[axis]
            normalized_k = k if k >= 0 else (axis_size - k)
            if not (0 <= normalized_k < axis_size):
                msg = (
                    f"Index {k} is out of bounds for axis {axis} with size {axis_size}"
                )
                raise IndexError(msg)
            normalized_key.append(k)

        return tuple(normalized_key)

    def _index_to_file(self, index: int) -> Path:
        """Return the filename associated with the given index."""
        return self.folder / self.filename_template.format(index)

    def _key_to_file(self, key: tuple[int, ...]) -> Path:
        """Return the filename associated with the given key."""
        index = sum(k * s for k, s in zip(key, self.strides))
        return self._index_to_file(index)

    def _files(self) -> Iterator[Path]:
        """Yield all the filenames that constitute the data in this array."""
        return (
            self._key_to_file(x) for x in itertools.product(*map(range, self.shape))
        )

    def __getitem__(self, key: tuple[int, ...]) -> np.ma.core.MaskedArray:
        key = self._normalize_key(key)
        if _key_has_slice(key):
            # XXX: need to figure out strides in order to implement this.  # noqa: FIX003, TD001
            msg = "Cannot yet slice subarrays"
            raise NotImplementedError(msg)

        f = self._key_to_file(key)
        if not f.is_file():
            return np.ma.core.masked
        return load(f)

    def to_array(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing all the data.

        The returned numpy array has dtype "object" and a mask for
        masking out missing data.
        """
        items = _load_all(map(self._index_to_file, range(self.size)))
        mask = [not self._index_to_file(i).is_file() for i in range(self.size)]
        arr = np.empty(self.size, dtype=object)
        arr[:] = items
        return np.ma.array(arr, mask=mask, dtype=object).reshape(self.shape)

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing the mask.

        The returned numpy array has dtype "bool" and a mask for
        masking out missing data.
        """
        mask = [not self._index_to_file(i).is_file() for i in range(self.size)]
        return np.ma.array(mask, mask=mask, dtype=bool).reshape(self.shape)

    def dump(self, key: tuple[int, ...], value: Any) -> None:
        """Dump 'value' into the file associated with 'key'.

        Examples
        --------
        >>> arr = FileArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        key = self._normalize_key(key)
        if not _key_has_slice(key):
            return dump(value, self._key_to_file(key))

        msg = "Cannot yet dump subarrays"
        raise NotImplementedError(msg)


def _key_has_slice(key: tuple[int, ...]) -> bool:
    return any(isinstance(x, slice) for x in key)


def _load_all(filenames: Iterator[Path]) -> list[Any]:
    def maybe_read(f: Path) -> Any | None:
        return read(f) if f.is_file() else None

    def maybe_load(x: str | None) -> Any | None:
        return cloudpickle.loads(x) if x is not None else None

    # Delegate file reading to the threadpool but deserialize sequentially,
    # as this is pure Python and CPU bound
    with concurrent.futures.ThreadPoolExecutor() as tex:
        return [maybe_load(x) for x in tex.map(maybe_read, filenames)]
