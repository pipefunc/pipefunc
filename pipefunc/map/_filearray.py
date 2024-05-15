# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import concurrent.futures
import functools
import itertools
import operator
import pathlib
from typing import Any, Sequence

import numpy as np

from . import serialize

filename_template = "__{:d}__.pickle"


class FileBasedObjectArray:
    """Array interface to a folder of files on disk.

    __getitem__ returns "np.ma.masked" for non-existant files.
    """

    def __init__(
        self,
        folder,
        shape,
        strides=None,
        filename_template=filename_template,
    ):
        self.folder = pathlib.Path(folder).absolute()
        self.shape = tuple(shape)
        self.strides = _make_strides(self.shape) if strides is None else tuple(strides)
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
            raise IndexError(
                f"too many indices for array: array is {self.rank}-dimensional, "
                "but {len(key)} were indexed"
            )

        if any(isinstance(k, slice) for k in key):
            raise NotImplementedError("Cannot yet slice subarrays")

        normalized_key = []
        for axis, k in enumerate(key):
            axis_size = self.shape[axis]
            normalized_k = k if k >= 0 else (axis_size - k)
            if not (0 <= normalized_k < axis_size):
                raise IndexError(
                    "index {k} is out of bounds for axis {axis} with size {axis_size}"
                )
            normalized_key.append(k)

        return tuple(normalized_key)

    def _index_to_file(self, index: int) -> pathlib.Path:
        """Return the filename associated with the given index."""
        return self.folder / self.filename_template.format(index)

    def _key_to_file(self, key: tuple[int, ...]) -> pathlib.Path:
        """Return the filename associated with the given key."""
        index = sum(k * s for k, s in zip(key, self.strides))
        return self._index_to_file(index)

    def _files(self):
        """Yield all the filenames that constitute the data in this array."""
        return map(self._key_to_file, itertools.product(*map(range, self.shape)))

    def __getitem__(self, key):
        key = self._normalize_key(key)
        if any(isinstance(x, slice) for x in key):
            # XXX: need to figure out strides in order to implement this.
            raise NotImplementedError("Cannot yet slice subarrays")

        f = self._key_to_file(key)
        if not f.is_file():
            return np.ma.core.masked
        return serialize.load(f)

    def to_array(self) -> np.ma.core.MaskedArray:
        """Return a masked numpy array containing all the data.

        The returned numpy array has dtype "object" and a mask for
        masking out missing data.
        """
        items = _load_all(map(self._index_to_file, range(self.size)))
        mask = [not self._index_to_file(i).is_file() for i in range(self.size)]
        return np.ma.array(items, mask=mask, dtype=object).reshape(self.shape)

    def dump(self, key, value):
        """Dump 'value' into the file associated with 'key'.

        Examples
        --------
        >>> arr = FileBasedObjectArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))
        """
        key = self._normalize_key(key)
        if not any(isinstance(x, slice) for x in key):
            return serialize.dump(value, self._key_to_file(key))

        raise NotImplementedError("Cannot yet dump subarrays")


def _tails(seq):
    while seq:
        seq = seq[1:]
        yield seq


def _make_strides(shape):
    return tuple(functools.reduce(operator.mul, s, 1) for s in _tails(shape))


def _load_all(filenames: Sequence[str]) -> list[Any]:
    def maybe_read(f):
        return serialize.read(f) if f.is_file() else None

    def maybe_load(x):
        return serialize.loads(x) if x is not None else None

    # Delegate file reading to the threadpool but deserialize sequentially,
    # as this is pure Python and CPU bound
    with concurrent.futures.ThreadPoolExecutor() as tex:
        return [maybe_load(x) for x in tex.map(maybe_read, filenames)]
