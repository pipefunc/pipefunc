"""Provides `zarr` integration for `pipefunc`."""

from __future__ import annotations

import gzip
import itertools
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import zarr
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.registry import register_codec

from pipefunc._utils import prod
from pipefunc.map._filearray import (
    FileArrayBase,
    _select_by_mask,
)
from pipefunc.map._mapspec import shape_to_strides

if TYPE_CHECKING:
    from pathlib import Path


class ZarrArray(FileArrayBase):
    """Array interface to a Zarr store."""

    def __init__(
        self,
        store: zarr.storage.Store | str | Path,
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
        if not isinstance(store, zarr.storage.Store):
            store = zarr.DirectoryStore(str(store))
        self.store = store
        self.shape = tuple(shape)
        self.strides = shape_to_strides(self.shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()
        self.full_shape = _select_by_mask(self.shape_mask, self.shape, self.internal_shape)

        if object_codec is None:
            object_codec = CloudPickleGzip()
        chunks = _select_by_mask(self.shape_mask, (1,) * len(self.shape), self.internal_shape)
        self.array = zarr.open(
            self.store,
            mode="a",
            path="/array",
            shape=self.full_shape,
            dtype=object,
            object_codec=object_codec,
            chunks=chunks,
        )
        self._mask = zarr.open(
            self.store,
            mode="a",
            path="/mask",
            shape=self.shape,
            dtype=bool,
            fill_value=True,
            object_codec=object_codec,
            chunks=1,
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
        full_index = _select_by_mask(
            self.shape_mask,
            np_index,
            (slice(None),) * len(self.internal_shape),
        )
        return self.array[full_index]

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        np_index = np.unravel_index(index, self.shape)
        return not self._mask[np_index]

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:
        """Return the data associated with the given key."""
        data = self.array[key]
        mask = self.mask[key]  # makes entire mask TODO make more efficient
        item: np.ma.MaskedArray = np.ma.masked_array(data, mask=mask, dtype=object)
        if item.shape == ():
            if item.mask:
                return np.ma.masked
            return item.item()
        return item

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        """Return the array as a NumPy masked array."""
        if splat_internal and not self.internal_shape:
            msg = "internal_shape must be provided if splat_internal is True"
            raise ValueError(msg)
        if splat_internal is None:
            splat_internal = True
        if not splat_internal:
            msg = "splat_internal must be True"
            raise NotImplementedError(msg)
        return np.ma.array(self.array[:], mask=self.mask, dtype=object)

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return the mask associated with the array."""
        # TODO: needs to add internal dims
        mask = self._mask[:]
        slc = _select_by_mask(
            self.shape_mask,
            (slice(None),) * len(self.shape),
            (None,) * len(self.internal_shape),
        )
        tile_shape = _select_by_mask(self.shape_mask, (1,) * len(self.shape), self.internal_shape)
        mask = np.tile(mask[slc], tile_shape)
        return np.ma.array(mask, dtype=bool)

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        return list(
            self.mask.flat,
        )  # TODO: this is surely wrong! needs to not include internal dims

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = ZarrArray(...)
        >>> arr.dump((2, 1, 5), dict(a=1, b=2))

        """
        if any(isinstance(k, slice) for k in key):
            for external_index in itertools.product(*self._slice_indices(key)):
                if self.internal_shape:
                    value = np.asarray(value)  # in case it's a list
                    assert value.shape == self.internal_shape
                    full_index = _select_by_mask(
                        self.shape_mask,
                        external_index,
                        (slice(None),) * len(self.internal_shape),
                    )
                    self.array[full_index] = value
                else:
                    self.array[external_index] = value
                self._mask[external_index] = False
            return

        if self.internal_shape:
            value = np.asarray(value)  # in case it's a list
            assert value.shape == self.internal_shape
            full_index = _select_by_mask(
                self.shape_mask,
                key,
                (slice(None),) * len(self.internal_shape),
            )
            self.array[full_index] = value
        else:
            self.array[key] = value
        self._mask[key] = False

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        slice_indices = []
        for size, k in zip(self.shape, key):
            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(size)))
            else:
                slice_indices.append(range(k, k + 1))
        return slice_indices


class CloudPickleGzip(Codec):
    """Codec to encode data as compressed cloudpickled bytes.

    Useful for encoding an array of Python objects while ensuring compatibility
    across different Python versions and environments.

    Parameters
    ----------
    protocol
        The protocol used to pickle data.
    compression_level
        The compression level used by gzip. Ranges from 0 to 9, with 9 being the highest compression.

    Examples
    --------
    >>> from pipefunc.map.zarr import CloudPickleGzip
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = CloudPickleGzip()
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    """

    codec_id = "cloudpickle_gzip"

    def __init__(
        self,
        protocol: int = cloudpickle.DEFAULT_PROTOCOL,
        compression_level: int = 9,
    ) -> None:
        """Initialize the CloudPickleGzip codec.

        Parameters
        ----------
        protocol
            The protocol used to pickle data, by default cloudpickle.DEFAULT_PROTOCOL
        compression_level
            The compression level used by gzip, by default 9

        """
        self.protocol = protocol
        self.compression_level = compression_level

    def encode(self, buf: Any) -> bytes:
        """Encode the input buffer using CloudPickle and gzip compression.

        Parameters
        ----------
        buf
            The input buffer to encode.

        Returns
        -------
        The compressed cloudpickled data.

        """
        data = cloudpickle.dumps(buf, protocol=self.protocol)
        return gzip.compress(data, compresslevel=self.compression_level)

    def decode(self, buf: np.ndarray, out: np.ndarray | None = None) -> Any:
        """Decode the input buffer using gzip decompression and CloudPickle.

        Parameters
        ----------
        buf
            The compressed cloudpickled data.
        out
            The output array to store the decoded data, by default None

        Returns
        -------
        Any
            The decoded data.

        """
        buf = ensure_contiguous_ndarray(buf)
        data = gzip.decompress(buf)  # type: ignore[arg-type]
        dec = cloudpickle.loads(data)

        if out is not None:
            np.copyto(out, dec)
            return out
        return dec

    def get_config(self) -> dict[str, Any]:
        """Get the configuration of the codec.

        Returns
        -------
        The configuration of the codec.

        """
        return {
            "id": self.codec_id,
            "protocol": self.protocol,
            "compression_level": self.compression_level,
        }

    def __repr__(self) -> str:
        """Return a string representation of the codec."""
        return (
            f"CloudPickleGzip(protocol={self.protocol}, compression_level={self.compression_level})"
        )


register_codec(CloudPickleGzip)
