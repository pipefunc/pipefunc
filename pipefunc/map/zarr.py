"""Provides `zarr` integration for `pipefunc`."""

from __future__ import annotations

import itertools
import multiprocessing.managers
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import zarr
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.registry import register_codec

from pipefunc._utils import prod
from pipefunc.map._storage_base import StorageBase, _select_by_mask, register_storage


class ZarrFileArray(StorageBase):
    """Array interface to a Zarr store."""

    storage_id = "zarr_file_array"

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: zarr.storage.Store | str | Path | None = None,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrFileArray."""
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)
        self.folder = Path(folder) if folder is not None else folder
        if not isinstance(store, zarr.storage.Store):
            store = zarr.DirectoryStore(str(self.folder))
        self.store = store
        self.shape = tuple(shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

        if object_codec is None:
            object_codec = CloudPickleCodec()

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

        if self.internal_shape:
            mask_key = tuple(x for x, m in zip(key, self.shape_mask) if m)
            if any(isinstance(k, slice) for k in key):
                mask = self._mask[mask_key]
                assert len(key) == len(self.shape_mask)
                slices = tuple(
                    slice(None) if m else None
                    for k, m in zip(key, self.shape_mask)
                    if isinstance(k, slice)
                )
                tile_shape = tuple(
                    1 if isinstance(sl, slice) else s for sl, s in zip(slices, data.shape)
                )
                mask = mask[slices]
                mask = np.tile(mask, tile_shape)
            else:
                mask = self._mask[mask_key]
        else:
            mask = self._mask[key]

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

        mask = self._mask[:]
        slc = _select_by_mask(
            self.shape_mask,
            (slice(None),) * len(self.shape),
            (None,) * len(self.internal_shape),  # Adds axes with size 1
        )
        tile_shape = _select_by_mask(
            self.shape_mask,
            (1,) * len(self.shape),
            self.internal_shape,
        )
        mask = np.tile(mask[slc], tile_shape)

        return np.ma.MaskedArray(self.array[:], mask=mask, dtype=object)

    @property
    def mask(self) -> np.ma.core.MaskedArray:
        """Return the mask associated with the array."""
        mask = self._mask[:]
        return np.ma.MaskedArray(mask, dtype=bool)

    def mask_linear(self) -> list[bool]:
        """Return a list of booleans indicating which elements are missing."""
        return list(self._mask[:].flat)

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        """Dump 'value' into the location associated with 'key'.

        Examples
        --------
        >>> arr = ZarrFileArray(...)
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
            assert len(key) == len(self.shape)
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

    @property
    def parallelizable(self) -> bool:
        """Return whether the storage is parallelizable."""
        return True


class _SharedDictStore(zarr.storage.KVStore):
    """Custom Store subclass using a shared dictionary."""

    def __init__(
        self,
        shared_dict: multiprocessing.managers.DictProxy | None = None,
    ) -> None:
        """Initialize the _SharedDictStore.

        Parameters
        ----------
        shared_dict
            Shared dictionary to use as the underlying storage, by default None
            If None, a new shared dictionary will be created.

        """
        if shared_dict is None:
            shared_dict = multiprocessing.Manager().dict()
        super().__init__(mutablemapping=shared_dict)


class ZarrMemoryArray(ZarrFileArray):
    """Array interface to an in-memory Zarr store."""

    storage_id = "zarr_memory"

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: zarr.storage.Store | None = None,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrMemoryArray."""
        if store is None:
            store = zarr.MemoryStore()
        super().__init__(
            folder=folder,
            shape=shape,
            internal_shape=internal_shape,
            shape_mask=shape_mask,
            store=store,
            object_codec=object_codec,
        )
        self.load()

    @property
    def persistent_store(self) -> zarr.storage.Store | None:
        """Return the persistent store."""
        if self.folder is None:  # pragma: no cover
            return None
        return zarr.DirectoryStore(self.folder)

    def persist(self) -> None:
        """Persist the memory storage to disk."""
        if self.folder is None:  # pragma: no cover
            return
        zarr.convenience.copy_store(self.store, self.persistent_store)

    def load(self) -> None:
        """Load the memory storage from disk."""
        if self.folder is None:  # pragma: no cover
            return
        if not self.folder.exists():
            return
        zarr.convenience.copy_store(self.persistent_store, self.store, if_exists="replace")

    @property
    def parallelizable(self) -> bool:
        """Return whether the storage is parallelizable."""
        return False


class ZarrSharedMemoryArray(ZarrMemoryArray):
    """Array interface to a shared memory Zarr store."""

    storage_id = "zarr_shared_memory"

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: zarr.storage.Store | None = None,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrSharedMemoryArray."""
        if store is None:
            store = _SharedDictStore()
        super().__init__(
            folder=folder,
            shape=shape,
            internal_shape=internal_shape,
            shape_mask=shape_mask,
            store=store,
            object_codec=object_codec,
        )

    @property
    def parallelizable(self) -> bool:
        """Return whether the storage is parallelizable."""
        return True


class CloudPickleCodec(Codec):
    """Codec to encode data as cloudpickled bytes.

    Useful for encoding an array of Python objects.

    Parameters
    ----------
    protocol
        The protocol used to pickle data.

    Examples
    --------
    >>> from pipefunc.map.zarr import CloudPickleCodec
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = CloudPickleCodec()
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    """

    codec_id = "cloudpickle"

    def __init__(
        self,
        protocol: int = cloudpickle.DEFAULT_PROTOCOL,
    ) -> None:
        """Initialize the CloudPickleCodec codec.

        Parameters
        ----------
        protocol
            The protocol used to pickle data, by default `cloudpickle.DEFAULT_PROTOCOL`

        """
        self.protocol = protocol

    def encode(self, buf: Any) -> bytes:
        """Encode the input buffer using CloudPickleCodec.

        Parameters
        ----------
        buf
            The input buffer to encode.

        Returns
        -------
            The cloudpickled data.

        """
        return cloudpickle.dumps(buf, protocol=self.protocol)

    def decode(self, buf: np.ndarray, out: np.ndarray | None = None) -> Any:
        """Decode the input buffer using CloudPickleCodec.

        Parameters
        ----------
        buf
            The cloudpickled data.
        out
            The output array to store the decoded data, by default None

        Returns
        -------
            The decoded data.

        """
        buf = ensure_contiguous_ndarray(buf)
        dec = cloudpickle.loads(buf)

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
        }

    def __repr__(self) -> str:
        """Return a string representation of the codec."""
        return f"CloudPickleCodec(protocol={self.protocol})"


register_codec(CloudPickleCodec)
register_storage(ZarrFileArray)
register_storage(ZarrMemoryArray)
register_storage(ZarrSharedMemoryArray)
