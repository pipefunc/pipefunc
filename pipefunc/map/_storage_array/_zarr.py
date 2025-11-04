"""Provides `zarr` integration for `pipefunc`."""

from __future__ import annotations

import itertools
import multiprocessing.managers
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import numpy as np
import zarr  # noqa: TC002
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
from numcodecs.registry import register_codec
from zarr.abc.store import Store
from zarr.api import synchronous as zs
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import sync
from zarr.dtype import VariableLengthBytes
from zarr.errors import UnstableSpecificationWarning
from zarr.storage import LocalStore, MemoryStore

from pipefunc._utils import prod

from ._base import StorageBase, normalize_key, register_storage, select_by_mask

if TYPE_CHECKING:
    from pipefunc.map._types import ShapeTuple


_FILL_VALUE = b""


def _open_or_create_array(
    store: Store,
    *,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    fill_value: Any,
) -> zarr.Array:
    """Open an array (creating it if missing) via the public Zarr API.

    Uses `zarr.api.synchronous.open_array(..., mode='a', ...)` to avoid
    reimplementing open-or-create behavior.
    """
    # Suppress Zarr's UnstableSpecificationWarning for VariableLengthBytes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        array = zs.open_array(
            store=store,
            path=name,
            mode="a",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            fill_value=fill_value,
        )
        # Maintain previous safety check: validate existing shape if already created
        if array.shape != shape:
            msg = f"Existing array '{name}' has unexpected shape {array.shape}, expected {shape}."
            raise ValueError(msg)
        return array


def _encode_scalar(codec: CloudPickleCodec, value: Any) -> bytes:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return codec.encode(value)


def _encode_array(codec: CloudPickleCodec, value: np.ndarray) -> np.ndarray:
    encoded = np.empty(value.shape, dtype=object)
    for idx in np.ndindex(value.shape):
        encoded[idx] = _encode_scalar(codec, value[idx])
    return encoded


def _decode_scalar(codec: CloudPickleCodec, value: Any) -> Any:
    """Decode a single stored element.

    For ``VariableLengthBytes``: slicing returns an ``ndarray`` (``dtype=object``)
    whose elements are Python ``bytes``; scalar indexing returns a 0-D
    ``numpy.ndarray``. Handle both forms and avoid extra branching.
    """
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    assert isinstance(value, (bytes, bytearray, np.bytes_))
    buffer = np.frombuffer(value, dtype="uint8")
    return codec.decode(buffer)


def _decode_array(codec: CloudPickleCodec, value: np.ndarray) -> np.ndarray:
    decoded = np.empty(value.shape, dtype=object)
    for idx in np.ndindex(value.shape):
        decoded[idx] = _decode_scalar(codec, value[idx])
    return decoded


def _decode_with_mask(
    codec: CloudPickleCodec,
    data: Any,
    mask: Any,
) -> tuple[np.ndarray, np.ndarray]:
    data_array = np.asarray(data, dtype=object)
    mask_array = np.asarray(mask, dtype=bool)
    if (mask_array.shape == () and data_array.shape != ()) or mask_array.shape != data_array.shape:
        mask_array = np.broadcast_to(mask_array, data_array.shape)

    decoded = np.empty(data_array.shape, dtype=object)
    for idx in np.ndindex(data_array.shape if data_array.shape else (1,)):
        index = idx if data_array.shape else ()
        if mask_array[index]:
            decoded[index] = None
        else:
            decoded[index] = _decode_scalar(codec, data_array[index])

    return decoded, mask_array


def _copy_store(source: Store, destination: Store) -> None:
    """Copy all keys from ``source`` into ``destination`` using the async Store API.

    Always overwrites existing keys in the destination ("replace" semantics).
    """

    async def _copy() -> None:
        # Use public Store API (lazy-open in get/set) without relying on privates
        prototype = default_buffer_prototype()
        async for key in source.list():
            buffer = await source.get(key, prototype=prototype)
            if buffer is not None:
                await destination.set(key, buffer)

    sync(_copy())


class ZarrFileArray(StorageBase):
    """Array interface to a Zarr store.

    Only exists if the `zarr` package is installed!
    """

    storage_id = "zarr_file_array"
    requires_serialization = True

    def __init__(
        self,
        folder: str | Path | None,
        shape: ShapeTuple,
        internal_shape: ShapeTuple | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: Store | str | Path | None = None,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrFileArray."""
        if internal_shape and shape_mask is None:
            msg = "shape_mask must be provided if internal_shape is provided"
            raise ValueError(msg)
        if internal_shape is not None and len(shape_mask) != len(shape) + len(internal_shape):  # type: ignore[arg-type]
            msg = "shape_mask must have the same length as shape + internal_shape"
            raise ValueError(msg)

        self.folder = Path(folder) if folder is not None else None
        if isinstance(store, Store):
            self.store = store
        else:
            if store is None and self.folder is None:
                msg = "Either a `store` or `folder` must be provided"
                raise ValueError(msg)
            root = Path(store) if store is not None else self.folder
            assert root is not None
            self.store = LocalStore(str(root))

        self.object_codec = (
            object_codec if object_codec is not None else CloudPickleCodec()
        )  # Used for encoding and in __repr__
        self.shape = tuple(shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

        raw_chunks = select_by_mask(self.shape_mask, (1,) * len(self.shape), self.internal_shape)
        chunks: tuple[int, ...] = tuple(int(value) for value in raw_chunks)

        self.array = _open_or_create_array(
            self.store,
            name="array",
            shape=self.full_shape,
            chunks=chunks,
            dtype=VariableLengthBytes(),
            fill_value=_FILL_VALUE,
        )
        self._mask = _open_or_create_array(
            self.store,
            name="mask",
            shape=self.resolved_shape,
            # Zarr v3 requires an empty tuple for 0-D arrays; using chunks=(1,)
            # raises ValueError. Use chunks=() for scalars.
            chunks=() if len(self.resolved_shape) == 0 else (1,) * len(self.resolved_shape),
            dtype=bool,
            fill_value=True,
        )

    def __repr__(self) -> str:
        folder = f"'{self.folder}'" if self.folder is not None else self.folder
        return (
            f"ZarrFileArray(folder={folder}, "
            f"shape={self.shape}, "
            f"internal_shape={self.internal_shape}, "
            f"shape_mask={self.shape_mask}, "
            f"store={self.store}, "
            f"object_codec={self.object_codec})"
        )

    @property
    def size(self) -> int:
        """Return number of elements in the array."""
        return prod(self.resolved_shape)

    @property
    def rank(self) -> int:
        """Return the rank of the array."""
        return len(self.resolved_shape)

    def get_from_index(self, index: int) -> Any:
        """Return the data associated with the given linear index."""
        np_index = np.unravel_index(index, self.resolved_shape)
        full_index = select_by_mask(
            self.shape_mask,
            np_index,
            (slice(None),) * len(self.internal_shape),
        )
        if self._mask[np_index]:
            return np.ma.masked
        data = self.array[full_index]
        # In Zarr v3 array indexing returns a NumPy ndarray (scalar
        # indices yield a 0-D ndarray); unwrap to a Python scalar for parity
        decoded = np.asarray(_decode_array(self.object_codec, data), dtype=object)
        return decoded.item() if decoded.shape == () else decoded

    def has_index(self, index: int) -> bool:
        """Return whether the given linear index exists."""
        np_index = np.unravel_index(index, self.resolved_shape)
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
                    1 if isinstance(sl, slice) else s
                    for sl, s in zip(slices, np.asarray(data).shape)
                )
                mask = mask[slices]
                mask = np.tile(mask, tile_shape)
            else:
                mask = self._mask[mask_key]
        else:
            mask = self._mask[key]

        decoded, mask_array = _decode_with_mask(self.object_codec, data, mask)
        if decoded.shape == ():
            if mask_array.item():
                return np.ma.masked
            decoded_value = decoded.item()
            if isinstance(decoded_value, list):
                return np.array(decoded_value, dtype=object)
            return decoded_value
        return np.ma.masked_array(decoded, mask=mask_array, dtype=object)

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

        data = self.array[:]
        mask = self._mask[:]
        slc = select_by_mask(
            self.shape_mask,
            (slice(None),) * len(self.shape),
            (None,) * len(self.internal_shape),  # Adds axes with size 1
        )
        tile_shape = select_by_mask(self.shape_mask, (1,) * len(self.shape), self.internal_shape)
        mask = np.tile(mask[slc], tile_shape)

        decoded, mask_array = _decode_with_mask(self.object_codec, data, mask)
        return np.ma.MaskedArray(decoded, mask=mask_array, dtype=object)

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
                    block = np.asarray(value, dtype=object)  # in case it's a list
                    if block.shape != self.internal_shape:
                        msg = "Value has incorrect internal_shape"
                        raise ValueError(msg)
                    full_index = select_by_mask(
                        self.shape_mask,
                        external_index,
                        (slice(None),) * len(self.internal_shape),
                    )
                    encoded = _encode_array(self.object_codec, block)
                    self.array[full_index] = encoded
                else:
                    encoded_value = _encode_scalar(self.object_codec, value)
                    self._store_scalar_encoded(external_index, encoded_value)
                self._mask[external_index] = False
            return

        if self.internal_shape:
            block = np.asarray(value, dtype=object)  # in case it's a list
            if block.shape != self.internal_shape:
                msg = "Value has incorrect internal_shape"
                raise ValueError(msg)
            assert len(key) == len(self.shape)
            full_index = select_by_mask(
                self.shape_mask,
                key,
                (slice(None),) * len(self.internal_shape),
            )
            self.array[full_index] = _encode_array(self.object_codec, block)
        else:
            encoded_value = _encode_scalar(self.object_codec, value)
            assert all(isinstance(k, int) for k in key)
            indices = cast(tuple[int, ...], key)
            self._store_scalar_encoded(indices, encoded_value)
        self._mask[key] = False

    def _slice_indices(self, key: tuple[int | slice, ...]) -> list[range]:
        slice_indices: list[range] = []
        for size, k in zip(self.resolved_shape, key, strict=False):
            if isinstance(k, slice):
                slice_indices.append(range(*k.indices(size)))
            else:
                slice_indices.append(range(k, k + 1))
        return slice_indices

    @property
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
        return True

    def _store_scalar_encoded(self, indices: tuple[int, ...], encoded: bytes) -> None:
        """Store a single serialized value at the provided indices."""
        normalized_indices = cast(
            tuple[int, ...],
            normalize_key(
                indices,
                self.resolved_shape,
                self.resolved_internal_shape,
                self.shape_mask,
                for_dump=True,
            ),
        )
        slice_key = tuple(slice(i, i + 1) for i in normalized_indices)
        shaped = np.array([encoded], dtype=object).reshape(*(1,) * len(indices))
        self.array[slice_key] = shaped


class _SharedDictStore(MemoryStore):
    """Custom Store subclass using a shared dictionary."""

    def __init__(self, shared_dict: multiprocessing.managers.DictProxy | None = None) -> None:
        """Initialize the _SharedDictStore.

        Parameters
        ----------
        shared_dict
            Shared dictionary to use as the underlying storage, by default None
            If None, a new shared dictionary will be created.

        """
        if shared_dict is None:
            shared_dict = multiprocessing.Manager().dict()
        super().__init__(store_dict=shared_dict)


class ZarrMemoryArray(ZarrFileArray):
    """Array interface to an in-memory Zarr store.

    Only exists if the `zarr` package is installed!
    """

    storage_id = "zarr_memory"
    requires_serialization = False

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: Store | None = None,
        object_codec: Any = None,
    ) -> None:
        """Initialize the ZarrMemoryArray."""
        if store is None:
            store = MemoryStore()
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
    def persistent_store(self) -> Store | None:
        """Return the persistent store."""
        if self.folder is None:  # pragma: no cover
            return None
        return LocalStore(self.folder)

    def persist(self) -> None:
        """Persist the memory storage to disk."""
        persistent = self.persistent_store
        if persistent is None:
            return
        _copy_store(self.store, persistent)

    def load(self) -> None:
        """Load the memory storage from disk."""
        persistent = self.persistent_store
        if persistent is None:  # pragma: no cover
            return
        folder = self.folder
        if folder is None or not folder.exists():  # pragma: no cover
            return
        _copy_store(persistent, self.store)

    @property
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
        return False


class ZarrSharedMemoryArray(ZarrMemoryArray):
    """Array interface to a shared memory Zarr store.

    Only exists if the `zarr` package is installed!
    """

    storage_id = "zarr_shared_memory"
    requires_serialization = True

    def __init__(
        self,
        folder: str | Path | None,
        shape: tuple[int, ...],
        internal_shape: tuple[int, ...] | None = None,
        shape_mask: tuple[bool, ...] | None = None,
        *,
        store: Store | None = None,
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
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
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
    >>> from pipefunc.map._storage._zarr import CloudPickleCodec
    >>> import numpy as np
    >>> x = np.array(['foo', 'bar', 'baz'], dtype='object')
    >>> f = CloudPickleCodec()
    >>> f.decode(f.encode(x))
    array(['foo', 'bar', 'baz'], dtype=object)

    """

    codec_id = "cloudpickle"

    def __init__(self, protocol: int = cloudpickle.DEFAULT_PROTOCOL) -> None:
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
        return {"id": self.codec_id, "protocol": self.protocol}

    def __repr__(self) -> str:
        """Return a string representation of the codec."""
        return f"CloudPickleCodec(protocol={self.protocol})"


register_codec(CloudPickleCodec)
register_storage(ZarrFileArray)
register_storage(ZarrMemoryArray)
register_storage(ZarrSharedMemoryArray)
