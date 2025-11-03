"""Provides `zarr` integration for `pipefunc` (Zarr v3 only)."""

from __future__ import annotations

import itertools
import multiprocessing.managers
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsInt, cast

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
from zarr.errors import ArrayNotFoundError, UnstableSpecificationWarning

try:  # pragma: no cover - import location changed in early v3 builds
    from zarr.dtype import VariableLengthBytes
except ImportError:  # pragma: no cover - fallback for pre-3.1
    from zarr.core.dtype import VariableLengthBytes  # type: ignore[attr-defined]

from zarr.storage import LocalStore, MemoryStore

from pipefunc.map._shapes import shape_is_resolved

from ._base import StorageBase, register_storage, select_by_mask

if TYPE_CHECKING:
    from pipefunc.map._types import ShapeTuple


# Zarr emits an "Unstable specification" warning for VariableLengthBytes. We rely on it
# intentionally, so silence that chatter for end users.
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

_FILL_VALUE = b""


def _ensure_int_tuple(
    values: tuple[int | str | SupportsInt, ...],
    *,
    what: str,
) -> tuple[int, ...]:
    ints: list[int] = []
    for value in values:
        if isinstance(value, str):
            msg = f"{what} contained unresolved values"
            raise TypeError(msg)
        ints.append(int(value))
    return tuple(ints)


def _open_or_create_array(
    store: Store,
    *,
    name: str,
    shape: tuple[Any, ...],
    chunks: tuple[Any, ...],
    dtype: Any,
    fill_value: Any,
) -> zarr.Array:
    """Open an array if it exists, otherwise create it."""
    try:
        array = zs.open_array(store=store, path=name)
    except ArrayNotFoundError:
        array = zs.create_array(
            store=store,
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            fill_value=fill_value,
            overwrite=False,
        )
    else:
        if array.shape != shape:
            msg = f"Existing array '{name}' has unexpected shape {array.shape}, expected {shape}."
            raise ValueError(msg)
    return array


def _encode_scalar(codec: CloudPickleCodec, value: Any) -> np.bytes_:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return np.bytes_(codec.encode(value))


def _encode_array(codec: CloudPickleCodec, value: np.ndarray) -> np.ndarray:
    encoded = np.empty(value.shape, dtype=object)
    for idx in np.ndindex(value.shape):
        encoded[idx] = _encode_scalar(codec, value[idx])
    return encoded


def _decode_scalar(codec: CloudPickleCodec, value: Any) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.bytes_):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        buffer = np.frombuffer(value, dtype="uint8")
        return codec.decode(buffer)
    return value


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

    decoded = decoded.reshape(data_array.shape)
    mask_array = mask_array.reshape(data_array.shape)
    return decoded, mask_array


def _copy_store(source: Store, destination: Store, *, if_exists: str = "replace") -> None:
    """Copy all keys from ``source`` into ``destination`` using the async Store API."""
    if if_exists not in {"raise", "replace", "skip"}:
        msg = "if_exists must be 'raise', 'replace', or 'skip'"
        raise ValueError(msg)

    async def _copy() -> None:
        await source._ensure_open()
        await destination._ensure_open()
        prototype = default_buffer_prototype()

        async for key in source.list():
            exists = await destination.exists(key)
            if exists and if_exists == "skip":
                continue
            if exists and if_exists == "raise":
                msg = f"Key {key!r} already exists in destination store"
                raise ValueError(msg)
            if exists and destination.supports_deletes:
                await destination.delete(key)
            buffer = await source.get(key, prototype=prototype)
            if buffer is not None:
                await destination.set(key, buffer)

    sync(_copy())


class ZarrFileArray(StorageBase):
    """Array interface to a Zarr store."""

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

        self.object_codec = object_codec if object_codec is not None else CloudPickleCodec()
        self.shape = tuple(shape)
        self.shape_mask = tuple(shape_mask) if shape_mask is not None else (True,) * len(self.shape)
        self.internal_shape = tuple(internal_shape) if internal_shape is not None else ()

        raw_chunks = select_by_mask(self.shape_mask, (1,) * len(self.shape), self.internal_shape)
        assert shape_is_resolved(
            raw_chunks,
        ), "Chunk sizes must be resolved before creating a Zarr array"
        chunks = cast("tuple[int, ...]", _ensure_int_tuple(raw_chunks, what="Chunk sizes"))

        full_shape_raw = self.full_shape
        assert shape_is_resolved(
            full_shape_raw,
        ), "Array shape must be resolved before creating a Zarr array"
        full_shape = cast("tuple[int, ...]", _ensure_int_tuple(full_shape_raw, what="Array shape"))
        self.array = _open_or_create_array(
            self.store,
            name="array",
            shape=full_shape,
            chunks=chunks,
            dtype=VariableLengthBytes(),
            fill_value=_FILL_VALUE,
        )
        self._mask = _open_or_create_array(
            self.store,
            name="mask",
            shape=self.shape,
            chunks=(1,) * len(self.shape) or (1,),
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
        if isinstance(data, np.ndarray):
            return np.asarray(
                _decode_array(self.object_codec, data),
                dtype=object,
            )
        return self._normalize_decoded(_decode_scalar(self.object_codec, data))

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
            return self._normalize_decoded(decoded.item())
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
            (None,) * len(self.internal_shape),
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
        """Dump ``value`` into the location associated with ``key``."""
        if any(isinstance(k, slice) for k in key):
            for external_index in itertools.product(*self._slice_indices(key)):
                if self.internal_shape:
                    block = np.asarray(value, dtype=object)
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
            block = np.asarray(value, dtype=object)
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
            indices = cast("tuple[int, ...]", key)
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
        """Indicates if the storage can be dumped in a subprocess."""
        return True

    def _store_scalar_encoded(self, indices: tuple[int, ...], encoded: np.bytes_) -> None:
        """Store a single serialized value at the provided indices."""
        slice_key = tuple(slice(i, i + 1) for i in indices)
        shaped = np.array([encoded], dtype=object).reshape(*(1,) * len(indices))
        self.array[slice_key] = shaped

    def _normalize_decoded(self, value: Any) -> Any:
        """Normalize decoded values to match legacy behaviour."""
        if isinstance(value, list):
            return np.array(value, dtype=object)
        return value


class _SharedDictStore(MemoryStore):
    """MemoryStore backed by a multiprocessing.Manager dictionary."""

    def __init__(self, shared_dict: multiprocessing.managers.DictProxy | None = None) -> None:
        if shared_dict is None:
            shared_dict = multiprocessing.Manager().dict()
        super().__init__(store_dict=shared_dict)


class ZarrMemoryArray(ZarrFileArray):
    """Array interface to an in-memory Zarr store."""

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
        if self.folder is None:  # pragma: no cover - defensive
            return None
        return LocalStore(self.folder)

    def persist(self) -> None:
        """Persist the memory storage to disk."""
        persistent = self.persistent_store
        if persistent is None:
            return
        _copy_store(self.store, persistent, if_exists="replace")

    def load(self) -> None:
        """Load the memory storage from disk."""
        persistent = self.persistent_store
        if persistent is None:  # pragma: no cover - defensive
            return
        folder = self.folder
        if folder is None or not folder.exists():  # pragma: no cover - defensive
            return
        _copy_store(persistent, self.store, if_exists="replace")

    @property
    def dump_in_subprocess(self) -> bool:
        """Indicates if the storage can be dumped in a subprocess and read by the main process."""
        return False


class ZarrSharedMemoryArray(ZarrMemoryArray):
    """Array interface to a shared memory Zarr store."""

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
        return True


class CloudPickleCodec(Codec):
    """Codec to encode data as cloudpickled bytes."""

    codec_id = "cloudpickle"

    def __init__(self, protocol: int = cloudpickle.DEFAULT_PROTOCOL) -> None:
        self.protocol = protocol

    def encode(self, buf: Any) -> bytes:
        return cloudpickle.dumps(buf, protocol=self.protocol)

    def decode(self, buf: np.ndarray, out: np.ndarray | None = None) -> Any:
        buf = ensure_contiguous_ndarray(buf)
        dec = cloudpickle.loads(buf)

        if out is not None:
            np.copyto(out, dec)
            return out
        return dec

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id, "protocol": self.protocol}

    def __repr__(self) -> str:
        return f"CloudPickleCodec(protocol={self.protocol})"


def _decode_array(
    codec: CloudPickleCodec,
    value: np.ndarray,
) -> np.ndarray:
    decoded = np.empty(value.shape, dtype=object)
    for idx in np.ndindex(value.shape):
        decoded[idx] = _decode_scalar(codec, value[idx])
    return decoded


register_codec(CloudPickleCodec)
register_storage(ZarrFileArray)
register_storage(ZarrMemoryArray)
register_storage(ZarrSharedMemoryArray)
