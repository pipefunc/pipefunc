"""pipefunc.map.storage: Modules that handle storage for MapSpecs."""

from contextlib import suppress as _suppress

from pipefunc.map.storage._base import StorageBase, register_storage, storage_registry
from pipefunc.map.storage._dict import DictArray, SharedMemoryDictArray
from pipefunc.map.storage._file import FileArray

__all__ = [
    "DictArray",
    "FileArray",
    "register_storage",
    "SharedMemoryDictArray",
    "storage_registry",
    "StorageBase",
]

with _suppress(ImportError):
    from pipefunc.map.storage._zarr import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
