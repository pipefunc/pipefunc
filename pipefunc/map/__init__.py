"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from ._map import RunInfo, load_outputs, load_xarray_dataset, map, map_async
from ._mapspec import MapSpec
from ._storage_array import (
    DictArray,
    FileArray,
    SharedMemoryDictArray,
    StorageBase,
    register_storage,
    storage_registry,
)

__all__ = [
    "DictArray",
    "FileArray",
    "load_outputs",
    "load_xarray_dataset",
    "MapSpec",
    "register_storage",
    "map",
    "map_async",
    "RunInfo",
    "SharedMemoryDictArray",
    "storage_registry",
    "StorageBase",
]

with _suppress(ImportError):
    from ._storage_array import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
