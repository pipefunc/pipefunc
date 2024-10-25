"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from ._map._io import load_outputs, load_xarray_dataset
from ._map._run import run_map, run_map_async
from ._map._run_info import RunInfo
from ._mapspec import MapSpec
from ._storage_array._base import StorageBase, register_storage, storage_registry
from ._storage_array._dict import DictArray, SharedMemoryDictArray
from ._storage_array._file import FileArray

__all__ = [
    "DictArray",
    "FileArray",
    "load_outputs",
    "load_xarray_dataset",
    "run_map_async",
    "run_map",
    "MapSpec",
    "register_storage",
    "RunInfo",
    "SharedMemoryDictArray",
    "storage_registry",
    "StorageBase",
]

with _suppress(ImportError):
    from ._storage_array import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
