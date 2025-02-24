"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from ._load import load_outputs, load_xarray_dataset
from ._mapspec import MapSpec
from ._run import run_map, run_map_async
from ._run_info import RunInfo
from ._storage_array._base import StorageBase, register_storage, storage_registry
from ._storage_array._dict import DictArray, SharedMemoryDictArray
from ._storage_array._file import FileArray

__all__ = [
    "DictArray",
    "FileArray",
    "MapSpec",
    "RunInfo",
    "SharedMemoryDictArray",
    "StorageBase",
    "load_outputs",
    "load_xarray_dataset",
    "register_storage",
    "run_map",
    "run_map_async",
    "storage_registry",
]

with _suppress(ImportError, AttributeError):
    # ImportError: zarr not installed
    # AttributeError: zarr v3 is not compatible with pipefunc
    #   Blocked by: https://github.com/zarr-developers/zarr-python/issues/2617
    #   TODO: Remove this block once zarr v3 is compatible with pipefunc
    from ._storage_array._zarr import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
