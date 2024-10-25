"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run._info import RunInfo
from pipefunc.map._run._run import load_outputs, load_xarray_dataset, run
from pipefunc.map._storage import (
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
    "run",
    "RunInfo",
    "SharedMemoryDictArray",
    "storage_registry",
    "StorageBase",
]

with _suppress(ImportError):
    from pipefunc.map._storage import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
