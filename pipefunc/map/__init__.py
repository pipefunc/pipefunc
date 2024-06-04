"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from pipefunc.map._dictarray import DictArray, SharedMemoryDictArray
from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import load_outputs, load_xarray_dataset, run
from pipefunc.map._run_info import RunInfo
from pipefunc.map._storage_base import StorageBase, register_storage

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
    "StorageBase",
]

with _suppress(ImportError):
    from pipefunc.map.zarr import ZarrFileArray, ZarrMemoryArray, ZarrSharedMemoryArray

    __all__ += ["ZarrFileArray", "ZarrMemoryArray", "ZarrSharedMemoryArray"]
