"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from contextlib import suppress as _suppress

from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import load_outputs, load_xarray_dataset, run
from pipefunc.map._storage_base import StorageBase, register_storage

__all__ = [
    "FileArray",
    "load_outputs",
    "load_xarray_dataset",
    "MapSpec",
    "register_storage",
    "run",
    "StorageBase",
]

with _suppress(ImportError):
    from pipefunc.map.zarr import ZarrFileArray, ZarrMemory

    __all__ += ["ZarrFileArray", "ZarrMemory"]
