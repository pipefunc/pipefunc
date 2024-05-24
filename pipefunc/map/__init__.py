"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import load_outputs, load_xarray_dataset, run

__all__ = ["FileArray", "load_outputs", "MapSpec", "run", "load_xarray_dataset"]
