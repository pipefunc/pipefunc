"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from pipefunc.map._filearray import FileArray
from pipefunc.map._map import run_pipeline
from pipefunc.map._mapspec import MapSpec

__all__ = ["MapSpec", "run_pipeline", "FileArray"]
