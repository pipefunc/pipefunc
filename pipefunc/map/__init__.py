"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import load_outputs, run_pipeline

__all__ = ["MapSpec", "run_pipeline", "FileArray", "load_outputs"]
