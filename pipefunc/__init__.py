"""PipeFunc: A Python library for defining, managing, and executing function pipelines."""

from pipefunc import lazy, map, sweep
from pipefunc._pipefunc import PipeFunc, pipefunc
from pipefunc._pipeline import Pipeline
from pipefunc._version import __version__

__all__ = [
    "__version__",
    "pipefunc",
    "PipeFunc",
    "Pipeline",
    "lazy",
    "map",
    "sweep",
]
