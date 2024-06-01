"""PipeFunc: A Python library for defining, managing, and executing function pipelines."""

from pipefunc._lazy import construct_dag, evaluate_lazy
from pipefunc._pipefunc import PipeFunc, pipefunc
from pipefunc._pipeline import Pipeline
from pipefunc._version import __version__
from pipefunc.sweep import Sweep

__all__ = [
    "construct_dag",
    "__version__",
    "evaluate_lazy",
    "pipefunc",
    "Pipeline",
    "PipeFunc",
    "Sweep",
]
