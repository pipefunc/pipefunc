"""PipeFunc: A Python library for defining, managing, and executing function pipelines."""

from pipefunc._lazy import construct_dag, evaluate_lazy
from pipefunc._pipefunc import PipeFunc, pipefunc
from pipefunc._pipeline import Pipeline
from pipefunc._sweep import (
    MultiSweep,
    Sweep,
    count_sweep,
    generate_sweep,
    get_precalculation_order,
    set_cache_for_sweep,
)
from pipefunc._version import __version__

__all__ = [
    "construct_dag",
    "__version__",
    "count_sweep",
    "evaluate_lazy",
    "generate_sweep",
    "get_precalculation_order",
    "MultiSweep",
    "pipefunc",
    "Pipeline",
    "PipeFunc",
    "set_cache_for_sweep",
    "Sweep",
]
