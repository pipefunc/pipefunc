"""PipeFunc: A Python library for defining, managing, and executing function pipelines."""

from pipefunc._pipefunc import Pipeline, PipelineFunction, evaluate_lazy, pipefunc
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
    "__version__",
    "count_sweep",
    "evaluate_lazy",
    "generate_sweep",
    "get_precalculation_order",
    "MultiSweep",
    "pipefunc",
    "Pipeline",
    "PipelineFunction",
    "set_cache_for_sweep",
    "Sweep",
]
