
from __future__ import annotations

import functools
import inspect
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import networkx as nx

from pipefunc._pipefunc import ErrorSnapshot, NestedPipeFunc, PipeFunc, _maybe_mapspec
from pipefunc._profile import print_profiling_stats
from pipefunc._simplify import _func_node_colors, _identify_combinable_nodes, simplified_pipeline
from pipefunc._utils import (
    assert_complete_kwargs,
    at_least_tuple,
    clear_cached_properties,
    handle_error,
)
from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache, to_hashable
from pipefunc.exceptions import UnusedParametersError
from pipefunc.lazy import _LazyFunction, task_graph
from pipefunc.map._mapspec import (
    ArraySpec,
    MapSpec,
    mapspec_axes,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run import AsyncMap, run_map, run_map_async
from pipefunc.resources import Resources
from pipefunc.typing import (
    Array,
    NoAnnotation,
    Unresolvable,
    is_object_array_type,
    is_type_compatible,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterable
    from concurrent.futures import Executor
    from pathlib import Path

    import graphviz
    import holoviews as hv
    import IPython.display

    from pipefunc._profile import ProfilingStats
    from pipefunc.map._result import Result
