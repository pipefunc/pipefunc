from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple

from ._mapspec import validate_consistent_axes
from ._progress import init_tracker
from ._run_info import RunInfo
from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    from pathlib import Path

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc._widgets import ProgressTracker

    from ._result import DirectValue, Result


def prepare_run(
    *,
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None,
    output_names: set[OUTPUT_TYPE] | None,
    parallel: bool,
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None,
    storage: str | dict[OUTPUT_TYPE, str],
    cleanup: bool,
    fixed_indices: dict[str, int | slice] | None,
    auto_subpipeline: bool,
    show_progress: bool,
    in_async: bool,
) -> tuple[
    Pipeline,
    RunInfo,
    dict[str, StorageBase | Path | DirectValue],
    OrderedDict[str, Result],
    bool,
    ProgressTracker | None,
]:
    if not parallel and show_progress:
        msg = "Cannot use `show_progress=True` with `parallel=False`."
        raise ValueError(msg)
    if not parallel and executor:
        msg = "Cannot use an executor without `parallel=True`."
        raise ValueError(msg)
    inputs = pipeline._flatten_scopes(inputs)
    if auto_subpipeline or output_names is not None:
        pipeline = pipeline.subpipeline(set(inputs), output_names)
    _validate_complete_inputs(pipeline, inputs)
    validate_consistent_axes(pipeline.mapspecs(ordered=False))
    _validate_fixed_indices(fixed_indices, inputs, pipeline)
    run_info = RunInfo.create(
        run_folder,
        pipeline,
        inputs,
        internal_shapes,
        storage=storage,
        cleanup=cleanup,
    )
    outputs: OrderedDict[str, Result] = OrderedDict()
    store = run_info.init_store()
    progress = init_tracker(store, pipeline.sorted_functions, show_progress, in_async)
    if executor is None and _cannot_be_parallelized(pipeline):
        parallel = False
    _check_parallel(parallel, store, executor)
    return pipeline, run_info, store, outputs, parallel, progress


def _cannot_be_parallelized(pipeline: Pipeline) -> bool:
    return all(f.mapspec is None for f in pipeline.functions) and all(
        len(fs) == 1 for fs in pipeline.topological_generations.function_lists
    )


def _check_parallel(
    parallel: bool,  # noqa: FBT001
    store: dict[str, StorageBase | Path | DirectValue],
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None,
) -> None:
    if isinstance(executor, dict):
        uses_default_executor: set[str] = set(store.keys()) - {
            n for name in executor for n in at_least_tuple(name)
        }
        for output_name, ex in executor.items():
            names = uses_default_executor if output_name == "" else at_least_tuple(output_name)
            _check_parallel(parallel, {n: store[n] for n in names}, ex)
        return
    if isinstance(executor, ThreadPoolExecutor):
        return
    if not parallel or not store:
        return
    for storage in store.values():
        if isinstance(storage, StorageBase) and not storage.parallelizable:
            recommendation = (
                "Consider\n - using a file-based storage or `shared_memory` / `zarr_shared_memory`"
                " for parallel execution,\n - disable parallel execution,\n - or use a different executor.\n"
            )
            default = f"The chosen storage type `{storage.storage_id}` does not support process-based parallel execution."
            if executor is None:
                msg = (
                    f"{default}"
                    f" PipeFunc defaults to using a `ProcessPoolExecutor`, which requires a parallelizable storage."
                    f" {recommendation}"
                )
                raise ValueError(msg)
            assert executor is not None
            msg = (
                f"{default}"
                f" If the current executor of type `{type(executor).__name__}` is process-based, it is incompatible."
                f" {recommendation}"
            )
            warnings.warn(msg, stacklevel=2)


def _validate_complete_inputs(pipeline: Pipeline, inputs: dict[str, Any]) -> None:
    """Validate that all required inputs are provided.

    Note that `output_name is None` means that all outputs are required!
    This is in contrast to some other functions, where ``None`` means that the
    `pipeline.unique_leaf_node` is used.
    """
    root_args = set(pipeline.topological_generations.root_args)
    inputs_with_defaults = set(inputs) | set(pipeline.defaults)
    if missing := root_args - set(inputs_with_defaults):
        missing_args = ", ".join(missing)
        msg = f"Missing inputs: `{missing_args}`."
        raise ValueError(msg)
    if extra := set(inputs_with_defaults) - root_args:
        extra_args = ", ".join(extra)
        msg = f"Got extra inputs: `{extra_args}` that are not accepted by this pipeline."
        raise ValueError(msg)


def _validate_fixed_indices(
    fixed_indices: dict[str, int | slice] | None,
    inputs: dict[str, Any],
    pipeline: Pipeline,
) -> None:
    if fixed_indices is None:
        return
    extra = set(fixed_indices)
    axes = pipeline.mapspec_axes
    for parameter, axes_ in axes.items():
        for axis in axes_:
            if axis in fixed_indices:
                extra.discard(axis)
        if parameter in inputs:
            key = tuple(fixed_indices.get(axis, slice(None)) for axis in axes_)
            if len(key) == 1:
                key = key[0]  # type: ignore[assignment]
            try:
                inputs[parameter][key]
            except IndexError as e:
                msg = f"Fixed index `{key}` for parameter `{parameter}` is out of bounds."
                raise IndexError(msg) from e
    if extra:
        msg = f"Got extra `fixed_indices`: `{extra}` that are not accepted by this map."
        raise ValueError(msg)

    reduced_axes = _reduced_axes(pipeline)
    for name, axes_set in reduced_axes.items():
        if reduced := set(axes_set) & set(fixed_indices):
            reduced_str = ", ".join(reduced)
            msg = f"Axis `{reduced_str}` in `{name}` is reduced and cannot be in `fixed_indices`."
            raise ValueError(msg)


def _reduced_axes(pipeline: Pipeline) -> dict[str, set[str]]:
    # TODO: check the overlap between this an `independent_axes_in_mapspecs`.
    # It might be that this function could be used instead.
    reduced_axes: dict[str, set[str]] = defaultdict(set)
    axes = pipeline.mapspec_axes
    for name in pipeline.mapspec_names:
        for func in pipeline.functions:
            if _is_parameter_reduced_by_function(func, name):
                reduced_axes[name].update(axes[name])
            elif _is_parameter_partially_reduced_by_function(func, name):
                _axes = _get_partially_reduced_axes(func, name, axes)
                reduced_axes[name].update(_axes)
    return dict(reduced_axes)


def _is_parameter_reduced_by_function(func: PipeFunc, name: str) -> bool:
    return name in func.parameters and (
        func.mapspec is None or name not in func.mapspec.input_names
    )


def _is_parameter_partially_reduced_by_function(func: PipeFunc, name: str) -> bool:
    if func.mapspec is None or name not in func.mapspec.input_names:
        return False
    spec = next(spec for spec in func.mapspec.inputs if spec.name == name)
    return None in spec.axes


def _get_partially_reduced_axes(
    func: PipeFunc,
    name: str,
    axes: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    assert func.mapspec is not None
    spec = next(spec for spec in func.mapspec.inputs if spec.name == name)
    return tuple(ax for ax, spec_ax in zip(axes[name], spec.axes) if spec_ax is None)
