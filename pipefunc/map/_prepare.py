from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pipefunc._pipeline._pydantic import maybe_pydantic_model_to_dict
from pipefunc._utils import at_least_tuple

from ._adaptive_scheduler_slurm_executor import validate_slurm_executor
from ._mapspec import validate_consistent_axes
from ._progress import init_tracker
from ._result import ResultDict
from ._run_info import RunInfo

if TYPE_CHECKING:
    from concurrent.futures import Executor
    from pathlib import Path

    import pydantic

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc._widgets import ProgressTracker
    from pipefunc.map._types import UserShapeDict

    from ._result import StoreType


def prepare_run(
    *,
    pipeline: Pipeline,
    inputs: dict[str, Any] | pydantic.BaseModel,
    run_folder: str | Path | None,
    internal_shapes: UserShapeDict | None,
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
    dict[str, StoreType],
    ResultDict,
    bool,
    dict[OUTPUT_TYPE, Executor] | None,
    ProgressTracker | None,
]:
    if not parallel and executor:
        msg = "Cannot use an executor without `parallel=True`."
        raise ValueError(msg)
    inputs = maybe_pydantic_model_to_dict(inputs)
    inputs = pipeline._flatten_scopes(inputs)
    if auto_subpipeline or output_names is not None:
        pipeline = pipeline.subpipeline(set(inputs), output_names)
    if executor is not None and not isinstance(executor, dict):
        executor = {"": executor}
    elif isinstance(executor, dict):
        executor = executor.copy()  # this dict might be mutated, so we copy it
    validate_slurm_executor(executor, in_async)
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
    outputs = ResultDict(_inputs_=inputs, _pipeline_=pipeline)
    store = run_info.init_store()
    progress = init_tracker(store, pipeline.sorted_functions, show_progress, in_async)
    if executor is None and _cannot_be_parallelized(pipeline):
        parallel = False
    _check_parallel(parallel, store, executor)
    if parallel and any(func.profile for func in pipeline.functions):
        msg = "`profile=True` is not supported with `parallel=True` using process-based executors."
        warnings.warn(msg, UserWarning, stacklevel=2)
    return pipeline, run_info, store, outputs, parallel, executor, progress


def _cannot_be_parallelized(pipeline: Pipeline) -> bool:
    return all(f.mapspec is None for f in pipeline.functions) and all(
        len(fs) == 1 for fs in pipeline.topological_generations.function_lists
    )


def _check_parallel(
    parallel: bool,  # noqa: FBT001
    store: dict[str, StoreType],
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
