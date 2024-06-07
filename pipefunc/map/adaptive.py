"""Provides `adaptive` integration for `pipefunc`."""

from __future__ import annotations

import functools
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, Union

import numpy as np
from adaptive import SequenceLearner, runner

from pipefunc._utils import at_least_tuple, prod
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import (
    _func_kwargs,
    _mask_fixed_axes,
    _maybe_load_single_output,
    _MockPipeline,
    _process_task,
    _reduced_axes,
    _run_iteration_and_process,
    _submit_single,
    _validate_fixed_indices,
    run,
)
from pipefunc.map._run_info import RunInfo, _external_shape, map_shapes
from pipefunc.map._storage_base import _iterate_shape_indices

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy.typing as npt

    from pipefunc import PipeFunc, Pipeline
    from pipefunc.map._storage_base import StorageBase
    from pipefunc.map.adaptive_scheduler import AdaptiveSchedulerDetails
    from pipefunc.resources import Resources
    from pipefunc.sweep import Sweep


_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]


class LearnerPipeFunc(NamedTuple):
    """A tuple with `~adaptive.SequenceLearner` and `~pipefunc.PipeFunc`."""

    learner: SequenceLearner
    pipefunc: PipeFunc


class AxisIndex(NamedTuple):
    """A named tuple to store the axis and index for a fixed axis."""

    axis: str
    idx: int | slice  # not called `index` to avoid shadowing the built-in


LearnersDictType: TypeAlias = UserDict[tuple[AxisIndex, ...] | None, list[list[LearnerPipeFunc]]]


class LearnersDict(LearnersDictType):
    """A dictionary of adaptive learners for a pipeline as returned by `create_learners`."""

    def __init__(self, learners_dict: LearnersDictType | None = None) -> None:
        """Create a dictionary of adaptive learners for a pipeline."""
        super().__init__(learners_dict or {})

    def flatten(self) -> dict[_OUTPUT_TYPE, list[SequenceLearner]]:
        """Flatten the learners into a dictionary with the output names as keys."""
        flat_learners: dict[_OUTPUT_TYPE, list[SequenceLearner]] = {}
        for learners_lists in self.data.values():
            for learners in learners_lists:
                for learner_with_pipefunc in learners:
                    output_name = learner_with_pipefunc.pipefunc.output_name
                    flat_learners.setdefault(output_name, []).append(learner_with_pipefunc.learner)
        return flat_learners

    def simple_run(self) -> None:
        """Run all the learners in the dictionary in order using `adaptive.runner.simple`."""
        for learner_list in self.flatten().values():
            for learner in learner_list:
                runner.simple(learner)

    def to_slurm_run(
        self,
        run_folder: str | Path,
        default_resources: dict | Resources | None = None,
        *,
        ignore_resources: bool = False,
    ) -> AdaptiveSchedulerDetails:
        """Uses `adaptive_scheduler.slurm_run` to create a `adaptive_scheduler.RunManager`."""
        from pipefunc.map.adaptive_scheduler import slurm_run_setup

        return slurm_run_setup(
            self,
            run_folder,
            default_resources,
            ignore_resources=ignore_resources,
        )


def create_learners(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    storage: str = "file_array",
    return_output: bool = False,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    split_independent_axes: bool = False,
) -> LearnersDict:
    """Create adaptive learners for a single `Pipeline.map` call.

    Creates learner(s) for each function node in the pipeline graph. The number of learners
    created for each node depends on the `fixed_indices` and `split_independent_axes` parameters:

    - If `fixed_indices` is provided or `split_independent_axes` is `False`, a single learner
      is created for each function node.
    - If `split_independent_axes` is `True`, multiple learners are created for each function
      node, corresponding to different combinations of the independent axes in the pipeline.

    Returns a dictionary where the keys represent specific combinations of indices for the
    independent axes, and the values are lists of lists of learners:

    - The outer lists represent different stages or generations of the pipeline, where the
      learners in each stage depend on the outputs of the learners in the previous stage.
    - The inner lists contain learners that can be executed independently within each stage.

    When `split_independent_axes` is `True`, each key in the dictionary corresponds to a
    different combination of indices for the independent axes, allowing for parallel
    execution across different subsets of the input data.

    If `fixed_indices` is `None` and `split_independent_axes` is `False`, the only key in
    the dictionary is `None`, indicating that all indices are being processed together.

    Parameters
    ----------
    pipeline
        The pipeline to create learners for.
    inputs
        The inputs to the pipeline, the same as passed to `pipeline.map`.
    run_folder
        The folder to store the run information.
    internal_shapes
        The internal shapes to use for the run.
    storage
        The storage class to use for the file arrays.
        Can use any registered storage class. See `pipefunc.map.storage_registry`.
    return_output
        Whether to return the output of the function in the learner.
    cleanup
        Whether to clean up the ``run_folder``.
    fixed_indices
        A dictionary mapping axes names to indices that should be fixed for the run.
        If not provided, all indices are iterated over.
    split_independent_axes
        Whether to split the independent axes into separate learners. Do not use
        in conjunction with ``fixed_indices``.

    See Also
    --------
    LearnersDict.to_slurm_run
        Convert the learners to variables that can be passed to `adaptive_scheduler.RunManager`.

    Returns
    -------
        A dictionary where the keys are the fixed indices, e.g., ``(("i", 0), ("j", 0))``,
        and the values are lists of lists of learners. The learners
        in the inner list can be executed in parallel, but the outer lists need
        to be executed in order. If ``fixed_indices`` is ``None`` and
        ``split_independent_axes`` is ``False``, then the only key is ``None``.

    """
    run_folder = Path(run_folder)
    run_info = RunInfo.create(
        run_folder,
        pipeline,
        inputs,
        internal_shapes,
        storage=storage,
        cleanup=cleanup,
    )
    run_info.dump(run_folder)
    store = run_info.init_store()
    learners: LearnersDict = LearnersDict()
    iterator = _maybe_iterate_axes(
        pipeline,
        inputs,
        fixed_indices,
        split_independent_axes,
        internal_shapes,
    )
    for fixed_indices in iterator:
        key = _key(fixed_indices) if fixed_indices else None
        for gen in pipeline.topological_generations.function_lists:
            _learners = []
            for func in gen:
                learner = _learner(
                    func=func,
                    run_info=run_info,
                    run_folder=run_folder,
                    store=store,
                    fixed_indices=fixed_indices,  # might be None
                    return_output=return_output,
                )
                _learners.append(LearnerPipeFunc(learner, func))
            learners.setdefault(key, []).append(_learners)
    return learners


def _learner(
    func: PipeFunc,
    run_info: RunInfo,
    run_folder: Path,
    store: dict[str, StorageBase],
    fixed_indices: dict[str, int | slice] | None,
    *,
    return_output: bool,
) -> SequenceLearner:
    if func.mapspec and func.mapspec.inputs:
        f = functools.partial(
            _execute_iteration_in_map_spec,
            func=func,
            run_info=run_info,
            run_folder=run_folder,
            store=store,
            return_output=return_output,
        )
        shape = run_info.shapes[func.output_name]
        mask = run_info.shape_masks[func.output_name]
        sequence = _sequence(fixed_indices, func.mapspec, shape, mask)
    else:
        f = functools.partial(
            _execute_iteration_in_single,
            func=func,
            run_info=run_info,
            run_folder=run_folder,
            store=store,
            return_output=return_output,
        )
        sequence = [None]  # type: ignore[list-item,assignment]
    return SequenceLearner(f, sequence)


def _key(fixed_indices: dict[str, int | slice]) -> tuple[AxisIndex, ...]:
    return tuple(AxisIndex(axis=axis, idx=idx) for axis, idx in sorted(fixed_indices.items()))


def _sequence(
    fixed_indices: dict[str, int | slice] | None,
    mapspec: MapSpec,
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
) -> npt.NDArray[np.int_] | range:
    if fixed_indices is None:
        return range(prod(shape))
    fixed_mask = _mask_fixed_axes(fixed_indices, mapspec, shape, mask)
    assert fixed_mask is not None
    assert len(fixed_mask) == prod(_external_shape(shape, mask))
    return np.flatnonzero(fixed_mask)


def _execute_iteration_in_single(
    _: Any,
    func: PipeFunc,
    run_info: RunInfo,
    run_folder: Path,
    store: dict[str, StorageBase],
    *,
    return_output: bool = False,
) -> Any | None:
    """Execute a single iteration of a single function.

    Meets the requirements of `adaptive.SequenceLearner`.
    """
    output, exists = _maybe_load_single_output(func, run_folder, return_output=return_output)
    if exists:
        return output
    kwargs = _func_kwargs(
        func,
        run_info.all_output_names,
        run_info.input_paths,
        run_info.shapes,
        run_info.shape_masks,
        store,
        run_folder,
    )
    task = _submit_single(func, kwargs, run_folder)
    result = _process_task(
        func,
        task,
        run_folder,
        store,
        kwargs,
    )
    if not return_output:
        return None
    output = tuple(result[name].output for name in at_least_tuple(func.output_name))
    return output if isinstance(func.output_name, tuple) else output[0]


def _execute_iteration_in_map_spec(
    index: int,
    func: PipeFunc,
    run_info: RunInfo,
    run_folder: Path,
    store: dict[str, StorageBase],
    *,
    return_output: bool = False,
) -> tuple[Any, ...] | None:
    """Execute a single iteration of a map spec.

    Performs a single iteration of the code in `_execute_map_spec`, however,
    it does not keep and return the output. This is meant to be used in the
    parallel execution of the map spec.

    Meets the requirements of `adaptive.SequenceLearner`.
    """
    file_arrays = [store[o] for o in at_least_tuple(func.output_name)]
    # Load the data if it exists
    if all(arr.has_index(index) for arr in file_arrays):
        if not return_output:
            return None
        return tuple(arr.get_from_index(index) for arr in file_arrays)
    # Otherwise, run the function
    assert isinstance(func.mapspec, MapSpec)
    kwargs = _func_kwargs(
        func,
        run_info.all_output_names,
        run_info.input_paths,
        run_info.shapes,
        run_info.shape_masks,
        store,
        run_folder,
    )
    shape = run_info.shapes[func.output_name]
    mask = run_info.shape_masks[func.output_name]
    outputs = _run_iteration_and_process(index, func, kwargs, shape, mask, file_arrays)
    return outputs if return_output else None


@dataclass
class _MapWrapper:
    """Wraps the `pipefunc.map.run` function and makes it a callable with a single unused argument.

    Uses a `_MockPipeline` that contains all the required information to run the pipeline but is
    cheaper to serialize and pass around.
    """

    mock_pipeline: _MockPipeline
    inputs: dict[str, Any]
    run_folder: Path
    internal_shapes: dict[str, int | tuple[int, ...]] | None
    parallel: bool
    cleanup: bool

    def __call__(self, _: Any) -> None:
        """Run the pipeline."""
        run(
            self.mock_pipeline,  # type: ignore[arg-type]
            self.inputs,
            self.run_folder,
            self.internal_shapes,
            parallel=self.parallel,
            cleanup=self.cleanup,
        )


def create_learners_from_sweep(
    pipeline: Pipeline,
    sweep: Sweep,
    run_folder: str | Path,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    parallel: bool = True,
    cleanup: bool = True,
) -> tuple[list[SequenceLearner], list[Path]]:
    """Create adaptive learners for a sweep.

    Creates an `adaptive.SequenceLearner` for each sweep run. These learners
    have a single iteration that executes the map in parallel. This means
    that here we rely on the internal parallelization of the pipeline. Each
    learner is fully independent of the others, and they can be executed in
    parallel.

    Note that this only parallelizes the nodes with a `MapSpec`, the rest of
    the nodes are executed in order. Only use this if the sequential execution
    of the nodes is not a bottleneck.

    Parameters
    ----------
    pipeline
        The pipeline to create learners for.
    sweep
        The sweep to create learners for, must generate ``input`` dictionaries as
        expected by `pipeline.map`.
    run_folder
        The folder to store the run information. Each sweep run will be stored in
        a subfolder of this folder.
    internal_shapes
        The internal shapes to use for the run, as expected by `pipeline.map`.
    parallel
        Whether to run the map in parallel.
    cleanup
        Whether to clean up the ``run_folder``.

    Returns
    -------
        A tuple of lists where the first list contains the learners and the second
        list contains the run folders for each sweep run.

    """
    run_folder = Path(run_folder)
    learners = []
    folders = []
    max_digits = len(str(len(sweep) - 1))
    for i, inputs in enumerate(sweep):
        sweep_run = run_folder / f"sweep_{str(i).zfill(max_digits)}"
        mock_pipeline = _MockPipeline.from_pipeline(pipeline)
        f = _MapWrapper(mock_pipeline, inputs, sweep_run, internal_shapes, parallel, cleanup)
        learner = SequenceLearner(f, sequence=[None])
        learners.append(learner)
        folders.append(sweep_run)
    return learners, folders


def _identify_cross_product_axes(pipeline: Pipeline) -> tuple[str, ...]:
    reduced = _reduced_axes(pipeline)
    impossible_axes: set[str] = set()  # Constructing this as a safety measure (for assert below)
    for func in pipeline.leaf_nodes:
        for output_name in pipeline.func_dependencies(func):
            for name in at_least_tuple(output_name):
                if name in reduced:
                    impossible_axes.update(reduced[name])

    possible_axes: set[str] = set()
    for func in pipeline.leaf_nodes:
        axes = pipeline.independent_axes_in_mapspecs(func.output_name)
        possible_axes.update(axes)

    assert not (possible_axes & impossible_axes)
    return tuple(sorted(possible_axes))


def _iterate_axes(
    independent_axes: tuple[str, ...],
    inputs: dict[str, Any],
    mapspec_axes: dict[str, tuple[str, ...]],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
) -> Generator[dict[str, Any], None, None]:
    shape: list[int] = []
    for axis in independent_axes:
        parameter, dim = next(
            (p, axes.index(axis))
            for p, axes in mapspec_axes.items()
            if axis in axes and p in inputs
        )
        shape.append(shapes[parameter][dim])

    for indices in _iterate_shape_indices(tuple(shape)):
        yield dict(zip(independent_axes, indices))


def _maybe_iterate_axes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    fixed_indices: dict[str, int | slice] | None,
    split_independent_axes: bool,  # noqa: FBT001
    internal_shapes: dict[str, int | tuple[int, ...]] | None,
) -> Generator[dict[str, Any] | None, None, None]:
    if fixed_indices:
        assert not split_independent_axes
        _validate_fixed_indices(fixed_indices, inputs, pipeline)
        yield fixed_indices
        return
    if not split_independent_axes:
        yield None
        return
    independent_axes = _identify_cross_product_axes(pipeline)
    axes = pipeline.mapspec_axes
    shapes = map_shapes(pipeline, inputs, internal_shapes).shapes
    for fixed_indices in _iterate_axes(independent_axes, inputs, axes, shapes):
        _validate_fixed_indices(fixed_indices, inputs, pipeline)
        yield fixed_indices
