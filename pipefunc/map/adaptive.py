"""Provides functions to create adaptive learners for a pipeline."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Tuple, Union

import adaptive

from pipefunc._utils import prod
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import (
    RunInfo,
    _execute_single,
    _func_kwargs,
    _init_file_arrays,
    _MockPipeline,
    _run_iteration_and_pick_output,
    _update_file_array,
    run,
)

if TYPE_CHECKING:
    import sys

    from pipefunc import PipeFunc, Pipeline, Sweep

    if sys.version_info < (3, 10):  # pragma: no cover
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias


_OUTPUT_TYPE: TypeAlias = Union[str, Tuple[str, ...]]


def create_learners(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    return_output: bool = False,
) -> list[dict[_OUTPUT_TYPE, adaptive.SequenceLearner]]:
    """Create adaptive learners for a single `Pipeline.map` call.

    Creates a learner for each function node in the graph. Which means that
    the returned lists of learners have to be executed in order.
    If a single list contains multiple learners, they can be executed in
    parallel.

    Parameters
    ----------
    pipeline
        The pipeline to create learners for.
    inputs
        The inputs to the pipeline, the same as passed to `pipeline.map`.
    run_folder
        The folder to store the run information.
    manual_shapes
        The manual shapes to use for the run.
    return_output
        Whether to return the output of the function in the learner.

    Returns
    -------
    A list of dictionaries where the keys are the output names of the
    functions and the values are the corresponding adaptive learners. As noted
    above, the learners have to be executed in order.

    """
    run_folder = Path(run_folder)
    run_info = RunInfo.create(run_folder, pipeline, inputs, manual_shapes)
    run_info.dump(run_folder)
    learners = []
    for gen in pipeline.topological_generations[1]:
        _learners = {}
        for func in gen:
            if func.mapspec:
                f = functools.partial(
                    _execute_iteration_in_map_spec,
                    func=func,
                    run_info=run_info,
                    run_folder=run_folder,
                    return_output=return_output,
                )
                sequence = list(range(prod(run_info.shapes[func.output_name])))
            else:
                f = functools.partial(
                    _execute_iteration_in_single,
                    func=func,
                    run_info=run_info,
                    run_folder=run_folder,
                    return_output=return_output,
                )
                sequence = [None]  # type: ignore[list-item]
            learner = adaptive.SequenceLearner(f, sequence)
            _learners[func.output_name] = learner
        learners.append(_learners)
    return learners


def _execute_iteration_in_single(
    _: Any,
    func: PipeFunc,
    run_info: RunInfo,
    run_folder: Path,
    *,
    return_output: bool = False,
) -> Any | None:
    """Execute a single iteration of a single function.

    Meets the requirements of `adaptive.SequenceLearner`.
    """
    kwargs = _func_kwargs(
        func,
        run_info.input_paths,
        run_info.shapes,
        run_info.manual_shapes,
        run_folder,
    )
    result = _execute_single(func, kwargs, run_folder)
    return result if return_output else None


def _execute_iteration_in_map_spec(
    index: int,
    func: PipeFunc,
    run_info: RunInfo,
    run_folder: Path,
    *,
    return_output: bool = False,
) -> Any | None:
    """Execute a single iteration of a map spec.

    Performs a single iteration of the code in `_execute_map_spec`, however,
    it does not keep and return the output. This is meant to be used in the
    parallel execution of the map spec.

    Meets the requirements of `adaptive.SequenceLearner`.
    """
    assert isinstance(func.mapspec, MapSpec)
    kwargs = _func_kwargs(
        func,
        run_info.input_paths,
        run_info.shapes,
        run_info.manual_shapes,
        run_folder,
    )
    shape = run_info.shapes[func.output_name]
    file_arrays = _init_file_arrays(func.output_name, shape, run_folder)
    outputs = _run_iteration_and_pick_output(index, func, kwargs, shape)
    _update_file_array(func, file_arrays, shape, index, outputs)
    return outputs if return_output else None


def _map_wrapper(
    pipeline: _MockPipeline,
    inputs: dict[str, Any],
    run_folder: Path,
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> Callable[[Any], None]:
    """Wraps the `pipefunc.map.run` function and makes it a callable with a single unused argument.

    Uses a `_MockPipeline` that contains all the required information to run the pipeline but is
    cheaper to serialize and pass around.
    """

    def wrapped(_: Any) -> None:
        run(pipeline, inputs, run_folder=run_folder, manual_shapes=manual_shapes)  # type: ignore[arg-type]

    return wrapped


def create_learners_from_sweep(
    pipeline: Pipeline,
    sweep: Sweep,
    run_folder: str | Path,
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> tuple[list[adaptive.SequenceLearner], list[Path]]:
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
        The sweep to create learners for, must generate `input` dictionaries as
        expected by `pipeline.map`.
    run_folder
        The folder to store the run information. Each sweep run will be stored in
        a subfolder of this folder.
    manual_shapes
        The manual shapes to use for the run, as expected by `pipeline.map`.

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
        f = _map_wrapper(mock_pipeline, inputs, sweep_run, manual_shapes)
        learner = adaptive.SequenceLearner(f, sequence=[None])
        learners.append(learner)
        folders.append(sweep_run)
    return learners, folders
