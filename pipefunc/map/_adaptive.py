from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple, Union

from pipefunc._utils import prod
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import (
    RunInfo,
    _execute_single,
    _func_kwargs,
    _init_file_arrays,
    _run_iteration_and_pick_output,
    _update_file_array,
)

if TYPE_CHECKING:
    import adaptive

    from pipefunc import PipeFunc, Pipeline

_OUTPUT_TYPE = Union[str, Tuple[str, ...]]


def make_learners(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    return_output: bool = False,
) -> list[dict[_OUTPUT_TYPE, adaptive.SequenceLearner]]:
    import adaptive

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
    outputs = _run_iteration_and_pick_output(func, kwargs, shape, index)
    _update_file_array(func, file_arrays, shape, index, outputs)
    return outputs if return_output else None
