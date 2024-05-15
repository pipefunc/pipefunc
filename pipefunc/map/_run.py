from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Tuple, Union

import numpy as np

from pipefunc._utils import at_least_tuple, dump, handle_error, load, prod
from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec, array_shape

if TYPE_CHECKING:
    from pipefunc import PipeFunc, Pipeline

_OUTPUT_TYPE = Union[str, Tuple[str, ...]]


def _dump_inputs(
    inputs: dict[str, Any],
    defaults: dict[str, Any],
    run_folder: Path,
) -> dict[str, Path]:
    folder = run_folder / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in defaults.items():
        inputs.setdefault(k, v)
    paths = {}
    for k, v in inputs.items():
        path = folder / f"{k}.cloudpickle"
        dump(v, path)
        paths[k] = path
    return paths


def _load_input(name: str, input_paths: dict[str, Path]) -> Any:
    path = input_paths[name]
    return load(path, cache=True)


def _output_path(output_name: str, folder: Path) -> Path:
    return folder / f"{output_name}.cloudpickle"


def _dump_output(func: PipeFunc, output: Any, run_folder: Path) -> Any:
    folder = run_folder / "outputs"
    folder.mkdir(parents=True, exist_ok=True)

    if isinstance(func.output_name, tuple):
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            path = _output_path(output_name, folder)
            dump(output, path)
        output = new_output
    else:
        path = _output_path(func.output_name, folder)
        dump(output, path)

    return output


def _load_output(output_name: str, run_folder: Path) -> Any:
    folder = run_folder / "outputs"
    path = _output_path(output_name, folder)
    return load(path)


class RunInfo(NamedTuple):
    input_paths: dict[str, Path]
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]
    manual_shapes: dict[str, int | tuple[int, ...]]
    run_folder: Path

    @classmethod
    def create(
        cls: type[RunInfo],
        run_folder: str | Path,
        pipeline: Pipeline,
        inputs: dict[str, Any],
        manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        cleanup: bool = True,
    ) -> RunInfo:
        run_folder = Path(run_folder)
        manual_shapes = manual_shapes or {}
        if cleanup:
            shutil.rmtree(run_folder, ignore_errors=True)
        input_paths = _dump_inputs(inputs, pipeline.defaults, run_folder)
        shapes = map_shapes(pipeline, inputs, manual_shapes)
        return cls(
            input_paths=input_paths,
            shapes=shapes,
            manual_shapes=manual_shapes,
            run_folder=run_folder,
        )

    def dump(self, run_folder: str | Path) -> None:
        path = Path(run_folder) / "run_info.cloudpickle"
        dump(self._asdict(), path)

    @classmethod
    def load(
        cls: type[RunInfo],
        run_folder: str | Path,
        *,
        cache: bool = True,
    ) -> RunInfo:
        path = Path(run_folder) / "run_info.cloudpickle"
        dct = load(path, cache=cache)
        return cls(**dct)


def _file_array_path(output_name: str, run_folder: Path) -> Path:
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


def _load_parameter(
    parameter: str,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    manual_shapes: dict[str, int | tuple[int, ...]],
    run_folder: Path,
) -> Any:
    if parameter in input_paths:
        return _load_input(parameter, input_paths)
    if parameter in manual_shapes or parameter not in shapes:
        return _load_output(parameter, run_folder)
    file_array_path = _file_array_path(parameter, run_folder)
    shape = shapes[parameter]
    return FileArray(file_array_path, shape)


def _func_kwargs(
    func: PipeFunc,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    manual_shapes: dict[str, int | tuple[int, ...]],
    run_folder: Path,
) -> dict[str, Any]:
    return {
        p: _load_parameter(p, input_paths, shapes, manual_shapes, run_folder)
        for p in func.parameters
    }


def _select_kwargs(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    input_keys = {
        k: v[0] if len(v) == 1 else v
        for k, v in func.mapspec.input_keys(shape, index).items()
    }
    selected = {
        k: v[input_keys[k]] if k in input_keys else v for k, v in kwargs.items()
    }
    _load_file_array(selected)
    return selected


def _init_file_arrays(
    output_name: _OUTPUT_TYPE,
    shape: tuple[int, ...],
    run_folder: Path,
) -> list[FileArray]:
    return [
        FileArray(_file_array_path(output_name, run_folder), shape)
        for output_name in at_least_tuple(output_name)
    ]


def _init_result_arrays(
    output_name: _OUTPUT_TYPE,
    shape: tuple[int, ...],
) -> list[np.ndarray]:
    return [np.empty(prod(shape), dtype=object) for _ in at_least_tuple(output_name)]


def _pick_output(func: PipeFunc, output: Any) -> list[Any]:
    return [
        (
            func.output_picker(output, output_name)
            if func.output_picker is not None
            else output
        )
        for output_name in at_least_tuple(func.output_name)
    ]


def _run_iteration(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    index: int,
) -> Any:
    selected = _select_kwargs(func, kwargs, shape, index)
    try:
        return func(**selected)
    except Exception as e:
        handle_error(e, func, selected)
        raise  # handle_error raises but mypy doesn't know that


def _run_iteration_and_pick_output(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    index: int,
) -> list[Any]:
    output = _run_iteration(func, kwargs, shape, index)
    return _pick_output(func, output)


def _update_file_array(
    func: PipeFunc,
    file_arrays: list[FileArray],
    shape: tuple[int, ...],
    index: int,
    output: list[Any],
) -> None:
    assert isinstance(func.mapspec, MapSpec)
    output_key = func.mapspec.output_key(shape, index)
    for file_array, _output in zip(file_arrays, output):
        file_array.dump(output_key, _output)


def _update_result_array(
    result_arrays: list[np.ndarray],
    index: int,
    output: list[Any],
) -> None:
    for result_array, _output in zip(result_arrays, output):
        result_array[index] = _output


def _execute_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    run_folder: Path,
) -> np.ndarray | list[np.ndarray]:
    assert isinstance(func.mapspec, MapSpec)
    shape = shapes[func.output_name]
    n = prod(shape)
    file_arrays = _init_file_arrays(func.output_name, shape, run_folder)
    result_arrays = _init_result_arrays(func.output_name, shape)
    for index in range(n):
        outputs = _run_iteration_and_pick_output(func, kwargs, shape, index)
        _update_file_array(func, file_arrays, shape, index, outputs)
        _update_result_array(result_arrays, index, outputs)
    result_arrays = [x.reshape(shape) for x in result_arrays]
    return result_arrays if isinstance(func.output_name, tuple) else result_arrays[0]


def _execute_single(func: PipeFunc, kwargs: dict[str, Any], run_folder: Path) -> Any:
    _load_file_array(kwargs)
    try:
        output = func(**kwargs)
    except Exception as e:
        handle_error(e, func, kwargs)
        raise  # handle_error raises but mypy doesn't know that
    return _dump_output(func, output, run_folder)


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> dict[_OUTPUT_TYPE, tuple[int, ...]]:
    if manual_shapes is None:
        manual_shapes = {}
    map_parameters: set[str] = pipeline.map_parameters

    input_parameters = set(pipeline.topological_generations[0])

    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]] = {
        p: array_shape(inputs[p]) for p in input_parameters if p in map_parameters
    }
    mapspec_funcs = [
        f for gen in pipeline.topological_generations[1] for f in gen if f.mapspec
    ]
    for func in mapspec_funcs:
        assert func.mapspec is not None
        input_shapes = {}
        for p in func.mapspec.parameters:
            if shape := shapes.get(p):
                input_shapes[p] = shape
            elif p in manual_shapes:
                input_shapes[p] = at_least_tuple(manual_shapes[p])
            else:
                msg = (
                    f"Parameter `{p}` is used in map but its shape"
                    " cannot be inferred from the inputs."
                    " Provide the shape manually in `manual_shapes`."
                )
                raise ValueError(msg)
        output_shape = func.mapspec.shape(input_shapes)
        shapes[func.output_name] = output_shape
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape

    assert all(k in shapes for k in map_parameters if k not in manual_shapes)
    return shapes


def _maybe_load_file_array(x: Any) -> Any:
    if isinstance(x, FileArray):
        return x.to_array()
    return x


def _load_file_array(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = _maybe_load_file_array(v)


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any


def _run_function(func: PipeFunc, run_folder: Path) -> list[Result]:
    run_info = RunInfo.load(run_folder)
    kwargs = _func_kwargs(
        func,
        run_info.input_paths,
        run_info.shapes,
        run_info.manual_shapes,
        run_folder,
    )
    if func.mapspec:
        output = _execute_map_spec(func, kwargs, run_info.shapes, run_folder)
    else:
        output = _execute_single(func, kwargs, run_folder)

    if isinstance(func.output_name, str):
        return [
            Result(
                function=func.__name__,
                kwargs=kwargs,
                output_name=func.output_name,
                output=output,
            ),
        ]

    return [
        Result(
            function=func.__name__,
            kwargs=kwargs,
            output_name=output_name,
            output=_output,
        )
        for output_name, _output in zip(func.output_name, output)
    ]


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
    manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> list[Result]:
    run_folder = Path(run_folder)
    run_info = RunInfo.create(run_folder, pipeline, inputs, manual_shapes)
    run_info.dump(run_folder)
    outputs = []
    for gen in pipeline.topological_generations[1]:
        # These evaluations can happen in parallel
        for func in gen:
            _outputs = _run_function(func, run_folder)
            outputs.extend(_outputs)
    return outputs


def load_outputs(
    *output_names: str,
    run_folder: str | Path,
) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    outputs = [
        _load_parameter(
            on,
            run_info.input_paths,
            run_info.shapes,
            run_info.manual_shapes,
            run_folder,
        )
        for on in output_names
    ]
    outputs = [_maybe_load_file_array(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs
