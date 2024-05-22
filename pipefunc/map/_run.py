from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Tuple, Union

import numpy as np

from pipefunc._utils import at_least_tuple, dump, equal_dicts, handle_error, load, prod
from pipefunc._version import __version__
from pipefunc.map._filearray import FileArray, _iterate_shape_indices, _select_by_mask
from pipefunc.map._mapspec import (
    MapSpec,
    _shape_to_key,
    array_shape,
    mapspec_dimensions,
    validate_consistent_axes,
)

if TYPE_CHECKING:
    import sys

    from pipefunc import PipeFunc, Pipeline

    if sys.version_info < (3, 10):  # pragma: no cover
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias

_OUTPUT_TYPE: TypeAlias = Union[str, Tuple[str, ...]]


@dataclass
class _MockPipeline:
    """An object that contains all information required to run a pipeline.

    Ensures that we're not pickling the entire pipeline object when not needed.
    """

    defaults: dict[str, Any]
    map_parameters: set[str]
    topological_generations: tuple[list[str], list[list[PipeFunc]]]

    @classmethod
    def from_pipeline(cls: type[_MockPipeline], pipeline: Pipeline) -> _MockPipeline:  # noqa: PYI019
        return cls(
            defaults=pipeline.defaults,
            map_parameters=pipeline.map_parameters,
            topological_generations=pipeline.topological_generations,
        )

    @property
    def functions(self) -> list[PipeFunc]:
        # Return all functions in topological order
        return [f for gen in self.topological_generations[1] for f in gen]

    def mapspecs(self, *, ordered: bool = True) -> list[MapSpec]:  # noqa: ARG002
        """Return the MapSpecs for all functions in the pipeline."""
        functions = self.functions  # topologically ordered
        return [f.mapspec for f in functions if f.mapspec]

    @property
    def sorted_functions(self) -> list[PipeFunc]:
        """Return the functions in the pipeline in topological order."""
        return self.functions

    def mapspec_dimensions(self) -> dict[str, int]:
        """Return the number of dimensions for each array parameter in the pipeline."""
        return mapspec_dimensions(self.mapspecs())


def _dump_inputs(
    inputs: dict[str, Any],
    defaults: dict[str, Any],
    run_folder: Path,
) -> dict[str, Path]:
    folder = run_folder / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    to_dump = dict(defaults, **inputs)
    for k, v in to_dump.items():
        path = folder / f"{k}.cloudpickle"
        dump(v, path)
        paths[k] = path
    return paths


def _load_input(name: str, input_paths: dict[str, Path]) -> Any:
    path = input_paths[name]
    return load(path, cache=True)


def _output_path(output_name: str, run_folder: Path) -> Path:
    return run_folder / "outputs" / f"{output_name}.cloudpickle"


def _dump_output(func: PipeFunc, output: Any, run_folder: Path) -> Any:
    folder = run_folder / "outputs"
    folder.mkdir(parents=True, exist_ok=True)

    if isinstance(func.output_name, tuple):
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            path = _output_path(output_name, run_folder)
            dump(_output, path)
        output = new_output
    else:
        path = _output_path(func.output_name, run_folder)
        dump(output, path)

    return output


def _load_output(output_name: str, run_folder: Path) -> Any:
    path = _output_path(output_name, run_folder)
    return load(path)


def cleanup_run_folder(run_folder: str | Path) -> None:
    """Remove the run folder and its contents."""
    run_folder = Path(run_folder)
    shutil.rmtree(run_folder, ignore_errors=True)


def _compare_to_previous_run_info(
    pipeline: Pipeline,
    run_folder: Path,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> None:
    if not RunInfo.path(run_folder).is_file():
        return
    old = RunInfo.load(run_folder, cache=False)
    if internal_shapes != old.internal_shapes:
        msg = "Internal shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)
    if shapes != old.shapes:
        msg = "Shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    old_inputs = {k: _load_input(k, old.input_paths) for k in inputs}
    equal_inputs = equal_dicts(inputs, old_inputs, verbose=True)
    if equal_inputs is None:
        print(
            "Could not compare new `inputs` to `inputs` from previous run."
            " Proceeding without `cleanup`, hoping for the best.",
        )
        return
    if not equal_inputs:
        msg = "Inputs do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)


def _check_inputs(pipeline: Pipeline, inputs: dict[str, Any]) -> None:
    input_dimensions = pipeline.mapspec_dimensions()
    for name, value in inputs.items():
        if (dim := input_dimensions.get(name, 0)) > 1 and isinstance(value, (list, tuple)):
            msg = f"Expected {dim}D `numpy.ndarray` for input `{name}`, got {type(value)}."
            raise ValueError(msg)


class RunInfo(NamedTuple):
    input_paths: dict[str, Path]
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]
    internal_shapes: dict[str, int | tuple[int, ...]] | None
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]
    run_folder: Path
    pipefunc_version: str = __version__

    @classmethod
    def create(
        cls: type[RunInfo],
        run_folder: str | Path,
        pipeline: Pipeline,
        inputs: dict[str, Any],
        internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        cleanup: bool = True,
    ) -> RunInfo:
        run_folder = Path(run_folder)
        if cleanup:
            cleanup_run_folder(run_folder)
        else:
            _compare_to_previous_run_info(pipeline, run_folder, inputs, internal_shapes)
        _check_inputs(pipeline, inputs)
        input_paths = _dump_inputs(inputs, pipeline.defaults, run_folder)
        shapes, masks = map_shapes(pipeline, inputs, internal_shapes or {})
        return cls(
            input_paths=input_paths,
            shapes=shapes,
            internal_shapes=internal_shapes,
            shape_masks=masks,
            run_folder=run_folder,
        )

    def dump(self, run_folder: str | Path) -> None:
        path = self.path(run_folder)
        dump(self._asdict(), path)

    @classmethod
    def load(cls: type[RunInfo], run_folder: str | Path, *, cache: bool = True) -> RunInfo:
        path = cls.path(run_folder)
        dct = load(path, cache=cache)
        return cls(**dct)

    @staticmethod
    def path(run_folder: str | Path) -> Path:
        return Path(run_folder) / "run_info.cloudpickle"


def _file_array_path(output_name: str, run_folder: Path) -> Path:
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


def _load_parameter(
    parameter: str,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    run_folder: Path,
) -> Any:
    if parameter in input_paths:
        return _load_input(parameter, input_paths)
    if parameter not in shapes or not any(shape_masks[parameter]):
        return _load_output(parameter, run_folder)
    file_array_path = _file_array_path(parameter, run_folder)
    external_shape = _external_shape(shapes[parameter], shape_masks[parameter])
    internal_shape = _internal_shape(shapes[parameter], shape_masks[parameter])
    return FileArray(
        file_array_path,
        external_shape,
        internal_shape,
        shape_masks[parameter],
    )


def _func_kwargs(
    func: PipeFunc,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    run_folder: Path,
) -> dict[str, Any]:
    return {
        p: _load_parameter(p, input_paths, shapes, shape_masks, run_folder) for p in func.parameters
    }


def _select_kwargs(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    input_keys = {
        k: v[0] if len(v) == 1 else v for k, v in func.mapspec.input_keys(shape, index).items()
    }
    selected = {k: v[input_keys[k]] if k in input_keys else v for k, v in kwargs.items()}
    _load_file_array(selected)
    return selected


def _internal_shape(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if not m)


def _external_shape(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if m)


def _init_file_arrays(
    output_name: _OUTPUT_TYPE,
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
    run_folder: Path,
) -> list[FileArray]:
    external_shape = _external_shape(shape, mask)
    internal_shape = _internal_shape(shape, mask)
    return [
        FileArray(
            _file_array_path(output_name, run_folder),
            external_shape,
            internal_shape,
            mask,
        )
        for output_name in at_least_tuple(output_name)
    ]


def _init_result_arrays(output_name: _OUTPUT_TYPE, shape: tuple[int, ...]) -> list[np.ndarray]:
    return [np.empty(prod(shape), dtype=object) for _ in at_least_tuple(output_name)]


def _pick_output(func: PipeFunc, output: Any) -> list[Any]:
    return [
        (func.output_picker(output, output_name) if func.output_picker is not None else output)
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


def _run_iteration_and_process(
    index: int,
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    file_arrays: list[FileArray],
) -> list[Any]:
    output = _run_iteration(func, kwargs, shape, index)
    outputs = _pick_output(func, output)
    _update_file_array(func, file_arrays, shape, shape_mask, index, outputs)
    return outputs


def _update_file_array(
    func: PipeFunc,
    file_arrays: list[FileArray],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
    outputs: list[Any],
) -> None:
    assert isinstance(func.mapspec, MapSpec)
    external_shape = _external_shape(shape, shape_mask)
    output_key = func.mapspec.output_key(external_shape, index)
    for file_array, _output in zip(file_arrays, outputs):
        file_array.dump(output_key, _output)


def _indices_to_flat_index(
    shape: tuple[int, ...],
    internal_shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    external_index: tuple[int, ...],
    internal_index: tuple[int, ...],
) -> np.int_:
    full_index = _select_by_mask(shape_mask, external_index, internal_index)
    full_shape = _select_by_mask(shape_mask, shape, internal_shape)
    return np.ravel_multi_index(full_index, full_shape)


def _set_output(
    arr: np.ndarray,
    output: np.ndarray,
    linear_index: int,
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
) -> None:
    external_shape = _external_shape(shape, shape_mask)
    internal_shape = _internal_shape(shape, shape_mask)
    external_index = _shape_to_key(external_shape, linear_index)
    for internal_index in _iterate_shape_indices(internal_shape):
        flat_index = _indices_to_flat_index(
            external_shape,
            internal_shape,
            shape_mask,
            external_index,
            internal_index,
        )
        arr[flat_index] = output[internal_index]


def _update_result_array(
    result_arrays: list[np.ndarray],
    index: int,
    output: list[Any],
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
) -> None:
    for result_array, _output in zip(result_arrays, output):
        if not all(mask):
            _output = np.asarray(_output)  # In case _output is a list
            _set_output(result_array, _output, index, shape, mask)
        else:
            result_array[index] = _output


def _existing_and_missing_indices(file_arrays: list[FileArray]) -> tuple[list[int], list[int]]:
    masks = (arr._mask_list() for arr in file_arrays)
    existing_indices = []
    missing_indices = []
    for i, mask_values in enumerate(zip(*masks)):
        if any(mask_values):  # rerun if any of the outputs are missing
            missing_indices.append(i)
        else:
            existing_indices.append(i)
    return existing_indices, missing_indices


def _execute_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    run_folder: Path,
    parallel: bool,  # noqa: FBT001
) -> np.ndarray | list[np.ndarray]:
    assert isinstance(func.mapspec, MapSpec)
    shape = shapes[func.output_name]
    mask = shape_masks[func.output_name]
    file_arrays = _init_file_arrays(func.output_name, shape, mask, run_folder)
    result_arrays = _init_result_arrays(func.output_name, shape)
    process_index = partial(
        _run_iteration_and_process,
        func=func,
        kwargs=kwargs,
        shape=shape,
        shape_mask=mask,
        file_arrays=file_arrays,
    )
    existing, missing = _existing_and_missing_indices(file_arrays)
    n = len(missing)
    if parallel and n > 1:
        with ProcessPoolExecutor() as ex:
            outputs_list = list(ex.map(process_index, missing))
    else:
        outputs_list = [process_index(index) for index in missing]

    for index, outputs in zip(missing, outputs_list):
        _update_result_array(result_arrays, index, outputs, shape, mask)

    for index in existing:
        outputs = [file_array.get_from_index(index) for file_array in file_arrays]
        _update_result_array(result_arrays, index, outputs, shape, mask)

    result_arrays = [x.reshape(shape) for x in result_arrays]
    return result_arrays if isinstance(func.output_name, tuple) else result_arrays[0]


def _maybe_load_single_output(
    func: PipeFunc,
    run_folder: Path,
    *,
    return_output: bool = True,
) -> tuple[Any, bool]:
    """Load the output if it exists.

    Returns the output and a boolean indicating whether the output exists.
    """
    output_paths = [_output_path(p, run_folder) for p in at_least_tuple(func.output_name)]
    if all(p.is_file() for p in output_paths):
        if not return_output:
            return None, True
        outputs = [load(p) for p in output_paths]
        if isinstance(func.output_name, tuple):
            return outputs, True
        return outputs[0], True
    return None, False


def _execute_single(func: PipeFunc, kwargs: dict[str, Any], run_folder: Path) -> Any:
    # Load the output if it exists
    output, exists = _maybe_load_single_output(func, run_folder)
    if exists:
        return output

    # Otherwise, run the function
    _load_file_array(kwargs)
    try:
        output = func(**kwargs)
    except Exception as e:
        handle_error(e, func, kwargs)
        raise  # handle_error raises but mypy doesn't know that
    return _dump_output(func, output, run_folder)


class Shapes(NamedTuple):
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]
    masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> Shapes:
    if internal_shapes is None:
        internal_shapes = {}
    internal = {k: at_least_tuple(v) for k, v in internal_shapes.items()}

    map_parameters: set[str] = pipeline.map_parameters

    input_parameters = set(pipeline.topological_generations[0])

    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]] = {
        p: array_shape(inputs[p]) for p in input_parameters if p in map_parameters
    }
    masks = {name: len(shape) * (True,) for name, shape in shapes.items()}
    mapspec_funcs = [f for f in pipeline.sorted_functions if f.mapspec]
    for func in mapspec_funcs:
        assert func.mapspec is not None  # mypy
        input_shapes = {p: shapes[p] for p in func.mapspec.input_names if p in shapes}
        output_shapes = {p: internal[p] for p in func.mapspec.output_names if p in internal}
        output_shape, mask = func.mapspec.shape(input_shapes, output_shapes)  # type: ignore[arg-type]
        shapes[func.output_name] = output_shape
        masks[func.output_name] = mask
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape
                masks[output_name] = mask

    assert all(k in shapes for k in map_parameters if k not in internal)
    return Shapes(shapes, masks)


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


def _run_function(func: PipeFunc, run_folder: Path, parallel: bool) -> list[Result]:  # noqa: FBT001
    run_info = RunInfo.load(run_folder)
    kwargs = _func_kwargs(
        func,
        run_info.input_paths,
        run_info.shapes,
        run_info.shape_masks,
        run_folder,
    )
    if func.mapspec and func.mapspec.inputs:
        output = _execute_map_spec(
            func,
            kwargs,
            run_info.shapes,
            run_info.shape_masks,
            run_folder,
            parallel,
        )
    else:
        output = _execute_single(func, kwargs, run_folder)

    # Note that the kwargs still contain the FileArray objects if _execute_map_spec
    # was used.
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
        Result(function=func.__name__, kwargs=kwargs, output_name=output_name, output=_output)
        for output_name, _output in zip(func.output_name, output)
    ]


def _ensure_run_folder(run_folder: str | Path | None) -> Path:
    if run_folder is None:
        tmp_dir = tempfile.mkdtemp()
        run_folder = Path(tmp_dir)
    return Path(run_folder)


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    parallel: bool = True,
    cleanup: bool = True,
) -> list[Result]:
    """Run a pipeline with `MapSpec` functions for given `inputs`.

    Parameters
    ----------
    pipeline
        The pipeline to run.
    inputs
        The inputs to the pipeline. The keys should be the names of the input
        parameters of the pipeline functions and the values should be the
        corresponding input data, these are either single values for functions without `mapspec`
        or lists of values or `numpy.ndarray`s for functions with `mapspec`.
    run_folder
        The folder to store the run information. If `None`, a temporary folder
        is created.
    internal_shapes
        The shapes for intermediary outputs that cannot be inferred from the inputs.
        You will receive an exception if the shapes cannot be inferred and need to be provided.
    parallel
        Whether to run the functions in parallel.
    cleanup
        Whether to clean up the `run_folder` before running the pipeline.

    """
    validate_consistent_axes(pipeline.mapspecs(ordered=False))
    run_folder = _ensure_run_folder(run_folder)
    run_info = RunInfo.create(run_folder, pipeline, inputs, internal_shapes, cleanup=cleanup)
    run_info.dump(run_folder)
    outputs = []
    for gen in pipeline.topological_generations[1]:
        # These evaluations *can* happen in parallel
        for func in gen:
            _outputs = _run_function(func, run_folder, parallel)
            outputs.extend(_outputs)
    return outputs


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    outputs = [
        _load_parameter(
            on,
            run_info.input_paths,
            run_info.shapes,
            run_info.shape_masks,
            run_folder,
        )
        for on in output_names
    ]
    outputs = [_maybe_load_file_array(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs
