from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path
from typing import Any, Literal, NamedTuple, Tuple, Union

import cloudpickle
import networkx as nx
import numpy as np

from pipefunc import PipeFunc, Pipeline
from pipefunc._filearray import FileArray
from pipefunc._mapspec import MapSpec

_OUTPUT_TYPE = Union[str, Tuple[str, ...]]


def _func_path(func: PipeFunc, folder: Path) -> Path:
    return folder / f"{func.__module__}.{func.__name__}.cloudpickle"


def _load(path: Path) -> Any:
    with path.open("rb") as f:
        return cloudpickle.load(f)


def _dump(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        cloudpickle.dump(obj, f)


_load_cached = functools.lru_cache(maxsize=None)(_load)


def _dump_functions(pipeline: Pipeline, run_folder: Path) -> list[Path]:
    folder = run_folder / "functions"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for func in pipeline.functions:
        path = _func_path(func, folder)
        _dump(func, path)
        paths.append(path)
    return paths


def _dump_inputs(inputs: dict[str, Any], run_folder: Path) -> dict[str, Path]:
    folder = run_folder / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, v in inputs.items():
        path = folder / f"{k}.cloudpickle"
        _dump(v, path)
        paths[k] = path
    return paths


def _load_input(name: str, input_paths: dict[str, Path]) -> Any:
    path = input_paths[name]
    return _load_cached(path)


def _output_path(output_name: str, folder: Path) -> Path:
    return folder / f"{output_name}.cloudpickle"


def _dump_output(output: Any, output_name: str, run_folder: Path) -> Path:
    folder = run_folder / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    path = _output_path(output_name, folder)
    _dump(output, path)
    return path


def _load_output(output_name: str, run_folder: Path) -> Any:
    folder = run_folder / "outputs"
    path = _output_path(output_name, folder)
    return _load_cached(path)


def _clean_run_folder(run_folder: Path) -> None:
    for folder in ["functions", "inputs", "outputs"]:
        shutil.rmtree(run_folder / folder, ignore_errors=True)


def _json_serializable(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    msg = f"Object {obj} is not JSON serializable"
    raise ValueError(msg)


def _dump_run_info(
    function_paths: list[Path],
    input_paths: dict[str, Path],
    run_folder: Path,
) -> None:
    path = run_folder / "run_info.json"
    info = {
        "functions": _json_serializable(function_paths),
        "inputs": _json_serializable(input_paths),
    }
    with path.open("w") as f:
        json.dump(info, f, indent=4)


def _file_array_path(output_name: _OUTPUT_TYPE, run_folder: Path) -> Path:
    if isinstance(output_name, tuple):
        output_name = "-".join(output_name)
    return run_folder / "outputs" / output_name


def _dump_file_array_shape(_file_array_path: Path, shape: tuple[int, ...]) -> None:
    path = _file_array_path / "shape"
    if not path.exists():
        with path.open("w") as f:
            json.dump(shape, f)


@functools.lru_cache(maxsize=None)
def _load_file_array_shape(_file_array_path: Path) -> tuple[int, ...]:
    path = _file_array_path / "shape"
    with path.open("r") as f:
        return tuple(json.load(f))


def _output_types(
    pipeline: Pipeline,
) -> dict[_OUTPUT_TYPE, Literal["single", "single_indexable", "file_array"]]:
    generations = list(nx.topological_generations(pipeline.graph))
    assert all(isinstance(x, str) for x in generations[0])
    otypes: dict[_OUTPUT_TYPE, Literal["single", "single_indexable", "file_array"]] = {}
    for gen in generations:
        for f in gen:
            if isinstance(f, str):
                otypes[f] = "single"
                continue
            assert isinstance(f, PipeFunc)
            if f.mapspec is None:
                otypes[f.output_name] = "single"
            else:
                otypes[f.output_name] = "file_array"
            for p in f.parameters:
                if (
                    otypes.get(p) == "single"
                    and f.mapspec is not None
                    and p in f.mapspec.parameters  # type: ignore[union-attr]
                ):
                    otypes[p] = "single_indexable"
    return otypes


def _func_kwargs(
    func: PipeFunc,
    output_types: dict[
        _OUTPUT_TYPE,
        Literal["single", "single_indexable", "file_array"],
    ],
    input_paths: dict[str, Path],
    run_folder: Path,
) -> dict[str, Any]:
    kwargs = {}
    for parameter in func.parameters:
        output_type = output_types[parameter]
        if parameter in input_paths:
            assert output_type in ("single", "single_indexable")
            value = _load_input(parameter, input_paths)
            kwargs[parameter] = value
        elif output_type in ("single_indexable", "single"):
            value = _load_output(parameter, run_folder)
            kwargs[parameter] = value
        else:
            assert output_type == "file_array"
            file_array_path = _file_array_path(parameter, run_folder)
            shape = _load_file_array_shape(file_array_path)
            file_array = FileArray(file_array_path, shape)
            kwargs[parameter] = file_array.to_array()
    return kwargs


def _execute_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    run_folder: Path,
) -> tuple[np.ndarray, dict[int, Path]]:
    assert isinstance(func.mapspec, MapSpec)
    shape = func.mapspec.shape_from_kwargs(kwargs)
    file_array_path = _file_array_path(func.output_name, run_folder)
    file_array = FileArray(file_array_path, shape)
    _dump_file_array_shape(file_array_path, shape)
    outputs = []
    output_paths = {}
    for index in range(np.prod(shape)):
        input_keys = {
            k: v[0] if len(v) == 1 else v
            for k, v in func.mapspec.input_keys(shape, index).items()
        }
        kwargs_ = {
            k: v[input_keys[k]] if k in input_keys else v for k, v in kwargs.items()
        }
        output = func(**kwargs_)
        output_key = func.mapspec.output_key(shape, index)
        file_array.dump(output_key, output)
        outputs.append(output)
        output_paths[index] = file_array._key_to_file(output_key)
    return np.asarray(outputs, dtype=object).reshape(shape), output_paths


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any


def run_pipeline(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
) -> list[Result]:
    run_folder = Path(run_folder)
    _clean_run_folder(run_folder)
    function_paths = _dump_functions(pipeline, run_folder)
    input_paths = _dump_inputs(inputs, run_folder)
    _dump_run_info(function_paths, input_paths, run_folder)
    output_types = _output_types(pipeline)

    generations = list(nx.topological_generations(pipeline.graph))
    assert all(isinstance(x, str) for x in generations[0])
    outputs = []
    for gen in generations[1:]:
        # These evaluations can happen in parallel
        for func in gen:
            kwargs = _func_kwargs(func, output_types, input_paths, run_folder)
            if func.mapspec:
                output, _ = _execute_map_spec(func, kwargs, run_folder)
            else:
                output = func(**kwargs)
                _dump_output(output, func.output_name, run_folder)
            outputs.append(
                Result(
                    function=func.__name__,
                    kwargs=kwargs,
                    output_name=func.output_name,
                    output=output,
                ),
            )
    return outputs
