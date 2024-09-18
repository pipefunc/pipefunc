from __future__ import annotations

import functools
import itertools
import time
import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from pipefunc._utils import at_least_tuple, dump, handle_error, load, prod
from pipefunc.cache import HybridCache, to_hashable
from pipefunc.map._mapspec import MapSpec, _shape_to_key, validate_consistent_axes
from pipefunc.map._run_info import DirectValue, RunInfo, _external_shape, _internal_shape
from pipefunc.map._storage_base import StorageBase, _iterate_shape_indices, _select_by_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import xarray as xr

    from pipefunc import PipeFunc, Pipeline
    from pipefunc.cache import _CacheBase


_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]


def _cannot_be_parallelized(pipeline: Pipeline) -> bool:
    return all(f.mapspec is None for f in pipeline.functions) and all(
        len(fs) == 1 for fs in pipeline.topological_generations.function_lists
    )


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None = None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    output_names: set[_OUTPUT_TYPE] | None = None,
    parallel: bool = True,
    executor: Executor | None = None,
    storage: str = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
) -> OrderedDict[str, Result]:
    """Run a pipeline with `MapSpec` functions for given ``inputs``.

    Parameters
    ----------
    pipeline
        The pipeline to run.
    inputs
        The inputs to the pipeline. The keys should be the names of the input
        parameters of the pipeline functions and the values should be the
        corresponding input data, these are either single values for functions without ``mapspec``
        or lists of values or `numpy.ndarray`s for functions with ``mapspec``.
    run_folder
        The folder to store the run information. If ``None``, either a temporary folder
        is created or no folder is used, depending on whether the storage class requires serialization.
    internal_shapes
        The shapes for intermediary outputs that cannot be inferred from the inputs.
        You will receive an exception if the shapes cannot be inferred and need to be provided.
    output_names
        The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
    parallel
        Whether to run the functions in parallel. Is ignored if provided ``executor`` is not ``None``.
    executor
        The executor to use for parallel execution. If ``None``, a `ProcessPoolExecutor`
        is used. Only relevant if ``parallel=True``.
    storage
        The storage class to use for the file arrays.
        Can use any registered storage class. See `pipefunc.map.storage_registry`.
    persist_memory
        Whether to write results to disk when memory based storage is used.
        Does not have any effect when file based storage is used.
    cleanup
        Whether to clean up the ``run_folder`` before running the pipeline.
    fixed_indices
        A dictionary mapping axes names to indices that should be fixed for the run.
        If not provided, all indices are iterated over.
    auto_subpipeline
        If ``True``, a subpipeline is created with the specified ``inputs``, using
        `Pipeline.subpipeline`. This allows to provide intermediate results in the ``inputs`` instead
        of providing the root arguments. If ``False``, all root arguments must be provided,
        and an exception is raised if any are missing.

    """
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
    if _cannot_be_parallelized(pipeline):
        parallel = False
    _check_parallel(parallel, store, executor)

    with _maybe_executor(executor, parallel) as ex:
        for gen in pipeline.topological_generations.function_lists:
            _run_and_process_generation(
                generation=gen,
                run_info=run_info,
                store=store,
                outputs=outputs,
                fixed_indices=fixed_indices,
                executor=ex,
                cache=pipeline.cache,
            )

    if persist_memory:  # Only relevant for memory based storage
        for arr in store.values():
            if isinstance(arr, StorageBase):
                arr.persist()

    return outputs


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StorageBase | Path | DirectValue


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    store = run_info.init_store()
    outputs = [_load_from_store(output_name, store).value for output_name in output_names]
    outputs = [_maybe_load_array(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs


def load_xarray_dataset(
    *output_name: str,
    run_folder: str | Path,
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the output(s) of a `pipeline.map` as an `xarray.Dataset`.

    Parameters
    ----------
    output_name
        The names of the outputs to load. If empty, all outputs are loaded.
    run_folder
        The folder where the pipeline run was stored.
    load_intermediate
        Whether to load intermediate outputs as coordinates.

    Returns
    -------
        An `xarray.Dataset` containing the outputs of the pipeline run.

    """
    from pipefunc.map.xarray import load_xarray_dataset

    run_info = RunInfo.load(run_folder)
    return load_xarray_dataset(
        run_info.mapspecs,
        run_info.inputs,
        run_folder=run_folder,
        output_names=output_name,  # type: ignore[arg-type]
        load_intermediate=load_intermediate,
    )


def _dump_single_output(
    func: PipeFunc,
    output: Any,
    store: dict[str, StorageBase | Path | DirectValue],
) -> tuple[Any, ...]:
    if isinstance(func.output_name, tuple):
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            _single_dump_single_output(_output, output_name, store)
        return tuple(new_output)
    _single_dump_single_output(output, func.output_name, store)
    return (output,)


def _single_dump_single_output(
    output: Any,
    output_name: str,
    store: dict[str, StorageBase | Path | DirectValue],
) -> None:
    storage = store[output_name]
    assert not isinstance(storage, StorageBase)
    if isinstance(storage, Path):
        dump(output, path=storage)
    else:
        assert isinstance(storage, DirectValue)
        storage.value = output


def _func_kwargs(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
) -> dict[str, Any]:
    kwargs = {}
    for p in func.parameters:
        if p in func._bound:
            kwargs[p] = func._bound[p]
        elif p in run_info.inputs:
            kwargs[p] = run_info.inputs[p]
        elif p in run_info.all_output_names:
            kwargs[p] = _load_from_store(p, store).value
        elif p in run_info.defaults and p not in run_info.all_output_names:
            kwargs[p] = run_info.defaults[p]
        else:  # pragma: no cover
            # In principle it should not be possible to reach this point because of
            # the checks in `run` and `_validate_complete_inputs`.
            msg = f"Parameter `{p}` not found in inputs, outputs, bound or defaults."
            raise ValueError(msg)
    return kwargs


def _select_kwargs(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    external_shape = _external_shape(shape, shape_mask)
    input_keys = func.mapspec.input_keys(external_shape, index)
    normalized_keys = {k: v[0] if len(v) == 1 else v for k, v in input_keys.items()}
    selected = {k: v[normalized_keys[k]] if k in normalized_keys else v for k, v in kwargs.items()}
    _load_file_arrays(selected)
    return selected


def _init_result_arrays(output_name: _OUTPUT_TYPE, shape: tuple[int, ...]) -> list[np.ndarray]:
    return [np.empty(prod(shape), dtype=object) for _ in at_least_tuple(output_name)]


def _pick_output(func: PipeFunc, output: Any) -> tuple[Any, ...]:
    return tuple(
        (func.output_picker(output, output_name) if func.output_picker is not None else output)
        for output_name in at_least_tuple(func.output_name)
    )


def _get_or_set_cache(
    func: PipeFunc,
    kwargs: dict[str, Any],
    cache: _CacheBase | None,
    compute_fn: Callable[[], Any],
) -> Any:
    if cache is None:
        return compute_fn()
    cache_key = (func.output_name, to_hashable(kwargs))

    if cache_key in cache:
        return cache.get(cache_key)
    if isinstance(cache, HybridCache):
        t = time.monotonic()
    result = compute_fn()
    if isinstance(cache, HybridCache):
        cache.put(cache_key, result, time.monotonic() - t)
    else:
        cache.put(cache_key, result)
    return result


_EVALUATED_RESOURCES = "__pipefunc_internal_evaluated_resources__"


def _run_iteration(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
    cache: _CacheBase | None,
) -> Any:
    selected = _select_kwargs(func, kwargs, shape, shape_mask, index)

    def compute_fn() -> Any:
        if callable(func.resources) and func.mapspec is not None and func.resources_scope == "map":  # type: ignore[has-type]
            selected[_EVALUATED_RESOURCES] = func.resources(kwargs)  # type: ignore[has-type]
        try:
            return func(**selected)
        except Exception as e:
            handle_error(e, func, selected)
            # handle_error raises but mypy doesn't know that
            raise  # pragma: no cover

    return _get_or_set_cache(func, selected, cache, compute_fn)


def _run_iteration_and_process(
    index: int,
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    file_arrays: Sequence[StorageBase],
    cache: _CacheBase | None = None,
) -> tuple[Any, ...]:
    output = _run_iteration(func, kwargs, shape, shape_mask, index, cache)
    outputs = _pick_output(func, output)
    _update_file_array(func, file_arrays, shape, shape_mask, index, outputs)
    return outputs


def _update_file_array(
    func: PipeFunc,
    file_arrays: Sequence[StorageBase],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
    outputs: tuple[Any, ...],
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
    assert np.shape(output) == internal_shape
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


def _existing_and_missing_indices(
    file_arrays: list[StorageBase],
    fixed_mask: np.flatiter[npt.NDArray[np.bool_]] | None,
) -> tuple[list[int], list[int]]:
    # TODO: when `fixed_indices` are used we could be more efficient by not
    # computing the full mask.
    masks = (arr.mask_linear() for arr in file_arrays)
    if fixed_mask is None:
        fixed_mask = itertools.repeat(object=True)  # type: ignore[assignment]

    existing_indices = []
    missing_indices = []
    for i, (*mask_values, select) in enumerate(zip(*masks, fixed_mask)):  # type: ignore[arg-type]
        if not select:
            continue
        if any(mask_values):  # rerun if any of the outputs are missing
            missing_indices.append(i)
        else:
            existing_indices.append(i)
    return existing_indices, missing_indices


@contextmanager
def _maybe_executor(
    executor: Executor | None,
    parallel: bool,  # noqa: FBT001
) -> Generator[Executor | None, None, None]:
    if executor is None and parallel:
        with ProcessPoolExecutor() as new_executor:  # shuts down the executor after use
            yield new_executor
    else:
        yield executor


class _MapSpecArgs(NamedTuple):
    process_index: functools.partial[tuple[Any, ...]]
    existing: list[int]
    missing: list[int]
    result_arrays: list[np.ndarray]
    shape: tuple[int, ...]
    mask: tuple[bool, ...]
    file_arrays: list[StorageBase]


def _prepare_submit_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    fixed_indices: dict[str, int | slice] | None,
    cache: _CacheBase | None = None,
) -> _MapSpecArgs:
    assert isinstance(func.mapspec, MapSpec)
    shape = run_info.shapes[func.output_name]
    mask = run_info.shape_masks[func.output_name]
    file_arrays: list[StorageBase] = [store[name] for name in at_least_tuple(func.output_name)]  # type: ignore[misc]
    result_arrays = _init_result_arrays(func.output_name, shape)
    process_index = functools.partial(
        _run_iteration_and_process,
        func=func,
        kwargs=kwargs,
        shape=shape,
        shape_mask=mask,
        file_arrays=file_arrays,
        cache=cache,
    )
    fixed_mask = _mask_fixed_axes(fixed_indices, func.mapspec, shape, mask)
    existing, missing = _existing_and_missing_indices(file_arrays, fixed_mask)  # type: ignore[arg-type]
    return _MapSpecArgs(process_index, existing, missing, result_arrays, shape, mask, file_arrays)


def _mask_fixed_axes(
    fixed_indices: dict[str, int | slice] | None,
    mapspec: MapSpec,
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
) -> np.flatiter[npt.NDArray[np.bool_]] | None:
    if fixed_indices is None:
        return None
    key = tuple(fixed_indices.get(axis, slice(None)) for axis in mapspec.output_indices)
    external_key = _external_shape(key, shape_mask)  # type: ignore[arg-type]
    external_shape = _external_shape(shape, shape_mask)
    select: npt.NDArray[np.bool_] = np.zeros(external_shape, dtype=bool)
    select[external_key] = True
    return select.flat


def _maybe_parallel_map(func: Callable[..., Any], seq: Sequence, executor: Executor | None) -> Any:
    if executor is not None:
        return executor.map(func, seq)
    return map(func, seq)


def _maybe_submit(func: Callable[..., Any], executor: Executor | None, *args: Any) -> Any:
    if executor:
        return executor.submit(func, *args)
    return func(*args)


class _StoredValue(NamedTuple):
    value: Any
    exists: bool


def _load_from_store(
    output_name: _OUTPUT_TYPE,
    store: dict[str, StorageBase | Path | DirectValue],
    *,
    return_output: bool = True,
) -> _StoredValue:
    outputs: list[Any] = []
    all_exist = True

    for name in at_least_tuple(output_name):
        storage = store[name]
        if isinstance(storage, StorageBase):
            outputs.append(storage)
        elif isinstance(storage, Path):
            if storage.is_file():
                outputs.append(load(storage) if return_output else None)
            else:
                all_exist = False
                outputs.append(None)
        else:
            assert isinstance(storage, DirectValue)
            if storage.exists():
                outputs.append(storage.value)
            else:
                all_exist = False
                outputs.append(None)

    if not return_output:
        outputs = None  # type: ignore[assignment]
    elif len(outputs) == 1:
        outputs = outputs[0]

    return _StoredValue(outputs, all_exist)


def _submit_single(
    func: PipeFunc,
    kwargs: dict[str, Any],
    store: dict[str, StorageBase | Path | DirectValue],
    cache: _CacheBase | None,
) -> Any:
    # Load the output if it exists
    output, exists = _load_from_store(func.output_name, store, return_output=True)
    if exists:
        return output

    # Otherwise, run the function
    _load_file_arrays(kwargs)

    def compute_fn() -> Any:
        try:
            return func(**kwargs)
        except Exception as e:
            handle_error(e, func, kwargs)
            # handle_error raises but mypy doesn't know that
            raise  # pragma: no cover

    return _get_or_set_cache(func, kwargs, cache, compute_fn)


def _maybe_load_array(x: Any) -> Any:
    if isinstance(x, StorageBase):
        return x.to_array()
    return x


def _load_file_arrays(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = _maybe_load_array(v)


class _KwargsTask(NamedTuple):
    kwargs: dict[str, Any]
    task: tuple[Any, _MapSpecArgs] | Any


def _run_and_process_generation(
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    outputs: dict[str, Result],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | None,
    cache: _CacheBase | None = None,
) -> None:
    tasks: dict[PipeFunc, _KwargsTask] = {}
    for func in generation:
        tasks[func] = _submit_func(func, run_info, store, fixed_indices, executor, cache)
    for func in generation:
        _outputs = _process_task(func, tasks[func], store, executor)
        outputs.update(_outputs)


def _submit_func(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | None,
    cache: _CacheBase | None = None,
) -> _KwargsTask:
    kwargs = _func_kwargs(func, run_info, store)
    if func.mapspec and func.mapspec.inputs:
        args = _prepare_submit_map_spec(func, kwargs, run_info, store, fixed_indices, cache)
        r = _maybe_parallel_map(args.process_index, args.missing, executor)
        task = r, args
    else:
        task = _maybe_submit(_submit_single, executor, func, kwargs, store, cache)
    return _KwargsTask(kwargs, task)


def _process_task(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    store: dict[str, StorageBase | Path | DirectValue],
    executor: Executor | None = None,
) -> dict[str, Result]:
    kwargs, task = kwargs_task
    if func.mapspec and func.mapspec.inputs:
        r, args = task
        outputs_list = list(r)

        for index, outputs in zip(args.missing, outputs_list):
            _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

        for index in args.existing:
            outputs = [file_array.get_from_index(index) for file_array in args.file_arrays]
            _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

        output = tuple(x.reshape(args.shape) for x in args.result_arrays)
    else:
        r = task.result() if executor else task  # type: ignore[union-attr]
        output = _dump_single_output(func, r, store)

    # Note that the kwargs still contain the StorageBase objects if _submit_map_spec
    # was used.
    return {
        output_name: Result(
            function=func.__name__,
            kwargs=kwargs,
            output_name=output_name,
            output=_output,
            store=store[output_name],
        )
        for output_name, _output in zip(at_least_tuple(func.output_name), output)
    }


def _check_parallel(
    parallel: bool,  # noqa: FBT001
    store: dict[str, StorageBase | Path | DirectValue],
    executor: Executor | None,
) -> None:
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
