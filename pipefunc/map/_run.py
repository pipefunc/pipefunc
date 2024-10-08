from __future__ import annotations

import asyncio
import functools
import itertools
import time
import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from pipefunc._utils import (
    at_least_tuple,
    dump,
    handle_error,
    is_running_in_ipynb,
    load,
    prod,
    requires,
)
from pipefunc.cache import HybridCache, to_hashable
from pipefunc.map._mapspec import MapSpec, _shape_to_key, validate_consistent_axes
from pipefunc.map._run_info import DirectValue, RunInfo, _external_shape, _internal_shape
from pipefunc.map._storage_base import StorageBase, _iterate_shape_indices, _select_by_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import xarray as xr

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._widgets import ProgressTracker
    from pipefunc.cache import _CacheBase


_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]


def _cannot_be_parallelized(pipeline: Pipeline) -> bool:
    return all(f.mapspec is None for f in pipeline.functions) and all(
        len(fs) == 1 for fs in pipeline.topological_generations.function_lists
    )


def _prepare_run(
    *,
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None,
    output_names: set[_OUTPUT_TYPE] | None,
    parallel: bool,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
    storage: str | dict[_OUTPUT_TYPE, str],
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
    progress = _init_tracker(store, pipeline.sorted_functions, show_progress, in_async)
    if executor is None and _cannot_be_parallelized(pipeline):
        parallel = False
    _check_parallel(parallel, store, executor)
    return pipeline, run_info, store, outputs, parallel, progress


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None = None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    output_names: set[_OUTPUT_TYPE] | None = None,
    parallel: bool = True,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None = None,
    storage: str | dict[_OUTPUT_TYPE, str] = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool = False,
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
        The ``internal_shape`` can also be provided via the ``PipeFunc(..., internal_shape=...)`` argument.
        If a `PipeFunc` has an ``internal_shape`` argument *and* it is provided here, the provided value is used.
    output_names
        The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
    parallel
        Whether to run the functions in parallel. Is ignored if provided ``executor`` is not ``None``.
    executor
        The executor to use for parallel execution. Can be specified as:

        1. ``None``: A `concurrent.futures.ProcessPoolExecutor` is used (only if ``parallel=True``).
        2. A `concurrent.futures.Executor` instance: Used for all outputs.
        3. A dictionary: Specify different executors for different outputs.

           - Use output names as keys and `~concurrent.futures.Executor` instances as values.
           - Use an empty string ``""`` as a key to set a default executor.

        If parallel is ``False``, this argument is ignored.
    storage
        The storage class to use for storing intermediate and final results.
        Can be specified as:

        1. A string: Use a single storage class for all outputs.
        2. A dictionary: Specify different storage classes for different outputs.

           - Use output names as keys and storage class names as values.
           - Use an empty string ``""`` as a key to set a default storage class.

        Available storage classes are registered in `pipefunc.map.storage_registry`.
        Common options include ``"file_array"``, ``"dict"``, and ``"shared_memory_dict"``.
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
    show_progress
        Whether to display a progress bar. Only works if ``parallel=True``.

    """
    pipeline, run_info, store, outputs, parallel, progress = _prepare_run(
        pipeline=pipeline,
        inputs=inputs,
        run_folder=run_folder,
        internal_shapes=internal_shapes,
        output_names=output_names,
        parallel=parallel,
        executor=executor,
        storage=storage,
        cleanup=cleanup,
        fixed_indices=fixed_indices,
        auto_subpipeline=auto_subpipeline,
        show_progress=show_progress,
        in_async=False,
    )
    if progress is not None:
        progress.display()
    with _maybe_executor(executor, parallel) as ex:
        for gen in pipeline.topological_generations.function_lists:
            _run_and_process_generation(
                generation=gen,
                run_info=run_info,
                store=store,
                outputs=outputs,
                fixed_indices=fixed_indices,
                executor=ex,
                progress=progress,
                cache=pipeline.cache,
            )
    if progress is not None:  # final update
        progress.update_progress(force=True)
    _maybe_persist_memory(store, persist_memory)
    return outputs


class AsyncRun(NamedTuple):
    task: asyncio.Task[OrderedDict[str, Result]]
    run_info: RunInfo
    progress: ProgressTracker | None

    def result(self) -> OrderedDict[str, Result]:
        if is_running_in_ipynb():  # pragma: no cover
            if self.task.done():
                return self.task.result()
            msg = (
                "Cannot block the event loop when running in a Jupyter notebook."
                " Use `await runner.task` instead."
            )
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()  # pragma: no cover
        return loop.run_until_complete(self.task)  # pragma: no cover


def run_async(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None = None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    output_names: set[_OUTPUT_TYPE] | None = None,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None = None,
    storage: str | dict[_OUTPUT_TYPE, str] = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool = False,
) -> AsyncRun:
    """Asynchronously run a pipeline with `MapSpec` functions for given ``inputs``.

    Returns immediately with an `AsyncRun` instance with a `task` attribute that can be awaited.

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
        The ``internal_shape`` can also be provided via the ``PipeFunc(..., internal_shape=...)`` argument.
        If a `PipeFunc` has an ``internal_shape`` argument *and* it is provided here, the provided value is used.
    output_names
        The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
    executor
        The executor to use for parallel execution. Can be specified as:

        1. ``None``: A `concurrent.futures.ProcessPoolExecutor` is used (only if ``parallel=True``).
        2. A `concurrent.futures.Executor` instance: Used for all outputs.
        3. A dictionary: Specify different executors for different outputs.

           - Use output names as keys and `~concurrent.futures.Executor` instances as values.
           - Use an empty string ``""`` as a key to set a default executor.
    storage
        The storage class to use for storing intermediate and final results.
        Can be specified as:

        1. A string: Use a single storage class for all outputs.
        2. A dictionary: Specify different storage classes for different outputs.

           - Use output names as keys and storage class names as values.
           - Use an empty string ``""`` as a key to set a default storage class.

        Available storage classes are registered in `pipefunc.map.storage_registry`.
        Common options include ``"file_array"``, ``"dict"``, and ``"shared_memory_dict"``.
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
    show_progress
        Whether to display a progress bar.

    """
    pipeline, run_info, store, outputs, _, progress = _prepare_run(
        pipeline=pipeline,
        inputs=inputs,
        run_folder=run_folder,
        internal_shapes=internal_shapes,
        output_names=output_names,
        parallel=True,
        executor=executor,
        storage=storage,
        cleanup=cleanup,
        fixed_indices=fixed_indices,
        auto_subpipeline=auto_subpipeline,
        show_progress=show_progress,
        in_async=True,
    )

    async def _run_pipeline() -> OrderedDict[str, Result]:
        with _maybe_executor(executor, parallel=True) as ex:
            assert ex is not None
            for gen in pipeline.topological_generations.function_lists:
                await _run_and_process_generation_async(
                    generation=gen,
                    run_info=run_info,
                    store=store,
                    outputs=outputs,
                    fixed_indices=fixed_indices,
                    executor=ex,
                    progress=progress,
                    cache=pipeline.cache,
                )
        _maybe_persist_memory(store, persist_memory)
        return outputs

    task = asyncio.create_task(_run_pipeline())
    if progress is not None:
        progress.attach_task(task)
        progress.display()
    return AsyncRun(task, run_info, progress)


def _maybe_persist_memory(
    store: dict[str, StorageBase | Path | DirectValue],
    persist_memory: bool,  # noqa: FBT001
) -> None:
    if persist_memory:  # Only relevant for memory based storage
        for arr in store.values():
            if isinstance(arr, StorageBase):
                arr.persist()


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
    requires("xarray", reason="load_xarray_dataset", extras="xarray")
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
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
    parallel: bool,  # noqa: FBT001
) -> Generator[Executor | dict[_OUTPUT_TYPE, Executor] | None, None, None]:
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


def _status_submit(
    func: Callable[..., Any],
    executor: Executor,
    status: _Status,
    progress: ProgressTracker,
    *args: Any,
) -> Future:
    status.mark_in_progress()
    fut = executor.submit(func, *args)
    fut.add_done_callback(status.mark_complete)
    if not progress.in_async:
        fut.add_done_callback(progress.update_progress)
    return fut


def _maybe_parallel_map(
    func: Callable[..., Any],
    seq: Sequence,
    executor: Executor | None,
    status: _Status | None,
    progress: ProgressTracker | None,
) -> list[Any]:
    if executor is not None:
        if status is not None:
            assert progress is not None
            return [_status_submit(func, executor, status, progress, x) for x in seq]
        return [executor.submit(func, x) for x in seq]
    return [func(x) for x in seq]


def _maybe_submit(
    func: Callable[..., Any],
    executor: Executor | None,
    status: _Status | None,
    progress: ProgressTracker | None,
    *args: Any,
) -> Any:
    if executor:
        if status is not None:
            assert progress is not None
            return _status_submit(func, executor, status, progress, *args)
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


@dataclass
class _Status:
    """A class to keep track of the progress of a function."""

    n_total: int
    n_in_progress: int = 0
    n_completed: int = 0
    n_failed: int = 0
    start_time: float | None = None
    end_time: float | None = None

    @property
    def n_left(self) -> int:
        return self.n_total - self.n_completed - self.n_failed

    def mark_in_progress(self) -> None:
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.n_in_progress += 1

    def mark_complete(self, _: Any) -> None:  # needs arg to be used as callback
        self.n_in_progress -= 1
        self.n_completed += 1
        if self.n_completed == self.n_total:
            self.end_time = time.monotonic()

    @property
    def progress(self) -> float:
        return self.n_completed / self.n_total

    def elapsed_time(self) -> float:
        assert self.start_time is not None
        if self.end_time is None:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time


def _init_tracker(
    store: dict[str, StorageBase | Path | DirectValue],
    functions: list[PipeFunc],
    show_progress: bool,  # noqa: FBT001
    in_async: bool,  # noqa: FBT001
) -> ProgressTracker | None:
    if not show_progress:
        return None
    requires("ipywidgets", reason="show_progress", extras="ipywidgets")
    from pipefunc._widgets import ProgressTracker

    progress = {}
    for func in functions:
        name, *_ = at_least_tuple(func.output_name)  # if multiple, the have equal size
        s = store[name]
        size = s.size if isinstance(s, StorageBase) else 1
        progress[func.output_name] = _Status(n_total=size)
    return ProgressTracker(progress, None, display=False, in_async=in_async)


# NOTE: A similar async version of this function is provided below.
def _run_and_process_generation(
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    outputs: dict[str, Result],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
    progress: ProgressTracker | None,
    cache: _CacheBase | None = None,
) -> None:
    tasks = _submit_generation(
        run_info,
        generation,
        store,
        fixed_indices,
        executor,
        progress,
        cache,
    )
    _process_generation(generation, tasks, store, outputs)


async def _run_and_process_generation_async(
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    outputs: dict[str, Result],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | dict[_OUTPUT_TYPE, Executor],
    progress: ProgressTracker | None,
    cache: _CacheBase | None = None,
) -> None:
    tasks = _submit_generation(
        run_info,
        generation,
        store,
        fixed_indices,
        executor,
        progress,
        cache,
    )
    await _process_generation_async(generation, tasks, store, outputs)


# NOTE: A similar async version of this function is provided below.
def _process_generation(
    generation: list[PipeFunc],
    tasks: dict[PipeFunc, _KwargsTask],
    store: dict[str, StorageBase | Path | DirectValue],
    outputs: dict[str, Result],
) -> None:
    for func in generation:
        _outputs = _process_task(func, tasks[func], store)
        outputs.update(_outputs)


async def _process_generation_async(
    generation: list[PipeFunc],
    tasks: dict[PipeFunc, _KwargsTask],
    store: dict[str, StorageBase | Path | DirectValue],
    outputs: dict[str, Result],
) -> None:
    for func in generation:
        _outputs = await _process_task_async(func, tasks[func], store)
        outputs.update(_outputs)


def _submit_func(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StorageBase | Path | DirectValue],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
    progress: ProgressTracker | None = None,
    cache: _CacheBase | None = None,
) -> _KwargsTask:
    kwargs = _func_kwargs(func, run_info, store)
    ex = _executor_for_func(func, executor)
    status = progress.progress_dict[func.output_name] if progress is not None else None
    if func.mapspec and func.mapspec.inputs:
        args = _prepare_submit_map_spec(func, kwargs, run_info, store, fixed_indices, cache)
        r = _maybe_parallel_map(args.process_index, args.missing, ex, status, progress)
        task = r, args
    else:
        task = _maybe_submit(_submit_single, ex, status, progress, func, kwargs, store, cache)
    return _KwargsTask(kwargs, task)


def _executor_for_func(
    func: PipeFunc,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
) -> Executor | None:
    if isinstance(executor, dict):
        if func.output_name in executor:
            return executor[func.output_name]
        if "" in executor:
            return executor[""]
        msg = (
            f"No executor found for output `{func.output_name}`."
            f" Please either specify an executor for this output using"
            f" `executor['{func.output_name}'] = ...`, or provide a default executor"
            f' using `executor[""] = ...`.'
        )
        raise ValueError(msg)
    return executor


def _submit_generation(
    run_info: RunInfo,
    generation: list[PipeFunc],
    store: dict[str, StorageBase | Path | DirectValue],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
    progress: ProgressTracker | None,
    cache: _CacheBase | None = None,
) -> dict[PipeFunc, _KwargsTask]:
    return {
        func: _submit_func(func, run_info, store, fixed_indices, executor, progress, cache)
        for func in generation
    }


def _output_from_mapspec_task(
    args: _MapSpecArgs,
    outputs_list: list[list[Any]],
) -> tuple[np.ndarray, ...]:
    for index, outputs in zip(args.missing, outputs_list):
        _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

    for index in args.existing:
        outputs = [file_array.get_from_index(index) for file_array in args.file_arrays]
        _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

    return tuple(x.reshape(args.shape) for x in args.result_arrays)


def _result(x: Any | Future) -> Any:
    return x.result() if isinstance(x, Future) else x


def _result_async(task: Future, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
    return asyncio.wrap_future(task, loop=loop)


def _to_result_dict(
    func: PipeFunc,
    kwargs: dict[str, Any],
    output: Any,
    store: dict[str, StorageBase | Path | DirectValue],
) -> dict[str, Result]:
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


# NOTE: A similar async version of this function is provided below.
def _process_task(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    store: dict[str, StorageBase | Path | DirectValue],
) -> dict[str, Result]:
    kwargs, task = kwargs_task
    if func.mapspec and func.mapspec.inputs:
        r, args = task
        outputs_list = [_result(x) for x in r]
        output = _output_from_mapspec_task(args, outputs_list)
    else:
        r = _result(task)
        output = _dump_single_output(func, r, store)
    return _to_result_dict(func, kwargs, output, store)


async def _process_task_async(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    store: dict[str, StorageBase | Path | DirectValue],
) -> dict[str, Result]:
    kwargs, task = kwargs_task
    loop = asyncio.get_event_loop()
    if func.mapspec and func.mapspec.inputs:
        r, args = task
        futs = [_result_async(x, loop) for x in r]
        outputs_list = await asyncio.gather(*futs)
        output = _output_from_mapspec_task(args, outputs_list)
    else:
        assert isinstance(task, Future)
        r = await _result_async(task, loop)
        output = _dump_single_output(func, r, store)
    return _to_result_dict(func, kwargs, output, store)


def _check_parallel(
    parallel: bool,  # noqa: FBT001
    store: dict[str, StorageBase | Path | DirectValue],
    executor: Executor | dict[_OUTPUT_TYPE, Executor] | None,
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
