from __future__ import annotations

import asyncio
import functools
import itertools
import math
import time
import warnings
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt

from pipefunc._utils import (
    at_least_tuple,
    dump,
    get_ncores,
    handle_error,
    is_running_in_ipynb,
    load,
    prod,
)
from pipefunc.cache import HybridCache, to_hashable

from ._adaptive_scheduler_slurm_executor import (
    maybe_finalize_slurm_executors,
    maybe_multi_run_manager,
    maybe_update_slurm_executor_map,
    maybe_update_slurm_executor_single,
)
from ._mapspec import MapSpec, _shape_to_key
from ._prepare import prepare_run
from ._result import DirectValue, Result, ResultDict
from ._shapes import external_shape_from_mask, internal_shape_from_mask, shape_is_resolved
from ._storage_array._base import StorageBase, iterate_shape_indices, select_by_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence

    import pydantic
    from adaptive_scheduler import MultiRunManager

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE, StorageType
    from pipefunc._widgets import ProgressTracker
    from pipefunc.cache import _CacheBase

    from ._progress import Status
    from ._result import StoreType
    from ._run_info import RunInfo
    from ._types import ShapeTuple, UserShapeDict


def run_map(
    pipeline: Pipeline,
    inputs: dict[str, Any] | pydantic.BaseModel,
    run_folder: str | Path | None = None,
    internal_shapes: UserShapeDict | None = None,
    *,
    output_names: set[OUTPUT_TYPE] | None = None,
    parallel: bool = True,
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
    storage: StorageType = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool = False,
    return_results: bool = True,
) -> ResultDict:
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
        If not provided, the shapes will be inferred from the first execution of the function.
        If provided, the shapes will be validated against the actual shapes of the outputs.
        The values can be either integers or "?" for unknown dimensions.
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
    chunksizes
        Controls batching of `~pipefunc.map.MapSpec` computations for parallel execution.
        Reduces overhead by grouping multiple function calls into single tasks.
        Can be specified as:

        - None: Automatically determine optimal chunk sizes (default)
        - int: Same chunk size for all outputs
        - dict: Different chunk sizes per output where:
            - Keys are output names (or ``""`` for default)
            - Values are either integers or callables
            - Callables take total execution count and return chunk size

        **Examples:**

        >>> chunksizes = None  # Auto-determine optimal chunk sizes
        >>> chunksizes = 100  # All outputs use chunks of 100
        >>> chunksizes = {"out1": 50, "out2": 100}  # Different sizes per output
        >>> chunksizes = {"": 50, "out1": lambda n: n // 20}  # Default and dynamic
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
    return_results
        Whether to return the results of the pipeline. If ``False``, the pipeline is run
        without keeping the results in memory. Instead the results are only kept in the set
        ``storage``. This is useful for very large pipelines where the results do not fit into memory.

    """
    pipeline, run_info, store, outputs, parallel, executor, progress = prepare_run(
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
                chunksizes=chunksizes,
                progress=progress,
                return_results=return_results,
                cache=pipeline.cache,
            )
    if progress is not None:  # final update
        progress.update_progress(force=True)
    _maybe_persist_memory(store, persist_memory)
    return outputs


@dataclass
class AsyncMap:
    task: asyncio.Task[ResultDict]
    run_info: RunInfo
    progress: ProgressTracker | None
    multi_run_manager: MultiRunManager | None

    def result(self) -> ResultDict:
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

    def display(self) -> None:  # pragma: no cover
        """Display the pipeline widget."""
        if is_running_in_ipynb():
            if self.progress is not None:
                self.progress.display()
            if self.multi_run_manager is not None:
                self.multi_run_manager.display()
        else:
            print("⚠️ Display is only supported in Jupyter notebooks.")


def run_map_async(
    pipeline: Pipeline,
    inputs: dict[str, Any] | pydantic.BaseModel,
    run_folder: str | Path | None = None,
    internal_shapes: UserShapeDict | None = None,
    *,
    output_names: set[OUTPUT_TYPE] | None = None,
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
    storage: StorageType = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool = False,
    return_results: bool = True,
) -> AsyncMap:
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
        If not provided, the shapes will be inferred from the first execution of the function.
        If provided, the shapes will be validated against the actual shapes of the outputs.
        The values can be either integers or "?" for unknown dimensions.
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
    chunksizes
        Controls batching of `~pipefunc.map.MapSpec` computations for parallel execution.
        Reduces overhead by grouping multiple function calls into single tasks.
        Can be specified as:

        - None: Automatically determine optimal chunk sizes (default)
        - int: Same chunk size for all outputs
        - dict: Different chunk sizes per output where:
            - Keys are output names (or ``""`` for default)
            - Values are either integers or callables
            - Callables take total execution count and return chunk size

        **Examples:**

        >>> chunksizes = None  # Auto-determine optimal chunk sizes
        >>> chunksizes = 100  # All outputs use chunks of 100
        >>> chunksizes = {"out1": 50, "out2": 100}  # Different sizes per output
        >>> chunksizes = {"": 50, "out1": lambda n: n // 20}  # Default and dynamic
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
    return_results
        Whether to return the results of the pipeline. If ``False``, the pipeline is run
        without keeping the results in memory. Instead the results are only kept in the set
        ``storage``. This is useful for very large pipelines where the results do not fit into memory.

    """
    pipeline, run_info, store, outputs, _, executor_dict, progress = prepare_run(
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

    multi_run_manager = maybe_multi_run_manager(executor_dict)

    async def _run_pipeline() -> ResultDict:
        with _maybe_executor(executor_dict, parallel=True) as ex:
            assert ex is not None
            for gen in pipeline.topological_generations.function_lists:
                await _run_and_process_generation_async(
                    generation=gen,
                    run_info=run_info,
                    store=store,
                    outputs=outputs,
                    fixed_indices=fixed_indices,
                    executor=ex,
                    chunksizes=chunksizes,
                    progress=progress,
                    return_results=return_results,
                    cache=pipeline.cache,
                    multi_run_manager=multi_run_manager,
                )
        _maybe_persist_memory(store, persist_memory)
        return outputs

    task = asyncio.create_task(_run_pipeline())
    if progress is not None:
        progress.attach_task(task)
    if is_running_in_ipynb():  # pragma: no cover
        if progress is not None:
            progress.display()
        if multi_run_manager is not None:
            multi_run_manager.display()
    return AsyncMap(task, run_info, progress, multi_run_manager)


def _maybe_persist_memory(
    store: dict[str, StoreType],
    persist_memory: bool,  # noqa: FBT001
) -> None:
    if persist_memory:  # Only relevant for memory based storage
        for arr in store.values():
            if isinstance(arr, StorageBase):
                arr.persist()


def _dump_single_output(
    func: PipeFunc,
    output: Any,
    store: dict[str, StoreType],
    run_info: RunInfo,
) -> tuple[Any, ...]:
    if isinstance(func.output_name, tuple):
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            _single_dump_single_output(_output, output_name, store, run_info)
        return tuple(new_output)
    _single_dump_single_output(output, func.output_name, store, run_info)
    return (output,)


def _single_dump_single_output(
    output: Any,
    output_name: str,
    store: dict[str, StoreType],
    run_info: RunInfo,
) -> None:
    run_info.resolve_downstream_shapes(output_name, store, output=output)
    storage = store[output_name]
    assert not isinstance(storage, StorageBase)
    if isinstance(storage, Path):
        dump(output, path=storage)
    else:
        assert isinstance(storage, DirectValue)
        storage.value = output


def _func_kwargs(func: PipeFunc, run_info: RunInfo, store: dict[str, StoreType]) -> dict[str, Any]:
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
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    external_shape = external_shape_from_mask(shape, shape_mask)
    input_keys = func.mapspec.input_keys(external_shape, index)  # type: ignore[arg-type]
    normalized_keys = {k: v[0] if len(v) == 1 else v for k, v in input_keys.items()}
    selected = {k: v[normalized_keys[k]] if k in normalized_keys else v for k, v in kwargs.items()}
    _load_arrays(selected)
    return selected


def _maybe_eval_resources_in_selected(
    kwargs: dict[str, Any],
    selected: dict[str, Any],
    func: PipeFunc,
) -> None:
    if callable(func.resources):  # type: ignore[has-type]
        kw = kwargs if func.resources_scope == "map" else selected
        selected[_EVALUATED_RESOURCES] = func.resources(kw)  # type: ignore[has-type]


def _select_kwargs_and_eval_resources(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    index: int,
) -> dict[str, Any]:
    selected = _select_kwargs(func, kwargs, shape, shape_mask, index)
    _maybe_eval_resources_in_selected(kwargs, selected, func)
    return selected


def _init_result_arrays(
    output_name: OUTPUT_TYPE,
    shape: ShapeTuple,
    return_results: bool,  # noqa: FBT001
) -> list[np.ndarray] | None:
    if not return_results or not shape_is_resolved(shape):
        return None
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
    cache_key = (func._cache_id, to_hashable(kwargs))

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


def _run_iteration(func: PipeFunc, selected: dict[str, Any], cache: _CacheBase | None) -> Any:
    def compute_fn() -> Any:
        try:
            return func(**selected)
        except Exception as e:
            handle_error(e, func, selected)
            # handle_error raises but mypy doesn't know that
            raise  # pragma: no cover

    return _get_or_set_cache(func, selected, cache, compute_fn)


def _try_shape(x: Any) -> tuple[int, ...]:
    try:
        return np.shape(x)
    except ValueError:
        # e.g., when inhomogeneous lists are passed
        return ()


@dataclass
class _InternalShape:
    shape: tuple[int, ...]

    @classmethod
    def from_outputs(cls, outputs: tuple[Any]) -> tuple[_InternalShape, ...]:
        return tuple(cls(_try_shape(output)) for output in outputs)


def _run_iteration_and_process(
    index: int,
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    arrays: Sequence[StorageBase],
    cache: _CacheBase | None = None,
    *,
    return_results: bool = True,
    force_dump: bool = False,
) -> tuple[Any, ...]:
    selected = _select_kwargs_and_eval_resources(func, kwargs, shape, shape_mask, index)
    output = _run_iteration(func, selected, cache)
    outputs = _pick_output(func, output)
    has_dumped = _update_array(
        func,
        arrays,
        shape,
        shape_mask,
        index,
        outputs,
        in_post_process=False,
        force_dump=force_dump,
    )
    if has_dumped and not return_results:
        return _InternalShape.from_outputs(outputs)
    return outputs


def _update_array(
    func: PipeFunc,
    arrays: Sequence[StorageBase],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    index: int,
    outputs: Iterable[Any],
    *,
    in_post_process: bool,
    force_dump: bool = False,  # Only true in `adaptive.py`
) -> bool:
    # This function is called both in the main process (in post processing) and in the executor process.
    # It needs to only dump the data once.
    # If the data can be written during the function call inside the executor (e.g., a file array),
    # we dump it in the executor. Otherwise, we dump it in the main process during the result array update.
    # We do this to offload the I/O and serialization overhead to the executor process if possible.
    assert isinstance(func.mapspec, MapSpec)
    output_key = None
    has_dumped = False
    for array, _output in zip(arrays, outputs):
        if not array.full_shape_is_resolved():
            _maybe_set_internal_shape(_output, array)
        if force_dump or (array.dump_in_subprocess != in_post_process):
            if output_key is None:  # Only calculate the output key if needed
                external_shape = external_shape_from_mask(shape, shape_mask)
                output_key = func.mapspec.output_key(external_shape, index)  # type: ignore[arg-type]
            array.dump(output_key, _output)
            has_dumped = True
    return has_dumped


def _indices_to_flat_index(
    shape: tuple[int, ...],
    internal_shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    external_index: tuple[int, ...],
    internal_index: tuple[int, ...],
) -> np.int_:
    full_index = select_by_mask(shape_mask, external_index, internal_index)
    full_shape = select_by_mask(shape_mask, shape, internal_shape)
    return np.ravel_multi_index(full_index, full_shape)


def _set_output(
    arr: np.ndarray,
    output: np.ndarray,
    linear_index: int,
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    func: PipeFunc,
) -> None:
    external_shape = external_shape_from_mask(shape, shape_mask)
    internal_shape = internal_shape_from_mask(shape, shape_mask)
    external_index = _shape_to_key(external_shape, linear_index)
    _validate_internal_shape(output, internal_shape, func)
    for internal_index in iterate_shape_indices(internal_shape):
        flat_index = _indices_to_flat_index(
            external_shape,
            internal_shape,
            shape_mask,
            external_index,
            internal_index,
        )
        arr[flat_index] = output[internal_index]


def _validate_internal_shape(
    output: np.ndarray,
    internal_shape: tuple[int, ...],
    func: PipeFunc,
) -> None:
    shape = np.shape(output)[: len(internal_shape)]
    if shape != internal_shape:
        msg = (
            f"Output shape {shape} of function '{func.__name__}'"
            f" (output '{func.output_name}') does not match the expected"
            f" internal shape {internal_shape} used in the `mapspec`"
            f" '{func.mapspec}'. This error typically occurs when"
            " a `PipeFunc` returns values with inconsistent shapes across"
            " different invocations. Ensure that the output shape is"
            " consistent for all inputs."
        )
        raise ValueError(msg)


def _update_result_array(
    result_arrays: list[np.ndarray] | None,
    index: int,
    output: list[Any],
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
    func: PipeFunc,
) -> None:
    if result_arrays is None:
        return
    for result_array, _output in zip(result_arrays, output):
        if not all(mask):
            _output = np.asarray(_output)  # In case _output is a list
            _set_output(result_array, _output, index, shape, mask, func)
        else:
            result_array[index] = _output


def _existing_and_missing_indices(
    arrays: list[StorageBase],
    fixed_mask: np.flatiter[npt.NDArray[np.bool_]] | None,
) -> tuple[list[int], list[int]]:
    # TODO: when `fixed_indices` are used we could be more efficient by not
    # computing the full mask.
    masks = (arr.mask_linear() for arr in arrays)
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
    executor: dict[OUTPUT_TYPE, Executor] | None,
    parallel: bool,  # noqa: FBT001
) -> Generator[dict[OUTPUT_TYPE, Executor] | None, None, None]:
    if executor is None and parallel:
        with ProcessPoolExecutor() as new_executor:  # shuts down the executor after use
            yield {"": new_executor}
    else:
        yield executor


@dataclass
class _MapSpecArgs:
    process_index: functools.partial[tuple[Any, ...]]
    existing: list[int]
    missing: list[int]
    result_arrays: list[np.ndarray] | None
    mask: tuple[bool, ...]
    arrays: list[StorageBase]


def _prepare_submit_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    run_info: RunInfo,
    store: dict[str, StoreType],
    fixed_indices: dict[str, int | slice] | None,
    status: Status | None,
    return_results: bool,  # noqa: FBT001
    cache: _CacheBase | None = None,
) -> _MapSpecArgs:
    assert isinstance(func.mapspec, MapSpec)
    shape = run_info.resolved_shapes[func.output_name]
    mask = run_info.shape_masks[func.output_name]
    arrays: list[StorageBase] = [store[name] for name in at_least_tuple(func.output_name)]  # type: ignore[misc]
    result_arrays = _init_result_arrays(func.output_name, shape, return_results)
    process_index = functools.partial(
        _run_iteration_and_process,
        func=func,
        kwargs=kwargs,
        shape=shape,
        shape_mask=mask,
        arrays=arrays,
        cache=cache,
        return_results=return_results,
    )
    fixed_mask = _mask_fixed_axes(fixed_indices, func.mapspec, shape, mask)
    existing, missing = _existing_and_missing_indices(arrays, fixed_mask)  # type: ignore[arg-type]
    _update_status_if_needed(status, existing, missing)
    return _MapSpecArgs(process_index, existing, missing, result_arrays, mask, arrays)


def _mask_fixed_axes(
    fixed_indices: dict[str, int | slice] | None,
    mapspec: MapSpec,
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
) -> np.flatiter[npt.NDArray[np.bool_]] | None:
    if fixed_indices is None:
        return None

    key = tuple(fixed_indices.get(axis, slice(None)) for axis in mapspec.output_indices)
    external_key = external_shape_from_mask(key, shape_mask)  # type: ignore[arg-type]
    external_shape = external_shape_from_mask(shape, shape_mask)

    if not shape_is_resolved(external_shape):
        unresolved_axes = [
            mapspec.output_indices[i]
            for i, dim in enumerate(external_shape)
            if not isinstance(dim, int)
        ]
        msg = (
            "Cannot mask fixed axes when unresolved dimensions are present in the external shape."
            f" The following axes are unresolved: {', '.join(unresolved_axes)}"
        )
        raise ValueError(msg)

    assert shape_is_resolved(external_shape)
    select: npt.NDArray[np.bool_] = np.zeros(external_shape, dtype=bool)  # type: ignore[assignment]
    select[external_key] = True
    return select.flat


def _submit(
    func: Callable[..., Any],
    executor: Executor,
    status: Status | None,
    progress: ProgressTracker | None,
    chunksize: int,
    *args: Any,
) -> Future:
    if status is None:
        return executor.submit(func, *args)
    assert progress is not None
    status.mark_in_progress(n=chunksize)
    fut = executor.submit(func, *args)
    mark_complete = functools.partial(status.mark_complete, n=chunksize)
    fut.add_done_callback(mark_complete)
    if not progress.in_async:
        fut.add_done_callback(progress.update_progress)
    return fut


def _process_chunk(
    chunk: list[int],
    process_index: functools.partial[tuple[Any, ...]],
) -> list[Any]:
    return list(map(process_index, chunk))


def _chunk_indices(indices: list[int], chunksize: int) -> Iterable[tuple[int, ...]]:
    # The same implementation as outlined in the itertools.batched() documentation
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # but itertools.batched() was only added in python 3.12
    assert chunksize >= 1

    iterator = iter(indices)
    while batch := tuple(itertools.islice(iterator, chunksize)):
        yield batch


def _chunksize_for_func(
    func: PipeFunc,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    num_iterations: int,
    executor: Executor,
) -> int:
    if isinstance(chunksizes, int):
        return chunksizes
    if chunksizes is not None:
        chunksize = chunksizes.get(func.output_name, None)
        if chunksize is None:
            chunksize = chunksizes.get("", 1)
        if callable(chunksize):
            chunksize = chunksize(num_iterations)
        if not isinstance(chunksize, int) or chunksize <= 0:
            msg = f"Invalid chunksize {chunksize} for {func.output_name}"
            raise ValueError(msg)
        return chunksize
    return _get_optimal_chunk_size(num_iterations, executor)


def _get_optimal_chunk_size(
    total_items: int,
    executor: Executor,
    min_chunks_per_worker: int = 20,
) -> int:
    """Calculate an optimal chunk size for parallel processing.

    Parameters
    ----------
    total_items
        Total number of items to process
    executor
        The executor to use for parallel processing
    min_chunks_per_worker
        Minimum number of chunks each worker should process. Default of 20 provides good
        balance between load distribution and overhead for most workloads

    """
    try:
        n_cores = get_ncores(executor)
        n_cores = max(1, n_cores)
    except TypeError as e:
        warnings.warn(f"Automatic chunksize calculation failed with: {e}", stacklevel=2)
        n_cores = 1

    if total_items < n_cores * 2:
        return 1

    chunk_size = math.ceil(total_items / (n_cores * min_chunks_per_worker))
    return max(1, chunk_size)


def _maybe_parallel_map(
    func: PipeFunc,
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    status: Status | None,
    progress: ProgressTracker | None,
) -> list[Any]:
    if not indices:
        return []
    ex = _executor_for_func(func, executor)
    if ex is not None:
        assert executor is not None
        ex = maybe_update_slurm_executor_map(func, ex, executor, process_index, indices)
        chunksize = _chunksize_for_func(func, chunksizes, len(indices), ex)
        chunks = list(_chunk_indices(indices, chunksize))
        process_chunk = functools.partial(_process_chunk, process_index=process_index)
        return [_submit(process_chunk, ex, status, progress, len(chunk), chunk) for chunk in chunks]
    if status is not None:
        assert progress is not None
        process_index = _wrap_with_status_update(process_index, status, progress)  # type: ignore[assignment]
    # Put the process_index result in a tuple to have consistent shapes when func has mapspec
    return [(process_index(i),) for i in indices]


def _wrap_with_status_update(
    func: Callable[..., Any],
    status: Status,
    progress: ProgressTracker,
) -> Callable[..., Any]:
    def wrapped(*args: Any) -> Any:
        status.mark_in_progress()
        result = func(*args)
        status.mark_complete()
        progress.update_progress()
        return result

    return wrapped


def _maybe_execute_single(
    executor: dict[OUTPUT_TYPE, Executor] | None,
    status: Status | None,
    progress: ProgressTracker | None,
    func: PipeFunc,
    kwargs: dict[str, Any],
    store: dict[str, StoreType],
    cache: _CacheBase | None,
) -> Any:
    args = (func, kwargs, store, cache)  # args for _execute_single
    ex = _executor_for_func(func, executor)
    if ex:
        assert executor is not None
        ex = maybe_update_slurm_executor_single(func, ex, executor, kwargs)
        return _submit(_execute_single, ex, status, progress, 1, *args)
    if status is not None:
        assert progress is not None
        return _wrap_with_status_update(_execute_single, status, progress)(*args)
    return _execute_single(*args)


class _StoredValue(NamedTuple):
    value: Any
    exists: bool


def _load_from_store(
    output_name: OUTPUT_TYPE,
    store: dict[str, StoreType],
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


def _execute_single(
    func: PipeFunc,
    kwargs: dict[str, Any],
    store: dict[str, StoreType],
    cache: _CacheBase | None,
) -> Any:
    # Load the output if it exists
    output, exists = _load_from_store(func.output_name, store, return_output=True)
    if exists:
        return output

    # Otherwise, run the function
    _load_arrays(kwargs)

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


def _load_arrays(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = _maybe_load_array(v)


class _KwargsTask(NamedTuple):
    kwargs: dict[str, Any]
    task: tuple[Any, _MapSpecArgs] | Any


# NOTE: A similar async version of this function is provided below.
def _run_and_process_generation(
    *,
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StoreType],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None = None,
) -> None:
    tasks = _submit_generation(
        run_info,
        generation,
        store,
        fixed_indices,
        executor,
        chunksizes,
        progress,
        return_results,
        cache,
    )
    _process_generation(generation, tasks, store, outputs, run_info, return_results)


async def _run_and_process_generation_async(
    *,
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StoreType],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor],
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None = None,
    multi_run_manager: MultiRunManager | None = None,
) -> None:
    tasks = _submit_generation(
        run_info,
        generation,
        store,
        fixed_indices,
        executor,
        chunksizes,
        progress,
        return_results,
        cache,
    )
    maybe_finalize_slurm_executors(generation, executor, multi_run_manager)
    await _process_generation_async(generation, tasks, store, outputs, run_info, return_results)


# NOTE: A similar async version of this function is provided below.
def _process_generation(
    generation: list[PipeFunc],
    tasks: dict[PipeFunc, _KwargsTask],
    store: dict[str, StoreType],
    outputs: ResultDict,
    run_info: RunInfo,
    return_results: bool,  # noqa: FBT001
) -> None:
    for func in generation:
        _outputs = _process_task(func, tasks[func], store, run_info, return_results)
        if return_results:
            assert _outputs is not None
            outputs.update(_outputs)


async def _process_generation_async(
    generation: list[PipeFunc],
    tasks: dict[PipeFunc, _KwargsTask],
    store: dict[str, StoreType],
    outputs: ResultDict,
    run_info: RunInfo,
    return_results: bool,  # noqa: FBT001
) -> None:
    for func in generation:
        _outputs = await _process_task_async(func, tasks[func], store, run_info, return_results)
        if return_results:
            assert _outputs is not None
            outputs.update(_outputs)


def _submit_func(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StoreType],
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
    progress: ProgressTracker | None = None,
    return_results: bool = True,  # noqa: FBT001, FBT002
    cache: _CacheBase | None = None,
) -> _KwargsTask:
    kwargs = _func_kwargs(func, run_info, store)
    status = progress.progress_dict[func.output_name] if progress is not None else None
    cache = cache if func.cache else None
    if func.requires_mapping:
        args = _prepare_submit_map_spec(
            func,
            kwargs,
            run_info,
            store,
            fixed_indices,
            status,
            return_results,
            cache,
        )
        r = _maybe_parallel_map(
            func,
            args.process_index,
            args.missing,
            executor,
            chunksizes,
            status,
            progress,
        )
        task = r, args
    else:
        task = _maybe_execute_single(executor, status, progress, func, kwargs, store, cache)
    return _KwargsTask(kwargs, task)


def _update_status_if_needed(
    status: Status | None,
    existing: list[int],
    missing: list[int],
) -> None:
    if status is not None and status.n_total is None:
        status.n_total = len(missing) + len(existing)


def _executor_for_func(
    func: PipeFunc,
    executor: dict[OUTPUT_TYPE, Executor] | None,
) -> Executor | None:
    if executor is None:
        return None
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


def _submit_generation(
    run_info: RunInfo,
    generation: list[PipeFunc],
    store: dict[str, StoreType],
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,  # noqa: FBT001
    cache: _CacheBase | None = None,
) -> dict[PipeFunc, _KwargsTask]:
    return {
        func: _submit_func(
            func,
            run_info,
            store,
            fixed_indices,
            executor,
            chunksizes,
            progress,
            return_results,
            cache,
        )
        for func in generation
    }


def _output_from_mapspec_task(
    func: PipeFunc,
    store: dict[str, StoreType],
    args: _MapSpecArgs,
    outputs_list: list[list[Any]],
    run_info: RunInfo,
    return_results: bool,  # noqa: FBT001
) -> tuple[np.ndarray, ...] | None:
    arrays: tuple[StorageBase, ...] = tuple(
        store[name]  # type: ignore[misc]
        for name in at_least_tuple(func.output_name)
    )

    first = True
    for index, outputs in zip(args.missing, outputs_list):
        if first:
            shape = _maybe_resolve_shapes_from_map(func, store, args, outputs, return_results)
            first = False
        _update_result_array(args.result_arrays, index, outputs, shape, args.mask, func)
        _update_array(func, arrays, shape, args.mask, index, outputs, in_post_process=True)

    first = True
    for index in args.existing:
        outputs = [array.get_from_index(index) for array in args.arrays]
        if first:
            shape = _maybe_resolve_shapes_from_map(func, store, args, outputs, return_results)
            first = False
        _update_result_array(args.result_arrays, index, outputs, shape, args.mask, func)

    if not args.missing and not args.existing:  # shape variable does not exist
        shape = args.arrays[0].full_shape

    for name in at_least_tuple(func.output_name):
        run_info.resolve_downstream_shapes(name, store, shape=shape)

    if args.result_arrays is None:
        return None
    return tuple(x.reshape(shape) for x in args.result_arrays)  # type: ignore[union-attr]


def _internal_shape(output: Any, storage: StorageBase) -> tuple[int, ...]:
    shape = output.shape if isinstance(output, _InternalShape) else np.shape(output)
    return shape[: len(storage.internal_shape)]


def _maybe_set_internal_shape(output: Any, storage: StorageBase) -> None:
    if not shape_is_resolved(storage.internal_shape):
        internal_shape = _internal_shape(output, storage)
        storage.internal_shape = internal_shape


def _result(x: Any | Future) -> Any:
    return x.result() if isinstance(x, Future) else x


def _result_async(task: Future, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
    return asyncio.wrap_future(task, loop=loop)


def _to_result_dict(
    func: PipeFunc,
    kwargs: dict[str, Any],
    output: tuple[Any, ...],
    store: dict[str, StoreType],
) -> ResultDict:
    # Note that the kwargs still contain the StorageBase objects if mapspec was used.
    data = {
        output_name: Result(
            function=func.__name__,
            kwargs=kwargs,
            output_name=output_name,
            output=_output,
            store=store[output_name],
        )
        for output_name, _output in zip(at_least_tuple(func.output_name), output)
    }
    return ResultDict(data)


# NOTE: A similar async version of this function is provided below.
def _process_task(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    store: dict[str, StoreType],
    run_info: RunInfo,
    return_results: bool,  # noqa: FBT001
) -> ResultDict | None:
    kwargs, task = kwargs_task
    if func.requires_mapping:
        r, args = task
        chunk_outputs_list = [_result(x) for x in r]
        # Flatten the list of chunked outputs
        chained_outputs_list = list(itertools.chain(*chunk_outputs_list))
        output = _output_from_mapspec_task(
            func,
            store,
            args,
            chained_outputs_list,
            run_info,
            return_results,
        )
    else:
        r = _result(task)
        output = _dump_single_output(func, r, store, run_info)

    if return_results:
        assert output is not None
        return _to_result_dict(func, kwargs, output, store)
    return None


def _maybe_resolve_shapes_from_map(
    func: PipeFunc,
    store: dict[str, StoreType],
    args: _MapSpecArgs,
    outputs: list[Any],
    return_results: bool,  # noqa: FBT001
) -> tuple[int, ...]:
    for output, name in zip(outputs, at_least_tuple(func.output_name)):
        array = store[name]
        assert isinstance(array, StorageBase)
        _maybe_set_internal_shape(output, array)
    # Outside the loop above, just needs to do this once ⬇️
    assert isinstance(array, StorageBase)
    if args.result_arrays is None:
        args.result_arrays = _init_result_arrays(func.output_name, array.full_shape, return_results)
    return array.full_shape


async def _process_task_async(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    store: dict[str, StoreType],
    run_info: RunInfo,
    return_results: bool,  # noqa: FBT001
) -> ResultDict | None:
    kwargs, task = kwargs_task
    loop = asyncio.get_event_loop()
    if func.requires_mapping:
        r, args = task
        futs = [_result_async(x, loop) for x in r]
        chunk_outputs_list = await asyncio.gather(*futs)
        # Flatten the list of chunked outputs
        chained_outputs_list = list(itertools.chain(*chunk_outputs_list))
        output = _output_from_mapspec_task(
            func,
            store,
            args,
            chained_outputs_list,
            run_info,
            return_results,
        )
    else:
        assert isinstance(task, Future)
        r = await _result_async(task, loop)
        output = _dump_single_output(func, r, store, run_info)
    if return_results:
        assert output is not None
        return _to_result_dict(func, kwargs, output, store)
    return None
