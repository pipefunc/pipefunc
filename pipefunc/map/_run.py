from __future__ import annotations

import asyncio
import functools
import itertools
import math
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from pipefunc._error_handling import (
    ErrorInfo,
    create_propagated_error,
    scan_inputs_for_errors,
)
from pipefunc._pipefunc_utils import handle_pipefunc_error
from pipefunc._utils import (
    at_least_tuple,
    dump,
    ensure_block_allowed,
    get_ncores,
    is_running_in_ipynb,
    prod,
)
from pipefunc._widgets.helpers import maybe_async_task_status_widget
from pipefunc.cache import HybridCache, to_hashable
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot

from ._adaptive_scheduler_slurm_executor import (
    is_slurm_executor,
    maybe_finalize_slurm_executors,
    maybe_multi_run_manager,
    maybe_update_slurm_executor_map,
    maybe_update_slurm_executor_single,
    should_filter_error_indices,
)
from ._load import _load_from_store, maybe_load_data
from ._mapspec import MapSpec, _shape_to_key
from ._prepare import prepare_run
from ._result import DirectValue, Result, ResultDict
from ._run_info import _handle_cleanup_deprecation
from ._shapes import external_shape_from_mask, internal_shape_from_mask, shape_is_resolved
from ._storage_array._base import StorageBase, iterate_shape_indices, select_by_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Generator, Iterable, Sequence

    import pydantic
    from adaptive_scheduler import MultiRunManager

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE, StorageType
    from pipefunc._widgets.async_status_widget import AsyncTaskStatusWidget
    from pipefunc._widgets.progress_headless import HeadlessProgressTracker
    from pipefunc._widgets.progress_ipywidgets import IPyWidgetsProgressTracker
    from pipefunc._widgets.progress_rich import RichProgressTracker
    from pipefunc.cache import _CacheBase

    from ._prepare import Prepared
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
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None = None,
    storage: StorageType | None = None,
    persist_memory: bool = True,
    cleanup: bool | None = None,
    resume: bool = False,
    resume_validation: Literal["auto", "strict", "skip"] = "auto",
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool | Literal["rich", "ipywidgets", "headless"] | None = None,
    return_results: bool = True,
    error_handling: Literal["raise", "continue"] = "raise",
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
        Defaults to ``"file_array"`` if ``run_folder`` is provided, otherwise ``"dict"``.
    persist_memory
        Whether to write results to disk when memory based storage is used.
        Does not have any effect when file based storage is used.
    cleanup
        .. deprecated:: 0.89.0
            Use `resume` parameter instead. Will be removed in version 1.0.0.

        Whether to clean up the ``run_folder`` before running the pipeline.
        When set, takes priority over ``resume`` parameter.
        ``cleanup=True`` is equivalent to ``resume=False``.
        ``cleanup=False`` is equivalent to ``resume=True``.
    resume
        Whether to resume data from a previous run in the ``run_folder``.

        - ``False`` (default): Clean up the ``run_folder`` before running (fresh start).
        - ``True``: Attempt to load and resume results from a previous run.

        Note: If ``cleanup`` is specified, it takes priority over this parameter.
    resume_validation
        Controls validation strictness when reusing data from a previous run
        (only applies when ``resume=True``):

        - ``"auto"`` (default): Validate that inputs/defaults match the previous run.
          If equality comparison fails (returns ``None``), warn but proceed anyway.
        - ``"strict"``: Validate that inputs/defaults match. Raise an error if
          equality comparison fails.
        - ``"skip"``: Skip input/default validation entirely. **Use when your input
          objects have broken ``__eq__`` implementations that return incorrect results.**
          You are responsible for ensuring inputs are actually identical.

        Note: Shapes and MapSpecs are always validated regardless of this setting.
        Ignored when ``resume=False``.
    fixed_indices
        A dictionary mapping axes names to indices that should be fixed for the run.
        If not provided, all indices are iterated over.
    auto_subpipeline
        If ``True``, a subpipeline is created with the specified ``inputs``, using
        `Pipeline.subpipeline`. This allows to provide intermediate results in the ``inputs`` instead
        of providing the root arguments. If ``False``, all root arguments must be provided,
        and an exception is raised if any are missing.
    show_progress
        Whether to display a progress bar. Can be:

        - ``True``: Display a progress bar. Auto-selects based on environment:
          `ipywidgets` in Jupyter (if installed), otherwise `rich` (if installed).
        - ``False``: No progress bar.
        - ``"ipywidgets"``: Force `ipywidgets` progress bar (HTML-based).
          Shown only if in a Jupyter notebook and `ipywidgets` is installed.
        - ``"rich"``: Force `rich` progress bar (text-based).
          Shown only if `rich` is installed.
        - ``"headless"``: No progress bar, but the progress is still tracked internally.
        - ``None`` (default): Shows `ipywidgets` progress bar *only if*
          running in a Jupyter notebook and `ipywidgets` is installed.
          Otherwise, no progress bar is shown.
    return_results
        Whether to return the results of the pipeline. If ``False``, the pipeline is run
        without keeping the results in memory. Instead the results are only kept in the set
        ``storage``. This is useful for very large pipelines where the results do not fit into memory.
    error_handling
        How to handle errors during function execution:

        - ``"raise"`` (default): Stop execution on first error and raise exception
        - ``"continue"``: Continue execution, collecting errors as ErrorSnapshot objects

    """
    resume = _handle_cleanup_deprecation(cleanup, resume, stacklevel=2)

    prep = prepare_run(
        pipeline=pipeline,
        inputs=inputs,
        run_folder=run_folder,
        internal_shapes=internal_shapes,
        output_names=output_names,
        parallel=parallel,
        executor=executor,
        chunksizes=chunksizes,
        storage=storage,
        cleanup=cleanup,
        resume=resume,
        resume_validation=resume_validation,
        fixed_indices=fixed_indices,
        auto_subpipeline=auto_subpipeline,
        show_progress=show_progress,
        error_handling=error_handling,
        in_async=False,
    )

    with _maybe_executor(prep.executor, prep.parallel) as ex:
        for gen in prep.pipeline.topological_generations.function_lists:
            _run_and_process_generation(
                generation=gen,
                run_info=prep.run_info,
                store=prep.store,
                outputs=prep.outputs,
                fixed_indices=fixed_indices,
                executor=ex,
                chunksizes=prep.chunksizes,
                progress=prep.progress,
                return_results=return_results,
                cache=prep.pipeline.cache,
            )
    return _finalize_run_map(prep, persist_memory)


def _finalize_run_map(prep: Prepared, persist_memory: bool) -> ResultDict:
    if prep.progress is not None:  # final update
        prep.progress.update_progress(force=True)
    _maybe_persist_memory(prep.store, persist_memory)
    return prep.outputs


@dataclass
class AsyncMap:
    """An object returned by `run_map_async` to manage an asynchronous pipeline execution."""

    run_info: RunInfo
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None
    multi_run_manager: MultiRunManager | None
    status_widget: AsyncTaskStatusWidget | None
    _run_pipeline: Callable[[], Coroutine[Any, Any, ResultDict]]
    _display_widgets: bool
    _prepared: Prepared
    _task: asyncio.Task[ResultDict] | None = None
    _result_cache: ResultDict | None = None

    @property
    def task(self) -> asyncio.Task[ResultDict]:
        if self._task is None:
            msg = (
                "The task has not been started. Call `start()` inside an event loop or use"
                " `runner.result()` from synchronous code."
            )
            raise RuntimeError(msg)
        return self._task

    def result(self) -> ResultDict:
        """Wait for the pipeline to complete and return the results."""
        if self._result_cache is not None:
            return self._result_cache

        if is_running_in_ipynb():  # pragma: no cover
            if self._task is not None and self.task.done():
                result = self.task.result()
                self._result_cache = result
                return result
            msg = (
                "Cannot block the event loop when running in a Jupyter notebook."
                " Use `await runner.task` instead."
            )
            raise RuntimeError(msg)

        ensure_block_allowed()
        if self._task is None:

            async def _run_and_return() -> ResultDict:
                task = asyncio.create_task(self._run_pipeline())
                self._task = task
                self._attach_to_task(task)
                try:
                    result = await task
                finally:
                    self._task = None
                return result

            result = asyncio.run(_run_and_return())
            self._result_cache = result
            return result

        async def _await_task(task: asyncio.Task[ResultDict]) -> ResultDict:
            return await task

        loop = self.task.get_loop()
        coro = _await_task(self.task)
        result = asyncio.run_coroutine_threadsafe(coro, loop).result()
        self._result_cache = result
        return result

    def display(self) -> None:  # pragma: no cover
        """Display the pipeline widget."""
        if is_running_in_ipynb():
            if self.status_widget is not None:
                self.status_widget.display()
            if self.progress is not None:
                self.progress.display()
            if self.multi_run_manager is not None:
                self.multi_run_manager.display()
        else:
            print("⚠️ Display is only supported in Jupyter notebooks.")

    def start(self) -> AsyncMap:
        """Start the pipeline execution."""
        if self._task is not None:
            warnings.warn("Task is already running.", stacklevel=2)
            return self

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            warnings.warn(
                "No running asyncio event loop; call `runner.result()` to run synchronously "
                "or start the task inside an async context.",
                UserWarning,
                stacklevel=2,
            )
            return self

        self._task = loop.create_task(self._run_pipeline())
        self._attach_to_task(self._task)
        return self

    def _attach_to_task(self, task: asyncio.Task[ResultDict]) -> None:
        task.add_done_callback(self._cache_task_result)
        if self.progress is not None:
            self.progress.attach_task(task)
        self.status_widget = maybe_async_task_status_widget(task)
        if self._display_widgets:
            self.display()

    def _cache_task_result(self, task: asyncio.Task[ResultDict]) -> None:
        if task.cancelled():
            return
        if task.exception() is None:
            self._result_cache = task.result()
        else:
            self._result_cache = None


def run_map_async(
    pipeline: Pipeline,
    inputs: dict[str, Any] | pydantic.BaseModel,
    run_folder: str | Path | None = None,
    internal_shapes: UserShapeDict | None = None,
    *,
    output_names: set[OUTPUT_TYPE] | None = None,
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None = None,
    storage: StorageType | None = None,
    persist_memory: bool = True,
    cleanup: bool | None = None,
    resume: bool = False,
    resume_validation: Literal["auto", "strict", "skip"] = "auto",
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool | Literal["rich", "ipywidgets", "headless"] | None = None,
    return_results: bool = True,
    error_handling: Literal["raise", "continue"] = "raise",
    display_widgets: bool = True,
    start: bool = True,
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
        Defaults to ``"file_array"`` if ``run_folder`` is provided, otherwise ``"dict"``.
    persist_memory
        Whether to write results to disk when memory based storage is used.
        Does not have any effect when file based storage is used.
    cleanup
        .. deprecated:: 0.89.0
            Use `resume` parameter instead. Will be removed in version 1.0.0.

        Whether to clean up the ``run_folder`` before running the pipeline.
        When set, takes priority over ``resume`` parameter.
        ``cleanup=True`` is equivalent to ``resume=False``.
        ``cleanup=False`` is equivalent to ``resume=True``.
    resume
        Whether to resume data from a previous run in the ``run_folder``.

        - ``False`` (default): Clean up the ``run_folder`` before running (fresh start).
        - ``True``: Attempt to load and resume results from a previous run.

        Note: If ``cleanup`` is specified, it takes priority over this parameter.
    resume_validation
        Controls validation strictness when reusing data from a previous run
        (only applies when ``resume=True``):

        - ``"auto"`` (default): Validate that inputs/defaults match the previous run.
          If equality comparison fails (returns ``None``), warn but proceed anyway.
        - ``"strict"``: Validate that inputs/defaults match. Raise an error if
          equality comparison fails.
        - ``"skip"``: Skip input/default validation entirely. **Use when your input
          objects have broken ``__eq__`` implementations that return incorrect results.**
          You are responsible for ensuring inputs are actually identical.

        Note: Shapes and MapSpecs are always validated regardless of this setting.
        Ignored when ``resume=False``.
    fixed_indices
        A dictionary mapping axes names to indices that should be fixed for the run.
        If not provided, all indices are iterated over.
    auto_subpipeline
        If ``True``, a subpipeline is created with the specified ``inputs``, using
        `Pipeline.subpipeline`. This allows to provide intermediate results in the ``inputs`` instead
        of providing the root arguments. If ``False``, all root arguments must be provided,
        and an exception is raised if any are missing.
    show_progress
        Whether to display a progress bar. Can be:

        - ``True``: Display a progress bar. Auto-selects based on environment:
          `ipywidgets` in Jupyter (if installed), otherwise `rich` (if installed).
        - ``False``: No progress bar.
        - ``"ipywidgets"``: Force `ipywidgets` progress bar (HTML-based).
          Shown only if in a Jupyter notebook and `ipywidgets` is installed.
        - ``"rich"``: Force `rich` progress bar (text-based).
          Shown only if `rich` is installed.
        - ``"headless"``: No progress bar, but the progress is still tracked internally.
        - ``None`` (default): Shows `ipywidgets` progress bar *only if*
          running in a Jupyter notebook and `ipywidgets` is installed.
          Otherwise, no progress bar is shown.
    display_widgets
        Whether to call ``IPython.display.display(...)`` on widgets.
        Ignored if **outside** of a Jupyter notebook.
    return_results
        Whether to return the results of the pipeline. If ``False``, the pipeline is run
        without keeping the results in memory. Instead the results are only kept in the set
        ``storage``. This is useful for very large pipelines where the results do not fit into memory.
    error_handling
        How to handle errors during function execution:

        - ``"raise"`` (default): Stop execution on first error and raise exception
        - ``"continue"``: Continue execution, collecting errors as ErrorSnapshot objects
    start
        Whether to start the pipeline immediately. If ``False``, the pipeline is not started until the
        `start()` method on the `AsyncMap` instance is called.

    """
    resume = _handle_cleanup_deprecation(cleanup, resume, stacklevel=2)

    prep = prepare_run(
        pipeline=pipeline,
        inputs=inputs,
        run_folder=run_folder,
        internal_shapes=internal_shapes,
        output_names=output_names,
        parallel=True,
        executor=executor,
        chunksizes=chunksizes,
        storage=storage,
        cleanup=cleanup,
        resume=resume,
        resume_validation=resume_validation,
        fixed_indices=fixed_indices,
        auto_subpipeline=auto_subpipeline,
        show_progress=show_progress,
        in_async=True,
        error_handling=error_handling,
    )

    multi_run_manager = maybe_multi_run_manager(prep.executor)

    async def _run_pipeline() -> ResultDict:
        with _maybe_executor(prep.executor, parallel=True) as ex:
            assert ex is not None
            for gen in prep.pipeline.topological_generations.function_lists:
                await _run_and_process_generation_async(
                    generation=gen,
                    run_info=prep.run_info,
                    store=prep.store,
                    outputs=prep.outputs,
                    fixed_indices=fixed_indices,
                    executor=ex,
                    chunksizes=prep.chunksizes,
                    progress=prep.progress,
                    return_results=return_results,
                    cache=prep.pipeline.cache,
                    multi_run_manager=multi_run_manager,
                )
        _maybe_persist_memory(prep.store, persist_memory)
        return prep.outputs

    return _finalize_run_map_async(
        _run_pipeline,
        prep,
        multi_run_manager,
        start,
        display_widgets,
        prep,
    )


def _finalize_run_map_async(
    run_pipeline: Callable[[], Coroutine[Any, Any, ResultDict]],
    prep: Prepared,
    multi_run_manager: MultiRunManager | None,
    start: bool,
    display_widgets: bool,
    prepared: Prepared,
) -> AsyncMap:
    async_map = AsyncMap(
        run_info=prep.run_info,
        progress=prep.progress,
        multi_run_manager=multi_run_manager,
        status_widget=None,
        _run_pipeline=run_pipeline,
        _display_widgets=display_widgets,
        _prepared=prepared,
    )
    if start:
        async_map.start()
    return async_map


def _maybe_persist_memory(
    store: dict[str, StoreType],
    persist_memory: bool,
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
        if _is_error_snapshot(output):
            for output_name in func.output_name:
                _single_dump_single_output(output, output_name, store, run_info)
            return tuple(output for _ in func.output_name)
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            try:
                _output = func.output_picker(output, output_name)
            except Exception as e:
                if run_info.error_handling == "continue":
                    # Create ErrorSnapshot for output_picker failure
                    _output = ErrorSnapshot(
                        func.output_picker,
                        e,
                        args=(output, output_name),
                        kwargs={},
                    )
                else:
                    raise
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
    _load_data(selected)
    return selected


class _ErrorInfos(NamedTuple):
    map: dict[str, ErrorInfo] | None
    element: dict[str, ErrorInfo] | None


# Cached singleton for raise mode - avoids creating new tuple each iteration
_ERROR_INFOS_NONE: _ErrorInfos = _ErrorInfos(None, None)


def _should_scan_map_error_info(
    func: PipeFunc,
    error_handling: Literal["raise", "continue"],
) -> bool:
    return (
        error_handling == "continue"
        and func.requires_mapping
        and func.resources_scope == "map"
        and callable(func.resources)  # type: ignore[has-type]
    )


def _create_error_infos(
    element_kwargs: dict[str, Any],
    error_handling: Literal["raise", "continue"],
    precomputed_map_error_info: dict[str, ErrorInfo] | None,
) -> _ErrorInfos:
    if error_handling != "continue":
        return _ERROR_INFOS_NONE  # Cached singleton for raise mode

    element_error_info = scan_inputs_for_errors(element_kwargs)
    map_error_info = precomputed_map_error_info

    return _ErrorInfos(map_error_info, element_error_info)


@dataclass(frozen=True)
class ResourcesEval:
    state: Literal["evaluated", "skipped", "error"]
    resources: Any | None = None
    snapshot: ErrorSnapshot | None = None


# Cached singleton for when resources are not callable - avoids creating new dataclass each iteration
_RESOURCES_SKIPPED: ResourcesEval = ResourcesEval("skipped")


def eval_resources(  # noqa: PLR0911
    *,
    func: PipeFunc,
    map_kwargs: dict[str, Any],
    element_kwargs: dict[str, Any],
    error_infos: _ErrorInfos,
    mode: Literal["raise", "continue"],
) -> ResourcesEval:
    # Most common case: resources is not callable (None or static value)
    # Check this first to avoid dict lookups on every iteration
    if not callable(func.resources):  # type: ignore[has-type]
        return _RESOURCES_SKIPPED

    # Short-circuit if resources were evaluated in a prior phase
    if _EVALUATED_RESOURCES in element_kwargs:
        return ResourcesEval("evaluated", resources=element_kwargs[_EVALUATED_RESOURCES])
    # Map-level pre-evaluated resources are not used; element scope holds them.

    # If a previous pass recorded a resource evaluation error, don't re-evaluate
    if _RESOURCE_EVALUATION_ERROR in element_kwargs:
        snap = element_kwargs[_RESOURCE_EVALUATION_ERROR]
        return ResourcesEval("error", snapshot=snap)

    # choose scope
    if func.resources_scope == "map":
        if error_infos.map and func.resources_variable is None:
            return _RESOURCES_SKIPPED
        kw: dict[str, Any] | None = map_kwargs
    else:
        kw = None if error_infos.element else element_kwargs

    if kw is None:
        return _RESOURCES_SKIPPED

    try:
        res = func.resources(kw)  # type: ignore[has-type]
        return ResourcesEval("evaluated", resources=res)
    except Exception as exc:
        if mode != "continue":
            raise
        snap = ErrorSnapshot(func.resources, exc, (kw,), {})
        return ResourcesEval("error", snapshot=snap)


def _eval_and_apply_resources(
    *,
    func: PipeFunc,
    map_kwargs: dict[str, Any],
    element_kwargs: dict[str, Any],
    error_infos: _ErrorInfos,
    mode: Literal["raise", "continue"],
) -> tuple[ResourcesEval, _ErrorInfos]:
    """Evaluate resources and apply side effects to kwargs and error infos."""
    res = eval_resources(
        func=func,
        map_kwargs=map_kwargs,
        element_kwargs=element_kwargs,
        error_infos=error_infos,
        mode=mode,
    )
    if res.state == "evaluated":
        element_kwargs[_EVALUATED_RESOURCES] = res.resources
        return res, error_infos
    if res.state == "error" and mode == "continue" and res.snapshot is not None:
        element_error_info = dict(error_infos.element or {})
        element_error_info[_RESOURCE_EVALUATION_ERROR] = ErrorInfo.from_full_error(res.snapshot)
        new_infos = _ErrorInfos(error_infos.map, element_error_info)
        element_kwargs[_RESOURCE_EVALUATION_ERROR] = res.snapshot
        return res, new_infos
    return res, error_infos


def _prepare_execution_environment(
    func: PipeFunc,
    element_kwargs: dict[str, Any],
    map_kwargs: dict[str, Any],
    error_handling: Literal["raise", "continue"],
    map_error_info: dict[str, ErrorInfo] | None,
) -> tuple[ResourcesEval, _ErrorInfos]:
    error_infos = _create_error_infos(element_kwargs, error_handling, map_error_info)
    return _eval_and_apply_resources(
        func=func,
        map_kwargs=map_kwargs,
        element_kwargs=element_kwargs,
        error_infos=error_infos,
        mode=error_handling,
    )


def _prepare_kwargs_for_execution(
    func: PipeFunc,
    map_kwargs: dict[str, Any],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    index: int,
    error_handling: Literal["raise", "continue"],
    map_error_info: dict[str, ErrorInfo] | None,
) -> tuple[dict[str, Any], _ErrorInfos]:
    selected_kwargs = _select_kwargs(func, map_kwargs, shape, shape_mask, index)
    _res, error_infos = _prepare_execution_environment(
        func,
        selected_kwargs,
        map_kwargs,
        error_handling,
        map_error_info,
    )
    return selected_kwargs, error_infos


def _init_result_arrays(
    output_name: OUTPUT_TYPE,
    shape: ShapeTuple,
    return_results: bool,
) -> list[np.ndarray] | None:
    if not return_results or not shape_is_resolved(shape):
        return None
    return [np.empty(prod(shape), dtype=object) for _ in at_least_tuple(output_name)]


def _is_error_snapshot(output: Any) -> bool:
    return isinstance(output, (ErrorSnapshot, PropagatedErrorSnapshot))


def _pick_single_output(
    func: PipeFunc,
    output: Any,
    output_name: str,
    error_handling: Literal["raise", "continue"],
) -> Any:
    """Pick a single output, with error handling for output_picker failures."""
    assert func.output_picker is not None
    try:
        return func.output_picker(output, output_name)
    except Exception as e:
        if error_handling == "continue":
            # Create ErrorSnapshot for output_picker failure
            return ErrorSnapshot(
                func.output_picker,
                e,
                args=(output, output_name),
                kwargs={},
            )
        raise


def _pick_output(
    func: PipeFunc,
    output: Any,
    error_handling: Literal["raise", "continue"] = "raise",
) -> tuple[Any, ...]:
    output_names = at_least_tuple(func.output_name)
    if _is_error_snapshot(output):
        return tuple(output for _ in output_names)
    if func.output_picker is None:
        return tuple(output for _ in output_names)

    # Apply output_picker with error handling
    return tuple(
        _pick_single_output(func, output, output_name, error_handling)
        for output_name in output_names
    )


def _get_or_set_cache(
    func: PipeFunc,
    kwargs: dict[str, Any],
    cache: _CacheBase | None,
    ctx: ErrorContext,
    error_handling: Literal["raise", "continue"],
) -> Any:
    """Compute with unified error guards and cache the result."""
    if cache is None:
        if ctx.mode == "continue" and ctx.error_info:
            return create_propagated_error(ctx.error_info, func, kwargs)
        return _call_user(func, kwargs, ctx)
    cache_key = (func._cache_id, error_handling, to_hashable(kwargs))

    if cache_key in cache:
        return cache.get(cache_key)
    if isinstance(cache, HybridCache):
        t = time.monotonic()
    if ctx.mode == "continue" and ctx.error_info:
        result = create_propagated_error(ctx.error_info, func, kwargs)
    else:
        result = _call_user(func, kwargs, ctx)
    if isinstance(cache, HybridCache):
        cache.put(cache_key, result, time.monotonic() - t)
    else:
        cache.put(cache_key, result)
    return result


_EVALUATED_RESOURCES = "__pipefunc_internal_evaluated_resources__"
_RESOURCE_EVALUATION_ERROR = "__pipefunc_internal_resource_error__"


@dataclass(frozen=True)
class ErrorContext:
    mode: Literal["raise", "continue"]
    error_info: dict[str, ErrorInfo] | None


_CTX_RAISE: ErrorContext = ErrorContext(mode="raise", error_info=None)


def _maybe_wrap_exception(
    ctx: ErrorContext,
    func: PipeFunc,
    kwargs: dict[str, Any],
    exc: Exception,
) -> ErrorSnapshot:
    snapshot = handle_pipefunc_error(exc, func, kwargs, ctx.mode)
    if ctx.mode == "continue":
        assert snapshot is not None
        return snapshot
    raise exc  # pragma: no cover


def _try_shape(x: Any) -> tuple[int, ...]:
    try:
        return np.shape(x)
    except ValueError:
        # e.g., when inhomogeneous lists are passed
        return ()


@dataclass
class ErrorStub:
    """Lightweight marker for errors when return_results=False."""

    is_propagated: bool  # True for PropagatedErrorSnapshot, False for ErrorSnapshot


@dataclass
class _InternalShape:
    shape: tuple[int, ...]

    @classmethod
    def from_outputs(cls, outputs: tuple[Any]) -> tuple[Any, ...]:
        return tuple(
            ErrorStub(is_propagated=isinstance(output, PropagatedErrorSnapshot))
            if isinstance(output, (ErrorSnapshot, PropagatedErrorSnapshot))
            else cls(_try_shape(output))
            for output in outputs
        )


def _call_user(
    func: PipeFunc,
    kwargs: dict[str, Any],
    ctx: ErrorContext,
) -> Any:
    try:
        return func(**kwargs)
    except Exception as e:  # noqa: BLE001
        return _maybe_wrap_exception(ctx, func, kwargs, e)


def _entry_contains_error(entry: Any) -> bool:
    if isinstance(entry, (ErrorSnapshot, PropagatedErrorSnapshot, ErrorStub)):
        return True
    return any(
        isinstance(value, (ErrorSnapshot, PropagatedErrorSnapshot, ErrorStub))
        for value in at_least_tuple(entry)
    )


def _count_errors_in_result(result: Any) -> int:
    if isinstance(result, list | tuple):
        return sum(_entry_contains_error(entry) for entry in result)
    return int(_entry_contains_error(result))


def _select_kwargs_and_eval_resources_raise(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    index: int,
) -> dict[str, Any]:
    """Select per-index kwargs and evaluate resources for raise mode."""
    selected_kwargs = _select_kwargs(func, kwargs, shape, shape_mask, index)
    if callable(func.resources):  # type: ignore[has-type]
        kw_for_resources = kwargs if func.resources_scope == "map" else selected_kwargs
        evaluated_resources = func.resources(kw_for_resources)  # type: ignore[has-type]
        selected_kwargs[_EVALUATED_RESOURCES] = evaluated_resources
    return selected_kwargs


def _run_iteration_and_process(
    index: int,
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: ShapeTuple,
    shape_mask: tuple[bool, ...],
    arrays: Sequence[StorageBase],
    cache: _CacheBase | None = None,
    *,
    error_handling: Literal["raise", "continue"],
    map_error_info: dict[str, ErrorInfo] | None = None,
    return_results: bool = True,
    force_dump: bool = False,
) -> tuple[Any, ...]:
    if error_handling == "raise":
        # Hot-path for default raise mode: avoid continue-mode scaffolding.
        selected_kwargs = _select_kwargs_and_eval_resources_raise(
            func,
            kwargs,
            shape,
            shape_mask,
            index,
        )
        ctx = _CTX_RAISE
    else:
        selected_kwargs, error_infos = _prepare_kwargs_for_execution(
            func,
            kwargs,
            shape,
            shape_mask,
            index,
            error_handling,
            map_error_info,
        )
        # Early error detection centralized through the guard helper
        ctx = ErrorContext(mode=error_handling, error_info=error_infos.element)
    output = _get_or_set_cache(func, selected_kwargs, cache, ctx, error_handling)
    outputs = _pick_output(func, output, error_handling)
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
        # XOR logic ensures dump happens exactly once:
        # - dump_in_subprocess=True + in_post_process=False → dump (in subprocess)
        # - dump_in_subprocess=True + in_post_process=True → skip (already dumped)
        # - dump_in_subprocess=False + in_post_process=False → skip (will dump later)
        # - dump_in_subprocess=False + in_post_process=True → dump (in main process)
        # Error objects follow the same logic - no special treatment needed.
        if force_dump or (array.dump_in_subprocess ^ in_post_process):
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
    parallel: bool,
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
    return_results: bool,
    cache: _CacheBase | None = None,
) -> _MapSpecArgs:
    assert isinstance(func.mapspec, MapSpec)
    shape = run_info.resolved_shapes[func.output_name]
    mask = run_info.shape_masks[func.output_name]
    arrays: list[StorageBase] = [store[name] for name in at_least_tuple(func.output_name)]  # type: ignore[misc]
    result_arrays = _init_result_arrays(func.output_name, shape, return_results)
    map_error_info = (
        scan_inputs_for_errors(kwargs)
        if _should_scan_map_error_info(func, run_info.error_handling)
        else None
    )
    process_index = functools.partial(
        _run_iteration_and_process,
        func=func,
        kwargs=kwargs,
        shape=shape,
        shape_mask=mask,
        arrays=arrays,
        cache=cache,
        error_handling=run_info.error_handling,
        map_error_info=map_error_info,
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
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
    chunksize: int,
    *args: Any,
) -> Future:
    if status is None:
        return executor.submit(func, *args)
    assert progress is not None
    status.mark_in_progress(n=chunksize)
    fut = executor.submit(func, *args)

    def _on_done(future: Future) -> None:
        exc = future.exception()
        if exc is not None:
            successes = 0
            failures = chunksize
        else:
            result = future.result()
            failures = _count_errors_in_result(result)
            successes = chunksize - failures
        status.mark_finished(successes=successes, failures=failures)
        if not progress.in_async:
            progress.update_progress()

    fut.add_done_callback(_on_done)
    return fut


def _process_chunk(
    chunk: list[int],
    process_index: functools.partial[tuple[Any, ...]],
) -> list[Any]:
    """Process a chunk of indices."""
    return [process_index(index) for index in chunk]


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
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
    num_iterations: int,
    executor: Executor,
) -> int:
    if isinstance(chunksizes, int):
        return chunksizes
    if chunksizes is None:
        return _get_optimal_chunk_size(num_iterations, executor)
    chunksize = None
    if func.output_name in chunksizes:
        chunksize = chunksizes[func.output_name]
    elif "" in chunksizes:
        chunksize = chunksizes[""]

    if callable(chunksize):
        chunksize = chunksize(num_iterations)

    if chunksize is None:
        chunksize = _get_optimal_chunk_size(num_iterations, executor)

    if not isinstance(chunksize, int) or chunksize <= 0:
        msg = f"Invalid chunksize {chunksize} for {func.output_name}"
        raise ValueError(msg)
    return chunksize


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
    if is_slurm_executor(executor):
        return 1
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


def _scan_error_flags(
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
) -> list[bool]:
    """Return booleans indicating whether each index has input errors."""
    kw = process_index.keywords
    func = kw["func"]
    flags: list[bool] = []
    for idx in indices:
        selected = _select_kwargs(func, kw["kwargs"], kw["shape"], kw["shape_mask"], idx)
        flags.append(bool(scan_inputs_for_errors(selected)))
    return flags


def _all_indices_propagate_errors(
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
) -> bool:
    kw = process_index.keywords
    if kw["error_handling"] != "continue":
        return False
    return all(_scan_error_flags(process_index, indices))


def _split_error_and_valid_indices(
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
) -> tuple[list[int], list[int]]:
    """Split indices into those with propagated errors and those that are valid.

    Returns
    -------
    (error_indices, valid_indices)

    """
    flags = _scan_error_flags(process_index, indices)
    error_indices = [i for i, is_err in zip(indices, flags) if is_err]
    valid_indices = [i for i, is_err in zip(indices, flags) if not is_err]
    return error_indices, valid_indices


def _route_indices_for_executor(
    executor: Executor,
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
    error_handling: str,
) -> tuple[list[int], list[int]]:
    """Decide which indices to run locally vs submit to the executor.

    - SLURM element-scope + continue: split error vs valid indices
    - Otherwise: if all indices propagate errors → process locally
    - Otherwise: submit everything to executor
    """
    if should_filter_error_indices(executor, error_handling):
        return _split_error_and_valid_indices(process_index, indices)
    if _all_indices_propagate_errors(process_index, indices):
        return indices, []
    return [], indices


def _submit_executor_indices(
    func: PipeFunc,
    ex: Executor,
    executor: dict[OUTPUT_TYPE, Executor],
    process_index: functools.partial[tuple[Any, ...]],
    executor_indices: list[int],
    status: Status | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
) -> list[_ChunkTask]:
    """Encapsulate the 'update → chunk → submit' path for executor indices."""
    ex = maybe_update_slurm_executor_map(func, ex, executor, process_index, executor_indices)
    chunksize = _chunksize_for_func(func, chunksizes, len(executor_indices), ex)
    chunks = list(_chunk_indices(executor_indices, chunksize))
    process_chunk = functools.partial(_process_chunk, process_index=process_index)
    return [
        _ChunkTask(
            _submit(process_chunk, ex, status, progress, len(chunk), chunk),
            tuple(chunk),
        )
        for chunk in chunks
    ]


def _maybe_parallel_map(
    func: PipeFunc,
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
    status: Status | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
) -> list[_ChunkTask]:
    """Submit map work, balancing local error-processing and executor tasks.

    Steps
    -----
    1. Route: decide local vs executor indices via route_indices_for_executor.
    2. Local: process error indices immediately (optionally with progress wrapping).
    3. Executor: update executor, chunk, and submit valid indices.
    """
    if not indices:
        return []
    ex = _executor_for_func(func, executor)
    if ex is not None:
        assert executor is not None

        # 1) Route indices to local vs executor according to policy
        local_indices, executor_indices = _route_indices_for_executor(
            ex,
            process_index,
            indices,
            process_index.keywords["error_handling"],
        )

        tasks: list[_ChunkTask] = []

        # 2) Process local (error) indices immediately without submitting to executor
        if local_indices:
            tasks.extend(_submit_local_indices(process_index, local_indices, status, progress))

        # 3) Submit only valid indices to the executor
        if executor_indices:
            tasks.extend(
                _submit_executor_indices(
                    func,
                    ex,
                    executor,
                    process_index,
                    executor_indices,
                    status,
                    progress,
                    chunksizes,
                ),
            )

        return tasks
    if status is not None:
        assert progress is not None
        process_index = _wrap_with_status_update(process_index, status, progress)  # type: ignore[assignment]
    # Put the process_index result in a tuple to have consistent shapes when func has mapspec
    return [_ChunkTask((process_index(i),), (i,)) for i in indices]


def _submit_local_indices(
    process_index: Callable[..., Any],
    local_indices: list[int],
    status: Status | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
) -> list[_ChunkTask]:
    """Create local tasks for error indices with consistent status handling."""
    wrapped = (
        _wrap_with_status_update(process_index, status, progress)  # type: ignore[arg-type]
        if status is not None
        else process_index
    )
    return [_ChunkTask((wrapped(i),), (i,)) for i in local_indices]


def _wrap_with_status_update(
    func: Callable[..., Any],
    status: Status,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker,
) -> Callable[..., Any]:
    def wrapped(*args: Any) -> Any:
        status.mark_in_progress()
        result = func(*args)
        failures = _count_errors_in_result(result)
        successes = max(0, 1 - failures)
        status.mark_finished(successes=successes, failures=failures)
        if not progress.in_async:
            progress.update_progress()
        return result

    return wrapped


def _maybe_execute_single(
    executor: dict[OUTPUT_TYPE, Executor] | None,
    status: Status | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
    func: PipeFunc,
    kwargs: dict[str, Any],
    store: dict[str, StoreType],
    cache: _CacheBase | None,
    error_handling: Literal["raise", "continue"],
) -> Any:
    args = (func, kwargs, store, cache, error_handling)  # args for _execute_single
    res, _error_infos = _prepare_execution_environment(func, kwargs, kwargs, error_handling, None)
    ex = _executor_for_func(func, executor)
    if ex is not None and res.state != "error":
        resolved_resources = kwargs.get(_EVALUATED_RESOURCES)
        if resolved_resources is None and not callable(func.resources):  # type: ignore[has-type]
            resolved_resources = func.resources
        assert executor is not None
        ex = maybe_update_slurm_executor_single(func, ex, executor, resolved_resources)
        return _submit(_execute_single, ex, status, progress, 1, *args)
    if status is not None:
        assert progress is not None
        return _wrap_with_status_update(_execute_single, status, progress)(*args)
    return _execute_single(*args)


def _execute_single(
    func: PipeFunc,
    kwargs: dict[str, Any],
    store: dict[str, StoreType],
    cache: _CacheBase | None,
    error_handling: Literal["raise", "continue"],
) -> Any:
    # Load the output if it exists
    output, exists = _load_from_store(func.output_name, store, return_output=True)
    if exists:
        return output

    # Otherwise, run the function
    _load_data(kwargs)
    if error_handling == "raise":
        return _get_or_set_cache(func, kwargs, cache, _CTX_RAISE, "raise")

    _res, error_infos = _prepare_execution_environment(func, kwargs, kwargs, error_handling, None)
    ctx = ErrorContext(mode=error_handling, error_info=error_infos.element)
    return _get_or_set_cache(func, kwargs, cache, ctx, error_handling)


def _load_data(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = maybe_load_data(v)


class _SingleTask(NamedTuple):
    value: Any


class _ChunkTask(NamedTuple):
    value: Any
    indices: tuple[int, ...]


class _MapTask(NamedTuple):
    chunk_tasks: list[_ChunkTask]
    args: _MapSpecArgs


class _KwargsTask(NamedTuple):
    kwargs: dict[str, Any]
    task: _MapTask | _SingleTask


# NOTE: A similar async version of this function is provided below.
def _run_and_process_generation(
    *,
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StoreType],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
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
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
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
    return_results: bool,
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
    return_results: bool,
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
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None = None,
    progress: IPyWidgetsProgressTracker
    | RichProgressTracker
    | HeadlessProgressTracker
    | None = None,
    return_results: bool = True,  # noqa: FBT002
    cache: _CacheBase | None = None,
) -> _KwargsTask:
    kwargs = _func_kwargs(func, run_info, store)
    status = progress.progress_dict[func.output_name] if progress is not None else None
    cache = cache if func.cache else None
    task: _MapTask | _SingleTask
    if func.requires_mapping:
        args = _prepare_submit_map_spec(
            func,
            kwargs,
            run_info,
            store,
            fixed_indices,
            status,
            return_results,
            cache=cache,
        )
        chunk_tasks = _maybe_parallel_map(
            func,
            args.process_index,
            args.missing,
            executor,
            chunksizes,
            status,
            progress,
        )
        task = _MapTask(chunk_tasks, args)
    else:
        single_value = _maybe_execute_single(
            executor,
            status,
            progress,
            func,
            kwargs,
            store,
            cache,
            run_info.error_handling,
        )
        task = _SingleTask(single_value)
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
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int] | None] | None,
    progress: IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None,
    return_results: bool,
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
    return_results: bool,
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


def _raise_and_set_error_snapshot(
    exc: Exception,
    func: PipeFunc,
    kwargs: dict[str, Any],
    *,
    index: int | None = None,
    run_info: RunInfo | None = None,
) -> None:
    if index is not None:
        assert run_info is not None, "run_info required when index is provided"
        shape = run_info.resolved_shapes[func.output_name]
        mask = run_info.shape_masks[func.output_name]
        kwargs = _select_kwargs(func, kwargs, shape, mask, index)
    handle_pipefunc_error(exc, func, kwargs)


def _result(
    x: Any | Future,
    func: PipeFunc,
    kwargs: dict[str, Any],
    index: int | None = None,
    run_info: RunInfo | None = None,
) -> Any:
    if isinstance(x, Future):
        try:
            return x.result()
        except Exception as e:
            _raise_and_set_error_snapshot(e, func, kwargs, index=index, run_info=run_info)
            raise  # pragma: no cover
    return x


async def _result_async(
    task: Future,
    loop: asyncio.AbstractEventLoop,
    func: PipeFunc,
    kwargs: dict[str, Any],
    index: int | None = None,
    run_info: RunInfo | None = None,
) -> Any:
    try:
        return await asyncio.wrap_future(task, loop=loop)
    except Exception as e:
        _raise_and_set_error_snapshot(e, func, kwargs, index=index, run_info=run_info)
        raise  # pragma: no cover


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
    return_results: bool,
) -> ResultDict | None:
    kwargs, task = kwargs_task
    if func.requires_mapping:
        assert isinstance(task, _MapTask)
        chunk_outputs_list = []
        for chunk_task in task.chunk_tasks:
            # TODO: Capture the precise failing index when a chunk contains multiple items.
            representative_index = chunk_task.indices[0] if chunk_task.indices else None
            chunk_outputs_list.append(
                _result(
                    chunk_task.value,
                    func,
                    kwargs,
                    representative_index,
                    run_info,
                ),
            )
        # Flatten the list of chunked outputs
        chained_outputs_list = list(itertools.chain(*chunk_outputs_list))
        output = _output_from_mapspec_task(
            func,
            store,
            task.args,
            chained_outputs_list,
            run_info,
            return_results,
        )
    else:
        assert isinstance(task, _SingleTask)
        r = _result(task.value, func, kwargs, run_info=run_info)
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
    return_results: bool,
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
    return_results: bool,
) -> ResultDict | None:
    kwargs, task = kwargs_task
    loop = asyncio.get_event_loop()
    if func.requires_mapping:
        assert isinstance(task, _MapTask)
        chunk_outputs_list: list[list[Any]] = []
        for chunk_task in task.chunk_tasks:
            representative_index = chunk_task.indices[0] if chunk_task.indices else None
            value = chunk_task.value
            if isinstance(value, Future):
                outputs = await _result_async(
                    value,
                    loop,
                    func,
                    kwargs,
                    representative_index,
                    run_info,
                )
            else:
                outputs = _result(
                    value,
                    func,
                    kwargs,
                    representative_index,
                    run_info,
                )
            chunk_outputs_list.append(outputs)
        # Flatten the list of chunked outputs
        chained_outputs_list = list(itertools.chain(*chunk_outputs_list))
        output = _output_from_mapspec_task(
            func,
            store,
            task.args,
            chained_outputs_list,
            run_info,
            return_results,
        )
    else:
        assert isinstance(task, _SingleTask)
        assert isinstance(task.value, Future)
        r = await _result_async(task.value, loop, func, kwargs, run_info=run_info)
        output = _dump_single_output(func, r, store, run_info)
    if return_results:
        assert output is not None
        return _to_result_dict(func, kwargs, output, store)
    return None
