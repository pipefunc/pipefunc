from __future__ import annotations

import asyncio
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipefunc._pipefunc import PipeFunc
from pipefunc.map._run import (
    _KwargsTask,
    _maybe_executor,
    _maybe_persist_memory,
    _process_task,
    _submit_func,
    prepare_run,
)

from ._adaptive_scheduler_slurm_executor import maybe_finalize_slurm_executors

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pydantic
    from adaptive_scheduler import MultiRunManager

    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE, StorageType
    from pipefunc._widgets import ProgressTracker
    from pipefunc.cache import _CacheBase

    from ._result import ResultDict
    from ._run_info import RunInfo
    from ._types import UserShapeDict


def run_map_eager(
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
    """Eagerly schedule pipeline functions as soon as their dependencies are met.

    This implementation replaces the generation-by-generation scheduling used in the
    original run_map. Instead, it maintains a dependency counter for each function and
    submits tasks as soon as all their upstream dependencies (other PipeFunc nodes) have finished.

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
    # Prepare the run (this call sets up the run folder, storage, progress, etc.)
    pipeline, run_info, store, outputs, parallel, executor_dict, progress = prepare_run(
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

    dependency_info = _build_dependency_graph(pipeline)

    with _maybe_executor(executor_dict, parallel) as ex:
        _eager_scheduler_loop(
            dependency_info=dependency_info,
            executor=ex,
            run_info=run_info,
            store=store,
            outputs=outputs,
            fixed_indices=fixed_indices,
            chunksizes=chunksizes,
            progress=progress,
            return_results=return_results,
            cache=pipeline.cache,
        )

    if progress is not None:  # final update
        progress.update_progress(force=True)

    _maybe_persist_memory(store, persist_memory)
    return outputs


def _build_dependency_graph(pipeline: Pipeline) -> _DependencyInfo:
    """Build the dependency graph for PipeFunc nodes.

    Returns
    -------
    DependencyInfo
        Contains the dependency counts, child relationships, and initially ready functions.

    """
    graph = pipeline.graph
    remaining_deps: dict[PipeFunc, int] = {}
    children: dict[PipeFunc, list[PipeFunc]] = {}

    for f in pipeline.functions:
        # Count only incoming edges from other PipeFunc nodes
        count = sum(1 for n in graph.predecessors(f) if isinstance(n, PipeFunc))
        remaining_deps[f] = count

        # Record all downstream functions (children) that are PipeFunc instances
        children[f] = [child for child in graph.successors(f) if isinstance(child, PipeFunc)]

    # Initially, functions with no PipeFunc dependencies are ready
    ready = [f for f in pipeline.functions if remaining_deps[f] == 0]

    return _DependencyInfo(remaining_deps, children, ready)


@dataclass
class _DependencyInfo:
    """Container for dependency graph information."""

    remaining_deps: dict[PipeFunc, int]
    children: dict[PipeFunc, list[PipeFunc]]
    ready: list[PipeFunc]


def _ensure_future(x: Any) -> Future[Any]:
    """Ensure that an object is a Future."""
    if isinstance(x, Future):
        return x
    fut: Future[Any] = Future()
    fut.set_result(x)
    return fut


class _FunctionTracker:
    """Tracks function execution state during eager scheduling with unified sync/async API."""

    def __init__(self, *, is_async: bool = False) -> None:
        """Initialize the function tracker."""
        self.tasks: dict[PipeFunc, _KwargsTask] = {}
        self.future_to_func: dict[Future, PipeFunc] = {}
        self.func_futures: dict[PipeFunc, set[Future]] = {}
        self.completed_funcs: set[PipeFunc] = set()
        self.is_async: bool = is_async

        # Async-specific attributes
        if self.is_async:
            self.future_to_async_task: dict[Future, asyncio.Task] = {}
            self.pending_async_tasks: set[asyncio.Task] = set()

    def submit_function(
        self,
        func: PipeFunc,
        run_info: RunInfo,
        store: dict[str, Any],
        fixed_indices: dict[str, int | slice] | None,
        executor: dict[OUTPUT_TYPE, Executor] | None,
        chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
        progress: ProgressTracker | None,
        return_results: bool,  # noqa: FBT001
        cache: _CacheBase | None,
        multi_run_manager: MultiRunManager | None = None,
    ) -> None:
        """Submit a function and track its futures."""
        kwargs_task = _submit_func(
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
        self.tasks[func] = kwargs_task

        # Initialize the set of futures for this function
        self.func_futures[func] = set()

        # Track futures for this function
        if func.requires_mapping:
            r, _ = kwargs_task.task
            for task in r:
                fut = _ensure_future(task)
                self.future_to_func[fut] = func
                self.func_futures[func].add(fut)
        else:
            task = kwargs_task.task
            fut = _ensure_future(task)
            self.future_to_func[fut] = func
            self.func_futures[func].add(fut)

        if multi_run_manager is not None:
            assert self.is_async
            assert executor is not None
            maybe_finalize_slurm_executors([func], executor, multi_run_manager)

    def has_active_futures(self) -> bool:
        """Check if there are any active futures."""
        if self.is_async:
            return bool(self.future_to_func) or bool(self.pending_async_tasks)
        return bool(self.future_to_func)

    async def wait_for_futures_async(self) -> list[PipeFunc]:
        """Wait for futures to complete in async mode and return completed functions."""
        assert self.is_async
        # Create asyncio tasks for all pending futures
        loop = asyncio.get_event_loop()
        pending_futures = list(self.future_to_func.keys())
        for fut in pending_futures:
            if fut not in self.future_to_async_task:
                task = asyncio.ensure_future(asyncio.wrap_future(fut, loop=loop))
                self.future_to_async_task[fut] = task
                self.pending_async_tasks.add(task)

        # Wait for any task to complete if there are pending tasks
        if self.pending_async_tasks:
            done, _ = await asyncio.wait(
                self.pending_async_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Remove completed tasks from pending set
            for task in done:
                self.pending_async_tasks.discard(task)

        # Return functions whose futures have all completed
        return self._get_completed_functions()

    def wait_for_futures_sync(self) -> list[PipeFunc]:
        """Wait for futures to complete in sync mode and return completed functions."""
        assert not self.is_async

        done, _ = wait(self.future_to_func.keys(), return_when=FIRST_COMPLETED)

        # Get functions with potentially all futures completed
        completed_funcs = set()
        for fut in done:
            func = self.future_to_func.pop(fut)
            if func in self.func_futures:
                self.func_futures[func].discard(fut)
                if not self.func_futures[func] and func not in self.completed_funcs:
                    completed_funcs.add(func)

        return list(completed_funcs)

    def _get_completed_functions(self) -> list[PipeFunc]:
        """Return functions whose futures are all complete but not yet processed."""
        newly_completed = []

        for func in list(self.func_futures.keys()):
            if func in self.completed_funcs:
                continue

            all_futures = list(self.func_futures[func])
            if all(fut.done() for fut in all_futures) and all_futures:  # ensure non-empty
                newly_completed.append(func)

        return newly_completed

    def mark_function_processed(self, func: PipeFunc) -> None:
        """Mark a function as processed and clean up its tracking data."""
        # Clean up future references
        all_futures = list(self.func_futures[func])
        for fut in all_futures:
            if fut in self.future_to_func:
                del self.future_to_func[fut]
            if self.is_async and fut in self.future_to_async_task:
                del self.future_to_async_task[fut]

        # Clear futures for this function
        self.func_futures[func] = set()

        # Mark as completed
        self.completed_funcs.add(func)


def _eager_scheduler_loop(
    *,
    dependency_info: _DependencyInfo,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    run_info: RunInfo,
    store: dict[str, Any],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None,
) -> None:
    """Dynamically submit tasks for functions as soon as they are ready."""
    tracker = _FunctionTracker()

    # Submit initial ready tasks
    for f in dependency_info.ready:
        tracker.submit_function(
            f,
            run_info,
            store,
            fixed_indices,
            executor,
            chunksizes,
            progress,
            return_results,
            cache,
        )

    # Process tasks as they complete
    while tracker.has_active_futures():
        _process_completed_futures(
            tracker=tracker,
            dependency_info=dependency_info,
            run_info=run_info,
            store=store,
            outputs=outputs,
            fixed_indices=fixed_indices,
            executor=executor,
            chunksizes=chunksizes,
            progress=progress,
            return_results=return_results,
            cache=cache,
        )


def _process_completed_futures(
    *,
    tracker: _FunctionTracker,
    dependency_info: _DependencyInfo,
    run_info: RunInfo,
    store: dict[str, Any],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None,
) -> None:
    """Process completed futures and schedule new tasks."""
    completed_funcs = tracker.wait_for_futures_sync()

    for func in completed_funcs:
        # Process the task and update outputs
        result = _process_task(func, tracker.tasks[func], store, run_info, return_results)

        if return_results and result is not None:
            outputs.update(result)

        # Mark this function as processed
        tracker.mark_function_processed(func)

        # Update dependencies and submit new tasks
        _update_dependencies_and_submit(
            func=func,
            tracker=tracker,
            dependency_info=dependency_info,
            run_info=run_info,
            store=store,
            fixed_indices=fixed_indices,
            executor=executor,
            chunksizes=chunksizes,
            progress=progress,
            return_results=return_results,
            cache=cache,
        )


def _update_dependencies_and_submit(
    *,
    func: PipeFunc,
    tracker: _FunctionTracker,
    dependency_info: _DependencyInfo,
    run_info: RunInfo,
    store: dict[str, Any],
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None,
    multi_run_manager: MultiRunManager | None = None,
) -> None:
    """Update dependencies after a function completes and submit newly ready functions."""
    for child in dependency_info.children.get(func, []):
        dependency_info.remaining_deps[child] -= 1
        if dependency_info.remaining_deps[child] == 0:
            # Submit child function if all dependencies are resolved
            tracker.submit_function(
                child,
                run_info,
                store,
                fixed_indices,
                executor,
                chunksizes,
                progress,
                return_results,
                cache,
                multi_run_manager,
            )
