from __future__ import annotations

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

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pydantic

    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc._widgets import ProgressTracker
    from pipefunc.cache import _CacheBase

    from ._result import ResultDict
    from ._run_info import RunInfo


def run_map_dynamic(
    pipeline: Pipeline,
    inputs: dict[str, Any] | pydantic.BaseModel,
    run_folder: str | Path | None = None,
    internal_shapes: dict[str, Any] | None = None,
    *,
    output_names: set[Any] | None = None,
    parallel: bool = True,
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
    storage: str | dict[OUTPUT_TYPE, str] = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
    show_progress: bool = False,
    return_results: bool = True,
) -> ResultDict:
    """Dynamically schedule pipeline functions as soon as their dependencies are met.

    This implementation replaces the generation-by-generation scheduling used in the
    original run_map. Instead, it maintains a dependency counter for each function and
    submits tasks as soon as all their upstream dependencies (other PipeFunc nodes) have finished.

    Parameters
    ----------
    pipeline
        The Pipeline to run.
    inputs
        The input arguments.
    run_folder
        Folder to store run data.
    internal_shapes
        Shapes for intermediary outputs.
    output_names
        Which outputs to compute.
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
    storage
        Storage backend (see existing run_map).
    persist_memory, cleanup, fixed_indices, auto_subpipeline, show_progress, return_results
        See run_map documentation.

    Returns
    -------
    ResultDict
        The dictionary with outputs from the pipeline.

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
        _dynamic_scheduler_loop(
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


class _FunctionTracker:
    """Tracks function execution state during dynamic scheduling."""

    def __init__(self) -> None:
        """Initialize the function tracker."""
        self.tasks: dict[PipeFunc, _KwargsTask] = {}
        self.future_to_func: dict[Future, PipeFunc] = {}
        self.func_futures: dict[PipeFunc, set[Future]] = {}
        self.completed_funcs: set[PipeFunc] = set()

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
            for future in r:
                if isinstance(future, Future):
                    self.future_to_func[future] = func
                    self.func_futures[func].add(future)
        else:
            task = kwargs_task.task
            if isinstance(task, Future):
                self.future_to_func[task] = func
                self.func_futures[func].add(task)  # Fixed: using task instead of undefined future

    def is_function_complete(self, func: PipeFunc) -> bool:
        """Check if all futures for a function are completed."""
        return (
            func in self.func_futures
            and not self.func_futures[func]
            and func not in self.completed_funcs
        )

    def mark_function_complete(self, func: PipeFunc) -> None:
        """Mark a function as completed."""
        self.completed_funcs.add(func)

    def has_active_futures(self) -> bool:
        """Check if there are any active futures."""
        return bool(self.future_to_func)


def _dynamic_scheduler_loop(
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
    done, _ = wait(tracker.future_to_func.keys(), return_when=FIRST_COMPLETED)

    for fut in done:
        # Get the function associated with this future
        func = tracker.future_to_func.pop(fut)

        # Remove this future from the function's futures
        if func in tracker.func_futures:
            tracker.func_futures[func].discard(fut)

            # If all futures for this function are done, process the results
            if tracker.is_function_complete(func):
                # Process the task and update outputs
                result = _process_task(func, tracker.tasks[func], store, run_info, return_results)
                if return_results and result is not None:
                    outputs.update(result)

                # Mark this function as completed
                tracker.mark_function_complete(func)

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
            )
