from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pipefunc.map._run import (
    AsyncMap,
    _maybe_executor,
    _maybe_persist_memory,
    _process_task_async,
    maybe_multi_run_manager,
    prepare_run,
)
from pipefunc.map._run_eager import (
    _build_dependency_graph,
    _DependencyInfo,
    _FunctionTracker,
    _update_dependencies_and_submit,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor
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


def run_map_eager_async(
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
    """Asynchronously run a pipeline with eager scheduling for optimal parallelism.

    This implementation dynamically schedules functions as soon as their dependencies
    are met, without waiting for an entire generation to complete.

    Parameters are identical to run_map_async.

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
            dependency_info = _build_dependency_graph(pipeline)
            await _eager_scheduler_loop_async(
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
                multi_run_manager=multi_run_manager,
            )
        _maybe_persist_memory(store, persist_memory)
        return outputs

    task = asyncio.create_task(_run_pipeline())
    if progress is not None:
        progress.attach_task(task)

    return AsyncMap(task, run_info, progress, multi_run_manager)


async def _eager_scheduler_loop_async(
    *,
    dependency_info: _DependencyInfo,
    executor: dict[OUTPUT_TYPE, Executor],
    run_info: RunInfo,
    store: dict[str, Any],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None,
    multi_run_manager: MultiRunManager | None = None,
) -> None:
    """Dynamically submit and await tasks for functions as soon as they are ready."""
    tracker = _FunctionTracker(is_async=True)

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
            multi_run_manager,
        )

    # Process tasks as they complete
    while tracker.has_active_futures():
        await _process_completed_futures_async(
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
            multi_run_manager=multi_run_manager,
        )


async def _process_completed_futures_async(
    *,
    tracker: _FunctionTracker,
    dependency_info: _DependencyInfo,
    run_info: RunInfo,
    store: dict[str, Any],
    outputs: ResultDict,
    fixed_indices: dict[str, int | slice] | None,
    executor: dict[OUTPUT_TYPE, Executor],
    chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None,
    progress: ProgressTracker | None,
    return_results: bool,
    cache: _CacheBase | None,
    multi_run_manager: MultiRunManager | None,
) -> None:
    """Process completed futures and schedule new tasks asynchronously."""
    completed_funcs = await tracker.wait_for_futures_async()

    for func in completed_funcs:
        # Process the function results
        result = await _process_task_async(
            func,
            tracker.tasks[func],
            store,
            run_info,
            return_results,
        )

        if return_results and result is not None:
            outputs.update(result)

        # Mark this function as processed
        tracker.mark_function_processed(func)

        # Schedule new tasks
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
            multi_run_manager=multi_run_manager,
        )
