"""Helpers for adaptive_scheduler.SlurmExecutor class introduced in 2.13.0."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

from pipefunc._utils import at_least_tuple, is_min_version

if TYPE_CHECKING:
    import functools
    from concurrent.futures import Executor

    from adaptive_scheduler import MultiRunManager, SlurmExecutor

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.resources import Resources


def _adaptive_scheduler_imported() -> bool:
    """Check if the adaptive_scheduler package is imported and at the correct version."""
    if "adaptive_scheduler" not in sys.modules:
        return False
    # The SlurmExecutor was introduced in version 2.13.0
    min_version = "2.13.0"
    if not is_min_version("adaptive_scheduler", min_version):
        msg = f"The 'adaptive_scheduler' package must be at least version {min_version}."
        warnings.warn(msg, stacklevel=2)
        return False
    return True


def maybe_convert_slurm_executor(
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None,
    pipeline: Pipeline,
    in_async: bool,  # noqa: FBT001
) -> Executor | dict[OUTPUT_TYPE, Executor] | None:
    """Convert a single SlurmExecutor to a dict of executors if needed."""
    if _adaptive_scheduler_imported():
        from adaptive_scheduler import SlurmExecutor

        if (isinstance(executor, SlurmExecutor) or executor is SlurmExecutor) and not in_async:
            msg = "Cannot use an `adaptive_scheduler.SlurmExecutor` in non-async mode, use `pipeline.run_async` instead."
            raise ValueError(msg)

        if isinstance(executor, SlurmExecutor):
            # If a single SlurmExecutor is provided, we need to create a new one for each output
            return {
                func.output_name: executor.new(update={"name": f"{executor.name}-{i}"})
                for i, func in enumerate(pipeline.sorted_functions)
            }
    return executor


def _executors_for_generation(
    generation: list[PipeFunc],
    executor: dict[OUTPUT_TYPE, Executor],
) -> list[Executor]:
    from ._run import _executor_for_func

    executors = []
    for func in generation:
        ex = _executor_for_func(func, executor)
        if ex is not None and ex not in executors:
            executors.append(ex)
    return executors


def maybe_finalize_slurm_executors(
    generation: list[PipeFunc],
    executor: dict[OUTPUT_TYPE, Executor] | None,
    multi_run_manager: MultiRunManager | None,
) -> None:
    if executor is None:
        return
    executors = _executors_for_generation(generation, executor)
    for ex in executors:
        if _adaptive_scheduler_imported():
            from adaptive_scheduler import SlurmExecutor

            if isinstance(ex, SlurmExecutor):
                assert multi_run_manager is not None
                run_manager = ex.finalize()
                multi_run_manager.add_run_manager(run_manager)
    return


def maybe_multi_run_manager(
    executor: dict[OUTPUT_TYPE, Executor] | None,
) -> MultiRunManager | None:
    if isinstance(executor, dict) and _adaptive_scheduler_imported():
        from adaptive_scheduler import MultiRunManager, SlurmExecutor

        for ex in executor.values():
            if isinstance(ex, SlurmExecutor) or ex is SlurmExecutor:
                return MultiRunManager()
    return None


def is_slurm_executor(executor: Executor | None) -> TypeGuard[SlurmExecutor]:
    if not _adaptive_scheduler_imported():
        return False
    from adaptive_scheduler import SlurmExecutor

    return isinstance(executor, SlurmExecutor)


def is_slurm_executor_type(executor: Executor | None) -> TypeGuard[type[SlurmExecutor]]:
    if not _adaptive_scheduler_imported():
        return False
    from adaptive_scheduler import SlurmExecutor

    return executor is SlurmExecutor


def _slurm_name(output_name: OUTPUT_TYPE) -> str:
    return "-".join(at_least_tuple(output_name))


def slurm_executor_for_map(
    executor: SlurmExecutor | type[SlurmExecutor],
    process_index: functools.partial[tuple[Any, ...]],
    seq: list[int],
) -> Executor:  # Actually SlurmExecutor, but mypy doesn't like it
    from adaptive_scheduler import SlurmExecutor

    func = process_index.keywords["func"]
    if func.resources is not None:
        resources_list: list[dict[str, Any]] = []
        for i in seq:
            resources = _resources_from_process_index(process_index, i)
            scheduler_resources = _adaptive_scheduler_resource_dict(resources)
            resources_list.append(scheduler_resources)
        executor_kwargs = _list_of_dicts_to_dict_of_tuples(resources_list)
    else:
        executor_kwargs = {}
    executor_kwargs["name"] = _slurm_name(func.output_name)  # type: ignore[assignment]
    if isinstance(executor, SlurmExecutor):
        return executor.new(update=executor_kwargs)
    return SlurmExecutor(**executor_kwargs)


def slurm_executor_for_single(
    executor: SlurmExecutor | type[SlurmExecutor],
    func: PipeFunc,
    kwargs: dict[str, Any],
) -> Executor:
    from adaptive_scheduler import SlurmExecutor

    resources: Resources | None = (
        func.resources(kwargs) if callable(func.resources) else func.resources  # type: ignore[has-type]
    )
    executor_kwargs = _adaptive_scheduler_resource_dict(resources) if resources is not None else {}
    executor_kwargs["name"] = _slurm_name(func.output_name)
    if isinstance(executor, SlurmExecutor):
        return executor.new(update=executor_kwargs)
    return SlurmExecutor(**executor_kwargs)


def _adaptive_scheduler_resource_dict(resources: Resources | None) -> dict[str, Any]:
    if resources is None:
        return {}
    assert _adaptive_scheduler_imported()
    from .adaptive_scheduler import __executor_type, __extra_scheduler

    return {
        "extra_scheduler": __extra_scheduler(resources),
        "executor_type": __executor_type(resources),
        "cores_per_node": resources.cpus_per_node or resources.cpus,
        "nodes": resources.nodes or 1,
        "partition": resources.partition,
    }


def _resources_from_process_index(
    process_index: functools.partial[tuple[Any, ...]],
    index: int,
) -> Resources | None:
    from ._run import _EVALUATED_RESOURCES, _select_kwargs_and_eval_resources
    # Import here to avoid circular imports

    kw = process_index.keywords
    assert kw["func"].resources is not None
    # NOTE: We are executing this line below 2 times for each index.
    # This is not ideal, if it becomes a performance issue we can cache
    # the result.
    selected = _select_kwargs_and_eval_resources(
        kw["func"],
        kw["kwargs"],
        kw["shape"],
        kw["shape_mask"],
        index,
    )
    return selected[_EVALUATED_RESOURCES]


T = TypeVar("T")


def _list_of_dicts_to_dict_of_tuples(
    list_of_dicts: list[dict[str, T]],
) -> dict[str, tuple[T, ...]]:
    return {k: tuple(d[k] for d in list_of_dicts) for k in list_of_dicts[0]}
