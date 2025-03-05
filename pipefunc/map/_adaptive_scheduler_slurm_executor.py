"""Helpers for adaptive_scheduler.SlurmExecutor class introduced in 2.13.3."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

from pipefunc._utils import at_least_tuple, is_imported, is_min_version

if TYPE_CHECKING:
    import functools
    from concurrent.futures import Executor

    from adaptive_scheduler import MultiRunManager, SlurmExecutor

    from pipefunc import PipeFunc
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.resources import Resources


def validate_slurm_executor(
    executor: dict[OUTPUT_TYPE, Executor] | None,
    in_async: bool,  # noqa: FBT001
) -> None:
    if executor is None or in_async or not _adaptive_scheduler_imported():
        return
    for ex in executor.values():
        if _is_slurm_executor(ex) or _is_slurm_executor_type(ex):
            msg = "Cannot use an `adaptive_scheduler.SlurmExecutor` in non-async mode, use `pipeline.map_async` instead."
            raise ValueError(msg)


def maybe_multi_run_manager(
    executor: dict[OUTPUT_TYPE, Executor] | None,
) -> MultiRunManager | None:
    if isinstance(executor, dict) and _adaptive_scheduler_imported():
        from adaptive_scheduler import MultiRunManager

        for ex in executor.values():
            if _is_slurm_executor(ex) or _is_slurm_executor_type(ex):
                return MultiRunManager()
    return None


def maybe_update_slurm_executor_single(
    func: PipeFunc,
    ex: Executor,
    executor: dict[OUTPUT_TYPE, Executor],
    kwargs: dict[str, Any],
) -> Executor:
    if _is_slurm_executor(ex) or _is_slurm_executor_type(ex):
        ex = _slurm_executor_for_single(ex, func, kwargs)
        assert isinstance(executor, dict)
        executor[func.output_name] = ex  # type: ignore[assignment]
    return ex


def maybe_update_slurm_executor_map(
    func: PipeFunc,
    ex: Executor,
    executor: dict[OUTPUT_TYPE, Executor],
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
) -> Executor:
    if _is_slurm_executor(ex) or _is_slurm_executor_type(ex):
        ex = _slurm_executor_for_map(ex, process_index, indices)
        assert isinstance(executor, dict)
        executor[func.output_name] = ex  # type: ignore[assignment]
    return ex


def maybe_finalize_slurm_executors(
    generation: list[PipeFunc],
    executor: dict[OUTPUT_TYPE, Executor],
    multi_run_manager: MultiRunManager | None,
) -> None:
    executors = _executors_for_generation(generation, executor)
    for ex in executors:
        if _adaptive_scheduler_imported() and _is_slurm_executor(ex):
            assert multi_run_manager is not None
            run_manager = ex.finalize()
            if run_manager is not None:  # is None if nothing was submitted
                multi_run_manager.add_run_manager(run_manager)


def _is_slurm_executor(executor: Executor | None) -> TypeGuard[SlurmExecutor]:
    if executor is None or not _adaptive_scheduler_imported():  # pragma: no cover
        return False
    from adaptive_scheduler import SlurmExecutor

    return isinstance(executor, SlurmExecutor)


def _is_slurm_executor_type(executor: Executor | None) -> TypeGuard[type[SlurmExecutor]]:
    if executor is None or not _adaptive_scheduler_imported():  # pragma: no cover
        return False
    from adaptive_scheduler import SlurmExecutor

    return isinstance(executor, type) and issubclass(executor, SlurmExecutor)


def _slurm_executor_for_map(
    executor: SlurmExecutor | type[SlurmExecutor],
    process_index: functools.partial[tuple[Any, ...]],
    indices: list[int],
) -> Executor:  # Actually SlurmExecutor, but mypy doesn't like it
    func = process_index.keywords["func"]
    executor_kwargs = _map_slurm_executor_kwargs(func, process_index, indices)
    executor_kwargs["name"] = _slurm_name(func.output_name, executor)  # type: ignore[assignment]
    return _new_slurm_executor(executor, **executor_kwargs)


def _slurm_executor_for_single(
    executor: SlurmExecutor | type[SlurmExecutor],
    func: PipeFunc,
    kwargs: dict[str, Any],
) -> Executor:
    resources: Resources | None = (
        func.resources(kwargs) if callable(func.resources) else func.resources  # type: ignore[has-type]
    )
    executor_kwargs = _adaptive_scheduler_resource_dict(resources)
    executor_kwargs["name"] = _slurm_name(func.output_name, executor)
    return _new_slurm_executor(executor, **executor_kwargs)


def _adaptive_scheduler_imported() -> bool:
    """Check if the adaptive_scheduler package is imported and at the correct version."""
    if not is_imported("adaptive_scheduler"):  # pragma: no cover
        return False
    # The SlurmExecutor was introduced in version 2.13.3
    min_version = "2.14.0"
    if not is_min_version("adaptive_scheduler", min_version):  # pragma: no cover
        msg = f"The 'adaptive_scheduler' package must be at least version {min_version}."
        raise ImportError(msg)
    return True


def _executors_for_generation(
    generation: list[PipeFunc],
    executor: dict[OUTPUT_TYPE, Executor],
) -> list[Executor]:
    from ._run import _executor_for_func
    # Import here to avoid circular imports

    executors = []
    for func in generation:
        ex = _executor_for_func(func, executor)
        if ex is not None and ex not in executors:
            executors.append(ex)
    return executors


def _slurm_name(output_name: OUTPUT_TYPE, executor: SlurmExecutor | type[SlurmExecutor]) -> str:
    from adaptive_scheduler import SlurmExecutor

    name = "-".join(at_least_tuple(output_name))
    if isinstance(executor, SlurmExecutor):
        return f"{executor.name}-{name}"
    return name


def _map_slurm_executor_kwargs(
    func: PipeFunc,
    process_index: functools.partial[tuple[Any, ...]],
    seq: list[int],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    size_per_learner = 1 if func.resources_scope == "element" else None
    kwargs["size_per_learner"] = size_per_learner
    resources = func.resources  # type: ignore[has-type]
    if resources is None:
        return kwargs  # type: ignore[return-value]

    # If resources is not callable, treat as static.
    if not callable(resources):
        if func.resources_scope == "element":
            # Replicate the static resource dict for each element.
            resources_dict = _adaptive_scheduler_resource_dict(resources)
            resources_list = [resources_dict] * len(seq)
            dict_of_tuples = _list_of_dicts_to_dict_of_tuples(resources_list)
            kwargs.update(dict_of_tuples)
            return kwargs
        assert func.resources_scope == "map"
        # Use the single static resource dict.
        kwargs.update(_adaptive_scheduler_resource_dict(resources))
        return kwargs

    # Now resources is callable.
    if func.resources_scope == "map":
        # Call the callable only once.
        evaluated_resources = _resources_from_process_index(process_index, seq[0])
        scheduler_resources = _adaptive_scheduler_resource_dict(evaluated_resources)
        kwargs.update(scheduler_resources)
        return kwargs
    assert func.resources_scope == "element"
    resources_list: list[dict[str, Any]] = []  # type: ignore[no-redef]
    for i in seq:
        evaluated_resources = _resources_from_process_index(process_index, i)
        scheduler_resources = _adaptive_scheduler_resource_dict(evaluated_resources)
        resources_list.append(scheduler_resources)
    dict_of_tuples = _list_of_dicts_to_dict_of_tuples(resources_list)
    kwargs.update(dict_of_tuples)
    return kwargs


def _new_slurm_executor(
    executor: SlurmExecutor | type[SlurmExecutor],
    **kwargs: Any,
) -> SlurmExecutor:
    from adaptive_scheduler import SlurmExecutor

    if _is_slurm_executor(executor):  # type: ignore[arg-type]
        return executor.new(update=kwargs)
    assert isinstance(executor, type)
    assert issubclass(executor, SlurmExecutor)
    return executor(**kwargs)


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
    tuples = {k: tuple(d[k] for d in list_of_dicts) for k in list_of_dicts[0]}
    # Remove keys with all None or [] values
    return {k: v for k, v in tuples.items() if any(v)}
