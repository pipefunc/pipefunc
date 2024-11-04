from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from pipefunc._utils import min_version_check

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from adaptive_scheduler import MultiRunManager

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE


def _adaptive_scheduler_imported() -> bool:
    """Check if the adaptive_scheduler package is imported and at the correct version."""
    if "adaptive_scheduler" not in sys.modules:
        return False
    # The SlurmExecutor was introduced in version 2.13.0
    min_version = "2.13.0"
    if not min_version_check("adaptive_scheduler", min_version):
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

        if isinstance(executor, SlurmExecutor):
            if not in_async:
                msg = "Cannot use an `adaptive_scheduler.SlurmExecutor` in non-async mode, use `pipeline.run_async` instead."
                raise ValueError(msg)
            # If a single SlurmExecutor is provided, we need to create a new one for each output
            return {
                func.output_name: executor.new(update={"name": f"{executor.name}-{i}"})
                for i, func in enumerate(pipeline.sorted_functions)
            }
    return executor


def _executors_for_generation(
    generation: list[PipeFunc],
    executor: Executor | dict[OUTPUT_TYPE, Executor],
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
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None,
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
    executor: Executor | dict[OUTPUT_TYPE, Executor] | None,
) -> MultiRunManager | None:
    if isinstance(executor, dict) and _adaptive_scheduler_imported():
        from adaptive_scheduler import MultiRunManager, SlurmExecutor

        for ex in executor.values():
            if isinstance(ex, SlurmExecutor):
                return MultiRunManager()
    return None
