"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pipefunc._utils import at_least_tuple
from pipefunc.map._run import _func_kwargs, _load_file_array, _select_kwargs
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from collections.abc import Callable

    import adaptive_scheduler
    from adaptive import SequenceLearner
    from adaptive_scheduler.utils import EXECUTOR_TYPES

    from pipefunc._pipefunc import PipeFunc
    from pipefunc.map._run_info import RunInfo
    from pipefunc.map.adaptive import LearnersDict


class AdaptiveSchedulerDetails(NamedTuple):
    """Details for the adaptive scheduler."""

    learners: list[SequenceLearner]
    fnames: list[Path]
    dependencies: dict[int, list[int]]
    nodes: tuple[int | None | Callable[[], int | None], ...] | None
    cores_per_node: tuple[int | None | Callable[[], int | None], ...] | None
    extra_scheduler: tuple[list[str] | Callable[[], list[str]], ...] | None
    partition: tuple[str | None | Callable[[], str | None], ...] | None
    executor_type: tuple[EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES], ...] | None = None

    def kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for `adaptive_scheduler.slurm_run`.

        Examples
        --------
        >>> learners = pipefunc.map.adaptive.create_learners(pipeline, ...)
        >>> info = learners.to_slurm_run(...)
        >>> kwargs = info.kwargs()
        >>> adaptive_scheduler.slurm_run(**kwargs)

        """
        dct = self._asdict()
        return {k: v for k, v in dct.items() if not _is_none(v)}

    def run_manager(self, kwargs: Any | None) -> adaptive_scheduler.RunManager:  # pragma: no cover
        """Get a `RunManager` for the adaptive scheduler."""
        import adaptive_scheduler

        return adaptive_scheduler.slurm_run(**(kwargs or self.kwargs()))


def _is_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, tuple):
        return all(_is_none(x) for x in value)
    return False


def _fname(run_folder: Path, func: PipeFunc, index: int) -> Path:
    output_name = "-".join(at_least_tuple(func.output_name))
    return run_folder / "adaptive_scheduler" / output_name / f"{index}.pickle"


def slurm_run_setup(
    learners_dict: LearnersDict,
    default_resources: dict | Resources | None = None,
    *,
    ignore_resources: bool = False,
) -> AdaptiveSchedulerDetails:
    """Set up the arguments for `adaptive_scheduler.slurm_run`."""
    assert learners_dict.run_info is not None
    default_resources = Resources.maybe_from_dict(default_resources)  # type: ignore[assignment]
    assert isinstance(default_resources, Resources) or default_resources is None
    tracker = _ResourcesContainer(default_resources)
    assert learners_dict.run_info is not None
    run_folder = Path(learners_dict.run_info.run_folder)
    learners: list[SequenceLearner] = []
    fnames: list[Path] = []
    dependencies: dict[int, list[int]] = {}
    for learners_lists in learners_dict.data.values():
        prev_indices: list[int] = []
        for learner_list in learners_lists:
            indices: list[int] = []
            for learner in learner_list:
                i = len(learners)
                indices.append(i)
                dependencies[i] = prev_indices
                learners.append(learner.learner)
                fnames.append(_fname(run_folder, learner.pipefunc, i))
                tracker.update(
                    resources=learner.pipefunc.resources if not ignore_resources else None,
                    func=learner.pipefunc,
                    learner=learner.learner,
                    run_info=learners_dict.run_info,
                )
            prev_indices = indices

    if not any(tracker.data["extra_scheduler"]):  # all are empty
        del tracker.data["extra_scheduler"]

    # Combine cores_per_node and cpus
    cores_per_node = tracker.get("cpus_per_node")
    cpus = tracker.get("cpus")
    if cores_per_node is not None and cpus is not None:
        assert len(cores_per_node) == len(cpus)
        cores_per_node = tuple(_or(cpn, _cpus) for cpn, _cpus in zip(cores_per_node, cpus))
    elif cpus is not None:
        cores_per_node = cpus

    return AdaptiveSchedulerDetails(
        learners=learners,
        fnames=fnames,
        dependencies=dependencies,
        nodes=tracker.get("nodes"),
        cores_per_node=cores_per_node,
        extra_scheduler=tracker.get("extra_scheduler"),
        partition=tracker.get("partition"),
        executor_type=tracker.get("executor_type"),
    )


@dataclass(frozen=True, slots=True)
class _ResourcesContainer:
    default_resources: Resources | None
    data: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))

    def get(self, key: str) -> tuple | None:
        if key not in self.data:
            return None
        value = tuple(self.data[key])
        if all(x is None for x in value):
            return None
        return value

    def update(
        self,
        resources: Resources | Callable[[dict[str, Any]], Resources] | None,
        func: PipeFunc,
        learner: SequenceLearner,
        run_info: RunInfo,
    ) -> None:
        if resources is None and self.default_resources is None:
            msg = "Either all `PipeFunc`s must have resources or `default_resources` must be provided."
            raise ValueError(msg)
        r: Resources | Callable[[dict[str, Any]], Resources]
        if resources is None:
            assert self.default_resources is not None
            r = self.default_resources
        elif self.default_resources is None:
            r = resources
        elif callable(resources):
            r = Resources.maybe_with_defaults(resources, self.default_resources)  # type: ignore[assignment]
            assert r is not None
        else:
            r = resources.with_defaults(self.default_resources)

        index = _get_index(learner, func)
        for name in ["cpus_per_node", "cpus", "nodes", "partition"]:
            if callable(r):
                # Note: we don't know if `resources.{name}` returns None or not
                value = functools.partial(
                    _getattr_from_resources,
                    name=name,
                    index=index,
                    resources=r,
                    func=func,
                    run_info=run_info,
                )
            else:
                value = getattr(r, name)
            self.data[name].append(value)
        # TODO: Allow setting any of EXECUTOR_TYPES
        self.data["executor_type"].append(_executor_type(index, r, func, run_info))
        self.data["extra_scheduler"].append(_extra_scheduler(index, r, func, run_info))


def _get_index(learner: SequenceLearner, func: PipeFunc) -> int | None:
    if func.resources_scope == "element" and func.mapspec is not None:
        # Assumes that the learner is already split up
        assert len(learner.sequence) == 1
        return learner.sequence[0]
    return None


def _eval_resources(
    index: int | None,
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Resources:
    kwargs = _func_kwargs(func, run_info, run_info.init_store())
    _load_file_array(kwargs)
    if index is not None:
        shape = run_info.shapes[func.output_name]
        shape_mask = run_info.shape_masks[func.output_name]
        kwargs = _select_kwargs(func, kwargs, shape, shape_mask, index)
    return resources(kwargs)


def _getattr_from_resources(
    *,
    name: str,
    index: int | None,
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Any | None:
    resources_instance = _eval_resources(index, resources, func, run_info)
    return getattr(resources_instance, name)


def _extra_scheduler(
    index: int | None,
    resources: Resources | Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> list[str] | Callable[[], list[str]]:
    if callable(resources):

        def _fn() -> list[str]:
            resources_instance = _eval_resources(index, resources, func, run_info)
            return _extra_scheduler(index, resources_instance, func, run_info)  # type: ignore[return-value]

        return _fn
    extra_scheduler = []
    if resources.memory:
        extra_scheduler.append(f"--mem={resources.memory}")
    if resources.gpus:
        extra_scheduler.append(f"--gres=gpu:{resources.gpus}")
    if resources.time:
        extra_scheduler.append(f"--time={resources.time}")
    if resources.extra_args:
        for key, value in resources.extra_args.items():
            extra_scheduler.append(f"--{key}={value}")
    return extra_scheduler


def _executor_type(
    index: int | None,
    resources: Resources | Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> EXECUTOR_TYPES | Callable[[], EXECUTOR_TYPES]:
    if callable(resources):

        def _fn() -> EXECUTOR_TYPES:
            resources_instance = _eval_resources(index, resources, func, run_info)
            return _executor_type(index, resources_instance, func, run_info)

        return _fn
    return "sequential" if resources.parallelization_mode == "internal" else "process-pool"


def _or(
    value_1: int | Callable[[], int | None] | None,
    value_2: int | Callable[[], int | None] | None,
) -> int | Callable[[], int | None] | None:
    if value_1 is None:
        return value_2
    if value_2 is None:
        return value_1
    if callable(value_1) and callable(value_2):
        return lambda: value_1() or value_2()
    if callable(value_1) and not callable(value_2):
        return lambda: value_1() or value_2
    if not callable(value_1) and callable(value_2):
        return lambda: value_1 or value_2()
    return value_1 or value_2
