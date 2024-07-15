"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from pipefunc._utils import at_least_tuple
from pipefunc.map._run import _func_kwargs
from pipefunc.resources import Resources

if TYPE_CHECKING:
    import adaptive_scheduler
    from adaptive import SequenceLearner

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

    def run_manager(self) -> adaptive_scheduler.RunManager:  # pragma: no cover
        """Get a `RunManager` for the adaptive scheduler."""
        import adaptive_scheduler

        return adaptive_scheduler.RunManager(**self.kwargs())


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
                    learner.pipefunc.resources if not ignore_resources else None,
                    learner.pipefunc,
                    learners_dict.run_info,
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
    )


@dataclass(frozen=True, slots=True)
class _ResourcesContainer:
    default_resources: Resources | None
    data: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))

    def _getattr_from_resources(
        self,
        resources: Resources | Callable[[dict[str, Any]], Resources],
        name: str,
        func: PipeFunc,
        run_info: RunInfo,
    ) -> Any | Callable[[], Any] | None:
        if callable(resources):
            assert name in ("cpus", "cpus_per_node", "nodes", "partition")
            # Note: we don't know if `resources.name` returns None or not
            return functools.partial(
                _getattr_from_resources,
                name=name,
                resources=resources,
                func=func,
                run_info=run_info,
            )
        return getattr(resources, name)

    def get(self, key: str) -> tuple | None:
        return tuple(self.data[key]) if key in self.data else None

    def update(
        self,
        resources: Resources | Callable[[dict[str, Any]], Resources] | None,
        func: PipeFunc,
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

        for name in ["cpus_per_node", "cpus", "nodes", "partition"]:
            value = self._getattr_from_resources(r, name, func, run_info)
            self.data[name].append(value)

        # There is no requirement for these to be defined for all `PipeFunc`s.
        self.data["extra_scheduler"].append(_extra_scheduler(r, func, run_info))


def _getattr_from_resources(
    *,
    name: str,
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Any | None:
    kwargs = _func_kwargs(func, run_info, run_info.init_store())
    return getattr(resources(kwargs), name)


def _extra_scheduler(
    resources: Resources | Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> list[str] | Callable[[], list[str]]:
    if callable(resources):

        def _fn() -> list[str]:
            kwargs = _func_kwargs(func, run_info, run_info.init_store())
            return _extra_scheduler(resources(kwargs), func, run_info)  # type: ignore[return-value]

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
