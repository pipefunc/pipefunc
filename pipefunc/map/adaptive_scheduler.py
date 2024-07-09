"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

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


class _UnknownBecauseDelayed:
    """A singleton to represent unknown values because of delayed evaluation."""


class AdaptiveSchedulerDetails(NamedTuple):
    """Details for the adaptive scheduler."""

    learners: list[SequenceLearner]
    fnames: list[Path]
    dependencies: dict[int, list[int]]
    nodes: tuple[int, ...] | None
    cores_per_node: tuple[int, ...] | None
    extra_scheduler: tuple[list[str], ...] | None
    partition: tuple[str, ...] | None
    exclusive: bool = False

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
        return {k: v for k, v in dct.items() if v is not None}

    def run_manager(self) -> adaptive_scheduler.RunManager:  # pragma: no cover
        """Get a `RunManager` for the adaptive scheduler."""
        import adaptive_scheduler

        return adaptive_scheduler.RunManager(**self.kwargs())


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
    tracker = _Tracker(default_resources)
    assert learners_dict.run_folder is not None
    run_folder = Path(learners_dict.run_folder)
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
                if not ignore_resources:
                    assert (
                        isinstance(learner.pipefunc.resources, Resources)
                        or learner.pipefunc.resources is None
                    )
                    tracker.update_resources(
                        learner.pipefunc.resources,
                        learner.pipefunc,
                        learners_dict.run_info,
                    )
                elif ignore_resources and default_resources is not None:
                    tracker.update_resources(
                        default_resources,
                        learner.pipefunc,
                        learners_dict.run_info,
                    )
            prev_indices = indices

    if not any(tracker.resources_dict["extra_scheduler"]):  # all are empty
        del tracker.resources_dict["extra_scheduler"]

    return AdaptiveSchedulerDetails(
        learners=learners,
        fnames=fnames,
        dependencies=dependencies,
        nodes=tracker.maybe_get("nodes"),
        cores_per_node=tracker.maybe_get("cores_per_node"),
        extra_scheduler=tracker.maybe_get("extra_scheduler"),
        partition=tracker.maybe_get("partition"),
    )


@dataclass(frozen=True, slots=True)
class _Tracker:
    """Ensures that iff a resource is defined for one `PipeFunc`, it is defined for all of them."""

    default_resources: Resources | None
    defined: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    resources_dict: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))

    def is_defined(self, key: str) -> None:
        self.defined.add(key)
        if key in self.missing:
            self.do_raise(key)

    def is_missing(self, key: str) -> None:
        if (
            self.default_resources is not None and getattr(self.default_resources, key) is not None
        ):  # pragma: no cover
            # Never happens because we strictly call `get` with combined defaults+resources
            return
        self.missing.add(key)
        if key in self.defined:
            self.do_raise(key)

    def get(
        self,
        resources: Resources | Callable[[dict[str, Any]], Resources],
        key: str,
        func: PipeFunc,
        run_info: RunInfo,
    ) -> Any | Callable[[], Any] | None:
        if callable(resources):
            if key == "num_cpus":
                return _num_cpus(resources, func, run_info)
            if key == "num_cpus_per_node":
                return _num_cpus_per_node(resources, func, run_info)
            if key == "num_nodes":
                return _num_nodes(resources, func, run_info)
            if key == "partition":
                return _partition(resources, func, run_info)
            msg = f"Unknown key: {key}"
            raise ValueError(msg)

        value = getattr(resources, key)
        if value is not None:
            self.is_defined(key)
            return value
        else:  # noqa: RET505
            self.is_missing(key)
            return None

    def maybe_get(self, key: str) -> tuple | None:
        return tuple(self.resources_dict[key]) if key in self.resources_dict else None

    def do_raise(self, key: str) -> None:
        msg = (
            f"At least one `PipeFunc` provides `{key}`."
            " It must either be defined for all `PipeFunc`s or in `default_resources`."
        )
        raise ValueError(msg)

    def update_resources(
        self,
        resources: Resources | Callable[[dict[str, Any]], Resources] | None,
        func: PipeFunc,
        run_info: RunInfo,
    ) -> None:
        if resources is None and self.default_resources is None:
            msg = "Either all `PipeFunc`s must have resources or `default_resources` must be provided."
            raise ValueError(msg)
        r: Resources | Callable[[dict[str, Any]], Resources] | None
        if resources is None:
            r = self.default_resources
        elif self.default_resources is None:
            r = resources
        elif callable(resources):
            r = Resources.maybe_with_defaults(resources, self.default_resources)
            # TODO: Create functions for cores_per_node, nodes, etc.
        else:
            r = resources.with_defaults(self.default_resources)
        assert r is not None
        if (v := self.get(r, "num_cpus", func, run_info)) is not None:
            self.resources_dict["cores_per_node"].append(v)
        if (v := self.get(r, "num_cpus_per_node", func, run_info)) is not None:
            self.resources_dict["cores_per_node"].append(v)
        if (v := self.get(r, "num_nodes", func, run_info)) is not None:
            self.resources_dict["nodes"].append(v)
        if (v := self.get(r, "partition", func, run_info)) is not None:
            self.resources_dict["partition"].append(v)

        # There is no requirement for these to be defined for all `PipeFunc`s.
        self.resources_dict["extra_scheduler"].append(_extra_scheduler(r, func, run_info))


def _num_cpus(
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Callable[[], int | None]:
    def _fn() -> int | None:
        kwargs = _func_kwargs(func, run_info, run_info.init_store())
        return resources(kwargs).num_cpus

    return _fn


def _num_cpus_per_node(
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Callable[[], int | None]:
    def _fn() -> int | None:
        kwargs = _func_kwargs(func, run_info, run_info.init_store())
        return resources(kwargs).num_cpus_per_node

    return _fn


def _num_nodes(
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Callable[[], int | None]:
    def _fn() -> int | None:
        kwargs = _func_kwargs(func, run_info, run_info.init_store())
        return resources(kwargs).num_nodes

    return _fn


def _partition(
    resources: Callable[[dict[str, Any]], Resources],
    func: PipeFunc,
    run_info: RunInfo,
) -> Callable[[], str | None]:
    def _fn() -> str | None:
        kwargs = _func_kwargs(func, run_info, run_info.init_store())
        return resources(kwargs).partition

    return _fn


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
    if resources.num_gpus:
        extra_scheduler.append(f"--gres=gpu:{resources.num_gpus}")
    if resources.wall_time:
        extra_scheduler.append(f"--time={resources.wall_time}")
    if resources.extra_args:
        for key, value in resources.extra_args.items():
            extra_scheduler.append(f"--{key}={value}")
    return extra_scheduler
