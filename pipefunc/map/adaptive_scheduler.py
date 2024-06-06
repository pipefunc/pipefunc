"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pipefunc._pipefunc import PipeFunc, _maybe_resources
from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from adaptive import SequenceLearner

    from pipefunc.map.adaptive import LearnersDict
    from pipefunc.resources import Resources


class AdaptiveSchedulerDetails(NamedTuple):
    """Details for the adaptive scheduler."""

    learners: list[SequenceLearner]
    fnames: list[Path]
    dependencies: dict[int, list[int]]
    nodes: tuple[int, ...] | None
    cores_per_node: tuple[int, ...] | None
    extra_scheduler: tuple[list[str], ...] | None
    partition: tuple[str, ...] | None


def _fname(run_folder: Path, func: PipeFunc, index: int) -> Path:
    output_name = "-".join(at_least_tuple(func.output_name))
    return run_folder / "adaptive_scheduler" / output_name / f"{index}.pickle"


def slurm_run_setup(
    learners_dict: LearnersDict,
    run_folder: str | Path,
    default_resources: dict | Resources | None,
    *,
    ignore_resources: bool = False,
) -> AdaptiveSchedulerDetails:
    """Set up the arguments for `adaptive_scheduler.slurm_run`."""
    default_resources = _maybe_resources(default_resources)
    tracker = _Tracker(default_resources)
    run_folder = Path(run_folder)
    learners: list[SequenceLearner] = []
    fnames: list[Path] = []
    resources_dict: dict[str, list[Any]] = defaultdict(list)
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
                    _update_resources(learner.pipefunc.resources, tracker, resources_dict)
            prev_indices = indices

    return AdaptiveSchedulerDetails(
        learners=learners,
        fnames=fnames,
        dependencies=dependencies,
        nodes=_maybe_get(resources_dict, "nodes"),
        cores_per_node=_maybe_get(resources_dict, "cores_per_node"),
        extra_scheduler=_maybe_get(resources_dict, "extra_scheduler"),
        partition=_maybe_get(resources_dict, "partition"),
    )


def _maybe_get(resources: dict[str, list[Any]], key: str) -> tuple | None:
    return tuple(resources[key]) if key in resources else None


def _update_resources(
    resources: Resources | None,
    tracker: _Tracker,
    resources_dict: dict[str, list[Any]],
) -> None:
    r = resources or tracker.default_resources
    if r is None:
        msg = "Either all `PipeFunc`s must have resources or `default_resources` must be provided."
        raise ValueError(msg)

    if (v := tracker.get(r, "num_cpus")) is not None:
        resources_dict["cores_per_node"].append(v)
    if (v := tracker.get(r, "num_cpus_per_node")) is not None:
        resources_dict["cores_per_node"].append(v)
    if (v := tracker.get(r, "num_nodes")) is not None:
        resources_dict["nodes"].append(v)
    if (v := tracker.get(r, "partition")) is not None:
        resources_dict["partition"].append(v)

    _extra_scheduler = []
    if (v := tracker.get(r, "memory")) is not None:
        _extra_scheduler.append(f"--mem={v}")
    if (v := tracker.get(r, "num_gpus")) is not None:
        _extra_scheduler.append(f"--gres=gpu:{v}")
    if (v := tracker.get(r, "wall_time")) is not None:
        _extra_scheduler.append(f"--time={v}")
    if (v := tracker.get(r, "extra_args")) is not None:
        for key, value in v.items():
            _extra_scheduler.append(f"--{key}={value}")
    resources_dict["extra_scheduler"].append(_extra_scheduler)


@dataclass
class _Tracker:
    """Ensures that iff a resource is defined for one `PipeFunc`, it is defined for all of them."""

    default_resources: Resources | None
    defined: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)

    def is_defined(self, key: str) -> None:
        self.defined.add(key)
        if key in self.missing:
            self.do_raise(key)

    def is_missing(self, key: str) -> None:
        if self.default_resources is not None and getattr(self.default_resources, key) is not None:
            return
        self.missing.add(key)
        if key in self.defined:
            self.do_raise(key)

    def get(self, resources: Resources, key: str) -> Any | None:
        value = getattr(resources, key)
        if value is not None:
            self.is_defined(key)
            return value
        else:  # noqa: RET505
            self.is_missing(key)
            return None

    def do_raise(self, key: str) -> None:
        msg = (
            f"At least one `PipeFunc` provides `{key}`."
            " It must either be defined for all `PipeFunc`s or in `default_resources`."
        )
        raise ValueError(msg)
