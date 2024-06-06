"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

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
    nodes: tuple[int, ...]
    cores_per_node: tuple[int, ...]
    extra_scheduler: tuple[list[str], ...]
    partition: tuple[str, ...]


def _fname(run_folder: Path, func: PipeFunc, index: int) -> Path:
    output_name = "-".join(at_least_tuple(func.output_name))
    return run_folder / "adaptive_scheduler" / output_name / f"{index}.pickle"


def slurm_run_setup(
    learners_dict: LearnersDict,
    run_folder: str | Path,
    default_resources: dict | Resources | None,
) -> AdaptiveSchedulerDetails:
    """Set up the arguments for `adaptive_scheduler.slurm_run`."""
    default_resources = _maybe_resources(default_resources)
    tracker = _Tracker(default_resources)
    run_folder = Path(run_folder)
    learners: list[SequenceLearner] = []
    fnames: list[Path] = []
    cores_per_node: list[int] = []
    num_nodes: list[int] = []
    extra_scheduler: list[list[str]] = []
    partition: list[str] = []
    dependencies: dict[int, list[int]] = {}
    for learners_lists in learners_dict.data.values():
        for learner_list in learners_lists:
            prev_indices: list[int] = []
            indices: list[int] = []
            for learner in learner_list:
                i = len(learners)
                indices.append(i)
                dependencies[i] = prev_indices
                learners.append(learner.learner)
                fnames.append(_fname(run_folder, learner.pipefunc, i))
                r = learner.pipefunc.resources or default_resources
                if r is None:
                    msg = "Either all `PipeFunc`s must have resources or `default_resources` must be provided."
                    raise ValueError(msg)

                if (v := tracker.get(r, "num_cpus")) is not None:
                    cores_per_node.append(v)
                if (v := tracker.get(r, "num_cpus_per_node")) is not None:
                    cores_per_node.append(v)
                if (v := tracker.get(r, "num_nodes")) is not None:
                    num_nodes.append(v)
                if (v := tracker.get(r, "partition")) is not None:
                    partition.append(v)

                _extra_scheduler = []
                if (v := tracker.get(r, "memory")) is not None:
                    _extra_scheduler.append(f"--mem={v}")
                if (v := tracker.get(r, "num_gpus")) is not None:
                    _extra_scheduler.append(f"--gres=gpu:{v}")
                if (v := tracker.get(r, "wall_time")) is not None:
                    _extra_scheduler.append(f"--time={v}")
                extra_scheduler.append(_extra_scheduler)

            prev_indices = indices
    return AdaptiveSchedulerDetails(
        learners=learners,
        fnames=fnames,
        dependencies=dependencies,
        nodes=tuple(num_nodes),
        cores_per_node=tuple(cores_per_node),
        extra_scheduler=tuple(extra_scheduler),
        partition=tuple(partition),
    )


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
