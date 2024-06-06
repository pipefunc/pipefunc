"""Provides `adaptive_scheduler` integration for `pipefunc`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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

                if r.num_cpus:
                    tracker.is_defined("num_cpus")
                    cores_per_node.append(r.num_cpus)
                else:
                    tracker.is_missing("num_cpus")
                if r.num_cpus_per_node:
                    tracker.is_defined("num_cpus_per_node")
                    cores_per_node.append(r.num_cpus_per_node)
                else:
                    tracker.is_missing("num_cpus_per_node")

                if r.num_nodes:
                    tracker.is_defined("num_nodes")
                    num_nodes.append(r.num_nodes)
                else:
                    tracker.is_missing("num_nodes")

                if r.partition:
                    tracker.is_defined("partition")
                    partition.append(r.partition)
                else:
                    tracker.is_missing("partition")

                _extra_scheduler = []
                if r.memory:
                    tracker.is_defined("memory")
                    _extra_scheduler.append(f"--mem={r.memory}")
                else:
                    tracker.is_missing("memory")
                if r.num_gpus:
                    tracker.is_defined("num_gpus")
                    _extra_scheduler.append(f"--gres=gpu:{r.num_gpus}")
                else:
                    tracker.is_missing("num_gpus")
                if r.wall_time:
                    tracker.is_defined("wall_time")
                    _extra_scheduler.append(f"--time={r.wall_time}")
                else:
                    tracker.is_missing("wall_time")
                if r.queue:
                    tracker.is_defined("queue")
                    _extra_scheduler.append(f"--partition={r.queue}")
                else:
                    tracker.is_missing("queue")
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

    def do_raise(self, key: str) -> None:
        msg = (
            f"At least one `PipeFunc` provides `{key}`."
            " It must either be defined for all `PipeFunc`s or in `default_resources`."
        )
        raise ValueError(msg)
