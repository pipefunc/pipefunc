"""Provides the `pipefunc.resources` module, containing the `Resources` class."""

from __future__ import annotations

import functools
import inspect
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True, eq=True)
class Resources:
    """A dataclass representing computational resources for a job.

    Parameters
    ----------
    cpus
        The number of CPUs required for the job. Must be a positive integer.
    cpus_per_node
        The number of CPUs per node required for the job. Must be a positive integer.
    nodes
        The number of nodes required for the job. Must be a positive integer.
    memory
        The memory required for the job. Must be a valid string (e.g., ``'2GB'``, ``'500MB'``).
    gpus
        The number of GPUs required for the job. Must be a non-negative integer.
    time
        The time required for the job. Must be a valid string (e.g., ``'2:00:00'``, ``'48:00:00'``).
    partition
        The partition to submit the job to.
    extra_args
        Extra arguments for the job. Default is an empty dictionary.
    parallelization_mode
        Specifies how parallelization should be handled.
        "internal": The function should use the resources (e.g., cpus) to handle its own parallelization.
        "external": The function should operate on a single core, with parallelization managed externally.
        Default is "external".

    Raises
    ------
    ValueError
        If any of the input parameters do not meet the specified constraints.

    Notes
    -----
    - `cpus` and `nodes` cannot be specified together.
    - `cpus_per_node` must be specified with `nodes`.

    Examples
    --------
    >>> resources = Resources(cpus=4, memory='16GB', time='2:00:00')
    >>> resources.cpus
    4
    >>> resources.memory
    '16GB'
    >>> resources.time
    '2:00:00'

    """

    cpus: int | None = None
    cpus_per_node: int | None = None
    nodes: int | None = None
    memory: str | None = None
    gpus: int | None = None
    time: str | None = None
    partition: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)
    parallelization_mode: Literal["internal", "external"] = "external"

    def __post_init__(self) -> None:
        """Validate input parameters after initialization.

        Raises
        ------
        ValueError
            If any of the input parameters do not meet the specified constraints.

        """
        if self.cpus is not None and self.cpus <= 0:
            msg = "`cpus` must be a positive integer."
            raise ValueError(msg)
        if self.gpus is not None and self.gpus < 0:
            msg = "`gpus` must be a non-negative integer."
            raise ValueError(msg)
        if self.nodes is not None and self.nodes <= 0:
            msg = "`nodes` must be a positive integer."
            raise ValueError(msg)
        if self.cpus_per_node is not None and self.cpus_per_node <= 0:
            msg = "`cpus_per_node` must be a positive integer."
            raise ValueError(msg)
        if self.memory is not None and not self._is_valid_memory(self.memory):
            msg = f"`memory` must be a valid string (e.g., '2GB', '500MB'), not '{self.memory}'."
            raise ValueError(msg)
        if self.time is not None and not self._is_valid_wall_time(self.time):
            msg = "`time` must be a valid string (e.g., '2:00:00', '48:00:00')."
            raise ValueError(msg)
        if self.nodes and self.cpus:
            msg = (
                "`nodes` and `cpus` cannot be specified together."
                " Either use nodes and `cpus_per_node` or use `cpus` alone."
            )
            raise ValueError(msg)
        if self.cpus_per_node and not self.nodes:
            msg = "`cpus_per_node` must be specified with `nodes`."
            raise ValueError(msg)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Resources:
        """Create a Resources instance from a dictionary.

        Parameters
        ----------
        data
            A dictionary containing the input parameters for the Resources instance.

        Returns
        -------
            A Resources instance created from the input dictionary.

        """
        assert isinstance(data, dict), "Input data must be a dictionary."
        try:
            return Resources(**data)
        except TypeError as e:
            parameters = list(inspect.signature(Resources.__init__).parameters)
            allowed_args = ", ".join(parameters[1:])
            msg = f"Error creating Resources instance: {e}.\n The following arguments are allowed: `{allowed_args}`"
            raise TypeError(msg) from e

    @staticmethod
    def maybe_from_dict(
        resources: dict[str, Any]
        | Resources
        | Callable[[dict[str, Any]], Resources | dict[str, Any]]
        | None,
    ) -> Resources | Callable[[dict[str, Any]], Resources] | None:
        """Create a Resources instance from a dictionary, if not already an instance and not None."""
        if resources is None:
            return None
        if isinstance(resources, Resources):
            return resources
        if callable(resources):
            return functools.partial(_ensure_resources, resources_callable=resources)
        return Resources.from_dict(resources)

    @staticmethod
    def _is_valid_memory(memory: str) -> bool:
        if not isinstance(memory, str):
            return False
        try:
            Resources._convert_to_gb(memory)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def _convert_to_gb(memory: str) -> float:
        units = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3, "PB": 1e6}
        match = re.match(r"^(\d+(?:\.\d+)?)([KMGTP]?B)$", memory.upper())
        if match:
            value, unit = match.groups()
            return float(value) * units[unit]
        msg = f"Invalid memory string '{memory}'. Expected format: <value> <unit>, e.g., '2GB', '500MB'."
        raise ValueError(msg)

    @staticmethod
    def _is_valid_wall_time(time: str) -> bool:
        pattern = re.compile(r"^(\d+:)?(\d{2}:)?\d{2}:\d{2}$")
        return bool(pattern.match(time))

    def to_slurm_options(self) -> str:
        """Convert the Resources instance to SLURM options.

        Returns
        -------
        str
            A string containing the SLURM options.

        """
        options = []
        if self.cpus:
            options.append(f"--cpus-per-task={self.cpus}")
        if self.gpus:
            options.append(f"--gres=gpu:{self.gpus}")
        if self.nodes:
            options.append(f"--nodes={self.nodes}")
        if self.cpus_per_node:
            options.append(f"--cpus-per-node={self.cpus_per_node}")
        if self.memory:
            options.append(f"--mem={self.memory}")
        if self.time:
            options.append(f"--time={self.time}")
        if self.partition:
            options.append(f"--partition={self.partition}")
        for key, value in self.extra_args.items():
            options.append(f"--{key}={value}")
        return " ".join(options)

    def update(self, **kwargs: Any) -> Resources:
        """Update the Resources instance with new values.

        Parameters
        ----------
        **kwargs
            Keyword arguments specifying the attributes to update and their new values.

        Returns
        -------
            A new Resources instance with the updated values.

        """
        data = self.__dict__.copy()
        for key, value in kwargs.items():
            if key == "extra_args":
                data["extra_args"] = {**data["extra_args"], **value}
            elif key in data:
                data[key] = value
            else:
                data["extra_args"][key] = value
        return Resources.from_dict(data)

    @staticmethod
    def combine_max(resources_list: list[Resources]) -> Resources:
        """Combine multiple Resources instances by taking the maximum value for each attribute.

        Parameters
        ----------
        resources_list
            A list of Resources instances to combine.

        Returns
        -------
            A new Resources instance with the maximum values from the input instances.

        """
        if not resources_list:
            return Resources()

        max_data: dict[str, Any] = {
            "cpus": None,
            "gpus": None,
            "memory": None,
            "time": None,
            "partition": None,
            "extra_args": {},
        }

        for resources in resources_list:
            if resources.cpus is not None:
                max_data["cpus"] = (
                    resources.cpus
                    if max_data["cpus"] is None
                    else max(max_data["cpus"], resources.cpus)
                )
            if resources.gpus is not None:
                max_data["gpus"] = (
                    resources.gpus
                    if max_data["gpus"] is None
                    else max(max_data["gpus"], resources.gpus)
                )
            if resources.memory is not None:
                max_memory_gb = (
                    Resources._convert_to_gb(max_data["memory"])
                    if max_data["memory"] is not None
                    else 0
                )
                current_memory_gb = Resources._convert_to_gb(resources.memory)
                if current_memory_gb > max_memory_gb:
                    max_data["memory"] = resources.memory
            if resources.time is not None:
                max_data["time"] = (
                    resources.time
                    if max_data["time"] is None
                    else max(max_data["time"], resources.time)
                )
            if resources.partition is not None:
                max_data["partition"] = resources.partition

            for key, value in resources.extra_args.items():
                if key not in max_data["extra_args"]:
                    max_data["extra_args"][key] = value

        return Resources(**max_data)

    def with_defaults(
        self,
        default_resources: Resources | None,
    ) -> Resources:
        """Combine the Resources instance with default resources."""
        if default_resources is None:
            return self
        return Resources(**dict(default_resources.dict(), **self.dict()))

    @staticmethod
    def maybe_with_defaults(
        resources: Resources | None | Callable[[dict[str, Any]], Resources],
        default_resources: Resources | None,
    ) -> Resources | Callable[[dict[str, Any]], Resources] | None:
        """Combine the Resources instance with default resources, if provided."""
        if resources is None and default_resources is None:
            return None
        if resources is None:
            return default_resources
        if default_resources is None:
            return resources
        if callable(resources):
            return functools.partial(
                _delayed_resources_with_defaults,
                _resources=resources,
                _default_resources=default_resources,
            )
        return resources.with_defaults(default_resources)

    def dict(self) -> dict[str, Any]:
        """Return the Resources instance as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the Resources instance.

        """
        return {k: v for k, v in asdict(self).items() if v is not None}


def _delayed_resources_with_defaults(
    kwargs: dict[str, Any],
    *,
    _resources: Callable[[dict[str, Any]], Resources],
    _default_resources: Resources,
) -> Resources:
    resources = _resources(kwargs)
    return resources.with_defaults(_default_resources)


def _ensure_resources(
    kwargs: dict[str, Any],
    *,
    resources_callable: Callable[[dict[str, Any]], Resources | dict[str, Any]],
) -> Resources:
    resources_instance = resources_callable(kwargs)
    if isinstance(resources_instance, dict):
        return Resources(**resources_instance)
    return resources_instance


@dataclass
class NodeInfo:
    """A dataclass representing the description of a node in a cluster."""

    name: str
    cpus: int
    memory: int  # Memory in GB
    gpu_model: str
    gpus: int
    partitions: list[str]


@functools.cache
def _slurm_node_info() -> list[NodeInfo]:
    result = subprocess.run(
        ["scontrol", "show", "node", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    node_info = json.loads(result.stdout)
    nodes = []
    for node in node_info["nodes"]:
        gres_info = node["gres"].split(":")
        gpu_model = gres_info[1] if len(gres_info) > 1 else "unknown"
        gpus = int(gres_info[2]) if len(gres_info) > 2 else 0  # noqa: PLR2004
        node_desc = NodeInfo(
            name=node["name"],
            cpus=node["cpus"],
            memory=node["real_memory"] / 1024,  # MB -> GB
            gpu_model=gpu_model,
            gpus=gpus,
            partitions=node["partitions"],
        )
        nodes.append(node_desc)
    return nodes


def _read_toml(resources_filename: str | Path | None) -> dict | None:
    if isinstance(resources_filename, str):
        resources_filename = Path(resources_filename)
    possible_paths = [
        resources_filename,
        Path.cwd() / "resources.toml",
        Path.home() / ".config" / "pipefunc" / "resources.toml",
        Path("/etc/pipefunc/resources.toml"),
    ]
    for path in possible_paths:
        if path is None:
            continue
        if path.exists():
            with path.open("rb") as file:
                return tomllib.load(file)
    return None


def node_info(resources_filename: str | Path | None = None) -> list[NodeInfo]:  # pragma: no cover
    """Get information about the nodes in the SLURM cluster from a TOML file or via the SLURM command.

    This function first attempts to load node information from a ``resources.toml`` file.
    The function looks for the file in the following locations, in order:

    1. The path specified by the `resources_filename` argument.
    2. The current working directory.
    3. The user's configuration directory (``~/.config/pipefunc/``).
    4. A common global path (``/etc/pipefunc/resources.toml``).

    If the TOML file is found, it parses the file and returns the node information as a dictionary of
    `NodeInfo` instances, where the key is the node name.

    If the TOML file is not found or cannot be parsed, the function falls back to querying the SLURM cluster
    directly using the ``scontrol show node --json`` command. The information from SLURM is also returned as
    a dictionary of `NodeInfo` instances.

    Returns
    -------
        A list of `NodeInfo` instances representing the nodes in the SLURM cluster.

    """
    toml_data = _read_toml(resources_filename)
    if toml_data is not None:
        return [
            NodeInfo(
                name=node_name,
                cpus=node_data["cpus"],
                memory=node_data["memory"],  # Assume memory is already in GB in TOML
                gpu_model=node_data["gpu_model"],
                gpus=node_data["gpus"],
                partitions=node_data["partitions"],
            )
            for node_name, node_data in toml_data.get("nodes", {}).items()
        ]

    # Fallback to using the SLURM command if no TOML file is found or an error occurred
    return _slurm_node_info()


@dataclass
class PartitionInfo:
    """A dataclass representing the description of a partition in a cluster.

    If nodes in a partition have different resources, this class will contain the minimum resources.
    """

    name: str
    nodes: list[str]
    cpus: int
    memory: int
    gpus: int
    gpu_model: str


def partition_info(resources_filename: str | Path | None = None) -> list[PartitionInfo]:
    """Get information about the partitions in the SLURM cluster.

    This function first attempts to load partition information from a ``resources.toml`` file.
    The function looks for the file in the following locations, in order:

    1. The path specified by the `resources_filename` argument.
    2. The current working directory.
    3. The user's configuration directory (``~/.config/pipefunc/``).
    4. A common global path (``/etc/pipefunc/resources.toml``).

    If the TOML file is found, it parses the file and returns the partition information as a list of
    `PartitionInfo` instances.

    If the TOML file is not found, the function falls back to generating the partition
    information by querying the SLURM cluster directly using the `node_info` function. The partition
    information is derived by grouping nodes by their associated partitions and determining the minimum
    resources (CPUs, memory, GPUs) available in each partition.

    The resulting partition information includes the following:

    - Partition name.
    - List of node names in the partition.
    - Minimum number of CPUs across all nodes in the partition.
    - Minimum amount of memory (in GB) across all nodes in the partition.
    - Minimum number of GPUs across all nodes in the partition.
    - GPU model, or "various" if nodes in the partition have different GPU models.

    Returns
    -------
        A list of `PartitionInfo` instances representing the partitions in the SLURM cluster, with each
        partition containing the minimum resources available across its nodes.

    """
    toml_data = _read_toml(resources_filename)
    if toml_data is not None:
        return [
            PartitionInfo(
                name=partition_name,
                nodes=data["nodes"],
                cpus=data["cpus"],
                memory=data["memory"],
                gpus=data["gpus"],
                gpu_model=data["gpu_model"],
            )
            for partition_name, data in toml_data.get("partition", {}).items()
        ]

    _node_info = node_info()
    partitions = defaultdict(list)

    # Group nodes by partition
    for node in _node_info:
        for partition in node.partitions:
            partitions[partition].append(node)

    # Create PartitionInfo objects
    partition_infos = []
    for partition_name, nodes in partitions.items():
        # Determine the minimum resources across all nodes in the partition
        min_cpus = min(node.cpus for node in nodes)
        min_memory = min(node.memory for node in nodes)
        min_gpus = min(node.gpus for node in nodes)
        gpu_model = (
            nodes[0].gpu_model
            if all(node.gpu_model == nodes[0].gpu_model for node in nodes)
            else "various"
        )

        partition_info = PartitionInfo(
            name=partition_name,
            nodes=[node.name for node in nodes],
            gpus=min_gpus,
            cpus=min_cpus,
            memory=min_memory,
            gpu_model=gpu_model,
        )
        partition_infos.append(partition_info)

    return partition_infos
