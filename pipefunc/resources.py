"""Provides the `pipefunc.resources` module, containing the `Resources` class."""

from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, eq=True)
class Resources:
    """A dataclass representing computational resources for a job.

    Parameters
    ----------
    num_cpus
        The number of CPUs required for the job. Must be a positive integer.
    num_gpus
        The number of GPUs required for the job. Must be a non-negative integer.
    num_nodes
        The number of nodes required for the job. Must be a positive integer.
    num_cpus_per_node
        The number of CPUs per node required for the job. Must be a positive integer.
    memory
        The memory required for the job. Must be a valid string (e.g., ``'2GB'``, ``'500MB'``).
    wall_time
        The wall time required for the job. Must be a valid string (e.g., ``'2:00:00'``, ``'48:00:00'``).
    partition
        The partition to submit the job to.
    extra_args
        Extra arguments for the job. Default is an empty dictionary.

    Raises
    ------
    ValueError
        If any of the input parameters do not meet the specified constraints.

    Notes
    -----
    - `num_cpus` and `num_nodes` cannot be specified together.
    - `num_cpus_per_node` must be specified with `num_nodes`.

    Examples
    --------
    >>> resources = Resources(num_cpus=4, memory='16GB', wall_time='2:00:00')
    >>> resources.num_cpus
    4
    >>> resources.memory
    '16GB'
    >>> resources.wall_time
    '2:00:00'

    """

    num_cpus: int | None = None
    num_gpus: int | None = None
    num_nodes: int | None = None
    num_cpus_per_node: int | None = None
    memory: str | None = None
    wall_time: str | None = None
    partition: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate input parameters after initialization.

        Raises
        ------
        ValueError
            If any of the input parameters do not meet the specified constraints.

        """
        if self.num_cpus is not None and self.num_cpus <= 0:
            msg = "num_cpus must be a positive integer."
            raise ValueError(msg)
        if self.num_gpus is not None and self.num_gpus < 0:
            msg = "num_gpus must be a non-negative integer."
            raise ValueError(msg)
        if self.num_nodes is not None and self.num_nodes <= 0:
            msg = "num_nodes must be a positive integer."
            raise ValueError(msg)
        if self.num_cpus_per_node is not None and self.num_cpus_per_node <= 0:
            msg = "num_cpus_per_node must be a positive integer."
            raise ValueError(msg)
        if self.memory is not None and not self._is_valid_memory(self.memory):
            msg = f"memory must be a valid string (e.g., '2GB', '500MB'), not '{self.memory}'."
            raise ValueError(msg)
        if self.wall_time is not None and not self._is_valid_wall_time(self.wall_time):
            msg = "wall_time must be a valid string (e.g., '2:00:00', '48:00:00')."
            raise ValueError(msg)
        if self.num_nodes and self.num_cpus:
            msg = (
                "num_nodes and num_cpus cannot be specified together."
                " Either use num_nodes and num_cpus_per_node or use num_cpus alone."
            )
            raise ValueError(msg)
        if self.num_cpus_per_node and not self.num_nodes:
            msg = "num_cpus_per_node must be specified with num_nodes."
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
    def _is_valid_wall_time(wall_time: str) -> bool:
        pattern = re.compile(r"^(\d+:)?(\d{2}:)?\d{2}:\d{2}$")
        return bool(pattern.match(wall_time))

    def to_slurm_options(self) -> str:
        """Convert the Resources instance to SLURM options.

        Returns
        -------
        str
            A string containing the SLURM options.

        """
        options = []
        if self.num_cpus:
            options.append(f"--cpus-per-task={self.num_cpus}")
        if self.num_gpus:
            options.append(f"--gres=gpu:{self.num_gpus}")
        if self.num_nodes:
            options.append(f"--nodes={self.num_nodes}")
        if self.num_cpus_per_node:
            options.append(f"--cpus-per-node={self.num_cpus_per_node}")
        if self.memory:
            options.append(f"--mem={self.memory}")
        if self.wall_time:
            options.append(f"--time={self.wall_time}")
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
            "num_cpus": None,
            "num_gpus": None,
            "memory": None,
            "wall_time": None,
            "partition": None,
            "extra_args": {},
        }

        for resources in resources_list:
            if resources.num_cpus is not None:
                max_data["num_cpus"] = (
                    resources.num_cpus
                    if max_data["num_cpus"] is None
                    else max(max_data["num_cpus"], resources.num_cpus)
                )
            if resources.num_gpus is not None:
                max_data["num_gpus"] = (
                    resources.num_gpus
                    if max_data["num_gpus"] is None
                    else max(max_data["num_gpus"], resources.num_gpus)
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
            if resources.wall_time is not None:
                max_data["wall_time"] = (
                    resources.wall_time
                    if max_data["wall_time"] is None
                    else max(max_data["wall_time"], resources.wall_time)
                )
            if resources.partition is not None:
                max_data["partition"] = resources.partition

            for key, value in resources.extra_args.items():
                if key not in max_data["extra_args"]:
                    max_data["extra_args"][key] = value

        return Resources(**max_data)

    def dict(self) -> dict[str, Any]:
        """Return the Resources instance as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the Resources instance.

        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    def with_defaults(self, default_resources: Resources | None) -> Resources:
        """Combine the Resources instance with default resources."""
        if default_resources is None:
            return self
        return Resources(**dict(default_resources.dict(), **self.dict()))
