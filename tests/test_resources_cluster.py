from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from pipefunc.resources import (
    NodeInfo,
    PartitionInfo,
    Resources,
    calculate_resources_fit,
    node_info,
    partition_info,
)

if TYPE_CHECKING:
    from pathlib import Path

# Sample mocked TOML content corresponding to the mock_scontrol_response
mock_node_toml_content = """
[nodes.mycluster-1]
cpus = 128
memory = 512  # GB
gpu_model = "a100"
gpus = 8
partitions = ["notebooks", "partition-1", "partition-all"]

[nodes.mycluster-2]
cpus = 256
memory = 1024  # GB
gpu_model = "a100"
gpus = 8
partitions = ["partition-2", "partition-all"]

[nodes.mycluster-3]
cpus = 64
memory = 512  # GB
gpu_model = "unknown"
gpus = 0
partitions = ["general", "compute"]
"""


def test_node_info_from_toml(tmp_path: Path) -> None:
    filename = tmp_path / "resources.toml"
    with filename.open("w") as f:
        f.write(mock_node_toml_content)
    # Call the function to test
    nodes = node_info(filename)

    # Expected results
    expected_nodes = [
        NodeInfo(
            name="mycluster-1",
            cpus=128,
            memory=512,  # GB
            gpu_model="a100",
            gpus=8,
            partitions=["notebooks", "partition-1", "partition-all"],
        ),
        NodeInfo(
            name="mycluster-2",
            cpus=256,
            memory=1024,  # GB
            gpu_model="a100",
            gpus=8,
            partitions=["partition-2", "partition-all"],
        ),
        NodeInfo(
            name="mycluster-3",
            cpus=64,
            memory=512,  # GB
            gpu_model="unknown",
            gpus=0,
            partitions=["general", "compute"],
        ),
    ]

    # Assertions to compare the expected results with the actual results
    assert nodes == expected_nodes


# Sample mocked JSON response from `scontrol show node --json`
mock_scontrol_response = {
    "nodes": [
        {
            "name": "mycluster-1",
            "cpus": 128,
            "real_memory": 1031891,
            "gres": "gpu:a100:8",
            "partitions": ["notebooks", "partition-1", "partition-all"],
        },
        {
            "name": "mycluster-2",
            "cpus": 256,
            "real_memory": 2051933,
            "gres": "gpu:a100:8",
            "partitions": ["partition-2", "partition-all"],
        },
        {
            "name": "mycluster-3",
            "cpus": 64,
            "real_memory": 512000,
            "gres": "",
            "partitions": ["general", "compute"],
        },
    ],
}


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mocked_run:
        # Mock the subprocess.run return value
        mocked_result = MagicMock()
        mocked_result.stdout = json.dumps(mock_scontrol_response)
        mocked_run.return_value = mocked_result
        yield mocked_run


def test_slurm_node_info(mock_subprocess_run):
    # Call the function to test
    nodes = node_info()

    # Expected results
    expected_nodes = [
        NodeInfo(
            name="mycluster-1",
            cpus=128,
            memory=1031891 / 1024,  # Memory in GB
            gpu_model="a100",
            gpus=8,
            partitions=["notebooks", "partition-1", "partition-all"],
        ),
        NodeInfo(
            name="mycluster-2",
            cpus=256,
            memory=2051933 / 1024,  # Memory in GB
            gpu_model="a100",
            gpus=8,
            partitions=["partition-2", "partition-all"],
        ),
        NodeInfo(
            name="mycluster-3",
            cpus=64,
            memory=512000 / 1024,  # Memory in GB
            gpu_model="unknown",
            gpus=0,
            partitions=["general", "compute"],
        ),
    ]

    # Assertions to compare the expected results with the actual results
    assert nodes == expected_nodes
    mock_subprocess_run.assert_called_once_with(
        ["scontrol", "show", "node", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )


# Sample mocked TOML content for partition information
mock_partition_toml_content = """
[partition.partition-1]
nodes = ["mycluster-1", "mycluster-2"]
cpus = 128
memory = 512  # GB
gpus = 8
gpu_model = "a100"

[partition.partition-2]
nodes = ["mycluster-2"]
cpus = 256
memory = 1024  # GB
gpus = 8
gpu_model = "a100"

[partition.general]
nodes = ["mycluster-3"]
cpus = 64
memory = 512  # GB
gpus = 0
gpu_model = "unknown"
"""


def test_partition_info_from_toml(tmp_path: Path) -> None:
    filename = tmp_path / "resources.toml"
    with filename.open("w") as f:
        f.write(mock_partition_toml_content)
    # Call the function to test
    partitions = partition_info(filename)

    # Expected results
    expected_partitions = [
        PartitionInfo(
            name="partition-1",
            nodes=["mycluster-1", "mycluster-2"],
            cpus=128,
            memory=512,
            gpus=8,
            gpu_model="a100",
        ),
        PartitionInfo(
            name="partition-2",
            nodes=["mycluster-2"],
            cpus=256,
            memory=1024,
            gpus=8,
            gpu_model="a100",
        ),
        PartitionInfo(
            name="general",
            nodes=["mycluster-3"],
            cpus=64,
            memory=512,
            gpus=0,
            gpu_model="unknown",
        ),
    ]

    # Assertions to compare the expected results with the actual results
    assert partitions == expected_partitions


def test_partition_info_from_node_info(tmp_path: Path) -> None:
    filename = tmp_path / "resources.toml"
    with filename.open("w") as f:
        f.write(mock_node_toml_content)

    # Mock the node_info function to use the _slurm_node_info response
    with patch("pipefunc.resources.node_info", return_value=node_info(filename)):
        # Call the function to test
        partitions = partition_info()

        # Expected results
        expected_partitions = [
            PartitionInfo(
                name="notebooks",
                nodes=["mycluster-1"],
                cpus=128,
                memory=512,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="partition-1",
                nodes=["mycluster-1"],
                cpus=128,
                memory=512,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="partition-all",
                nodes=["mycluster-1", "mycluster-2"],
                cpus=128,
                memory=512,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="partition-2",
                nodes=["mycluster-2"],
                cpus=256,
                memory=1024,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="general",
                nodes=["mycluster-3"],
                cpus=64,
                memory=512,
                gpus=0,
                gpu_model="unknown",
            ),
            PartitionInfo(
                name="compute",
                nodes=["mycluster-3"],
                cpus=64,
                memory=512,
                gpus=0,
                gpu_model="unknown",
            ),
        ]

        # Assertions to compare the expected results with the actual results
        assert partitions == expected_partitions


def test_calculate_resources_fit_basic():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(cpus=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 4  # Expect 4 instances to fit


def test_calculate_resources_fit_cpu_limited():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=32,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(cpus=16, memory="32GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # CPU is the limiting factor


def test_calculate_resources_fit_memory_limited():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=128,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(cpus=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # Memory is the limiting factor


def test_calculate_resources_fit_gpu_limited():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=2,
        gpu_model="A100",
    )
    resources = Resources(cpus=16, memory="32GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # GPU is the limiting factor


def test_calculate_resources_fit_no_gpus_required():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=0,
        gpu_model="None",
    )
    resources = Resources(cpus=16, memory="64GB")

    fit = calculate_resources_fit(partition, resources)
    assert fit == 4  # Fit is not limited by GPUs


def test_calculate_resources_fit_gpus_required_but_not_available():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=0,
        gpu_model="None",
    )
    resources = Resources(cpus=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 0  # Can't fit any because GPUs are required but not available


def test_calculate_resources_fit_nodes_constraint():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2", "node3"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(nodes=2, cpus_per_node=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 1  # Can fit one instance of 2 nodes


def test_calculate_resources_fit_nodes_constraint_not_enough_nodes():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(nodes=3, cpus_per_node=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 0  # Can't fit any because not enough nodes


def test_calculate_resources_fit_cpus_per_node_constraint():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=32,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(nodes=1, cpus_per_node=16, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # Can fit two instances, each using one node


def test_calculate_resources_fit_cpus_per_node_constraint_too_high():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=32,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(nodes=1, cpus_per_node=64, memory="64GB", gpus=1)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 0  # Can't fit any because cpus_per_node is higher than available


def test_calculate_resources_fit_no_constraints():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources()

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # Should return the number of nodes in the partition


def test_calculate_resources_fit_memory_units():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(memory="128000MB")

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # 256 GB / 128 GB = 2


def test_calculate_resources_fit_all_constraints():
    partition = PartitionInfo(
        name="compute",
        nodes=["node1", "node2", "node3", "node4"],
        cpus=64,
        memory=256,
        gpus=4,
        gpu_model="A100",
    )
    resources = Resources(nodes=2, cpus_per_node=32, memory="128GB", gpus=2)

    fit = calculate_resources_fit(partition, resources)
    assert fit == 2  # Can fit two instances, each using two nodes
