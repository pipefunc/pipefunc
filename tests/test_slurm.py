from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from pipefunc.resources import (
    NodeInfo,
    PartitionInfo,
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
partitions = ["notebooks", "mycluster-64", "mycluster-all"]

[nodes.mycluster-2]
cpus = 256
memory = 1024  # GB
gpu_model = "a100"
gpus = 8
partitions = ["mycluster-128", "mycluster-all"]

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
            partitions=["notebooks", "mycluster-64", "mycluster-all"],
        ),
        NodeInfo(
            name="mycluster-2",
            cpus=256,
            memory=1024,  # GB
            gpu_model="a100",
            gpus=8,
            partitions=["mycluster-128", "mycluster-all"],
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
            "partitions": ["notebooks", "mycluster-64", "mycluster-all"],
        },
        {
            "name": "mycluster-2",
            "cpus": 256,
            "real_memory": 2051933,
            "gres": "gpu:a100:8",
            "partitions": ["mycluster-128", "mycluster-all"],
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


@pytest.fixture()
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
            partitions=["notebooks", "mycluster-64", "mycluster-all"],
        ),
        NodeInfo(
            name="mycluster-2",
            cpus=256,
            memory=2051933 / 1024,  # Memory in GB
            gpu_model="a100",
            gpus=8,
            partitions=["mycluster-128", "mycluster-all"],
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
[partition.mycluster-64]
nodes = ["mycluster-1", "mycluster-2"]
cpus = 128
memory = 512  # GB
gpus = 8
gpu_model = "a100"

[partition.mycluster-128]
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
            name="mycluster-64",
            nodes=["mycluster-1", "mycluster-2"],
            cpus=128,
            memory=512,
            gpus=8,
            gpu_model="a100",
        ),
        PartitionInfo(
            name="mycluster-128",
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
                name="mycluster-64",
                nodes=["mycluster-1"],
                cpus=128,
                memory=512,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="mycluster-all",
                nodes=["mycluster-1", "mycluster-2"],
                cpus=128,
                memory=512,
                gpus=8,
                gpu_model="a100",
            ),
            PartitionInfo(
                name="mycluster-128",
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
