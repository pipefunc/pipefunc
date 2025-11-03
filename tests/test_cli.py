import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._pipeline._cli import (
    _add_map_arguments,
    _add_pydantic_arguments,
    _process_map_kwargs,
    _validate_inputs,
)
from pipefunc.map import load_outputs
from pipefunc.typing import Array


def test_cli_add_pydantic_arguments() -> None:
    from pydantic import BaseModel, Field

    parser = argparse.ArgumentParser()

    class MockInputModel(BaseModel):
        param1: int = Field(..., description="Parameter 1 description")
        param2: str = Field("default_value", description="Parameter 2 description")
        param3: float | None = Field(None, description="Parameter 3 description")

    _add_pydantic_arguments(parser, MockInputModel)
    actions = parser._actions  # Access private actions for testing

    # Expecting 1 help action + 3 parameter actions = 4 actions.
    assert len(actions) == 4

    param1_action = actions[1]
    assert param1_action.dest == "param1"
    assert param1_action.type is str
    assert param1_action.help == "Parameter 1 description"
    assert param1_action.default is None

    param2_action = actions[2]
    assert param2_action.dest == "param2"
    assert param2_action.type is str
    # Expected help text now includes the default value.
    assert param2_action.help == "Parameter 2 description (default: default_value)"
    assert param2_action.default == "default_value"

    param3_action = actions[3]
    assert param3_action.dest == "param3"
    assert param3_action.type is str
    assert param3_action.help == "Parameter 3 description (default: null)"
    assert param3_action.default == "null"


def test_cli_add_map_arguments() -> None:
    parser = argparse.ArgumentParser()
    _add_map_arguments(parser)
    actions = parser._actions

    assert any(action.dest == "map_run_folder" for action in actions)
    assert any(action.dest == "map_parallel" for action in actions)
    assert any(action.dest == "map_storage" for action in actions)
    assert any(action.dest == "map_cleanup" for action in actions)


def test_cli_validate_inputs() -> None:
    from pydantic import BaseModel

    class MockInputModel(BaseModel):
        param1: int
        param2: list[str]
        param3: float | None

    # Include mode so that _validate_inputs sees mode "cli"
    namespace = argparse.Namespace(param1="10", param2='["a", "b"]', param3="1.5", mode="cli")
    inputs = _validate_inputs(namespace, MockInputModel)
    assert inputs == {"param1": 10, "param2": ["a", "b"], "param3": 1.5}


def test_cli_validate_inputs_validation_error() -> None:
    from pydantic import BaseModel, ValidationError

    class MockInputModel(BaseModel):
        param1: int

    namespace = argparse.Namespace(param1="invalid", mode="cli")
    with pytest.raises(ValidationError):
        _validate_inputs(namespace, MockInputModel)


def test_cli_process_map_kwargs() -> None:
    namespace = argparse.Namespace(
        map_run_folder="test_run",
        map_parallel="False",
        map_storage="dict",
        mode="cli",
    )
    map_kwargs = _process_map_kwargs(namespace)
    assert map_kwargs == {
        "run_folder": "test_run",
        "parallel": False,
        "storage": "dict",
    }


def _monkeypatch_cli(cli_args_dict: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ns = argparse.Namespace(**cli_args_dict)

    def fake_parse_args(self, args=None, namespace=None):
        return fake_ns

    def fake_parse_known_args(self, args=None, namespace=None):
        return (fake_ns, [])

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", fake_parse_known_args)


def test_cli_pipeline_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the cli function with a simple pipeline end-to-end in CLI mode.

    The pipeline doubles each element of a list (using a mapspec) and then sums the
    resulting array. We simulate command-line input by monkey-patching both
    parse_args and parse_known_args to return a fake namespace.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        assert isinstance(y, np.ndarray)
        return int(np.sum(y))

    pipeline = Pipeline([double_it, take_sum])

    # CLI inputs: only "x" is used by the pipeline.
    inputs_cli = {
        "x": "[0, 1, 2, 3]",  # Provided as a JSON string.
    }
    # CLI map arguments.
    map_kwargs_cli = {
        "map_run_folder": str(tmp_path),
        "map_parallel": "False",
        "map_storage": "dict",
    }
    # Combine all CLI arguments including the positional "mode".
    cli_args_dict = {"mode": "cli", **inputs_cli, **map_kwargs_cli}
    _monkeypatch_cli(cli_args_dict, monkeypatch)
    printed = []

    def fake_print(*args, **kwargs):
        printed.append(args)

    monkeypatch.setattr("rich.print", fake_print)

    pipeline.cli("CLI integration test")
    final_sum = load_outputs("sum", run_folder=tmp_path)
    assert final_sum == 12
    # Verify that the printed inputs include key "x".
    assert "x" in printed[0][1]


def test_cli_pipeline_integration_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the cli function with a simple pipeline end-to-end in JSON mode.

    The pipeline doubles each element of a list (using a mapspec) and then sums the resulting array.
    This test creates a temporary JSON file with inputs and simulates JSON mode via monkey-patching
    both parse_args and parse_known_args.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        assert isinstance(y, np.ndarray)
        return int(np.sum(y))

    pipeline = Pipeline([double_it, take_sum])
    # Create a temporary JSON file with the input "x".
    inputs = {"x": [0, 1, 2, 3]}
    json_file = tmp_path / "inputs.json"
    with json_file.open("w") as f:
        json.dump(inputs, f)

    map_kwargs = {
        "map_run_folder": str(tmp_path),
        "map_parallel": "False",
        "map_storage": "dict",
        "map_cleanup": "True",
    }
    cli_args_dict: dict[str, str] = {
        "mode": "json",
        "json_file": json_file,  # type: ignore[dict-item]
        **map_kwargs,
    }
    _monkeypatch_cli(cli_args_dict, monkeypatch)

    printed = []

    def fake_print(*args, **kwargs):
        printed.append(args)

    monkeypatch.setattr("rich.print", fake_print)

    pipeline.cli("CLI integration test")
    final_sum = load_outputs("sum", run_folder=tmp_path)
    assert final_sum == 12
    # Verify that the printed inputs (loaded from JSON) include key "x".
    assert "x" in printed[0][1]


def test_none_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @pipefunc("foo")
    def foo(x: int, y: int | None = None) -> int:
        return x + (y or 1)

    pipeline = Pipeline([foo])
    cli_args = {
        "mode": "cli",
        "x": "10",
        "y": "null",
    }
    map_kwargs_cli = {
        "map_run_folder": str(tmp_path),
        "map_parallel": "False",
        "map_storage": "dict",
    }
    cli_args_dict = {**cli_args, **map_kwargs_cli}
    _monkeypatch_cli(cli_args_dict, monkeypatch)
    pipeline.cli("CLI integration test")
    final_sum = load_outputs("foo", run_folder=tmp_path)
    assert final_sum == 11
