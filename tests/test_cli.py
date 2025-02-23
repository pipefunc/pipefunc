import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from pipefunc._pipeline._base import Pipeline
from pipefunc._pipeline._cli import (
    _add_map_arguments,
    _add_pydantic_arguments,
    _create_parser,
    _maybe_bool,
    _process_map_kwargs,
    _validate_inputs,
    cli,
)


def test_cli_create_parser() -> None:
    parser = _create_parser("Test Description")
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.description == "Test Description"


def test_cli_add_pydantic_arguments() -> None:
    parser = argparse.ArgumentParser()

    class MockInputModel(BaseModel):
        param1: int = Field(..., description="Parameter 1 description")
        param2: str = Field("default_value", description="Parameter 2 description")
        param3: float | None = Field(None, description="Parameter 3 description")

    _add_pydantic_arguments(parser, MockInputModel)
    actions = parser._actions  # Access private actions for testing

    assert len(actions) == 4  # help + 3 params
    param1_action = actions[1]
    assert param1_action.dest == "param1"
    assert param1_action.type is str
    assert param1_action.help == "Parameter 1 description"
    assert param1_action.default is None

    param2_action = actions[2]
    assert param2_action.dest == "param2"
    assert param2_action.type is str
    assert param2_action.help == "Parameter 2 description"
    assert param2_action.default == "default_value"

    param3_action = actions[3]
    assert param3_action.dest == "param3"
    assert param3_action.type is str
    assert param3_action.help == "Parameter 3 description"
    assert param3_action.default is None


def test_cli_add_map_arguments() -> None:
    parser = argparse.ArgumentParser()
    _add_map_arguments(parser)
    actions = parser._actions

    assert any(action.dest == "map_run_folder" for action in actions)
    assert any(action.dest == "map_parallel" for action in actions)
    assert any(action.dest == "map_storage" for action in actions)
    assert any(action.dest == "map_cleanup" for action in actions)


def test_cli_validate_inputs() -> None:
    class MockInputModel(BaseModel):
        param1: int
        param2: list[str]
        param3: float | None

    namespace = argparse.Namespace(param1="10", param2='["a", "b"]', param3="1.5")
    inputs = _validate_inputs(namespace, MockInputModel)
    assert inputs == {"param1": 10, "param2": ["a", "b"], "param3": 1.5}


def test_cli_validate_inputs_validation_error() -> None:
    class MockInputModel(BaseModel):
        param1: int

    namespace = argparse.Namespace(param1="invalid")  # "invalid" cannot be coerced to int
    with pytest.raises(ValidationError):
        _validate_inputs(namespace, MockInputModel)


def test_cli_process_map_kwargs() -> None:
    namespace = argparse.Namespace(
        map_run_folder="test_run",
        map_parallel="False",
        map_storage="dict",
    )
    map_kwargs = _process_map_kwargs(namespace)
    assert map_kwargs == {
        "run_folder": "test_run",
        "parallel": False,
        "storage": "dict",
    }


@pytest.mark.parametrize(
    ("value_str", "expected"),
    [
        ("True", True),
        ("true", True),
        ("TRUE", True),
        ("False", False),
        ("false", False),
        ("FALSE", False),
        ("None", None),
        ("none", None),
        ("NONE", None),
        ("123", 123),
        ("123.45", 123.45),
        ("abc", "abc"),
        (123, 123),  # Not a string
        (True, True),  # Not a string
        (False, False),  # Not a string
        (None, None),  # Not a string
    ],
)
def test_maybe_bool(value_str: Any, expected: Any) -> None:
    assert _maybe_bool(value_str) == expected


@patch("pipefunc._pipeline._cli._create_parser")
@patch("pipefunc._pipeline._cli._add_pydantic_arguments")
@patch("pipefunc._pipeline._cli._add_map_arguments")
@patch("pipefunc._pipeline._cli._parse_arguments")
@patch("pipefunc._pipeline._cli._validate_inputs")
@patch("pipefunc._pipeline._cli._process_map_kwargs")
@patch("pipefunc._pipeline._cli.rich")  # Mock rich.print
def test_cli_integration(
    mock_rich,
    mock_process_map_kwargs,
    mock_validate_inputs,
    mock_parse_arguments,
    mock_add_map_arguments,
    mock_add_pydantic_arguments,
    mock_create_parser,
) -> None:
    mock_pipeline = MagicMock(spec=Pipeline)
    mock_pipeline.pydantic_model.return_value = MagicMock(spec=BaseModel)

    description = "Test Pipeline CLI"
    cli(mock_pipeline, description)

    mock_create_parser.assert_called_once_with(description)
    mock_add_pydantic_arguments.assert_called_once()
    mock_add_map_arguments.assert_called_once()
    mock_parse_arguments.assert_called_once()
    mock_validate_inputs.assert_called_once()
    mock_process_map_kwargs.assert_called_once()
    mock_pipeline.map.assert_called_once()
    assert mock_rich.print.call_count >= 3  # Check for at least 3 rich.print calls
