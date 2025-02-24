from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

if TYPE_CHECKING:
    from pydantic import BaseModel

    from pipefunc import Pipeline


DEFAULT_DESCRIPTION = """PipeFunc Pipeline CLI

This command-line interface (CLI) provides an easy and flexible way to execute a PipeFunc pipeline
directly from the terminal. The CLI is auto-generated based on your Pipeline's definition and input schema,
allowing you to supply parameters interactively or via a JSON file.

Input Modes:
  - `cli`: Supply individual input parameters as command-line options (e.g., --x, --y, etc.).
  - `json`: Load all input parameters from a JSON file by specifying the `--json-file` option.

Mapping Options:
  In addition to input parameters, you can configure mapping options (e.g., --map-parallel, --map-run_folder,
  --map-storage, --map-cleanup) to control parallel execution, storage method, and cleanup behavior.

Usage Examples:
  CLI mode:
    `python cli-example.py cli --V_left "[0, 1]" --V_right "[1, 2]" --mesh_size 1 --x 0 --y 1 --map-parallel false --map-cleanup true`

  JSON mode:
    `python cli-example.py json --json-file my_inputs.json --map-parallel false --map-cleanup true`

For more details, run the CLI with the `--help` flag.
"""


def cli(pipeline: Pipeline, description: str | None = None) -> None:
    """Execute a PipeFunc pipeline via a command-line interface.

    This function auto-generates a CLI for the provided PipeFunc pipeline using its underlying
    Pydantic input model for validation and coercion. It supports two modes of input:

      1. "cli" mode: Input parameters are provided directly via command-line options.
         Example:
           python cli-example.py cli --x 0 --y 1 --V_left "[0]" --V_right "[1]" --mesh_size 1 --coarse_mesh_size 0.1 --map-parallel false --map-cleanup true

      2. "json" mode: A single JSON file containing all input parameters is used.
         In this mode, you must supply the JSON file using the '--json-file' option.
         Example:
           python cli-example.py json --json-file my_inputs.json --map-parallel false --map-cleanup true

    Additionally, mapping options (prefixed with "--map-") allow configuration of parallel execution,
    result storage, and cleanup behaviors.

    On execution, the CLI:
      - Parses a positional mode argument (either "cli" or "json").
      - Adds input arguments based on the selected mode (individual parameters for CLI mode or a required
        JSON file for JSON mode).
      - Processes any mapping options provided.
      - Validates and coerces the inputs using the pipeline's Pydantic model.
      - Executes the pipeline with the provided inputs and mapping options.
      - Outputs the results to the console using Rich formatting.

    Parameters
    ----------
    pipeline : Pipeline
        The PipeFunc pipeline instance to be executed.
    description : str, optional
        A custom description for the CLI help message. If not provided, a default description is used.

    Raises
    ------
    ValueError
        If an invalid mode is specified or if required arguments (such as the JSON file in JSON mode) are missing.
    FileNotFoundError
        If the JSON input file specified in JSON mode does not exist.
    json.JSONDecodeError
        If the JSON input file is not formatted correctly.

    """
    requires("rich", "griffe", "pydantic", reason="cli", extras="cli")
    import rich

    parser = _create_parser(description)
    input_model = pipeline.pydantic_model()

    # First, do a partial parse to check the selected mode.
    args_partial, _ = parser.parse_known_args()
    mode = args_partial.mode

    if mode == "json":
        parser.add_argument(
            "--json-file",
            type=Path,
            help="Path to a JSON file containing inputs for the pipeline.",
            required=True,
        )
    elif mode == "cli":
        _add_pydantic_arguments(parser, input_model)
    else:  # pragma: no cover
        msg = f"Invalid mode: {mode}. Must be 'cli' or 'json'."
        raise ValueError(msg)

    _add_map_arguments(parser)
    args_cli = parser.parse_args()
    inputs = _validate_inputs(args_cli, input_model)
    map_kwargs = _process_map_kwargs(args_cli)

    rich.print("Inputs from CLI:", inputs)
    rich.print("Map kwargs from CLI:", map_kwargs)
    results = pipeline.map(inputs, **map_kwargs)
    rich.print("\n\n[bold blue]Results:")
    rich.print(results)


def _create_parser(description: str | None) -> argparse.ArgumentParser:
    """Create and return an ArgumentParser instance with a positional 'mode' argument.

    The parser is configured to support two modes:
      - 'cli': Accepts individual command-line input parameters.
      - 'json': Requires a JSON file containing the inputs via the '--json-file' option.

    Parameters
    ----------
    description : str or None
        The description for the help message; if None, DEFAULT_DESCRIPTION is used.

    Returns
    -------
    argparse.ArgumentParser
        The constructed ArgumentParser instance.

    """
    try:  # pragma: no cover
        from rich_argparse import RawTextRichHelpFormatter as _HelpFormatter
    except ImportError:  # pragma: no cover
        from argparse import (
            RawDescriptionHelpFormatter as _HelpFormatter,  # type: ignore[assignment]
        )
    if description is None:  # pragma: no cover
        description = DEFAULT_DESCRIPTION
    parser = argparse.ArgumentParser(description=description, formatter_class=_HelpFormatter)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cli", "json"],
        default="cli",
        help="Input mode: 'cli' to specify parameters via command-line options or 'json' to load from a JSON file.",
    )
    return parser


def _add_pydantic_arguments(
    parser: argparse.ArgumentParser,
    input_model: type[BaseModel],
) -> None:
    """Add arguments from Pydantic Model fields to the parser."""
    from pydantic.fields import PydanticUndefined

    for field_name, field_info in input_model.model_fields.items():
        help_text = field_info.description or ""
        default = field_info.default if field_info.default is not PydanticUndefined else None
        if default is not None:
            help_text += f" (default: {default})"
        parser.add_argument(
            f"--{field_name}",
            type=str,  # CLI always receives strings, Pydantic will coerce
            default=default,
            help=help_text,
        )


def _add_map_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the Pipeline.map method to the parser."""
    from pipefunc._pipeline._autodoc import _create_parameter_row, parse_function_docstring
    from pipefunc._pipeline._base import Pipeline

    doc_map = parse_function_docstring(Pipeline.map)
    sig_map = inspect.signature(Pipeline.map)
    defaults = {
        "run_folder": "run_folder",
        "parallel": True,
        "storage": "file_array",
        "cleanup": True,
    }
    include_only = {"run_folder", "parallel", "storage", "cleanup"}
    for arg, p in sig_map.parameters.items():
        if arg not in include_only:
            continue
        default = defaults[arg]
        row = _create_parameter_row(
            arg,
            [doc_map.parameters[arg]],
            {arg: default},
            {arg: p.annotation},
            skip_optional=True,
        )
        help_text = str(row[-1]).replace("``", "`")
        parser.add_argument(
            f"--map-{arg}",
            type=str,
            default=default,
            help=help_text,
        )


def _validate_inputs_from_cli(
    args_cli: argparse.Namespace,
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Create Pydantic Model instance for validation and coercion of inputs from CLI args."""
    input_data = {}
    for arg, field_info in input_model.model_fields.items():
        value = getattr(args_cli, arg)
        try:
            input_data[arg] = json.loads(value) if field_info.annotation is not str else value
        except json.JSONDecodeError:
            input_data[arg] = value
    model_instance = input_model.model_validate(input_data)
    return model_instance.model_dump()


def _validate_inputs_from_json(
    args_cli: argparse.Namespace,
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Create Pydantic Model instance for validation and coercion of inputs from a JSON file."""
    json_file_path = args_cli.json_file
    assert isinstance(json_file_path, Path)
    try:
        with json_file_path.open() as f:
            input_data = json.load(f)
    except FileNotFoundError:  # pragma: no cover
        msg = f"JSON input file not found: {json_file_path}"
        raise FileNotFoundError(msg) from None
    except json.JSONDecodeError:  # pragma: no cover
        raise
    model_instance = input_model.model_validate(input_data)
    return model_instance.model_dump()


def _validate_inputs(
    args_cli: argparse.Namespace,
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Dispatch to the correct input validation function based on mode (deprecated)."""
    if args_cli.mode == "json":
        return _validate_inputs_from_json(args_cli, input_model)
    return _validate_inputs_from_cli(args_cli, input_model)


def _process_map_kwargs(args_cli: argparse.Namespace) -> dict[str, Any]:
    """Process and convert map_kwargs from argparse.Namespace."""
    map_kwargs: dict[str, Any] = {}
    for arg, value in vars(args_cli).items():
        if arg.startswith("map_"):
            map_kwargs[arg[4:]] = _maybe_bool(value)
    return map_kwargs


def _maybe_bool(value: Any) -> bool | Any:
    """Convert string values to boolean if they represent boolean literals."""
    if not isinstance(value, str):  # pragma: no cover
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value
