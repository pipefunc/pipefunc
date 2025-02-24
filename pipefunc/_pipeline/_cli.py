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
allowing you to supply parameters interactively (via the `cli` subcommand) or load them from a JSON file
(via the `json` subcommand).

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
    """Automatically construct a command-line interface using argparse.

    This method creates an `argparse.ArgumentParser` instance, adds arguments for each
    root parameter in the pipeline using a Pydantic model, sets the default values if they exist,
    parses the command-line arguments, and runs `pipeline.map` with the parsed arguments.
    Mapping options (prefixed with "--map-") are available in both subcommands to control
    parallel execution, storage method, and cleanup behavior.

    It constructs a CLI with two subcommands:
    - "cli": for specifying individual input parameters as command-line options.
    - "json": for loading input parameters from a JSON file.


    Usage Examples:
    CLI mode:
        python cli-example.py cli --V_left "[0, 1]" --V_right "[1, 2]" --mesh_size 1 --x 0 --y 1 --map-parallel false --map-cleanup true
    JSON mode:
        python cli-example.py json --json-file my_inputs.json --map-parallel false --map-cleanup true

    Parameters
    ----------
    pipeline
        The PipeFunc pipeline instance to be executed.
    description
        A custom description for the CLI help message. If not provided, a default description is used.

    Raises
    ------
    ValueError
        If an invalid subcommand is specified.
    FileNotFoundError
        If the JSON input file does not exist (in JSON mode).
    json.JSONDecodeError
        If the JSON input file is not formatted correctly.

    Examples
    --------
    >>> if __name__ == "__main__":
    ...     pipeline = create_my_pipeline()
    ...     pipeline.cli()

    """
    requires("rich", "griffe", "pydantic", reason="cli", extras="cli")
    import rich

    # Create the base parser.
    parser = argparse.ArgumentParser(
        description=description or DEFAULT_DESCRIPTION,
        formatter_class=_formatter_class(),
    )

    # Create subparsers for the two input modes.
    subparsers = parser.add_subparsers(
        title="Input Modes",
        dest="mode",
        required=True,
        help="Choose an input mode: 'cli' for individual options or 'json' to load from a JSON file.",
    )

    # Subparser for CLI mode: add individual parameter arguments.
    cli_parser = subparsers.add_parser(
        "cli",
        help="Supply individual input parameters as command-line options.",
        formatter_class=_formatter_class(),
    )
    input_model = pipeline.pydantic_model()
    _add_pydantic_arguments(cli_parser, input_model)
    _add_map_arguments(cli_parser)

    # Subparser for JSON mode: require a JSON file.
    json_parser = subparsers.add_parser(
        "json",
        help="Load all input parameters from a JSON file.",
        formatter_class=_formatter_class(),
    )
    json_parser.add_argument(
        "--json-file",
        type=Path,
        help="Path to a JSON file containing inputs for the pipeline.",
        required=True,
    )
    _add_map_arguments(json_parser)

    # Parse arguments from the command line.
    args = parser.parse_args()

    # Validate and parse inputs using the pydantic model.
    inputs = _validate_inputs(args, input_model)

    # Process mapping-related arguments.
    map_kwargs = _process_map_kwargs(args)

    rich.print("Inputs from CLI:", inputs)
    rich.print("Map kwargs from CLI:", map_kwargs)
    results = pipeline.map(inputs, **map_kwargs)
    rich.print("\n\n[bold blue]Results:")
    rich.print(results)


def _formatter_class() -> type[argparse.RawDescriptionHelpFormatter]:
    try:  # pragma: no cover
        from rich_argparse import RawTextRichHelpFormatter

        return RawTextRichHelpFormatter  # noqa: TRY300
    except ImportError:  # pragma: no cover
        return argparse.RawDescriptionHelpFormatter


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
            type=str,  # CLI always receives strings; Pydantic will coerce them.
            default=default,
            help=help_text,
        )


def _add_map_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mapping options for Pipeline.map to the parser."""
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
    """Validate CLI-provided inputs using a Pydantic model."""
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
    """Validate inputs loaded from a JSON file using a Pydantic model."""
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
    """Dispatch input validation based on the chosen mode."""
    if args_cli.mode == "json":
        return _validate_inputs_from_json(args_cli, input_model)
    if args_cli.mode == "cli":
        return _validate_inputs_from_cli(args_cli, input_model)
    msg = f"Invalid mode: {args_cli.mode}. Must be 'cli' or 'json'."  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _process_map_kwargs(args_cli: argparse.Namespace) -> dict[str, Any]:
    """Process and convert map_kwargs from the argparse.Namespace."""
    map_kwargs: dict[str, Any] = {}
    for arg, value in vars(args_cli).items():
        if arg.startswith("map_"):
            map_kwargs[arg[4:]] = _maybe_bool(value)
    return map_kwargs


def _maybe_bool(value: Any) -> bool | Any:
    """Convert string values to booleans if they represent boolean literals."""
    if not isinstance(value, str):  # pragma: no cover
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value
