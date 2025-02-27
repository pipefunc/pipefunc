from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

if TYPE_CHECKING:
    from pydantic import BaseModel

    from pipefunc import Pipeline

DEFAULT_DESCRIPTION = """PipeFunc Pipeline CLI

This command-line interface (CLI) provides an easy and flexible way to execute a PipeFunc pipeline
directly from the terminal. The CLI is auto-generated based on your Pipeline's definition and input schema,
allowing you to supply parameters interactively (via the `cli` subcommand), load them from a JSON file
(via the `json` subcommand), or simply view the pipeline documentation (via the `docs` subcommand).
Mapping options (prefixed with `--map-`) allow you to configure parallel execution, storage method,
and cleanup behavior. In `cli` or `json` mode it runs `pipeline.map` with the provided inputs and
mapping options.

Usage Examples:
  CLI mode:
    `python cli-example.py cli --x 2 --y 3`
  JSON mode:
    `python cli-example.py json --json-file inputs.json`
  Docs mode:
    `python cli-example.py docs`

For more details, run the CLI with the `--help` flag.
"""


def cli(pipeline: Pipeline, description: str | None = None) -> None:
    """Automatically construct a command-line interface using argparse.

    This method creates an `argparse.ArgumentParser` instance, adds arguments for each
    root parameter in the pipeline using a Pydantic model, sets default values if they exist,
    parses the command-line arguments, and runs one of three subcommands:

    - ``cli``: Supply individual input parameters as command-line options.
    - ``json``: Load all input parameters from a JSON file.
    - ``docs``: Display the pipeline documentation (using `pipeline.print_documentation`).

    Mapping options (prefixed with `--map-`) are available for the `cli` and `json` subcommands to control
    parallel execution, storage method, and cleanup behavior.

    Usage Examples:

    **CLI mode:**
        ``python cli-example.py cli --x 2 --y 3 --map-parallel false --map-cleanup true``

    **JSON mode:**
        ``python cli-example.py json --json-file inputs.json --map-parallel false --map-cleanup true``

    **Docs mode:**
        ``python cli-example.py docs``

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
    from rich.traceback import install

    install()
    # Create the base parser.
    parser = argparse.ArgumentParser(
        description=description or DEFAULT_DESCRIPTION,
        formatter_class=_formatter_class(),
    )

    # Create subparsers for the three input modes.
    subparsers = parser.add_subparsers(
        title="Input Modes",
        dest="mode",
        help="Choose an input mode: `cli` for individual options, `json` to load from a JSON file, or `docs` to print documentation.",
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

    # Subparser for Docs mode: print pipeline documentation.
    _docs_parser = subparsers.add_parser(
        "docs",
        help="Print the pipeline documentation.",
        formatter_class=_formatter_class(),
    )

    # Parse arguments from the command line.
    args = parser.parse_args()

    # If no arguments are provided, show the help message and exit.
    if args.mode is None:  # pragma: no cover
        parser.print_help()
        sys.exit(0)

    # Docs mode: print documentation and exit.
    if args.mode == "docs":  # pragma: no cover
        pipeline.print_documentation()
        sys.exit(0)

    # Validate and parse inputs using the pydantic model.
    inputs = _validate_inputs(args, input_model)

    # Process mapping-related arguments.
    map_kwargs = _process_map_kwargs(args)

    rich.print("Inputs from CLI:", inputs)
    rich.print("Map kwargs from CLI:", map_kwargs)
    results = pipeline.map(inputs, **map_kwargs)
    rich.print("\n\n[bold blue]Results:")
    for key, value in results.items():
        rich.print(f"[bold yellow]Output `{key}`:[/]", "\n", value.output, "\n")


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

        if field_info.default is PydanticUndefined:
            default = None
        else:
            default = field_info.default
            if default is None:
                default = "null"

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
    import rich

    input_data = {}
    for arg, field_info in input_model.model_fields.items():
        value = getattr(args_cli, arg)
        try:
            input_data[arg] = (
                json.loads(value)
                if field_info.annotation is not str and isinstance(value, str)
                else value
            )
        except json.JSONDecodeError:
            msg = (
                f"[red bold]Error decoding JSON:[/] for `{arg}`: `{value!r}` with"
                f" type `{type(value)}` and annotation `{field_info.annotation}`"
            )
            rich.print(msg)
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
    msg = f"Invalid mode: {args_cli.mode}. Must be 'cli', 'json', or 'docs'."  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _process_map_kwargs(args_cli: argparse.Namespace) -> dict[str, Any]:
    """Process and convert map_kwargs from the argparse.Namespace."""
    map_kwargs: dict[str, Any] = {}
    for arg, value in vars(args_cli).items():
        if arg.startswith("map_"):
            map_kwargs[arg[4:]] = _maybe_bool_or_none(value)
    return map_kwargs


def _maybe_bool_or_none(value: Any) -> bool | Any:
    if not isinstance(value, str):  # pragma: no cover
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("none", "null"):  # pragma: no cover
        return None
    return value
