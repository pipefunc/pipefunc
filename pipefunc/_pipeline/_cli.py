from __future__ import annotations

import argparse
import inspect
import json
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

if TYPE_CHECKING:
    from pydantic import BaseModel

    from pipefunc import Pipeline


def cli(pipeline: Pipeline, description: str | None) -> None:
    """Run the pipeline from the command-line."""
    requires("rich", "griffe", "pydantic", reason="cli", extras="cli")
    import rich

    parser = _create_parser(description)
    input_model = pipeline.pydantic_model()
    _add_pydantic_arguments(parser, input_model)
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
    """Create the argparse.ArgumentParser instance."""
    try:  # pragma: no cover
        from rich_argparse import RichHelpFormatter as _HelpFormatter
    except ImportError:  # pragma: no cover
        from argparse import HelpFormatter as _HelpFormatter  # type: ignore[assignment]
    return argparse.ArgumentParser(description=description, formatter_class=_HelpFormatter)


def _add_pydantic_arguments(
    parser: argparse.ArgumentParser,
    input_model: type[BaseModel],
) -> None:
    """Add arguments from Pydantic Model fields to the parser."""
    from pydantic.fields import PydanticUndefined

    for field_name, field_info in input_model.model_fields.items():
        help_text = field_info.description or ""
        parser.add_argument(
            f"--{field_name}",
            type=str,  # CLI always receives strings, Pydantic will coerce
            default=field_info.default if field_info.default is not PydanticUndefined else None,
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
    for arg, p in sig_map.parameters.items():
        if arg not in {"run_folder", "parallel", "storage", "cleanup"}:
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


def _validate_inputs(
    args_cli: argparse.Namespace,
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Create Pydantic Model instance for validation and coercion of inputs."""
    input_data = {}
    for arg, field_info in input_model.model_fields.items():
        value = getattr(args_cli, arg)
        try:
            # Attempt to parse string as JSON (list, dict, number, bool, etc.)
            input_data[arg] = json.loads(value) if field_info.annotation is not str else value
        except json.JSONDecodeError:
            # If JSON parsing fails, use the string value directly
            input_data[arg] = value

    model_instance = input_model.model_validate(input_data)
    return model_instance.model_dump()


def _process_map_kwargs(args_cli: argparse.Namespace) -> dict[str, Any]:
    """Process and convert map_kwargs from argparse.Namespace."""
    map_kwargs: dict[str, Any] = {}
    for arg, value in vars(args_cli).items():
        if arg.startswith("map_"):
            map_kwargs[arg[4:]] = _maybe_bool(value)
    return map_kwargs


def _maybe_bool(value: Any) -> bool | Any:
    """Convert string values to boolean if they represent boolean literals."""
    if not isinstance(value, str):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value
