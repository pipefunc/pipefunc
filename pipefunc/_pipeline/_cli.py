from __future__ import annotations

import argparse
import inspect
import json
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

if TYPE_CHECKING:
    from pipefunc import Pipeline


def cli(pipeline: Pipeline, description: str | None) -> None:
    """Run the pipeline from the command-line."""
    requires("rich", "griffe", "pydantic", "rich_argparse", reason="cli", extras="cli")
    import rich
    from rich_argparse import RichHelpFormatter

    from ._autodoc import _create_parameter_row, parse_function_docstring

    parser = argparse.ArgumentParser(description=description, formatter_class=RichHelpFormatter)

    # Generate Pydantic Model
    InputModel = pipeline.pydantic_model()  # noqa: N806

    # Add arguments from Pydantic Model fields
    for field_name, field_info in InputModel.model_fields.items():
        help_text = field_info.description or ""
        parser.add_argument(
            f"--{field_name}",
            type=str,  # CLI always receives strings, Pydantic will coerce
            default=field_info.default
            if field_info.default is not inspect.Parameter.empty
            else None,
            help=help_text,
        )

    doc_map = parse_function_docstring(type(pipeline).map)
    sig_map = inspect.signature(type(pipeline).map)
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

    # Parse the arguments
    args_cli = parser.parse_args()

    # Create Pydantic Model instance for validation and coercion
    input_data = {}
    for arg, field_info in InputModel.model_fields.items():
        value = getattr(args_cli, arg)
        try:
            # Attempt to parse string as JSON (list, dict, number, bool, etc.)
            input_data[arg] = json.loads(value) if field_info.annotation is not str else value
        except json.JSONDecodeError:
            # If JSON parsing fails, use the string value directly
            input_data[arg] = value

    model_instance = InputModel.model_validate(input_data)
    inputs = model_instance.model_dump()

    rich.print("Inputs from CLI:", model_instance)
    map_kwargs: dict[str, Any] = {}
    for arg, value in vars(args_cli).items():
        if arg.startswith("map_"):
            map_kwargs[arg[4:]] = _maybe_bool(value)

    rich.print("Map kwargs from CLI:", map_kwargs)
    results = pipeline.map(inputs, **map_kwargs)
    rich.print("\n\n[bold blue]Results:")
    rich.print(results)


def _maybe_bool(value: Any) -> bool | Any:
    if not isinstance(value, str):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value
