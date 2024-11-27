from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from pipefunc._pipefunc import PipeFunc
from pipefunc._utils import at_least_tuple
from pipefunc.typing import (
    Array,
    NoAnnotation,
    Unresolvable,
    is_object_array_type,
    is_type_compatible,
)

if TYPE_CHECKING:
    from ._types import OUTPUT_TYPE


def validate_consistent_defaults(
    functions: list[PipeFunc],
    output_to_func: dict[OUTPUT_TYPE, PipeFunc],
) -> None:
    """Check that the default values for shared arguments are consistent."""
    arg_defaults = {}
    for f in functions:
        for arg, default_value in f.defaults.items():
            if arg in f._bound or arg in output_to_func:
                continue
            if arg not in arg_defaults:
                arg_defaults[arg] = default_value
            elif default_value != arg_defaults[arg]:
                msg = (
                    f"Inconsistent default values for argument '{arg}' in"
                    " functions. Please make sure the shared input arguments have"
                    " the same default value or are set only for one function."
                )
                raise ValueError(msg)


def validate_consistent_type_annotations(graph: nx.DiGraph) -> None:
    """Check that the type annotations for shared arguments are consistent."""
    for node in graph.nodes:
        if not isinstance(node, PipeFunc):
            continue
        deps = nx.descendants_at_distance(graph, node, 1)
        output_types = node.output_annotation
        for dep in deps:
            assert isinstance(dep, PipeFunc)
            for parameter_name, input_type in dep.parameter_annotations.items():
                if parameter_name not in output_types:
                    continue
                if _mapspec_is_generated(node, dep):
                    # NOTE: We cannot check the type-hints for auto-generated MapSpecs
                    continue
                if _mapspec_with_internal_shape(node, parameter_name):
                    # NOTE: We cannot verify the type hints because the output
                    # might be any iterable instead of an Array as returned by
                    # a map operation.
                    continue
                output_type = output_types[parameter_name]
                if (
                    _axis_is_reduced(node, dep, parameter_name)
                    and not is_object_array_type(output_type)
                    and not isinstance(output_type, Unresolvable)
                    and output_type is not NoAnnotation
                ):
                    output_type = Array[output_type]  # type: ignore[valid-type]
                if not is_type_compatible(output_type, input_type):
                    msg = (
                        f"Inconsistent type annotations for:"
                        f"\n  - Argument `{parameter_name}`"
                        f"\n  - Function `{node.__name__}(...)` returns:\n      `{output_type}`."
                        f"\n  - Function `{dep.__name__}(...)` expects:\n      `{input_type}`."
                        "\nPlease make sure the shared input arguments have the same type."
                        "\nNote that the output type displayed above might be wrapped in"
                        " `pipefunc.typing.Array` if using `MapSpec`s."
                        " Disable this check by setting `validate_type_annotations=False`."
                    )
                    raise TypeError(msg)


def validate_scopes(functions: list[PipeFunc], new_scope: str | None = None) -> None:
    all_scopes = {scope for f in functions for scope in f.parameter_scopes}
    if new_scope is not None:
        all_scopes.add(new_scope)
    all_parameters = {p for f in functions for p in f.parameters + at_least_tuple(f.output_name)}
    if overlap := all_scopes & all_parameters:
        overlap_str = ", ".join(overlap)
        msg = f"Scope(s) `{overlap_str}` are used as both parameter and scope."
        raise ValueError(msg)


def _axis_is_reduced(f_out: PipeFunc, f_in: PipeFunc, parameter_name: str) -> bool:
    """Whether the output was the result of a map, and the input takes the entire result."""
    output_mapspec_names = f_out.mapspec.output_names if f_out.mapspec else ()
    input_mapspec_names = f_in.mapspec.input_names if f_in.mapspec else ()
    if f_in.mapspec:
        input_spec_axes = next(
            (s.axes for s in f_in.mapspec.inputs if s.name == parameter_name),
            None,
        )
    else:
        input_spec_axes = None
    return parameter_name in output_mapspec_names and (
        parameter_name not in input_mapspec_names
        or (input_spec_axes is not None and None in input_spec_axes)
    )


def _mapspec_is_generated(f_out: PipeFunc, f_in: PipeFunc) -> bool:
    if f_out.mapspec is None or f_in.mapspec is None:
        return False
    return f_out.mapspec._is_generated or f_in.mapspec._is_generated


def _mapspec_with_internal_shape(f_out: PipeFunc, parameter_name: str) -> bool:
    """Whether the output was not from a map operation but returned an array with internal shape."""
    if f_out.mapspec is None or parameter_name not in f_out.mapspec.output_names:
        return False
    output_spec = next(s for s in f_out.mapspec.outputs if s.name == parameter_name)
    all_inputs_in_outputs = f_out.mapspec.input_indices.issuperset(output_spec.indices)
    return not all_inputs_in_outputs


def validate_unique_output_names(
    output_name: OUTPUT_TYPE,
    output_to_func: dict[OUTPUT_TYPE, PipeFunc],
) -> None:
    for name in at_least_tuple(output_name):
        for other, func in output_to_func.items():
            if name in at_least_tuple(other):
                msg = f"The function with output name `{name!r}` already exists in the pipeline (`{func}`)."
                raise ValueError(msg)
