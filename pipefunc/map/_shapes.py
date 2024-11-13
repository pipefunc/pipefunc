from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from pipefunc._utils import at_least_tuple

from ._mapspec import array_shape

if TYPE_CHECKING:
    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE

    from ._types import ShapeTuple, UserShapeDict


class Shapes(NamedTuple):
    shapes: dict[OUTPUT_TYPE, ShapeTuple]
    masks: dict[OUTPUT_TYPE, tuple[bool, ...]]


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    internal_shapes: UserShapeDict | None = None,
) -> Shapes:
    if internal_shapes is None:
        internal_shapes = {}
    internal = {k: at_least_tuple(v) for k, v in internal_shapes.items()}

    input_parameters = set(pipeline.topological_generations.root_args)

    inputs_with_defaults = pipeline.defaults | inputs
    shapes: dict[OUTPUT_TYPE, tuple[int, ...]] = {
        p: array_shape(inputs_with_defaults[p], p)
        for p in input_parameters
        if p in pipeline.mapspec_names
    }
    masks = {name: len(shape) * (True,) for name, shape in shapes.items()}
    mapspec_funcs = [f for f in pipeline.sorted_functions if f.mapspec]
    for func in mapspec_funcs:
        assert func.mapspec is not None  # mypy
        input_shapes = {p: shapes[p] for p in func.mapspec.input_names if p in shapes}
        output_shapes = {p: internal[p] for p in func.mapspec.output_names if p in internal}
        output_shape, mask = func.mapspec.shape(input_shapes, output_shapes)  # type: ignore[arg-type]
        shapes[func.output_name] = output_shape
        masks[func.output_name] = mask
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape
                masks[output_name] = mask

    assert all(k in shapes for k in pipeline.mapspec_names if k not in internal)
    return Shapes(shapes, masks)


def internal_shape_from_mask(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if not m)


def external_shape_from_mask(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if m)
