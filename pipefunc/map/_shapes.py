from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypeGuard, TypeVar

from pipefunc._utils import at_least_tuple

from ._mapspec import MapSpec, array_shape

if TYPE_CHECKING:
    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE

    from ._types import ShapeDict, ShapeTuple, UserShapeDict


class Shapes(NamedTuple):
    shapes: dict[OUTPUT_TYPE, ShapeTuple]
    masks: dict[OUTPUT_TYPE, tuple[bool, ...]]


def _input_shapes_and_masks(
    pipeline: Pipeline,
    inputs: dict[str, Any],
) -> Shapes:
    input_parameters = set(pipeline.topological_generations.root_args)
    inputs_with_defaults = pipeline.defaults | inputs
    # The type of the shapes is `int | Literal["?"]` but we only use ints in this function
    shapes: dict[OUTPUT_TYPE, ShapeTuple] = {
        p: array_shape(inputs_with_defaults[p], p)
        for p in input_parameters
        if p in pipeline.mapspec_names
    }
    masks = {name: len(shape) * (True,) for name, shape in shapes.items()}
    return Shapes(shapes, masks)


def shape_and_mask_from_mapspec(
    mapspec: MapSpec,
    shapes: dict[OUTPUT_TYPE, ShapeTuple],
    internal_shapes: ShapeDict,
) -> tuple[ShapeTuple, tuple[bool, ...]]:
    """Determine the shape and mask from a mapspec.

    Only requires the key-value pairs (need to be resolved) in `shapes` that appear in
    `mapspec.input_names` and `mapspec.output_names`.
    """
    input_shapes = {p: shapes[p] for p in mapspec.input_names if p in shapes}
    output_shapes = {p: internal_shapes[p] for p in mapspec.output_names if p in internal_shapes}
    output_shape, mask = mapspec.shape(input_shapes, output_shapes)  # type: ignore[arg-type]
    return output_shape, mask


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    internal_shapes: UserShapeDict | None = None,
) -> Shapes:
    if internal_shapes is None:
        internal_shapes = {}
    internal = {k: at_least_tuple(v) for k, v in internal_shapes.items()}
    shapes: dict[OUTPUT_TYPE, ShapeTuple] = {}
    masks: dict[OUTPUT_TYPE, tuple[bool, ...]] = {}
    input_shapes, input_masks = _input_shapes_and_masks(pipeline, inputs)
    shapes.update(input_shapes)
    masks.update(input_masks)

    mapspec_funcs = [f for f in pipeline.sorted_functions if f.mapspec]
    for func in mapspec_funcs:
        assert func.mapspec is not None  # mypy
        output_shape, mask = shape_and_mask_from_mapspec(func.mapspec, shapes, internal)
        shapes[func.output_name] = output_shape
        masks[func.output_name] = mask
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape
                masks[output_name] = mask

    assert all(k in shapes for k in pipeline.mapspec_names if k not in internal)
    return Shapes(shapes, masks)


S = TypeVar("S")


def internal_shape_from_mask(shape: tuple[S, ...], mask: tuple[bool, ...]) -> tuple[S, ...]:
    return tuple(s for s, m in zip(shape, mask) if not m)


def external_shape_from_mask(shape: tuple[S, ...], mask: tuple[bool, ...]) -> tuple[S, ...]:
    return tuple(s for s, m in zip(shape, mask) if m)


def shape_is_resolved(shape: ShapeTuple) -> TypeGuard[tuple[int, ...]]:
    return all(isinstance(i, int) for i in shape)
