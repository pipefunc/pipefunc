from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np

from pipefunc._utils import requires
from pipefunc.typing import ArrayElementType

if TYPE_CHECKING:
    from pydantic import BaseModel

    from pipefunc import Pipeline
    from pipefunc.map._mapspec import ArraySpec, MapSpec

has_griffe = importlib.util.find_spec("griffe") is not None


def pipeline_to_pydantic(pipeline: Pipeline, model_name: str = "InputModel") -> type[BaseModel]:
    """Create a Pydantic model from a pipeline for the root arguments.

    Parameters
    ----------
    pipeline
        The pipeline to create the Pydantic model from.
    model_name
        The name of the Pydantic model.

    """
    requires("pydantic", reason="pydantic_model")
    from pydantic import Field, create_model

    if has_griffe:
        from pipefunc._pipeline._autodoc import PipelineDocumentation

        doc = PipelineDocumentation.from_pipeline(pipeline)
    else:
        doc = None

    defaults = pipeline.defaults
    parameter_annotations = {}
    root_args = pipeline.root_args()
    field_definitions = {}
    for p in root_args:
        for f in pipeline.functions:
            if p in f.parameters:
                parameter_annotations[p] = f.parameter_annotations[p]
    mapspecs = pipeline.mapspecs()
    for p in root_args:
        type_annotation = parameter_annotations.get(p, Any)
        type_annotation = _maybe_ndarray_type_annotation_from_mapspec(
            p,
            type_annotation,
            mapspecs,
        )
        description = None if doc is None else "\n".join(doc.parameters.get(p, []))
        default = defaults.get(p, ...)
        field = Field(default, description=description)
        field_definitions[p] = (type_annotation, field)

    return create_model(model_name, **field_definitions)


def _maybe_ndarray_type_annotation_from_mapspec(
    parameter_name: str,
    type_annotation: Any,
    mapspecs: list[MapSpec],
) -> Any:
    array_spec = _select_array_spec(parameter_name, mapspecs)
    if array_spec is None:
        return type_annotation
    ndim = len(array_spec.axes)
    if ndim == 1:
        return list[type_annotation]
    # Use NDArray for higher dimensions
    shape_any_tuple = (Any,) * ndim
    shape = tuple[shape_any_tuple]  # type: ignore[valid-type]
    # Same as pipefunc.typing.Array but with shape
    return Annotated[
        np.ndarray[shape, np.dtype[np.object_]],
        ArrayElementType[type_annotation],  # type: ignore[valid-type]
    ]


def _select_array_spec(parameter_name: str, mapspecs: list[MapSpec]) -> ArraySpec | None:
    for m in mapspecs:
        for spec in m.inputs:
            if spec.name == parameter_name:
                return spec
    return None
