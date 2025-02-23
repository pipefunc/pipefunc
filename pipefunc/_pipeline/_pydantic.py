from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np

from pipefunc._utils import is_imported, requires

if TYPE_CHECKING:
    from collections.abc import Callable

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
    from pydantic import ConfigDict, Field, create_model

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
            if p not in f.parameters:
                continue
            if p not in f.parameter_annotations:
                msg = (
                    f"Parameter '{p}' is not annotated, using `typing.Any`."
                    " This may lead to unexpected behavior and incorrect type coercion."
                )
                warnings.warn(msg, stacklevel=2)
            parameter_annotations[p] = f.parameter_annotations.get(p, Any)
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

    return create_model(
        model_name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **field_definitions,
    )


def maybe_pydantic_model_to_dict(x: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Convert a Pydantic model to a dictionary if needed."""
    if isinstance(x, dict):
        return x
    if not is_imported("pydantic"):  # pragma: no cover
        # Means it is not a Pydantic model
        msg = "Unknown type, expected a Pydantic model or a dictionary."
        raise ValueError(msg)
    import pydantic

    assert isinstance(x, pydantic.BaseModel)
    return x.model_dump()


def _maybe_ndarray_type_annotation_from_mapspec(
    parameter_name: str,
    type_annotation: Any,
    mapspecs: list[MapSpec],
) -> Any:
    array_spec = _select_array_spec(parameter_name, mapspecs)
    if array_spec is None:
        return type_annotation
    from pydantic import AfterValidator
    from pydantic.functional_serializers import PlainSerializer

    ndim = len(array_spec.axes)
    list_type = _nested_list_type(ndim, type_annotation)
    return Annotated[
        list_type,
        AfterValidator(_nd_array_with_ndim(ndim)),
        PlainSerializer(lambda x: x.tolist(), return_type=list_type, when_used="json"),
    ]


def _nested_list_type(ndim: int, inner_type: Any) -> Any:
    """Recursively build a nested list type annotation.

    For ndim == 1, returns list[inner_type].
    For ndim == 2, returns list[list[inner_type]], etc.
    """
    if ndim < 1:  # pragma: no cover
        msg = "ndim must be at least 1"
        raise ValueError(msg)
    if ndim == 1:
        return list[inner_type]
    type_ = _nested_list_type(ndim - 1, inner_type)
    return list[type_]  # type: ignore[valid-type]


def _nd_array_with_ndim(ndim: int) -> Callable[[Any], np.ndarray]:
    def _as_ndarray(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim != ndim:  # pragma: no cover
            # This should never happen, as the pydantic model should prevent it.
            msg = f"Expected an array with {ndim} dimensions, got {arr.ndim}."
            raise ValueError(msg)
        return arr

    return _as_ndarray


def _select_array_spec(parameter_name: str, mapspecs: list[MapSpec]) -> ArraySpec | None:
    for m in mapspecs:
        for spec in m.inputs:
            if spec.name == parameter_name:
                return spec
    return None
