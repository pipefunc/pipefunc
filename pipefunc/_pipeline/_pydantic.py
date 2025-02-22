from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

if TYPE_CHECKING:
    from pydantic import BaseModel

    from pipefunc import Pipeline

has_griffe = importlib.util.find_spec("griffe") is not None


def pipeline_to_pydantic(pipeline: Pipeline, model_name: str = "InputModel") -> type[BaseModel]:
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

    for p in root_args:
        type_annotation = parameter_annotations.get(p, Any)
        description = None if doc is None else "\n".join(doc.parameters.get(p, []))
        default = defaults.get(p, ...)
        field = Field(default, description=description)
        field_definitions[p] = (type_annotation, field)

    return create_model(model_name, **field_definitions)
