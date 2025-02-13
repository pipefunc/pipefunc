from pydantic import BaseModel, Field

from pipefunc import PipeFunc, Pipeline
from pipefunc.typing import safe_get_type_hints


def test_pydantic_annotations() -> None:
    class Foo(BaseModel):
        x: int
        y: int = 1
        z: dict = Field(default_factory=dict)

    foo = PipeFunc(Foo, "foo", defaults={"x": 2})

    type_hints = safe_get_type_hints(Foo)
    assert type_hints == {"x": int, "y": int, "z": dict}
    assert foo.defaults == {"x": 2, "y": 1, "z": {}}
    assert foo.parameter_annotations == {"x": int, "y": int, "z": dict}

    class Bar(BaseModel):
        foo: Foo

    bar = PipeFunc(Bar, "bar")
    assert bar.parameter_annotations == {"foo": Foo}
    assert bar.defaults == {}


def test_pydantic_model_with__call__() -> None:
    class Foo(BaseModel):
        x: int
        y: int = 1
        z: dict = Field(default_factory=dict)

        def __call__(self, a: int = 42) -> int:
            return self.x + self.y + a

    foo = PipeFunc(Foo(x=1, y=2, z={1: 2}), "foo")
    assert foo.__name__ == "Foo"
    assert foo(3) == foo(a=3) == 6
    assert foo.parameter_annotations == {"a": int}
    assert foo.output_annotation == {"foo": int}
    assert foo.defaults == {"a": 42}
    pipeline = Pipeline([foo])
    assert pipeline(a=3) == 6
    assert pipeline.parameter_annotations == {"a": int}
    assert pipeline.defaults == {"a": 42}
    assert pipeline.output_annotations == {"foo": int}


def test_dataclass_annotations() -> None:
    # Just for reference, pydantic.BaseModel should behave the same way as dataclasses.dataclass
    from dataclasses import dataclass, field

    @dataclass
    class Foo:
        x: int
        y: int = 1
        z: dict = field(default_factory=dict)

    foo = PipeFunc(Foo, "foo", defaults={"x": 2})
    assert foo.defaults == {"x": 2, "y": 1, "z": {}}
    assert foo.parameter_annotations == {"x": int, "y": int, "z": dict}
    assert foo.__name__ == "Foo"
