"""Tests for `pipefunc.Pipeline` using `pipefunc.resources.Resources`."""

from __future__ import annotations

from typing import Any

import pytest

from pipefunc import NestedPipeFunc, Pipeline, pipefunc
from pipefunc.resources import Resources


def test_default_resources_from_pipeline() -> None:
    @pipefunc(output_name="c", resources={"memory": "1GB", "cpus": 2})
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    pipeline1 = Pipeline([f, g], default_resources={"memory": "2GB", "cpus": 1})

    @pipefunc(output_name="e")
    def h(d):
        return d

    pipeline2 = Pipeline([h], default_resources={"memory": "3GB", "cpus": 3})

    @pipefunc(output_name="f", resources={"memory": "4GB", "cpus": 4})
    def i(e):
        return e

    pipeline3 = Pipeline([i])

    pipeline = pipeline1 | pipeline2 | pipeline3
    assert pipeline._default_resources is None
    assert isinstance(pipeline["c"].resources, Resources)
    assert isinstance(pipeline["d"].resources, Resources)
    assert isinstance(pipeline["e"].resources, Resources)
    assert isinstance(pipeline["f"].resources, Resources)

    assert pipeline["c"].resources.cpus == 2
    assert pipeline["c"].resources.memory == "1GB"
    assert pipeline["d"].resources.cpus == 1
    assert pipeline["d"].resources.memory == "2GB"
    assert pipeline["e"].resources.cpus == 3
    assert pipeline["e"].resources.memory == "3GB"
    assert pipeline["f"].resources.cpus == 4


def test_resources_variable():
    @pipefunc(output_name="c", resources_variable="resources", resources={"gpus": 8})
    def f_c(a, b, resources):
        return resources.gpus

    assert f_c(a=1, b=2) == 8

    pipeline = Pipeline([f_c])
    assert pipeline(a=1, b=2) == 8

    with pytest.raises(ValueError, match="Unexpected keyword arguments: `{'resources'}`"):
        f_c(a=1, b=2, resources={"gpus": 4})

    with pytest.raises(ValueError, match="Unused keyword arguments: `resources`"):
        pipeline(a=1, b=2, resources={"gpus": 4})


def test_resources_variable_nested_func():
    @pipefunc(output_name="c", resources_variable="resources", resources={"gpus": 8})
    def f_c(a, b, resources):
        return resources.gpus

    @pipefunc(output_name="d")
    def f_d(c):
        return c

    nf = NestedPipeFunc([f_c, f_d], output_name="d")
    assert nf.resources.gpus == 8
    assert nf(a=1, b=2) == 8

    pipeline = Pipeline([nf])
    assert pipeline(a=1, b=2) == 8


def test_resources_variable_with_callable_resources() -> None:
    @pipefunc(
        output_name="c",
        resources=lambda kwargs: Resources(gpus=kwargs["a"] + kwargs["b"]),
        resources_variable="resources",
    )
    def f_c(a, b, resources):
        return resources

    @pipefunc(output_name="d", resources=lambda kwargs: kwargs["c"])  # 'c' is the resources of f_c
    def f_d(c):
        assert isinstance(c, Resources)
        return c

    @pipefunc(
        output_name="e",
        resources=lambda kwargs: kwargs["d"],  # 'd' is the resources of f_c
        resources_variable="resources",
    )
    def f_e(d, resources):
        assert isinstance(resources, Resources)
        assert isinstance(d, Resources)
        assert resources == d
        return resources

    pipeline = Pipeline([f_c, f_d, f_e])
    r = pipeline(a=1, b=2)
    assert isinstance(r, Resources)
    assert r.gpus == 3

    with pytest.raises(
        ValueError,
        match="A `NestedPipeFunc` cannot have nested functions with callable `resources`.",
    ):
        NestedPipeFunc([f_c, f_d, f_e], output_name="e")


def test_resources_variable_in_nested_func_with_defaults() -> None:
    def resources_func(kwargs) -> Resources:
        msg = "Should not be called"
        raise ValueError(msg)

    @pipefunc("c", resources=resources_func)
    def f(a, b):
        return a + b

    @pipefunc("d", resources_variable="resources", resources=resources_func)
    def g(c, resources):
        assert isinstance(resources, Resources)
        return resources

    nf = NestedPipeFunc([f, g], output_name="d", resources={"gpus": 3})
    pipeline = Pipeline([nf], default_resources={"memory": "4GB", "gpus": 1})

    assert isinstance(pipeline["d"].resources, Resources)
    assert pipeline["d"].resources == Resources(gpus=3, memory="4GB")
    r = pipeline(a=1, b=2)
    assert isinstance(r, Resources)
    assert r.gpus == 3
    assert r.memory == "4GB"


def test_resources_func_with_variable() -> None:
    def resources_with_cpu(kwargs) -> dict[str, Any]:
        cpus = kwargs["a"] + kwargs["b"]
        return {"cpus": cpus}  # Also tests that a dict is converted to Resources

    @pipefunc(
        output_name="i",
        resources=resources_with_cpu,
        resources_variable="resources",
    )
    def j(a, b, resources):
        assert isinstance(resources, Resources)
        assert resources.cpus == a + b
        return a * b

    result = j(a=2, b=3)
    assert result == 6

    pipeline = Pipeline([j])
    result = pipeline(a=2, b=3)
    assert result == 6
    result = pipeline.map(inputs={"a": 2, "b": 3}, parallel=False, storage="dict")
    assert result["i"].output == 6


def test_with_resource_func_with_defaults() -> None:
    def resources_with_cpu(kwargs) -> Resources:
        cpus = kwargs["a"] + kwargs["b"]
        return Resources(cpus=cpus)

    @pipefunc(
        output_name="i",
        resources=resources_with_cpu,
        resources_variable="resources",
    )
    def j(a, b, resources):
        assert isinstance(resources, Resources)
        assert resources.cpus == a + b
        return resources

    pipeline = Pipeline([j], default_resources={"gpus": 5, "cpus": 1000})
    result = pipeline(a=2, b=3)
    assert result.cpus == 5
    assert result.gpus == 5
    result = pipeline.map(inputs={"a": 2, "b": 3}, parallel=False, storage="dict")
    assert result["i"].output.cpus == 5
    assert result["i"].output.gpus == 5
