"""Tests for pipefunc.PipeFunc."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest

from pipefunc import NestedPipeFunc, PipeFunc, pipefunc
from pipefunc.resources import Resources


def test_pipe_func_profile() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c", profile=True)
    assert pipe_func.profile
    assert pipe_func.profiling_stats is not None
    pipe_func.profile = False
    assert not pipe_func.profile
    assert pipe_func.profiling_stats is None


def test_pipe_func_str() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c")
    assert str(pipe_func) == "f1(...) → c"


def test_pipe_func_getstate_setstate() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c")
    state = pipe_func.__getstate__()

    # We'll validate getstate by asserting that 'func' in the state
    # is a bytes object (dumped by cloudpickle) and other attributes
    # are as expected
    assert isinstance(state["func"], bytes)
    assert state["output_name"] == "c"

    # Now we'll test setstate by creating a new instance, applying setstate and
    # verifying that the object attributes match the original
    new_pipe_func = PipeFunc.__new__(PipeFunc)
    new_pipe_func.__setstate__(state)

    assert new_pipe_func.output_name == pipe_func.output_name
    assert new_pipe_func.parameters == pipe_func.parameters
    assert new_pipe_func.func(2, 3) == pipe_func.func(
        2,
        3,
    )  # the functions behave the same


def f(a, b):
    return a + b


@dataclass
class DataClass:
    a: int


def test_pickle_pipefunc() -> None:
    func = PipeFunc(f, output_name="c")
    p = pickle.dumps(func)
    func2 = pickle.loads(p)  # noqa: S301
    assert func(1, 2) == func2(1, 2)

    func = PipeFunc(DataClass, output_name="c")  # type: ignore[arg-type]
    p = pickle.dumps(func)
    func2 = pickle.loads(p)  # noqa: S301
    assert func(a=1) == func2(a=1)


def test_update_defaults_and_renames_and_bound() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    # Test initial parameters and defaults
    assert f.parameters == ("a1", "b")
    assert f.defaults == {"a1": 42, "b": 1}

    # Update defaults
    f.update_defaults({"b": 2})
    assert f.defaults == {"a1": 42, "b": 2}

    # Call function with updated defaults
    assert f(a1=3) == 5

    # Overwrite defaults
    f.update_defaults({"a1": 1, "b": 3}, overwrite=True)
    assert f.defaults == {"a1": 1, "b": 3}
    assert f.parameters == ("a1", "b")

    # Call function with new defaults
    assert f(a1=2) == 5
    assert f() == 4
    assert f(a1=2, b=3) == 5

    # Update renames
    assert f.renames == {"a": "a1"}
    f.update_renames({"a": "a2"}, update_from="original")
    assert f.renames == {"a": "a2"}
    assert f.parameters == ("a2", "b")

    # Call function with updated renames
    assert f(a2=4) == 7
    assert f(b=0) == 1

    # Overwrite renames
    f.update_renames({"a": "a3"}, overwrite=True, update_from="original")
    assert f.parameters == ("a3", "b")

    # Call function with new renames
    assert f(a3=1) == 4

    assert f.defaults == {"a3": 1, "b": 3}  # need to reset defaults before updating bound
    f.update_defaults({}, overwrite=True)
    f.update_bound({"a3": "yolo", "b": "swag"})
    assert f(a3=88, b=1) == "yoloswag"
    assert f.bound == {"a3": "yolo", "b": "swag"}
    f.update_renames({"a": "a4"}, update_from="original")
    assert f.bound == {"a4": "yolo", "b": "swag"}
    f.update_bound({}, overwrite=True)
    assert f(a4=88, b=1) == 89

    f.update_renames({"a4": "a5"}, update_from="current")
    assert f(a5=88, b=1) == 89
    f.update_renames({"b": "b1"}, update_from="current")
    assert f.renames == {"a": "a5", "b": "b1"}

    f.update_renames({}, overwrite=True)
    assert f.parameters == ("a", "b")
    assert f.renames == {}


def test_update_renames_with_mapspec() -> None:
    @pipefunc(output_name="c", renames={"a": "a1"}, mapspec="a1[i], b[j] -> c[i, j]")
    def f(a=42, b=69):
        return a + b

    # Test initial parameters and defaults
    assert f.parameters == ("a1", "b")
    assert str(f.mapspec) == "a1[i], b[j] -> c[i, j]"

    f.update_renames({"a": "a2"}, update_from="original")
    assert f.renames == {"a": "a2"}
    assert f.parameters == ("a2", "b")
    assert str(f.mapspec) == "a2[i], b[j] -> c[i, j]"
    f.update_renames({"a": "a3"}, overwrite=True, update_from="original")
    assert f.parameters == ("a3", "b")
    assert str(f.mapspec) == "a3[i], b[j] -> c[i, j]"
    f.update_renames({"a": "a4"}, update_from="original")
    assert str(f.mapspec) == "a4[i], b[j] -> c[i, j]"
    f.update_renames({"a4": "a5"}, update_from="current")
    assert str(f.mapspec) == "a5[i], b[j] -> c[i, j]"
    f.update_renames({"b": "b1"}, update_from="current")
    assert str(f.mapspec) == "a5[i], b1[j] -> c[i, j]"
    f.update_renames({}, overwrite=True)
    assert str(f.mapspec) == "a[i], b[j] -> c[i, j]"

    # Test updating output_name
    f.update_renames({"c": "c1"}, update_from="original")
    assert str(f.mapspec) == "a[i], b[j] -> c1[i, j]"
    assert f.output_name == "c1"
    f.update_renames({"c1": "c2"}, update_from="current")
    assert str(f.mapspec) == "a[i], b[j] -> c2[i, j]"
    assert f.output_name == "c2"
    f.update_renames({"c": "c3"}, update_from="original")
    assert str(f.mapspec) == "a[i], b[j] -> c3[i, j]"
    assert f.output_name == "c3"
    f.update_renames({}, overwrite=True)
    assert str(f.mapspec) == "a[i], b[j] -> c[i, j]"
    assert f.output_name == "c"


def test_validate_update_defaults_and_renames_and_bound() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_defaults({"does_not_exist": 1})
    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_renames({"does_not_exist": "1"}, update_from="original")
    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_bound({"does_not_exist": 1})


@pytest.mark.parametrize("output_name", [("a.1", "b"), "#a", "1"])
def test_invalid_output_name_identifier(output_name):
    with pytest.raises(
        ValueError,
        match="The `output_name` should contain/be valid Python identifier",
    ):

        @pipefunc(output_name=output_name)
        def f(): ...


def test_invalid_output_name() -> None:
    with pytest.raises(
        TypeError,
        match="The `output_name` should be a string or a tuple of strings, not",
    ):

        @pipefunc(output_name=["a"])  # type: ignore[arg-type]
        def f(): ...


def test_nested_func() -> None:
    def f(a, b):
        return a + b

    def g(f):
        return f

    def h(g, x):
        return g

    nf = NestedPipeFunc([PipeFunc(f, "f"), PipeFunc(g, "g")])
    assert str(nf) == "NestedPipeFunc_f_g(...) → f, g"
    assert repr(nf) == "NestedPipeFunc(pipefuncs=[PipeFunc(f), PipeFunc(g)])"
    assert nf(a=1, b=2) == (3, 3)

    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", mapspec="a[i], b[i] -> f[i]"),
            PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
        ],
    )
    assert str(nf.mapspec) == "a[i], b[i] -> f[i], g[i]"
    nf_copy = nf.copy()
    assert str(nf.mapspec) == str(nf_copy.mapspec)

    # Test not returning all outputs by providing a output_name
    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", mapspec="a[i], b[i] -> f[i]"),
            PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
        ],
        output_name="g",
    )
    assert str(nf.mapspec) == "a[i], b[i] -> g[i]"
    assert nf(a=1, b=2) == 3

    # Check all exceptions
    with pytest.raises(ValueError, match="The provided `output_name='not_exist'` should"):
        nf = NestedPipeFunc(
            [PipeFunc(f, "f"), PipeFunc(g, "g")],
            output_name="not_exist",
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different input and output mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="... -> f[i]"),
                PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine a mix of None and MapSpec instances",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="... -> f[i]"),
                PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
                PipeFunc(h, "z", mapspec=None),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different input mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="a[i], b[j] -> f[i, j]"),
                PipeFunc(g, "g", mapspec="f[i, :] -> g[i]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different output mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="a[i], b[j] -> f[i, j]"),
                PipeFunc(g, "g", mapspec="f[i, j] -> g[j, i]"),
            ],
        )

    with pytest.raises(ValueError, match="should have at least two"):
        NestedPipeFunc([PipeFunc(f, "f")])

    with pytest.raises(
        TypeError,
        match="All elements in `pipefuncs` should be instances of `PipeFunc`.",
    ):
        NestedPipeFunc([f, PipeFunc(g, "g")])  # type: ignore[list-item]


def test_nested_func_renames_defaults_and_bound() -> None:
    def f(a, b=99):
        return a + b

    def g(f):
        return f

    # Test renaming
    nf = NestedPipeFunc([PipeFunc(f, "f"), PipeFunc(g, "g")], output_name="g")

    assert nf.renames == {}
    assert nf.defaults == {"b": 99}
    nf.update_renames({"a": "a1", "b": "b1"}, update_from="original")
    assert nf.defaults == {"b1": 99}
    assert nf.renames == {"a": "a1", "b": "b1"}
    assert nf(a1=1, b1=2) == 3
    assert nf(a1=1) == 100
    nf.update_defaults({"b1": 2, "a1": 2})
    assert nf() == 4
    assert nf.renames == {"a": "a1", "b": "b1"}
    assert nf.defaults == {"b1": 2, "a1": 2}
    # Reset defaults to update bound
    nf.update_defaults({}, overwrite=True)
    nf.update_bound({"a1": "a", "b1": "b"})
    assert nf(a1=3, b1=4) == "ab"  # will ignore the input values now


def test_nested_pipefunc_with_resources() -> None:
    def f(a, b=99):
        return a + b

    def g(f):
        return f

    # Test the resources are combined correctly
    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "cpus": 2}),
            PipeFunc(g, "g", resources={"memory": "2GB", "cpus": 1}),
        ],
        output_name="g",
    )
    assert isinstance(nf.resources, Resources)
    assert nf.resources.cpus == 2
    assert nf.resources.memory == "2GB"

    # Test that the resources specified in NestedPipeFunc are used
    nf2 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "cpus": 2}),
            PipeFunc(g, "g", resources={"memory": "2GB", "cpus": 1}),
        ],
        output_name="g",
        resources={"memory": "3GB", "cpus": 3},
    )
    assert isinstance(nf2.resources, Resources)
    assert nf2.resources.cpus == 3
    assert nf2.resources.memory == "3GB"

    # Test that the resources specified in PipeFunc are used, with the other None
    nf3 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "cpus": 2}),
            PipeFunc(g, "g", resources=None),
        ],
        output_name="g",
    )
    assert isinstance(nf3.resources, Resources)
    assert nf3.resources.cpus == 2
    assert nf3.resources.memory == "1GB"

    # Test that Resources instance in NestedPipeFunc is used
    nf3 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "cpus": 2}),
            PipeFunc(g, "g", resources=None),
        ],
        output_name="g",
        resources=Resources(cpus=3, memory="3GB"),
    )
    assert isinstance(nf3.resources, Resources)
    assert nf3.resources.cpus == 3
    assert nf3.resources.memory == "3GB"


def test_pipefunc_scope() -> None:
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f(a, b):
        return a + b

    scope = "x"
    f.update_scope(scope, "*")
    assert f(x={"a": 1, "b": 1}) == 2
    assert f(**{"x.a": 1, "x.b": 1}) == 2
    assert f(**{"x.b": 1, "x": {"a": 1}}) == 2


def test_set_pipefunc_scope_on_init() -> None:
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]", scope="x")
    def f(a, b):
        return a + b

    assert f.unscoped_parameters == ("a", "b")
    assert f.parameter_scopes == {"x"}
    assert f.renames == {"a": "x.a", "b": "x.b", "c": "x.c"}
    assert str(f.mapspec) == "x.a[i] -> x.c[i]"
    assert f(x={"a": 1, "b": 1}) == 2
    f.update_scope(None, "*", "*")
    assert f.unscoped_parameters == ("a", "b")
    assert f.parameters == ("a", "b")
    assert f(a=1, b=1) == 2


def test_incorrect_resources_variable():
    with pytest.raises(
        ValueError,
        match="The `resources_variable='missing'` should be a parameter of the function.",
    ):

        @pipefunc(output_name="c", resources_variable="missing")
        def f_c(a):
            return a


def test_delayed_resources_in_nested_func() -> None:
    @pipefunc("c")
    def f(a, b):
        return a + b

    @pipefunc("d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], resources={"gpus": 3})
    assert isinstance(nf.resources, Resources)
    assert nf.resources.gpus == 3
    with pytest.raises(TypeError, match="`NestedPipeFunc` cannot have callable `resources`."):
        NestedPipeFunc(
            [f, g],
            resources=lambda kwargs: Resources(gpus=kwargs["c"]),  # type: ignore[arg-type]
        )


def test_mapping_over_bound() -> None:
    def f(a, b):
        return a + b

    with pytest.raises(
        ValueError,
        match="The bound arguments cannot be part of the MapSpec input names",
    ):
        PipeFunc(f, output_name="out", mapspec="a[i], b[i] -> out[i]", bound={"b": [1, 2, 3]})

    pf = PipeFunc(f, output_name="out", mapspec="a[i], b[i] -> out[i]")
    with pytest.raises(
        ValueError,
        match="The bound arguments cannot be part of the MapSpec input names",
    ):
        pf.update_bound({"b": [1, 2, 3]})


def test_arg_and_output_name_identical_error():
    with pytest.raises(
        ValueError,
        match="The `output_name` cannot be the same as any of the input parameter names",
    ):
        PipeFunc(lambda x: x, output_name="x")


def test_picklable_resources() -> None:
    @pipefunc(output_name="c", resources=lambda kwargs: Resources(memory="1GB"))  # noqa: ARG005
    def f(a, b):
        return a + b

    p = pickle.dumps(f)
    del f
    f2 = pickle.loads(p)  # noqa: S301
    assert f2.resources({}).memory == "1GB"
    assert f2(a=1, b=2) == 3
