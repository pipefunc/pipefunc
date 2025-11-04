from __future__ import annotations

from typing import Any, cast

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.helpers import chain


@pipefunc(output_name="m1")
def f1(src: int) -> int:
    return src + 1


@pipefunc(output_name="m2")
def f2(something: int, k: int = 2) -> int:
    return something * k


@pipefunc(output_name="out")
def f3(value: int) -> int:
    return value - 3


def test_chain_basic() -> None:
    # Automatically connect f1 -> f2 -> f3 by renaming the primary parameter
    chained = chain([f1, f2, f3])
    pipeline = Pipeline(cast("list[Any]", chained))

    # Root args are f1's first input (src) and the untouched extra input k for f2
    assert set(pipeline.root_args()) == {"src", "k"}

    r = pipeline.run("out", kwargs={"src": 2, "k": 3})
    assert r == (2 + 1) * 3 - 3


@pipefunc(output_name="m2b")
def f2b(m1: int, k: int = 5) -> int:  # already matches upstream output name
    return m1 * k


def test_chain_keeps_matching_name() -> None:
    chained = chain([f1, f2b, f3])
    # Ensure no conflicting rename was introduced; parameter list still contains 'm1'
    assert "m1" in chained[1].parameters
    pipeline = Pipeline(cast("list[Any]", chained))
    r = pipeline.run("out", kwargs={"src": 4, "k": 2})
    assert r == (4 + 1) * 2 - 3


@pipefunc(output_name=("a", "b"))
def f_multi(x: int) -> tuple[int, int]:
    return x, 10 * x


@pipefunc(output_name="out2")
def f_sink(y: int) -> int:
    return y


def test_chain_multi_output_default_first() -> None:
    # By default, first output ('a') is used
    chained = chain([f_multi, f_sink])
    pipeline = Pipeline(cast("list[Any]", chained))
    assert pipeline.run("out2", kwargs={"x": 7}) == 7


def test_chain_multi_output_match_by_name() -> None:
    # If downstream parameter matches an upstream output name, that output is used.
    @pipefunc(output_name="out2b")
    def f_sink_b(b: int) -> int:  # matches the second output name of f_multi
        return b

    chained = chain([f_multi, f_sink_b])
    pipeline = Pipeline(cast("list[Any]", chained))
    assert pipeline.run("out2b", kwargs={"x": 7}) == 70


@pipefunc(output_name="m2c", bound={"skip": 1})
def f2c(skip: int, real_input: int) -> int:
    # First parameter is bound; chain should raise an error
    return real_input + skip


def test_chain_rejects_bound_first_parameter() -> None:
    # When first parameter is bound and there's no explicit match, should raise
    with pytest.raises(ValueError, match="First parameter 'skip'.*is bound"):
        chain([f1, f2c])


def test_chain_bound_first_param_with_explicit_match() -> None:
    # But if there's an explicit name match, it should work
    @pipefunc(output_name="m2e", bound={"config": 1})
    def f2e(config: int, m1: int) -> int:  # m1 matches upstream output
        return m1 + config

    chained = chain([f1, f2e])
    pipeline = Pipeline(cast("list[Any]", chained))
    # Should work because 'm1' explicitly matches the upstream output name
    assert pipeline.run("m2e", kwargs={"src": 10}) == (10 + 1) + 1


def test_chain_raises_on_zero_param_middle() -> None:
    @pipefunc(output_name="const")
    def const_func() -> int:  # zero-arg function cannot accept upstream value
        return 42

    with pytest.raises(ValueError, match="has no parameters"):
        _ = chain([f1, const_func])


def test_chain_accepts_plain_callables() -> None:
    def g(z: int) -> int:
        return z * 2

    def h(t: int) -> int:
        return t + 5

    chained = chain([g, h])  # wraps callables as PipeFuncs
    assert all(isinstance(pf, PipeFunc) for pf in chained)
    pipeline = Pipeline(cast("list[Any]", chained))
    assert pipeline.run(chained[-1].output_name, kwargs={"z": 3}) == (3 * 2) + 5


def test_chain_requires_function() -> None:
    with pytest.raises(ValueError, match="requires at least one function"):
        chain([])


def test_chain_single_function_returns_copy() -> None:
    chained = chain([f1])
    assert len(chained) == 1
    pipeline = Pipeline(cast("list[Any]", chained))
    assert pipeline.run(chained[0].output_name, kwargs={"src": 3}) == 4


def test_chain_bound_match_still_connects() -> None:
    @pipefunc(output_name="m2d", bound={"m1": 10})
    def f2d(m1: int, value: int) -> int:
        return m1 + value

    # Even if the bound parameter matches the upstream output name,
    # we still reject it because the first parameter is bound
    with pytest.raises(ValueError, match="First parameter 'm1'.*is bound"):
        chain([f1, f2d])


def test_chain_all_params_bound_error() -> None:
    @pipefunc(output_name="bound", bound={"fixed": 7})
    def all_bound(fixed: int) -> int:
        return fixed

    with pytest.raises(ValueError, match="All parameters.*bound"):
        chain([f1, all_bound])
