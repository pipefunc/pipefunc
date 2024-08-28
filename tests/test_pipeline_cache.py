"""Tests for `pipefunc.Pipeline` using cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._cache import LRUCache

if TYPE_CHECKING:
    from pathlib import Path


def test_tuple_outputs_with_cache() -> None:
    @pipefunc(
        output_name=("c", "_throw"),
        profile=True,
        debug=True,
        cache=True,
        output_picker=dict.__getitem__,
    )
    def f_c(a, b):
        return {"c": a + b, "_throw": 1}

    @pipefunc(output_name=("d", "e"), cache=True)
    def f_d(b, c, x=1):
        return b * c, 1

    @pipefunc(output_name=("g", "h"), output_picker=getattr, cache=True)
    def f_g(c, e, x=1):
        from types import SimpleNamespace

        print(f"Called f_g with c={c} and e={e}")
        return SimpleNamespace(g=c + e, h=c - e)

    @pipefunc(output_name="i", cache=True)
    def f_i(h, g):
        return h + g

    pipeline = Pipeline(
        [f_c, f_d, f_g, f_i],
        cache_type="lru",
        cache_kwargs={"shared": False},
    )
    f = pipeline.func("i")
    r = f.call_full_output(a=1, b=2, x=3)["i"]
    assert r == f(a=1, b=2, x=3)
    key = (("d", "e"), (("a", 1), ("b", 2), ("x", 3)))
    assert pipeline.cache is not None
    assert pipeline.cache.cache[key] == (6, 1)


def test_full_output_disk_cash(tmp_path: Path) -> None:
    from pipefunc import Pipeline

    @pipefunc(output_name="f1", cache=True)
    def f1(a, b):
        return a + b

    @pipefunc(output_name=("f2i", "f2j"), cache=True)
    def f2(f1):
        return 2 * f1, 1

    @pipefunc(output_name="f3", cache=True)
    def f3(a, f2i):
        return a + f2i

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    pipeline = Pipeline([f1, f2, f3], cache_type="disk", cache_kwargs={"cache_dir": cache_dir})
    pipeline("f3", a=1, b=2)
    func = pipeline.func("f3")
    assert func.call_full_output(a=1, b=2) == {
        "a": 1,
        "b": 2,
        "f1": 3,
        "f2i": 6,
        "f2j": 1,
        "f3": 7,
    }
    assert len(list(cache_dir.glob("*.pkl"))) == 3


def test_cache_regression() -> None:
    def f(a):
        return a

    pipeline = Pipeline([PipeFunc(f, output_name="c", cache=True)], cache_type="lru")
    ff = pipeline.func("c")
    assert ff(a=1) == 1
    assert ff(a=1) == 1  # should not raise an error


def test_full_output_cache() -> None:
    ran_f1 = False
    ran_f2 = False

    @pipefunc(output_name="c", cache=True)
    def f1(a, b):
        nonlocal ran_f1
        if ran_f2:
            raise RuntimeError
        ran_f1 = True
        return a + b

    @pipefunc(output_name="d", cache=True)
    def f2(b, c, x=1):
        nonlocal ran_f2
        if ran_f2:
            raise RuntimeError
        ran_f2 = True
        return b * c * x

    pipeline = Pipeline([f1, f2], cache_type="hybrid")
    f = pipeline.func("d")
    r = f.call_full_output(a=1, b=2, x=3)
    expected = {"a": 1, "b": 2, "c": 3, "d": 18, "x": 3}
    assert r == expected
    assert pipeline.cache is not None
    assert len(pipeline.cache) == 2
    r = f.call_full_output(a=1, b=2, x=3)
    assert r == expected
    r = f(a=1, b=2, x=3)
    assert r == 18


def test_simple_cache() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.raises(ValueError, match="Invalid cache type"):
        Pipeline([f], cache_type="not_exist")  # type: ignore[arg-type]
    pipeline = Pipeline([f], cache_type="simple")
    assert pipeline("c", a=1, b=2) == (1, 2)
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {("c", (("a", 1), ("b", 2))): (1, 2)}
    pipeline.cache.clear()
    assert pipeline("c", a={"a": 1}, b=[2]) == ({"a": 1}, [2])
    assert pipeline.cache.cache == {("c", (("a", (("a", 1),)), ("b", (2,)))): ({"a": 1}, [2])}
    pipeline.cache.clear()
    assert pipeline("c", a={"a"}, b=[2]) == ({"a"}, [2])
    assert pipeline.cache.cache == {("c", (("a", ("a",)), ("b", (2,)))): ({"a"}, [2])}


def test_cache_non_root_args() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def g(c, b):
        return c + b

    pipeline = Pipeline([f, g], cache_type="simple")
    # Won't populate cache because `c` is not a root argument
    assert pipeline("d", c=1, b=2) == 3
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {}


def test_sharing_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, cache=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def g(b, c):
        return b + c

    pipeline = Pipeline([f, g], cache_type="simple")
    assert pipeline("d", a=1) == 3
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {("c", (("a", 1), ("b", 1))): 2, ("d", (("a", 1), ("b", 1))): 3}
    assert pipeline.map(inputs={"a": 1})["d"].output == 3
    assert pipeline.map(inputs={"a": 1, "b": 2})["d"].output == 5


def test_autoset_cache() -> None:
    @pipefunc(output_name="y", cache=True)
    def f(a):
        return a

    pipeline = Pipeline([f])
    assert pipeline.cache is not None
    assert isinstance(pipeline.cache, LRUCache)
