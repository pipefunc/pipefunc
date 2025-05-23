import re

import pytest

from pipefunc import Pipeline, pipefunc


def test_axis_is_reduced_and_noannotation():
    # Bug fixed in: https://github.com/pipefunc/pipefunc/pull/275
    @pipefunc("y", mapspec="x[i] -> y[i]")
    def f(x):
        return x

    @pipefunc("z")
    def g(y: list[int]) -> list[int]:
        return y

    Pipeline([f, g])


def test_axis_is_reduced_and_unresolvable():
    # Bug fixed in: https://github.com/pipefunc/pipefunc/pull/278
    @pipefunc("y", mapspec="x[i] -> y[i]")
    def f(x) -> "UnresolvableBecauseNotExists":  # type: ignore[name-defined]  # noqa: F821
        return x

    @pipefunc("z")
    def g(y: list[int]) -> list[int]:
        return y

    with pytest.warns(
        UserWarning,
        match=re.escape("Unresolvable type hint: `UnresolvableBecauseNotExists`"),
    ):
        Pipeline([f, g])


def test_returning_tuple_autogenerated_axis():
    # Bug fixed in: https://github.com/pipefunc/pipefunc/pull/276
    # Mapspec is autogenerated
    @pipefunc(output_name="x")
    def f() -> list[int]:
        return [1, 2, 3]

    # Uses elements of the lists in map
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def g(x: int) -> int:
        return x

    # Takes entire list
    @pipefunc(output_name="z")
    def h(x: list[int]) -> list[int]:
        return x

    Pipeline([f, g, h])
