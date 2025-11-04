from __future__ import annotations

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot


ALLOWED = {"input_is_error", "array_contains_errors"}


def test_reason_domain_for_elementwise_propagation() -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_on_two(x: int) -> int:
        if x == 2:
            raise ValueError("two is bad")
        return x * 2

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def plus_one(y: int) -> int:
        return y + 1

    p = Pipeline([fail_on_two, plus_one])
    res = p.map({"x": [1, 2, 3]}, error_handling="continue", parallel=False)

    y = res["y"].output
    z = res["z"].output

    assert isinstance(y[1], ErrorSnapshot)
    assert isinstance(z[1], PropagatedErrorSnapshot)
    assert z[1].reason in ALLOWED and z[1].reason == "input_is_error"


def test_reason_domain_for_reduction_propagation() -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_on_three(x: int) -> int:
        if x == 3:
            raise ValueError("three is bad")
        return x * 2

    @pipefunc(output_name="total")  # full-array reduction
    def sum_values(y):
        import numpy as np
        return int(np.sum(y))

    p = Pipeline([fail_on_three, sum_values])
    res = p.map({"x": [1, 2, 3]}, error_handling="continue", parallel=False)

    total = res["total"].output
    assert isinstance(total, PropagatedErrorSnapshot)
    assert total.reason in ALLOWED and total.reason == "array_contains_errors"
