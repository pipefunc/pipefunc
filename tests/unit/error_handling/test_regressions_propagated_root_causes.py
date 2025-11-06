from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot

if TYPE_CHECKING:
    import numpy as np


@pytest.mark.xfail(reason="Reduction get_root_causes only returns roots for full-error inputs")
def test_get_root_causes_in_reduction_returns_upstream_errors():
    """Regression: root causes should be discoverable for reduction errors.

    We create an element-wise stage that fails for one element, followed by a
    full-array reduction. The reduction should produce a PropagatedErrorSnapshot
    with reason "array_contains_errors". Calling `.get_root_causes()` on this
    snapshot should return the upstream ErrorSnapshot(s) that triggered the
    reduction to skip.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 3:
            msg = "cannot process 3"
            raise ValueError(msg)
        return x * 2

    @pipefunc(output_name="total")  # reduction over full array
    def sum_all(y: np.ndarray) -> int:  # accepts array
        # This should be skipped if y contains any errors
        return int(sum(y))  # pragma: no cover (should not execute)

    pipeline = Pipeline([may_fail, sum_all])

    result = pipeline.map({"x": [1, 2, 3, 4]}, error_handling="continue", parallel=False)

    # y has an element error at index 2
    y = result["y"].output
    assert isinstance(y[2], ErrorSnapshot)

    # total is a propagated error from the reduction over an array containing errors
    total = result["total"].output
    assert isinstance(total, PropagatedErrorSnapshot)
    assert total.reason == "array_contains_errors"

    # Expected: get_root_causes should include the upstream ErrorSnapshot(s)
    roots = total.get_root_causes()
    # At least one upstream error with the original message
    assert any(
        isinstance(r, ErrorSnapshot) and "cannot process 3" in str(r.exception) for r in roots
    )
