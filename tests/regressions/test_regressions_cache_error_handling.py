from __future__ import annotations

import pytest

from pipefunc import Pipeline, pipefunc


def test_cache_does_not_mask_raise_mode_after_continue_cache():
    """Regression: cached ErrorSnapshot from `continue` must not mask `raise`.

    Scenario:
    - First run uses `error_handling="continue"` to populate the cache with an
      ErrorSnapshot.
    - Second run uses `error_handling="raise"` with identical inputs.

    Expected:
    - The second run must raise the original exception instead of returning a
      cached ErrorSnapshot. This asserts the cache key must differ across
      error-handling modes (or raise if a cached snapshot is seen under "raise").
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", cache=True)
    def may_fail(x: int) -> int:
        if x == 3:
            msg = "boom at 3"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # 1) Populate cache with an ErrorSnapshot for x == 3
    result = pipeline.map({"x": [1, 3]}, error_handling="continue", parallel=False)
    y = result["y"].output
    assert y[0] == 2
    # Ensure the continue-run produced an error object at index 1
    from pipefunc.exceptions import ErrorSnapshot

    assert isinstance(y[1], ErrorSnapshot)

    # 2) Now, with error_handling="raise", identical inputs should raise
    with pytest.raises(ValueError, match="boom at 3"):
        pipeline.map({"x": [1, 3]}, error_handling="raise", parallel=False)
