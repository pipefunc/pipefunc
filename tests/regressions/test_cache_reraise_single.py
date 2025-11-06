from __future__ import annotations

import pytest

from pipefunc import Pipeline, pipefunc


def test_cache_reraises_on_cached_error_snapshot_single() -> None:
    """If a cached ErrorSnapshot exists from continue mode, raise in raise mode.

    This exercises the cache short-circuit in the single (non-mapspec) path
    to ensure that a cached ErrorSnapshot does not mask raising semantics.
    """

    call_count = {"n": 0}

    @pipefunc(output_name="y", cache=True)
    def may_fail(x: int) -> int:
        call_count["n"] += 1
        if x == 3:
            msg = "boom at 3 (single)"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # 1) Populate cache under continue mode
    res = pipeline.map({"x": 3}, error_handling="continue", parallel=False)
    from pipefunc.exceptions import ErrorSnapshot

    assert isinstance(res["y"].output, ErrorSnapshot)
    # called exactly once so far
    assert call_count["n"] == 1

    # 2) Now with raise mode, identical input should re-raise, not return cached snapshot
    with pytest.raises(ValueError, match=r"boom at 3 \(single\)"):
        pipeline.map({"x": 3}, error_handling="raise", parallel=False)
