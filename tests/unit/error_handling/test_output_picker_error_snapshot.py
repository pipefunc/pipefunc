"""Ensure the default output_picker handles ErrorSnapshot outputs."""

from __future__ import annotations

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot


def test_default_picker_passes_through_error_snapshot() -> None:
    @pipefunc(output_name=("a", "b"))
    def will_error(x: int) -> ErrorSnapshot:
        return ErrorSnapshot(will_error, ValueError("boom"), (x,), {})

    pipeline = Pipeline([will_error])

    a, b = pipeline.run(["a", "b"], kwargs={"x": 1})

    assert isinstance(a, ErrorSnapshot)
    assert isinstance(b, ErrorSnapshot)
    assert a is b
