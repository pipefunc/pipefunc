"""Regression tests for continue-mode error handling in chunked map runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

ERROR_INPUT = 13
INPUT_VALUES = list(range(30))


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def _double_or_fail(x: int) -> int:
    """Return ``2 * x`` except raise on the designated ``ERROR_INPUT``."""

    if x == ERROR_INPUT:
        msg = f"boom at {x}"
        raise ValueError(msg)
    return x * 2


def _assert_expected_outputs(raw_outputs: Sequence) -> None:
    """Validate that the outputs align per-index and capture the real kwargs."""

    outputs = list(raw_outputs)
    assert len(outputs) == len(INPUT_VALUES)
    for idx, source_value in enumerate(INPUT_VALUES):
        entry = outputs[idx]
        if idx == ERROR_INPUT:
            assert isinstance(entry, ErrorSnapshot)
            assert entry.kwargs.get("x") == ERROR_INPUT
        else:
            assert entry == source_value * 2


def test_parallel_continue_preserves_alignment_with_chunk_errors() -> None:
    """Parallel map with chunking should keep per-index ordering and metadata."""

    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = pipeline.map(
            {"x": INPUT_VALUES},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=10,
        )

    outputs = result["y"].output
    _assert_expected_outputs(outputs)
    assert all(entry is not None for entry in outputs)


@pytest.mark.asyncio
async def test_map_async_continue_returns_per_index_outputs() -> None:
    """Async continue-mode should surface chunk failures without crashing."""

    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=4) as executor:
        async_run = pipeline.map_async(
            {"x": INPUT_VALUES},
            error_handling="continue",
            chunksizes=6,
            executor=executor,
            start=True,
        )
        result = await async_run.task

    outputs = result["y"].output
    _assert_expected_outputs(outputs)
