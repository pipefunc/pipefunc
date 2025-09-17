from __future__ import annotations

import asyncio

import pytest

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="value")
async def _async_identity(x: int) -> int:
    await asyncio.sleep(0)
    return x


@pipefunc(output_name="double")
def _double(value: int) -> int:
    return value * 2


def test_pipefunc_async_call_guard() -> None:
    with pytest.raises(RuntimeError):
        _async_identity(1)


def test_pipefunc_call_async_executes() -> None:
    result = asyncio.run(_async_identity.__call_async__(1))
    assert result == 1


def test_pipeline_run_async_mixed_nodes() -> None:
    pipeline = Pipeline([_async_identity, _double])
    with pytest.raises(RuntimeError):
        pipeline.run("double", kwargs={"x": 2})

    result = asyncio.run(pipeline.run_async("double", kwargs={"x": 2}))
    assert result == 4


@pipefunc(output_name="out", mapspec="items[i] -> out[i]")
async def _async_increment(items: int) -> int:
    await asyncio.sleep(0)
    return items + 1


def test_map_async_with_async_node_returns_results() -> None:
    pipeline = Pipeline([_async_increment])

    async def _run_map() -> list[int]:
        async_map = pipeline.map_async(
            {"items": [0, 1, 2]},
            display_widgets=False,
        )
        results = await async_map.task
        return list(results["out"].output)

    outputs = asyncio.run(_run_map())
    assert outputs == [1, 2, 3]
