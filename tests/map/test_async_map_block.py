import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def _increment(x: int) -> int:
    time.sleep(0.05)
    return x + 1


def _build_pipeline() -> Pipeline:
    return Pipeline([_increment])


def test_async_map_block_synchronous(capsys: pytest.CaptureFixture[str]) -> None:
    pipeline = _build_pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
        runner = pipeline.map_async(
            {"x": range(3)},
            executor=executor,
            start=False,
            display_widgets=False,
        )
        result = runner.block()

    assert result["y"].output.tolist() == [1, 2, 3]
    captured = capsys.readouterr().out
    assert captured == ""


@pytest.mark.asyncio
async def test_async_map_block_raises_in_running_loop() -> None:
    pipeline = _build_pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
        runner = pipeline.map_async({"x": range(3)}, executor=executor, display_widgets=False)
        with pytest.raises(
            RuntimeError,
            match=r"Cannot call `block\(\)` while an event loop is running",
        ):
            runner.block()
        await runner.task
