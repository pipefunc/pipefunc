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

        class _DummyRunManager:
            def info(self, format: str = "text") -> str:  # noqa: ARG002, A002 - signature matches real API
                return "dummy info"

        class _DummyMultiRunManager:
            def __init__(self) -> None:
                self.run_managers = {"y": _DummyRunManager()}

        runner.multi_run_manager = _DummyMultiRunManager()
        result = runner.block(poll_interval=0.01)

    assert result["y"].output.tolist() == [1, 2, 3]
    captured = capsys.readouterr().out
    assert "----- y -----" in captured
    assert "Current time:" in captured


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
