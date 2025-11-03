import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pipefunc import Pipeline, pipefunc

CALLS: list[int] = []


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def _increment(x: int) -> int:
    CALLS.append(x)
    time.sleep(0.05)
    return x + 1


def _build_pipeline() -> Pipeline:
    return Pipeline([_increment])


def test_async_map_block_synchronous(capsys: pytest.CaptureFixture[str]) -> None:
    CALLS.clear()
    pipeline = _build_pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
        runner = pipeline.map_async(
            {"x": range(3)},
            executor=executor,
            start=False,
            display_widgets=False,
        )
        result = runner.result()
        second = runner.result()

    assert result["y"].output.tolist() == [1, 2, 3]
    assert second["y"].output.tolist() == [1, 2, 3]
    assert sorted(CALLS) == [0, 1, 2]
    captured = capsys.readouterr().out
    assert captured == ""


@pytest.mark.asyncio
async def test_async_map_block_raises_in_running_loop() -> None:
    CALLS.clear()
    pipeline = _build_pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
        runner = pipeline.map_async({"x": range(3)}, executor=executor, display_widgets=False)
        with pytest.raises(
            RuntimeError,
            match=r"Cannot call `result\(\)` while an event loop is running",
        ):
            runner.result()
        await runner.task


def test_async_map_start_warns_without_loop() -> None:
    pipeline = _build_pipeline()
    with ThreadPoolExecutor(max_workers=2) as executor:
        runner = pipeline.map_async(
            {"x": range(2)},
            executor=executor,
            start=False,
            display_widgets=False,
        )
        with pytest.warns(UserWarning, match=r"call `runner.result\(\)` to run synchronously"):
            returned = runner.start()
        assert returned is runner
        result = runner.result()
    assert result["y"].output.tolist() == [1, 2]


def test_async_map_result_waits_for_running_task() -> None:
    CALLS.clear()

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def slow_increment(x: int) -> int:
        CALLS.append(x)
        time.sleep(0.2)
        return x + 1

    pipeline = Pipeline([slow_increment])
    executor = ThreadPoolExecutor(max_workers=2)
    runner = pipeline.map_async(
        {"x": range(3)},
        start=False,
        display_widgets=False,
        executor=executor,
    )

    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop)
    loop_thread.start()

    def start_runner() -> None:
        runner.start()

    try:
        loop.call_soon_threadsafe(start_runner)

        # Wait for the task to be attached to the runner
        for _ in range(50):
            if runner._task is not None:
                break
            time.sleep(0.01)
        else:  # pragma: no cover - defensive guard
            pytest.fail("Async task was not started in time")

        start_time = time.time()
        result = runner.result()
        duration = time.time() - start_time

        assert duration >= 0.2
        assert result["y"].output.tolist() == [1, 2, 3]
        # Second call should be instantaneous and use the cache
        assert runner.result()["y"].output.tolist() == [1, 2, 3]
        assert CALLS == [0, 1, 2]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join()
        executor.shutdown(wait=True)
