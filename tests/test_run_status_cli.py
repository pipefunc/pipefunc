from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._run_status import list_run_statuses, status_from_run_folder
from pipefunc._run_status_cli import main

if TYPE_CHECKING:
    from pathlib import Path

    from pipefunc.typing import Array


@pytest.fixture
def simple_pipeline() -> Pipeline:
    @pipefunc(output_name="result")
    def add_numbers(x: float, y: float) -> float:
        return x + y

    return Pipeline([add_numbers])


@pytest.fixture
def slow_pipeline() -> Pipeline:
    @pipefunc(output_name="values", mapspec="x[i] -> values[i]")
    def slow_square(x: float) -> float:
        time.sleep(0.02)
        return x**2

    @pipefunc(output_name="total")
    def aggregate(values: Array[float]) -> float:
        return float(sum(values))

    return Pipeline([slow_square, aggregate])


async def _wait_for_heartbeat_status(
    run_folder: Path,
    *,
    timeout: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = status_from_run_folder(run_folder)
        if status["status_source"] == "heartbeat":
            return status
        await asyncio.sleep(0.01)
    msg = "Timed out waiting for heartbeat-backed status"
    raise AssertionError(msg)


def test_status_from_run_folder_completed(simple_pipeline: Pipeline, tmp_path: Path) -> None:
    run_folder = tmp_path / "completed"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    status = status_from_run_folder(run_folder)

    assert status["status"] == "completed"
    assert status["status_source"] == "disk_heuristic"
    assert status["all_complete"] is True
    assert status["progress_fraction"] == 1.0
    assert status["n_outputs"] == 1
    assert status["n_outputs_completed"] == 1
    assert status["outputs"]["result"]["complete"] is True
    assert status["outputs"]["result"]["progress"] == 1.0


@pytest.mark.asyncio
async def test_status_from_run_folder_incomplete(slow_pipeline: Pipeline, tmp_path: Path) -> None:
    run_folder = tmp_path / "incomplete"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": [1, 2, 3, 4, 5]},
        run_folder=run_folder,
        executor=executor,
        show_progress="headless",
    )
    try:
        status = status_from_run_folder(run_folder)
        assert status["status"] in {"pending", "running", "incomplete"}
        assert status["all_complete"] is False
        assert status["n_outputs"] == 2
        assert "values" in status["outputs"]
        assert "total" in status["outputs"]
        await runner.task
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_status_from_run_folder_uses_heartbeat_by_default(
    slow_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "heartbeat"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": list(range(40))},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        status = await _wait_for_heartbeat_status(run_folder)

        assert runner.progress is not None
        assert status["status_source"] == "heartbeat"
        assert (run_folder / "pipefunc_status.json").exists()
        assert status["outputs"]["values"]["function_name"] == "slow_square"
        assert status["outputs"]["total"]["function_name"] == "aggregate"

        saw_active_functions = False
        for _ in range(100):
            status = status_from_run_folder(run_folder)
            if status["status"] == "running" and status["active_functions"]:
                saw_active_functions = True
                break
            await asyncio.sleep(0.01)

        assert saw_active_functions is True
        assert "slow_square" in status["active_functions"]

        await runner.task
        final_status = status_from_run_folder(run_folder)

        assert final_status["status"] == "completed"
        assert final_status["status_source"] == "heartbeat"
        assert final_status["active_functions"] == []
        assert final_status["stale"] is False
    finally:
        executor.shutdown(wait=True)


def test_list_run_statuses(simple_pipeline: Pipeline, tmp_path: Path) -> None:
    run_folder = tmp_path / "run-1"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    payload = list_run_statuses(tmp_path)

    assert payload["error"] is None
    assert payload["total_count"] == 1
    assert payload["runs"][0]["run_folder"] == str(run_folder.absolute())
    assert payload["runs"][0]["status"] == "completed"


def test_status_cli(
    simple_pipeline: Pipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-status"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    code = main(["status", str(run_folder)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "completed"
    assert payload["run_folder"] == str(run_folder.absolute())


def test_list_runs_cli(
    simple_pipeline: Pipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-list"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    code = main(["list-runs", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["total_count"] == 1
    assert payload["runs"][0]["status"] == "completed"


def test_watch_cli(
    simple_pipeline: Pipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-watch"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    code = main(["watch", str(run_folder), "--interval", "0.01", "--timeout", "0.1"])
    lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(lines[-1])

    assert code == 0
    assert payload["status"] == "completed"


def test_status_cli_missing_run_folder(capsys: pytest.CaptureFixture) -> None:
    code = main(["status", "does-not-exist"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "missing"
    assert "error" in payload
