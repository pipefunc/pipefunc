from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._run_status import (
    list_historical_runs,
    list_run_statuses,
    load_outputs,
    run_info,
    status_from_run_folder,
)
from pipefunc._run_status_cli import main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pipefunc.typing import Array


@dataclass
class BlockingScalarPipeline:
    pipeline: Pipeline
    release: threading.Event


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
        time.sleep(0.01)
        return x**2

    @pipefunc(output_name="total")
    def aggregate(values: Array[float]) -> float:
        return float(sum(values))

    return Pipeline([slow_square, aggregate])


@pytest.fixture
def fast_scalar_pipeline() -> Pipeline:
    @pipefunc(output_name="result")
    def increment(x: int) -> int:
        return x + 1

    return Pipeline([increment])


@pytest.fixture
def blocking_scalar_pipeline() -> BlockingScalarPipeline:
    release = threading.Event()

    @pipefunc(output_name="result")
    def block(x: int) -> int:
        release.wait(timeout=1.0)
        return x + 1

    return BlockingScalarPipeline(Pipeline([block]), release)


@pytest.fixture
def mapspec_pipeline() -> Pipeline:
    @pipefunc(output_name="values", mapspec="x[i] -> values[i]")
    def square(x: int) -> int:
        return x**2

    return Pipeline([square])


@pytest.fixture
def multi_output_mapspec_pipeline() -> Pipeline:
    @pipefunc(output_name=("a", "b"), mapspec="x[i] -> a[i], b[i]")
    def split(x: int) -> tuple[int, int]:
        return x, 2 * x

    return Pipeline([split])


@pytest.fixture
def failing_pipeline() -> Pipeline:
    @pipefunc(output_name="values", mapspec="x[i] -> values[i]")
    def sometimes_fail(x: int) -> int:
        time.sleep(0.01)
        if x == 2:
            msg = "boom"
            raise RuntimeError(msg)
        return x

    return Pipeline([sometimes_fail])


@pytest.fixture(scope="module")
def completed_run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    @pipefunc(output_name="result")
    def add_numbers(x: float, y: float) -> float:
        return x + y

    run_folder = tmp_path_factory.mktemp("run-status-completed") / "completed"
    Pipeline([add_numbers]).map(inputs={"x": 2, "y": 5}, run_folder=run_folder, parallel=False)
    return run_folder


@pytest.fixture(scope="module")
def completed_runs_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    @pipefunc(output_name="result")
    def add_numbers(x: float, y: float) -> float:
        return x + y

    runs_folder = tmp_path_factory.mktemp("run-status-list")
    pipeline = Pipeline([add_numbers])
    pipeline.map(inputs={"x": 1, "y": 2}, run_folder=runs_folder / "run-1", parallel=False)
    pipeline.map(inputs={"x": 3, "y": 4}, run_folder=runs_folder / "run-2", parallel=False)
    return runs_folder


async def _wait_for_status(
    run_folder: Path,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
    timeout: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_status = status_from_run_folder(run_folder)
        if predicate is None:
            if last_status["status_source"] == "heartbeat":
                return last_status
        elif predicate(last_status):
            return last_status
        await asyncio.sleep(0.005)
    msg = f"Timed out waiting for run status condition, last_status={last_status!r}"
    raise AssertionError(msg)


async def _wait_for_heartbeat_status(
    run_folder: Path,
    *,
    timeout: float = 1.0,
) -> dict[str, Any]:
    return await _wait_for_status(run_folder, timeout=timeout)


def _load_run_info_json(run_folder: Path) -> dict[str, Any]:
    return json.loads((run_folder / "run_info.json").read_text())


def _write_run_info_json(run_folder: Path, payload: dict[str, Any]) -> None:
    (run_folder / "run_info.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_status_from_run_folder_completed(completed_run_folder: Path) -> None:
    status = status_from_run_folder(completed_run_folder)

    assert status["status"] == "completed"
    assert status["status_source"] == "disk_heuristic"
    assert status["all_complete"] is True
    assert status["progress_fraction"] == 1.0
    assert status["n_outputs"] == 1
    assert status["n_outputs_completed"] == 1
    assert status["outputs"]["result"]["complete"] is True
    assert status["outputs"]["result"]["progress"] == 1.0


def test_status_from_relocated_run_folder(simple_pipeline: Pipeline, tmp_path: Path) -> None:
    original_run_folder = tmp_path / "original-run"
    relocated_run_folder = tmp_path / "relocated-run"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=original_run_folder, parallel=False)

    shutil.copytree(original_run_folder, relocated_run_folder)
    shutil.rmtree(original_run_folder)

    status = status_from_run_folder(relocated_run_folder)

    assert status["status"] == "completed"
    assert status["run_folder"] == str(relocated_run_folder.absolute())
    assert status["outputs"]["result"]["bytes"] > 0


@pytest.mark.asyncio
async def test_status_from_run_folder_incomplete(slow_pipeline: Pipeline, tmp_path: Path) -> None:
    run_folder = tmp_path / "incomplete"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": [1, 2, 3, 4]},
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
        inputs={"x": list(range(8))},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        status = await _wait_for_heartbeat_status(run_folder)
        status_with_run_info = status_from_run_folder(run_folder, include_run_info=True)

        assert runner.progress is not None
        assert status["status_source"] == "heartbeat"
        assert (run_folder / "pipefunc_status.json").exists()
        assert status["outputs"]["values"]["function_name"] == "slow_square"
        assert status["outputs"]["total"]["function_name"] == "aggregate"
        assert "run_info" in status_with_run_info

        running_status = await _wait_for_status(
            run_folder,
            predicate=lambda s: s["status"] == "running" and bool(s["active_functions"]),
        )
        assert "slow_square" in running_status["active_functions"]

        await runner.task
        final_status = status_from_run_folder(run_folder)

        assert final_status["status"] == "completed"
        assert final_status["status_source"] == "heartbeat"
        assert final_status["active_functions"] == []
        assert final_status["stale"] is False
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_list_run_statuses_compact_heartbeat_view(
    slow_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "heartbeat-list"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": list(range(6))},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_heartbeat_status(run_folder)
        payload = list_run_statuses(tmp_path)

        assert payload["runs"][0]["status_source"] == "heartbeat"
        assert "outputs" not in payload["runs"][0]
    finally:
        await runner.task
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_list_run_statuses_skips_disk_scan_for_heartbeat_runs(
    blocking_scalar_pipeline: BlockingScalarPipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "heartbeat-compact"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = blocking_scalar_pipeline.pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_status(run_folder, predicate=lambda s: s["status"] == "running")

        run_info_json = _load_run_info_json(run_folder)
        run_info_json["storage"] = {}
        _write_run_info_json(run_folder, run_info_json)

        status = status_from_run_folder(run_folder, include_outputs=False)
        payload = list_run_statuses(tmp_path)

        assert status["status_source"] == "heartbeat"
        assert "outputs" not in status
        assert payload["runs"][0]["status_source"] == "heartbeat"
        assert payload["runs"][0]["run_folder"] == str(run_folder.absolute())
    finally:
        blocking_scalar_pipeline.release.set()
        await runner.task
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_heartbeat_loop_updates_during_single_running_task(
    blocking_scalar_pipeline: BlockingScalarPipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "single-heartbeat"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = blocking_scalar_pipeline.pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
        start=False,
    )
    try:
        assert runner.heartbeat is not None
        runner.heartbeat.request_write()
        assert not (run_folder / "pipefunc_status.json").exists()

        runner.heartbeat.heartbeat_interval_seconds = 0.02
        runner.start()

        first_status = await _wait_for_status(
            run_folder,
            predicate=lambda s: s["status_source"] == "heartbeat" and s["status"] == "running",
        )
        first_updated_at = first_status["updated_at"]

        second_status = await _wait_for_status(
            run_folder,
            predicate=lambda s: s["updated_at"] != first_updated_at,
        )

        assert second_status["status"] == "running"
        assert second_status["updated_at"] != first_updated_at
        blocking_scalar_pipeline.release.set()
        await runner.task
    finally:
        blocking_scalar_pipeline.release.set()
        executor.shutdown(wait=True)


def test_heartbeat_request_write_after_sync_result(
    fast_scalar_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "sync-heartbeat"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = fast_scalar_pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
        start=False,
    )
    try:
        runner.result()
        assert runner.heartbeat is not None
        runner.heartbeat.request_write()

        status = status_from_run_folder(run_folder)

        assert status["status"] == "completed"
        assert status["status_source"] == "heartbeat"
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_heartbeat_cancelled_status(
    blocking_scalar_pipeline: BlockingScalarPipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "cancelled-heartbeat"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = blocking_scalar_pipeline.pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_status(run_folder, predicate=lambda s: bool(s.get("active_functions")))

        runner.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner.task

        status = await _wait_for_status(run_folder, predicate=lambda s: s["status"] == "cancelled")
        assert status["stale"] is False
    finally:
        blocking_scalar_pipeline.release.set()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_empty_pipeline_heartbeat_progress_fraction_none(tmp_path: Path) -> None:
    run_folder = tmp_path / "empty-pipeline"
    runner = Pipeline([]).map_async(inputs={}, run_folder=run_folder, display_widgets=False)
    await runner.task

    status = status_from_run_folder(run_folder)

    assert status["status"] == "completed"
    assert status["status_source"] == "heartbeat"
    assert status["progress_fraction"] is None
    assert status["n_outputs"] == 0


@pytest.mark.asyncio
async def test_status_from_run_folder_disk_heuristic_pending_and_running(
    slow_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "disk-heuristic"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": list(range(6))},
        run_folder=run_folder,
        executor=executor,
        show_progress=False,
        start=False,
    )
    try:
        pending_status = status_from_run_folder(run_folder)

        assert pending_status["status_source"] == "disk_heuristic"
        assert pending_status["status"] == "pending"
        assert pending_status["progress_fraction"] == 0.0

        runner.start()

        running_status = await _wait_for_status(
            run_folder,
            predicate=lambda s: s["status_source"] == "disk_heuristic" and s["status"] == "running",
        )

        assert running_status["status_source"] == "disk_heuristic"
        assert 0.0 < running_status["progress_fraction"] < 1.0

        await runner.task
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_heartbeat_failed_run_reports_failed_function_counters(
    failing_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "failed-heartbeat"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = failing_pipeline.map_async(
        inputs={"x": [1, 2, 3]},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await runner.task

        status = await _wait_for_status(run_folder, predicate=lambda s: s["status"] == "failed")

        assert status["status_source"] == "heartbeat"
        assert status["functions"][0]["state"] == "running"
        assert status["functions"][0]["n_failed"] == 1
        assert status["functions"][0]["n_completed"] == 1
    finally:
        executor.shutdown(wait=True)


def test_run_status_helpers_error_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")

    missing_list = list_run_statuses(missing)
    file_list = list_run_statuses(file_path)

    assert missing_list["error"] == f"Folder '{missing}' does not exist"
    assert file_list["error"] == f"'{file_path}' is not a directory"
    assert run_info(missing)["error"]
    assert list_historical_runs(missing)["error"] == f"Folder '{missing}' does not exist"
    assert load_outputs(missing)["error"]


def test_list_runs_skips_broken_entries(
    simple_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    good_run = tmp_path / "good-run"
    broken_run = tmp_path / "broken-run"
    ignored_dir = tmp_path / "ignored-dir"
    ignored_file = tmp_path / "ignored.txt"

    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=good_run, parallel=False)
    broken_run.mkdir()
    ignored_dir.mkdir()
    ignored_file.write_text("ignore me")
    (broken_run / "run_info.json").write_text("{")

    statuses = list_run_statuses(tmp_path)
    history = list_historical_runs(tmp_path)

    assert statuses["error"] is None
    assert statuses["scanned_directories"] == 3
    assert statuses["total_count"] == 2
    assert any("error" in run for run in statuses["runs"])
    assert history["error"] is None
    assert history["total_count"] == 1
    assert history["runs"][0]["run_folder"] == str(good_run.absolute())


def test_load_outputs_subset(completed_run_folder: Path) -> None:
    payload = load_outputs(completed_run_folder, ["result"])

    assert payload == {"result": 7}


def test_load_outputs_all(completed_run_folder: Path) -> None:
    payload = load_outputs(completed_run_folder)

    assert payload == {"result": 7}


def test_load_outputs_unknown_output(completed_run_folder: Path) -> None:
    payload = load_outputs(completed_run_folder, ["does_not_exist"])

    assert payload["error"] == "Unknown output names: ['does_not_exist']"


def test_run_info_success(completed_run_folder: Path) -> None:
    payload = run_info(completed_run_folder)

    assert payload["all_complete"] is True
    assert payload["outputs"]["result"]["bytes"] > 0
    assert payload["run_info"]["run_folder"] == str(completed_run_folder.absolute())


def test_status_from_run_folder_dict_storage_unknown_progress(
    mapspec_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "dict-storage"
    mapspec_pipeline.map(
        inputs={"x": [1, 2, 3]},
        run_folder=run_folder,
        storage={"values": "dict"},
        parallel=False,
    )

    status = status_from_run_folder(run_folder)

    assert status["status"] == "incomplete"
    assert status["progress_fraction"] is None
    assert status["outputs"]["values"] == {
        "progress": "unknown",
        "complete": False,
        "bytes": 0,
    }


def test_status_from_run_folder_tuple_storage_override(
    multi_output_mapspec_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "tuple-storage"
    multi_output_mapspec_pipeline.map(
        inputs={"x": [1, 2, 3]},
        run_folder=run_folder,
        storage={"a": "file_array"},
        parallel=False,
    )

    run_info_json = _load_run_info_json(run_folder)
    status = status_from_run_folder(run_folder)

    assert "a,b" in run_info_json["storage"]
    assert status["status"] == "completed"
    assert status["outputs"]["a"]["complete"] is True
    assert status["outputs"]["b"]["complete"] is True


def test_status_from_run_folder_corrupted_run_info_shapes(
    mapspec_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "corrupt-shapes"
    mapspec_pipeline.map(inputs={"x": [1, 2, 3]}, run_folder=run_folder, parallel=False)

    run_info_json = _load_run_info_json(run_folder)
    run_info_json["resolved_shapes"]["values"] = ["?"]
    _write_run_info_json(run_folder, run_info_json)

    status = status_from_run_folder(run_folder)

    assert status["status"] == "incomplete"
    assert status["progress_fraction"] is None
    assert status["outputs"]["values"]["progress"] == "unknown"
    assert status["outputs"]["values"]["bytes"] > 0


def test_status_from_run_folder_corrupted_run_info_json_returns_error(tmp_path: Path) -> None:
    run_folder = tmp_path / "corrupt-run-info"
    run_folder.mkdir()
    (run_folder / "run_info.json").write_text("{")

    status = status_from_run_folder(run_folder)

    assert status["status"] == "error"
    assert "Expecting property name enclosed in double quotes" in status["error"]


def test_list_runs_handles_corrupted_storage_metadata(
    mapspec_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "broken-storage"
    mapspec_pipeline.map(inputs={"x": [1, 2, 3]}, run_folder=run_folder, parallel=False)

    run_info_json = _load_run_info_json(run_folder)
    run_info_json["storage"] = {}
    _write_run_info_json(run_folder, run_info_json)

    statuses = list_run_statuses(tmp_path)

    assert statuses["runs"][0]["status"] == "error"
    assert "Cannot find storage class" in statuses["runs"][0]["error"]


def test_status_from_run_folder_corrupted_storage_metadata_returns_error(
    mapspec_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "broken-storage-status"
    mapspec_pipeline.map(inputs={"x": [1, 2, 3]}, run_folder=run_folder, parallel=False)

    run_info_json = _load_run_info_json(run_folder)
    run_info_json["storage"] = {}
    _write_run_info_json(run_folder, run_info_json)

    status = status_from_run_folder(run_folder)

    assert status["status"] == "error"
    assert "Cannot find storage class" in status["error"]


@pytest.mark.asyncio
async def test_status_from_run_folder_heartbeat_reports_output_metadata_error(
    slow_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "heartbeat-broken-storage"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = slow_pipeline.map_async(
        inputs={"x": list(range(6))},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_heartbeat_status(run_folder)

        run_info_json = _load_run_info_json(run_folder)
        run_info_json["storage"] = {}
        _write_run_info_json(run_folder, run_info_json)

        status = status_from_run_folder(run_folder)

        assert status["status_source"] == "heartbeat"
        assert status["status"] in {"pending", "running", "completed"}
        assert "Cannot find storage class" in status["error"]
        assert "values" in status["outputs"]
        assert "function_name" in status["outputs"]["values"]
    finally:
        await runner.task
        executor.shutdown(wait=True)


def test_status_from_run_folder_invalid_heartbeat_falls_back_to_disk(
    simple_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "invalid-heartbeat"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    heartbeat_path = run_folder / "pipefunc_status.json"
    heartbeat_path.write_text("{invalid json")

    status = status_from_run_folder(run_folder)

    assert status["status_source"] == "disk_heuristic"
    assert status["status"] == "completed"


def test_status_from_run_folder_invalid_heartbeat_timestamp(
    simple_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "invalid-heartbeat-timestamp"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    heartbeat_path = run_folder / "pipefunc_status.json"
    heartbeat_payload = {
        "run_folder": str(run_folder.absolute()),
        "status": "running",
        "status_source": "heartbeat",
        "all_complete": False,
        "progress_fraction": 0.0,
        "n_outputs": 1,
        "n_outputs_completed": 0,
        "outputs": {"result": {"progress": 0.0, "complete": False}},
        "functions": [],
        "active_outputs": [],
        "active_functions": [],
        "updated_at": "not-a-timestamp",
        "last_modified": "not-a-timestamp",
        "heartbeat_interval_seconds": 30.0,
        "stale_after_seconds": "not-a-number",
        "pipefunc_version": "test-version",
    }
    heartbeat_path.write_text(json.dumps(heartbeat_payload))

    status = status_from_run_folder(run_folder)

    assert status["status_source"] == "heartbeat"
    assert status["stale"] is False
    assert status["outputs"]["result"]["bytes"] > 0


def test_status_from_run_folder_invalid_heartbeat_payload_returns_error(
    simple_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    run_folder = tmp_path / "invalid-heartbeat-payload"
    simple_pipeline.map(inputs={"x": 1, "y": 2}, run_folder=run_folder, parallel=False)

    heartbeat_path = run_folder / "pipefunc_status.json"
    heartbeat_path.write_text(json.dumps(["not-a-mapping"]))

    status = status_from_run_folder(run_folder)

    assert status["status"] == "error"
    assert "dictionary update sequence element" in status["error"]


def test_list_run_statuses(completed_run_folder: Path) -> None:
    payload = list_run_statuses(completed_run_folder.parent)

    assert payload["error"] is None
    assert payload["total_count"] == 1
    assert payload["runs"][0]["run_folder"] == str(completed_run_folder.absolute())
    assert payload["runs"][0]["status"] == "completed"


def test_list_run_statuses_respects_max_runs(completed_runs_folder: Path) -> None:
    payload = list_run_statuses(completed_runs_folder, max_runs=1)

    assert payload["total_count"] == 1


def test_status_cli(
    completed_run_folder: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["status", str(completed_run_folder)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "completed"
    assert payload["run_folder"] == str(completed_run_folder.absolute())


def test_status_cli_pretty_with_run_info(
    completed_run_folder: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["status", str(completed_run_folder), "--include-run-info", "--pretty"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert "run_info" in payload
    assert payload["outputs"]["result"]["bytes"] > 0


def test_list_runs_cli(
    completed_run_folder: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["list-runs", str(completed_run_folder.parent)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["total_count"] == 1
    assert payload["runs"][0]["status"] == "completed"


def test_watch_cli(
    completed_run_folder: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    code = main(["watch", str(completed_run_folder), "--interval", "0.01", "--timeout", "0.1"])
    lines = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(lines[-1])

    assert code == 0
    assert payload["status"] == "completed"


def test_watch_cli_missing_run_folder(capsys: pytest.CaptureFixture) -> None:
    code = main(["watch", "does-not-exist", "--interval", "0.01", "--timeout", "0.02"])
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])

    assert code == 1
    assert payload["status"] == "missing"


@pytest.mark.asyncio
async def test_watch_cli_timeout_for_running_job(
    blocking_scalar_pipeline: BlockingScalarPipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-watch-timeout"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = blocking_scalar_pipeline.pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_status(run_folder, predicate=lambda s: s["status"] == "running")

        code = main(["watch", str(run_folder), "--interval", "0.01", "--timeout", "0.02"])
        payload = json.loads(capsys.readouterr().out.splitlines()[-1])

        assert code == 2
        assert payload["status"] == "running"
    finally:
        blocking_scalar_pipeline.release.set()
        await runner.task
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_watch_cli_cancelled_job(
    blocking_scalar_pipeline: BlockingScalarPipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-watch-cancelled"
    executor = ThreadPoolExecutor(max_workers=1)
    runner = blocking_scalar_pipeline.pipeline.map_async(
        inputs={"x": 1},
        run_folder=run_folder,
        executor=executor,
        display_widgets=False,
    )
    try:
        await _wait_for_heartbeat_status(run_folder)
        runner.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner.task

        code = main(["watch", str(run_folder), "--interval", "0.01", "--timeout", "0.05"])
        payload = json.loads(capsys.readouterr().out.splitlines()[-1])

        assert code == 1
        assert payload["status"] == "cancelled"
    finally:
        blocking_scalar_pipeline.release.set()
        executor.shutdown(wait=True)


def test_status_cli_missing_run_folder(capsys: pytest.CaptureFixture) -> None:
    code = main(["status", "does-not-exist"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "missing"
    assert "error" in payload


def test_status_cli_corrupted_storage_returns_json_error(
    mapspec_pipeline: Pipeline,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_folder = tmp_path / "cli-broken-storage"
    mapspec_pipeline.map(inputs={"x": [1, 2, 3]}, run_folder=run_folder, parallel=False)

    run_info_json = _load_run_info_json(run_folder)
    run_info_json["storage"] = {}
    _write_run_info_json(run_folder, run_info_json)

    code = main(["status", str(run_folder)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "error"
    assert "Cannot find storage class" in payload["error"]
