from __future__ import annotations

import asyncio
import shutil
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal
from unittest import mock

import adaptive
import numpy as np
import pytest
from adaptive_scheduler import RunManager, SlurmExecutor, SlurmTask
from adaptive_scheduler._executor import TaskID

from pipefunc import NestedPipeFunc, Pipeline, pipefunc
from pipefunc.map._result import ResultDict
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from collections.abc import Callable

bash_path = Path("/bin/bash")
has_slurm = shutil.which("srun") is not None and bash_path.exists()


@pytest.fixture(autouse=True)
def patch_slurm_partitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch `slurm_partitions` to avoid `StopIteration`."""
    if has_slurm:
        return

    def slurm_partitions():
        return {"default": 1}

    monkeypatch.setattr(
        "adaptive_scheduler._server_support.slurm_run.slurm_partitions",
        slurm_partitions,
    )
    monkeypatch.setattr(
        "adaptive_scheduler._scheduler.slurm.slurm_partitions",
        slurm_partitions,
    )
    monkeypatch.setattr(
        "shutil.which",
        lambda x: "/usr/bin/squeue" if x == "squeue" else None,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=0),  # noqa: ARG005
    )


class MockScheduler:
    def queue(self, *args, **kwargs):  # noqa: ARG002
        return {}


def _dummy_run_manager() -> RunManager:
    return RunManager(
        scheduler=MockScheduler(),  # type: ignore[arg-type]
        learners=[adaptive.SequenceLearner(lambda x: x, [0])],
        fnames=["yo"],
        job_name=f"test-{uuid.uuid4().hex}",
    )


class MockSlurmTask(SlurmTask):
    """A mock task that wraps a ThreadPoolExecutor future.

    This is simpler than trying to mock all the learner file infrastructure
    that the plain SlurmTask expects.
    """

    def __init__(
        self,
        executor: MockSlurmExecutor,
        thread_future: Future[Any],
    ) -> None:
        super().__init__(
            executor=executor,
            task_id=TaskID(0, 0),  # dummy task_id since we don't use it
        )
        self._thread_future = thread_future


@dataclass
class MockSlurmExecutor(SlurmExecutor):
    """A mock executor for testing."""

    _finalized: bool = False
    _thread_pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    _mock_tasks: list[MockSlurmTask] = field(default_factory=list)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> MockSlurmTask:
        if kwargs:
            msg = "Keyword arguments are not supported"
            raise ValueError(msg)
        fut = self._thread_pool.submit(fn, *args)
        task = MockSlurmTask(executor=self, thread_future=fut)
        self._mock_tasks.append(task)
        self._all_tasks.append(task)  # Needed for parent class tracking
        return task

    async def _monitor_files(self) -> None:
        """Override the file monitoring to check thread futures instead."""
        while self._run_manager is not None:
            await asyncio.sleep(0.01)  # Check more frequently for tests

            if self._run_manager.task is not None and self._run_manager.task.cancelled():
                break

            # Check all mock tasks
            for task in self._mock_tasks:
                if not task.done() and task._thread_future.done():
                    try:
                        result = task._thread_future.result()
                        task.set_result(result)
                    except Exception as e:  # noqa: BLE001
                        task.set_exception(e)

    def finalize(
        self,
        *,
        start: bool = True,  # noqa: ARG002
    ) -> RunManager:  # type: ignore[return]
        if self._finalized:
            msg = "Already finalized"
            raise RuntimeError(msg)
        self._finalized = True
        self._run_manager = _dummy_run_manager()
        self._file_monitor_task = asyncio.create_task(self._monitor_files())
        return self._run_manager

    def new(
        self,
        update: dict[str, Any] | None = None,
    ) -> MockSlurmExecutor:
        return MockSlurmExecutor(**(update or {}))


@pytest.fixture
def pipeline() -> Pipeline:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 1})
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": kw["y"] + 1},
        resources_scope="element",
    )
    def add_one(y):
        return y + 1

    @pipefunc(output_name="z_sum")
    def sum_it(z):
        return sum(z)

    return Pipeline([double_it, add_one, sum_it])


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mock", [True, False])
@pytest.mark.parametrize("use_instance", [True, False])
@pytest.mark.parametrize("scheduling_strategy", ["eager", "generation"])
async def test_adaptive_slurm_executor(
    pipeline: Pipeline,
    tmp_path: Path,
    use_mock: bool,
    use_instance: bool,
    scheduling_strategy: Literal["eager", "generation"],
) -> None:
    if not has_slurm and not use_mock:
        pytest.skip("Slurm not available")
    inputs = {"x": range(10)}
    run_folder = tmp_path / "my_run_folder"
    if use_mock:
        ex = MockSlurmExecutor() if use_instance else MockSlurmExecutor
    else:
        ex = SlurmExecutor(cores_per_node=1) if use_instance else SlurmExecutor
    runner = pipeline.map_async(
        inputs,
        run_folder,
        executor=ex,  # type: ignore[arg-type]
        show_progress="ipywidgets",
        scheduling_strategy=scheduling_strategy,
    )
    result = await runner.task
    assert isinstance(result, ResultDict)
    assert len(result) == 3
    assert result["z_sum"].output == 100
    assert result["y"].output[0] == 0
    assert result["y"].output[-1] == 18
    assert result["z"].output[0] == 1
    assert result["z"].output[-1] == 19


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mock", [True, False])
async def test_adaptive_mock_slurm_executor(
    use_mock: bool,
) -> None:
    if not has_slurm and not use_mock:
        pytest.skip("Slurm not available")
    ex = SlurmExecutor(cores_per_node=1) if not use_mock else MockSlurmExecutor(cores_per_node=1)
    fut = ex.submit(lambda x: x, "echo 'Hello World'")
    rm = ex.finalize()
    if not use_mock:
        assert rm.scheduler.executor_type is not None
    await fut


def test_slurm_executor_map_exception(
    pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Cannot use an `adaptive_scheduler.SlurmExecutor` in non-async mode, use `pipeline.map_async` instead.",
    ):
        pipeline.map(
            {},
            tmp_path,
            executor=MockSlurmExecutor(cores_per_node=1),
            parallel=True,
            storage="dict",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mock", [True, False])
@pytest.mark.parametrize("use_instance", [True, False])
async def test_pipeline_no_resources(
    tmp_path: Path,
    use_mock: bool,
    use_instance: bool,
) -> None:
    if not has_slurm and not use_mock:
        pytest.skip("Slurm not available")

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
    )
    def add_one(y):
        return y + 1

    @pipefunc(output_name="z_sum")
    def sum_it(z):
        return sum(z)

    pipeline = Pipeline([double_it, add_one, sum_it])
    inputs = {"x": range(10)}
    if use_mock:
        ex = MockSlurmExecutor() if use_instance else MockSlurmExecutor
    else:
        ex = SlurmExecutor(cores_per_node=1) if use_instance else SlurmExecutor
    run_folder = tmp_path / "my_run_folder"
    runner = pipeline.map_async(
        inputs,
        run_folder,
        executor=ex,  # type: ignore[arg-type]
        show_progress="ipywidgets",
    )
    result = await runner.task
    assert isinstance(result, ResultDict)
    assert len(result) == 3
    assert result["z_sum"].output == 100
    assert result["y"].output[0] == 0
    assert result["y"].output[-1] == 18
    assert result["z"].output[0] == 1
    assert result["z"].output[-1] == 19


@pytest.mark.parametrize("resources", [lambda kw: {"cpus": 2}, {"cpus": 2}])  # noqa: ARG005
@pytest.mark.parametrize("resources_scope", ["element", "map"])
@pytest.mark.asyncio
async def test_number_of_jobs_created_with_resources(resources, resources_scope):
    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=resources,
        resources_scope=resources_scope,
    )
    def add_one(y):
        return y + 1

    pipeline = Pipeline([add_one])
    # Create an instance of our mock executor so we can inspect calls
    executor_instance = MockSlurmExecutor(cores_per_node=1)
    with mock.patch(
        "pipefunc.map._adaptive_scheduler_slurm_executor._new_slurm_executor",
        autospec=True,
    ) as mock_new_executor:
        # When _new_slurm_executor is called, return our executor instance.
        mock_new_executor.return_value = executor_instance
        runner = pipeline.map_async({"y": range(10)}, executor=executor_instance)
        await runner.task
        # Inspect all calls to _new_slurm_executor
        calls = mock_new_executor.call_args_list
        # For each call, check the keyword arguments:
        for call in calls:
            kw = call.kwargs
            # We expect the key "cores_per_node" to be present
            assert "cores_per_node" in kw
            value = kw["cores_per_node"]
            if resources_scope == "element":
                # In element scope, the resource dict should be replicated per element.
                # So cores_per_node should be a tuple with length 10.
                assert isinstance(value, tuple)
                assert len(value) == 10
            else:  # resources_scope == "map"
                # In map scope, only one set of resources is used, so it should be an int.
                assert isinstance(value, int)
                assert value == 2

    # Now rerun the test with without mocking to see if it works
    # This will actually submit jobs to the mock scheduler.
    runner = pipeline.map_async({"y": range(10)}, executor=executor_instance)
    result = await runner.task
    assert isinstance(result, ResultDict)
    assert len(result["z"].output) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("resources_scope", ["element", "map"])
@pytest.mark.parametrize("use_mock", [True, False])
async def test_with_nested_pipefunc(
    tmp_path: Path,
    resources_scope: Literal["element", "map"],
    use_mock: bool,
):
    if not has_slurm and not use_mock:
        pytest.skip("Slurm not available")

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def add_one(y):
        return y + 1

    nested_pipefunc = NestedPipeFunc(
        [double_it, add_one],
        resources={"cpus": 1},
        resources_scope=resources_scope,
    )

    pipeline = Pipeline([nested_pipefunc])
    runner = pipeline.map_async(
        {"x": range(10)},
        tmp_path,
        executor=MockSlurmExecutor() if use_mock else SlurmExecutor(),
    )
    result = await runner.task
    assert isinstance(result, ResultDict)
    assert len(result["z"].output) == 10


@pytest.mark.asyncio
async def test_inputs_serialized_to_disk(pipeline: Pipeline, tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]", resources_scope="element")
    def double_it(x: int, b: np.ndarray) -> int:
        return 2 * x + b.sum()

    x = np.random.random((10, 10))  # noqa: NPY002
    b = np.random.random((10, 10))  # noqa: NPY002
    assert x.nbytes > 100
    assert b.nbytes > 100
    pipeline = Pipeline([double_it])
    with (
        mock.patch("pipefunc.map._run_info.is_slurm_executor", return_value=True),
        mock.patch("pipefunc.map._run_info._MAX_SIZE_BYTES_INPUT", new=100),
    ):
        with pytest.warns(UserWarning, match="dumping to disk instead of serializing"):
            # Set executor to anything not None to trigger the right branch
            runner = pipeline.map_async({"x": x, "b": b}, tmp_path, executor=ThreadPoolExecutor())
        result = await runner.task
        assert result["y"].output.shape == x.shape
        assert isinstance(runner.run_info.inputs["x"], FileArray)


@pytest.mark.asyncio
async def test_setting_executor_type_in_resources(pipeline: Pipeline, tmp_path: Path) -> None:
    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources={"cpus_per_node": 2, "nodes": 2, "extra_args": {"executor_type": "ipyparallel"}},
    )
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    pipeline = Pipeline([double_it])
    executor_instance = MockSlurmExecutor(cores_per_node=1)
    with mock.patch(
        "pipefunc.map._adaptive_scheduler_slurm_executor._new_slurm_executor",
        autospec=True,
    ) as mock_new_executor:
        # When _new_slurm_executor is called, return our executor instance.
        mock_new_executor.return_value = executor_instance
        runner = pipeline.map_async(
            {"x": range(10)},
            tmp_path,
            executor=executor_instance,
        )
        await runner.task
        # Check that _new_slurm_executor was called with the expected arguments
        mock_new_executor.assert_called_once()
        call_args = mock_new_executor.call_args

        # Check that the first argument is the executor instance
        assert call_args[0][0] == executor_instance

        # Check the keyword arguments we care about
        assert call_args.kwargs["executor_type"] == "ipyparallel"
        assert call_args.kwargs["cores_per_node"] == 2
        assert call_args.kwargs["nodes"] == 2


@pytest.mark.asyncio
async def test_slurm_executor_simple(
    pipeline: Pipeline,
    tmp_path: Path,
):
    if not has_slurm:
        pytest.skip("Slurm not available")
    runner = pipeline.map_async(
        {"x": range(10)},
        tmp_path,
        executor=SlurmExecutor(),
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(runner.task, timeout=0.2)

    # Give the event loop a moment to process the cancellation
    await asyncio.sleep(0)
    assert runner.task.cancelled()
