from __future__ import annotations

import asyncio
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import adaptive
import pytest
from adaptive_scheduler import RunManager, SlurmExecutor, SlurmTask
from adaptive_scheduler._executor import TaskID

from pipefunc import Pipeline, pipefunc
from pipefunc.map._result import ResultDict

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

has_slurm = shutil.which("srun") is not None


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
    """A mock task that wraps a ThreadPoolExecutor future."""

    def __init__(
        self,
        executor: MockSlurmExecutor,
        thread_future: asyncio.Future[Any],
        min_load_interval: float = 1.0,
    ) -> None:
        super().__init__(
            executor=executor,
            task_id=TaskID(0, 0),  # dummy task_id since we don't use it
            min_load_interval=min_load_interval,
        )
        self._thread_future = thread_future

    async def _background_check(self) -> None:
        """Check the thread future for completion."""
        while not self.done():
            self._get()
            await asyncio.sleep(0.1)  # can be shorter for tests

    def _get(self) -> Any | None:
        """Check if the thread future is done and set our result."""
        if self.done():
            return self._result

        if self._thread_future.done():
            try:
                result = self._thread_future.result()
            except Exception as e:
                self.set_exception(e)
                raise
            else:
                self.set_result(result)
                return result
        return None


@dataclass
class MockSlurmExecutor(SlurmExecutor):
    """A mock executor for testing."""

    _finalized: bool = False
    _thread_pool: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> MockSlurmTask:
        if kwargs:
            msg = "Keyword arguments are not supported"
            raise ValueError(msg)
        fut = self._thread_pool.submit(fn, *args)
        return MockSlurmTask(executor=self, thread_future=fut)  # type: ignore[arg-type]

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
async def test_adaptive_slurm_executor(
    pipeline: Pipeline,
    tmp_path: Path,
    use_mock: bool,  # noqa: FBT001
    use_instance: bool,  # noqa: FBT001
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
        show_progress=True,
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
    use_mock: bool,  # noqa: FBT001
) -> None:
    if not has_slurm and not use_mock:
        pytest.skip("Slurm not available")
    ex = SlurmExecutor(cores_per_node=1) if not use_mock else MockSlurmExecutor(cores_per_node=1)
    fut = ex.submit(lambda x: x, "echo 'Hello World'")
    ex.finalize()
    await fut


def test_slurm_executor_map_exception(
    pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Cannot use an `adaptive_scheduler.SlurmExecutor` in non-async mode, use `pipeline.map_async` instead.",
    ):
        pipeline.map({}, tmp_path, executor=MockSlurmExecutor(cores_per_node=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("use_mock", [True, False])
@pytest.mark.parametrize("use_instance", [True, False])
async def test_pipeline_no_resources(
    tmp_path: Path,
    use_mock: bool,  # noqa: FBT001
    use_instance: bool,  # noqa: FBT001
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
        show_progress=True,
    )
    result = await runner.task
    assert isinstance(result, ResultDict)
    assert len(result) == 3
    assert result["z_sum"].output == 100
    assert result["y"].output[0] == 0
    assert result["y"].output[-1] == 18
    assert result["z"].output[0] == 1
    assert result["z"].output[-1] == 19
