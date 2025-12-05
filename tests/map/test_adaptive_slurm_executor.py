from __future__ import annotations

import asyncio
import functools
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

import pipefunc.map._adaptive_scheduler_slurm_executor as slurm_mod
import pipefunc.map._prepare as prepare_mod
from pipefunc import NestedPipeFunc, Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot
from pipefunc.map._result import ResultDict
from pipefunc.map._run import _RESOURCE_EVALUATION_ERROR
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
async def test_map_scope_resources_populated_with_error_inputs():
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def invert(x: int) -> float:
        if x == 0:
            raise ZeroDivisionError
        return 1 / x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": 1},  # noqa: ARG005
        resources_scope="map",
    )
    def passthrough(y: float) -> float:
        return y

    pipeline = Pipeline([invert, passthrough])
    executor = MockSlurmExecutor(cores_per_node=1)

    runner = pipeline.map_async(
        {"x": [0, 1]},
        executor=executor,
        error_handling="continue",
    )

    result = await runner.task

    assert isinstance(result, ResultDict)
    assert isinstance(result["z"].output[0], PropagatedErrorSnapshot)
    assert result["z"].output[1] == 1.0


@pytest.mark.asyncio
async def test_map_scope_all_error_inputs_skip_executor_submission() -> None:
    """Document current behaviour when every map index already has an error.

    Context (October 2025): a prior failure report showed that when upstream map
    entries produce `ErrorSnapshot` / `PropagatedErrorSnapshot` objects and the
    downstream `PipeFunc` uses map-scope resources, we still submit work to the
    executor—even though every element will immediately propagate the error.
    The minimal test below recreates that situation using the in-test
    `MockSlurmExecutor` (no actual Slurm cluster required):

    1. `always_fail` guarantees both items raise, so the downstream map receives
       only propagated errors when `error_handling="continue"`.
    2. `passthrough` declares map-scope resources, which is where the bug was
       originally reported.
    3. We patch `_submit` (shared by all executors) so we can count how many
       pieces of work are actually scheduled. In the desired behaviour this
       should remain zero; today it still increments, so we hard-fail with a
       descriptive assertion message.

    Keep this test failing until the executor submission logic short-circuits
    the all-error case.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def always_fail(x: int) -> int:  # pragma: no cover - executed via pipeline
        msg = "boom"
        raise RuntimeError(msg)

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": 1},  # noqa: ARG005
        resources_scope="map",
    )
    def passthrough(y: int) -> int:  # pragma: no cover - executed via pipeline
        return y

    pipeline = Pipeline([always_fail, passthrough])
    executor = MockSlurmExecutor(cores_per_node=1)

    from pipefunc.map import _run as map_run

    original_submit = map_run._submit
    submission_funcs: list[str] = []

    def wrapped_submit(func, *args, **kwargs):
        process_index_callable = func.keywords.get("process_index")
        if process_index_callable is not None:
            captured_func = process_index_callable.keywords.get("func")
            if captured_func is not None:
                submission_funcs.append(captured_func.__name__)
        return original_submit(func, *args, **kwargs)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            {"x": [0, 1]},
            executor=executor,
            error_handling="continue",
        )

        result = await runner.task

    assert isinstance(result, ResultDict)
    # Upstream `always_fail` must still be submitted, but downstream `passthrough`
    # should short-circuit instead of enqueuing work on the executor.
    assert "passthrough" not in submission_funcs


@pytest.mark.asyncio
async def test_map_scope_all_error_inputs_with_progress_avoids_executor() -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def always_fail(x: int) -> int:  # pragma: no cover - executed via pipeline
        msg = "boom"
        raise RuntimeError(msg)

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kw: {"cpus": 1},  # noqa: ARG005
        resources_scope="map",
    )
    def passthrough(y: int) -> int:  # pragma: no cover - executed via pipeline
        return y

    pipeline = Pipeline([always_fail, passthrough])
    executor = MockSlurmExecutor(cores_per_node=1)

    from pipefunc.map import _run as map_run

    original_submit = map_run._submit
    submission_funcs: list[str] = []

    def wrapped_submit(func, *args, **kwargs):
        process_index_callable = func.keywords.get("process_index")
        if process_index_callable is not None:
            captured_func = process_index_callable.keywords.get("func")
            if captured_func is not None:
                submission_funcs.append(captured_func.__name__)
        return original_submit(func, *args, **kwargs)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            {"x": [0, 1]},
            executor=executor,
            error_handling="continue",
            show_progress="headless",
        )

        result = await runner.task

    assert isinstance(result, ResultDict)
    assert "passthrough" not in submission_funcs


@pytest.mark.asyncio
async def test_element_scope_filters_error_indices_with_mock_slurm() -> None:
    """Element-scope resources must not submit error indices to the executor.

    This recreates a mixed-success scenario where upstream map has a failure at
    one index and the downstream element-scope function declares callable
    resources. Only valid indices should be submitted to the SLURM executor;
    error indices are processed locally and propagate as error snapshots.
    """

    from pipefunc.map import _run as map_run

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 1})
    def double_it(x: int) -> int:  # pragma: no cover - executed via pipeline
        if x == 5:
            msg = "intentional error"
            raise ValueError(msg)
        return 2 * x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kwargs: {"cpus": int(kwargs["y"] % 3) + 1},
        resources_scope="element",
    )
    def add_one(y: int) -> int:  # pragma: no cover - executed via pipeline
        return y + 1

    @pipefunc(output_name="z_sum")
    def sum_it(z):  # pragma: no cover - executed via pipeline
        return sum(z)

    pipeline = Pipeline([double_it, add_one, sum_it])
    inputs = {"x": list(range(10))}
    executor = MockSlurmExecutor(cores_per_node=2)

    # Capture submitted indices by function name
    original_submit = map_run._submit
    submission_funcs: list[tuple[str, list[int]]] = []

    def wrapped_submit(func, *args, **kwargs):
        process_index_callable = getattr(func, "keywords", {}).get("process_index")
        if process_index_callable is not None:
            captured_func = process_index_callable.keywords.get("func")
            if captured_func is not None:
                chunk = args[4] if len(args) > 4 else []
                submission_funcs.append((captured_func.__name__, list(chunk)))
        return original_submit(func, *args, **kwargs)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            inputs,
            executor=executor,
            resume=False,
            error_handling="continue",
            show_progress="headless",
        )
        result = await runner.task

    # Upstream produced an error at index 5
    assert len(result["y"].output) == 10
    assert isinstance(result["y"].output[5], ErrorSnapshot)

    # Submission behavior: downstream element-scope must exclude the error index
    double_submissions = [s for s in submission_funcs if s[0] == "double_it"]
    double_indices = [idx for _, indices in double_submissions for idx in indices]
    assert sorted(double_indices) == list(range(10))

    add_one_submissions = [s for s in submission_funcs if s[0] == "add_one"]
    add_one_indices = [idx for _, indices in add_one_submissions for idx in indices]
    expected_add_one = [i for i in range(10) if i != 5]
    assert sorted(add_one_indices) == expected_add_one

    # A downstream reduce without element mapping must not be submitted when inputs contain errors
    sum_it_submissions = [s for s in submission_funcs if s[0] == "sum_it"]
    assert not sum_it_submissions


@pytest.mark.asyncio
async def test_eager_element_scope_filters_error_indices_with_mock_slurm() -> None:
    """Eager scheduler should apply the same SLURM element-scope routing policy."""

    from pipefunc.map import _run as map_run

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 1})
    def double_it(x: int) -> int:  # pragma: no cover
        if x == 5:
            msg = "boom"
            raise ValueError(msg)
        return 2 * x

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=lambda kwargs: {"cpus": int(kwargs["y"] % 3) + 1},
        resources_scope="element",
    )
    def add_one(y: int) -> int:  # pragma: no cover
        return y + 1

    pipeline = Pipeline([double_it, add_one])
    executor = MockSlurmExecutor(cores_per_node=1)
    inputs = {"x": list(range(10))}

    original_submit = map_run._submit
    submission: list[tuple[str, list[int]]] = []

    def wrapped_submit(func, *args, **kwargs):
        process_index_callable = getattr(func, "keywords", {}).get("process_index")
        if process_index_callable is not None:
            captured_func = process_index_callable.keywords.get("func")
            if captured_func is not None:
                chunk = args[4] if len(args) > 4 else []
                submission.append((captured_func.__name__, list(chunk)))
        return original_submit(func, *args, **kwargs)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            inputs,
            executor=executor,
            error_handling="continue",
            show_progress="headless",
            scheduling_strategy="eager",
        )
        result = await runner.task

    # Downstream should exclude the error index 5
    add_chunks = [idx for name, idxs in submission if name == "add_one" for idx in idxs]
    assert sorted(add_chunks) == [i for i in range(10) if i != 5]
    # Upstream still submitted all
    y_chunks = [idx for name, idxs in submission if name == "double_it" for idx in idxs]
    assert sorted(y_chunks) == list(range(10))
    # Outputs consistent with error propagation
    assert isinstance(result["y"].output[5], ErrorSnapshot)


@pytest.mark.asyncio
async def test_slurm_chunksize_is_one_for_executor_submissions() -> None:
    """SLURM executor should submit chunks of size 1 regardless of inputs."""

    from pipefunc.map import _run as map_run

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def id_fn(x: int) -> int:  # pragma: no cover
        return x

    pipeline = Pipeline([id_fn])
    executor = MockSlurmExecutor(cores_per_node=1)
    inputs = {"x": list(range(10))}

    original_submit = map_run._submit
    chunk_lengths: list[int] = []

    def wrapped_submit(func, _ex, _status, _progress, chunksize, *args):
        # record the declared chunksize and that it equals the real chunk length
        chunk = args[0] if args else []
        chunk_lengths.append(len(chunk))
        assert chunksize == len(chunk)
        return original_submit(func, _ex, _status, _progress, chunksize, *args)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            inputs,
            executor=executor,
            show_progress="headless",
        )
        result = await runner.task

    assert len(result["y"].output) == 10
    # Every submitted chunk must be of size 1 for SLURM
    assert chunk_lengths
    assert all(n == 1 for n in chunk_lengths)


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


def test_single_resources_failure_slurm_no_retry_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource hook failures should not be retried in the Slurm helper."""

    call_count = 0

    def failing_resources(_kwargs: dict[str, Any]) -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        msg = "resource boom"
        raise RuntimeError(msg)

    @pipefunc(output_name="fail", resources=failing_resources)
    def process(x: int) -> int:
        return x * 2

    pipeline = Pipeline([process])
    executor = MockSlurmExecutor()

    # NOTE: pipefunc forbids using a Slurm executor with the synchronous `Pipeline.map`
    # API. We patch the guard here so we can exercise the single-function execution
    # path while still going through the public pipeline entry point.
    def _allow_slurm_validation(_executor_arg: Any, _in_async: bool) -> None:
        return None

    monkeypatch.setattr(slurm_mod, "validate_slurm_executor", _allow_slurm_validation)
    monkeypatch.setattr(prepare_mod, "validate_slurm_executor", _allow_slurm_validation)

    result = pipeline.map(
        {"x": 5},
        executor={"fail": executor},  # type: ignore[arg-type]
        error_handling="continue",
    )

    assert call_count == 1
    assert not executor._mock_tasks, "executor should not receive tasks when resources fail"
    output = result["fail"].output
    assert isinstance(output, PropagatedErrorSnapshot)
    assert process.error_snapshot is None
    assert _RESOURCE_EVALUATION_ERROR in output.error_info
    resource_info = output.error_info[_RESOURCE_EVALUATION_ERROR]
    assert resource_info.type == "full"
    assert isinstance(resource_info.error, ErrorSnapshot)
    error_func = resource_info.error.function
    while isinstance(error_func, functools.partial) and "resources_callable" in error_func.keywords:
        error_func = error_func.keywords["resources_callable"]
    assert error_func is failing_resources


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


@pytest.mark.asyncio
async def test_map_scope_filters_error_indices_with_mock_slurm() -> None:
    """Map-scope resources must not submit error indices to the executor.

    This tests that `resources_scope="map"` also filters out error indices
    when using a SLURM executor with error_handling="continue".
    """
    from pipefunc.map import _run as map_run

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 1})
    def double_it(x: int) -> int:
        if x == 5:
            msg = "intentional error"
            raise ValueError(msg)
        return 2 * x

    # IMPORTANT: This function uses resources_scope="map" (default)
    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources={"cpus": 1},
        resources_scope="map",
    )
    def add_one(y: int) -> int:
        return y + 1

    pipeline = Pipeline([double_it, add_one])
    inputs = {"x": list(range(10))}
    executor = MockSlurmExecutor(cores_per_node=2)

    # Capture submitted indices by function name
    original_submit = map_run._submit
    submission_funcs: list[tuple[str, list[int]]] = []

    def wrapped_submit(func, *args, **kwargs):
        process_index_callable = getattr(func, "keywords", {}).get("process_index")
        if process_index_callable is not None:
            captured_func = process_index_callable.keywords.get("func")
            if captured_func is not None:
                chunk = args[4] if len(args) > 4 else []
                submission_funcs.append((captured_func.__name__, list(chunk)))
        return original_submit(func, *args, **kwargs)

    with mock.patch("pipefunc.map._run._submit", side_effect=wrapped_submit):
        runner = pipeline.map_async(
            inputs,
            executor=executor,
            resume=False,
            error_handling="continue",
            show_progress="headless",
        )
        result = await runner.task

    # Upstream produced an error at index 5
    assert len(result["y"].output) == 10
    assert isinstance(result["y"].output[5], ErrorSnapshot)

    # Submission behavior: downstream map-scope must ALSO exclude the error index
    add_one_submissions = [s for s in submission_funcs if s[0] == "add_one"]
    add_one_indices = [idx for _, indices in add_one_submissions for idx in indices]
    expected_add_one = [i for i in range(10) if i != 5]

    # This assertion would fail before the fix
    assert sorted(add_one_indices) == expected_add_one
    assert 5 not in add_one_indices
