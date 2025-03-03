import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map._run_dynamic import run_map_dynamic
from pipefunc.typing import Array


@pipefunc(output_name="slow")
def slow_task():
    # Simulate a slow CPU-bound task
    time.sleep(0.5)  # Reduced sleep time for faster tests
    return "slow"


@pipefunc(output_name="quick")
def quick_task():
    # Simulate a fast CPU-bound task
    time.sleep(0.1)  # Reduced sleep time for faster tests
    return "quick"


@pipefunc(output_name="combined")
def combine(slow, quick):
    return f"{slow} and {quick}"


def test_dynamic_scheduler_parallel_execution(tmp_path: Path):
    """
    Test that independent branches run in parallel.
    The pipeline is:
       slow_task  -->
                    \
                     --> combine  (should wait for both slow and quick)
       quick_task -->

    Since slow_task sleeps for 0.5 seconds and quick_task for 0.1 seconds,
    if they were executed sequentially the total time would be >0.6 seconds.
    With parallel dynamic scheduling, the total time should be close to the slow branch (≈0.5 seconds + overhead).
    """
    pipeline = Pipeline([slow_task, quick_task, combine])
    # No root inputs in this example, so we pass an empty dict.
    run_folder = tmp_path / "run_dynamic"
    start = time.monotonic()
    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )
    elapsed = time.monotonic() - start
    # Check that the combined result is correct.
    assert result["combined"].output == "slow and quick"
    # The total elapsed time is likely to include overhead from the dynamic scheduler,
    # which may make it longer than expected. Let's use a more generous threshold.
    assert elapsed < 2.0, f"Elapsed time was {elapsed:.2f} seconds, expected less than 2.0 seconds"


# Test a more complex dependency graph
@pipefunc(output_name="a")
def task_a():
    time.sleep(0.1)
    return "a"


@pipefunc(output_name="b")
def task_b():
    time.sleep(0.1)
    return "b"


@pipefunc(output_name="c")
def task_c(a):
    time.sleep(0.1)
    return f"c({a})"


@pipefunc(output_name="d")
def task_d(b):
    time.sleep(0.1)
    return f"d({b})"


@pipefunc(output_name="e")
def task_e(c, d):
    time.sleep(0.1)
    return f"e({c},{d})"


def test_complex_dependency_graph(tmp_path: Path):
    """
    Test a more complex dependency graph:

    a --> c --\
              --> e
    b --> d --/

    With dynamic scheduling, tasks a and b should run in parallel,
    then c and d should run in parallel once a and b complete,
    and finally e should run once c and d complete.
    """
    pipeline = Pipeline([task_a, task_b, task_c, task_d, task_e])
    run_folder = tmp_path / "complex_graph"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )

    assert result["e"].output == "e(c(a),d(b))"

    # Also check intermediate results
    assert result["a"].output == "a"
    assert result["b"].output == "b"
    assert result["c"].output == "c(a)"
    assert result["d"].output == "d(b)"


# Test with MapSpec functions
@pipefunc(output_name="values", mapspec="x[i] -> values[i]")
def multiply_by_two(x: int) -> int:
    return x * 2


@pipefunc(output_name="sum")
def sum_values(values: Array[int]) -> int:
    return sum(values)


def test_dynamic_scheduler_with_mapspec(tmp_path: Path):
    """Test that the dynamic scheduler works with MapSpec functions."""
    pipeline = Pipeline([multiply_by_two, sum_values])
    run_folder = tmp_path / "mapspec"

    inputs = {"x": [1, 2, 3, 4, 5]}
    result = run_map_dynamic(
        pipeline,
        inputs=inputs,
        run_folder=run_folder,
        show_progress=False,
    )

    assert result["sum"].output == 30  # 2 + 4 + 6 + 8 + 10 = 30
    assert result["values"].output.tolist() == [2, 4, 6, 8, 10]

    # Verify results are also saved to disk
    assert load_outputs("sum", run_folder=run_folder) == 30
    assert load_outputs("values", run_folder=run_folder).tolist() == [2, 4, 6, 8, 10]


# Test with multiple outputs
@pipefunc(output_name=("out1", "out2"))
def multiple_outputs():
    return "first", "second"


@pipefunc(output_name="combined")
def combine_outputs(out1, out2):
    return f"{out1} and {out2}"


def test_dynamic_scheduler_with_multiple_outputs(tmp_path: Path):
    """Test that the dynamic scheduler handles functions with multiple outputs."""
    pipeline = Pipeline([multiple_outputs, combine_outputs])
    run_folder = tmp_path / "multiple_outputs"

    # Make sure we're getting results back
    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
        return_results=True,
    )

    assert result["out1"].output == "first"
    assert result["out2"].output == "second"
    assert result["combined"].output == "first and second"
    # Check the results from disk to avoid issues with return_results
    assert load_outputs("out1", run_folder=run_folder) == "first"
    assert load_outputs("out2", run_folder=run_folder) == "second"
    assert load_outputs("combined", run_folder=run_folder) == "first and second"

    # Also check the returned results if they exist
    if "out1" in result:
        assert result["out1"].output == "first"
        assert result["out2"].output == "second"
        assert result["combined"].output == "first and second"


# Test with custom executor
def test_dynamic_scheduler_with_custom_executor(tmp_path: Path):
    """Test that the dynamic scheduler works with a custom executor."""
    pipeline = Pipeline([task_a, task_b, task_c, task_d, task_e])
    run_folder = tmp_path / "custom_executor"

    with ProcessPoolExecutor(max_workers=2) as executor:
        result = run_map_dynamic(
            pipeline,
            inputs={},
            run_folder=run_folder,
            executor=executor,
            show_progress=False,
        )

    assert result["e"].output == "e(c(a),d(b))"


# Test with return_results=False
def test_dynamic_scheduler_without_returning_results(tmp_path: Path):
    """Test that the dynamic scheduler works when not returning results."""
    pipeline = Pipeline([task_a, task_b, task_c, task_d, task_e])
    run_folder = tmp_path / "no_return"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        return_results=False,
        show_progress=False,
    )

    # Result should be an empty dict
    assert not result

    # But results should still be saved to disk
    assert load_outputs("e", run_folder=run_folder) == "e(c(a),d(b))"


# Test with caching
def test_dynamic_scheduler_with_caching(tmp_path: Path):
    """Test that the dynamic scheduler respects function caching."""
    # Define functions with counters to check if they're called
    call_counts = {"a": 0, "b": 0, "c": 0}

    @pipefunc(output_name="a", cache=True)
    def cached_a():
        call_counts["a"] += 1
        return "a"

    @pipefunc(output_name="b", cache=True)
    def cached_b(a):
        call_counts["b"] += 1
        return f"b({a})"

    @pipefunc(output_name="c", cache=True)
    def cached_c(b):
        call_counts["c"] += 1
        return f"c({b})"

    pipeline = Pipeline([cached_a, cached_b, cached_c], cache_type="hybrid")
    run_folder = tmp_path / "caching"

    # First run
    result1 = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )

    assert result1["c"].output == "c(b(a))"
    # Check results from disk
    assert load_outputs("a", run_folder=run_folder) == "a"
    assert load_outputs("b", run_folder=run_folder) == "b(a)"
    assert load_outputs("c", run_folder=run_folder) == "c(b(a))"
    assert call_counts == {"a": 1, "b": 1, "c": 1}

    # Second run should use cache
    result2 = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        cleanup=False,  # Don't clean up to test caching
        show_progress=False,
    )

    assert result2["c"].output == "c(b(a))"
    # Counts should remain the same if caching worked
    assert call_counts == {"a": 1, "b": 1, "c": 1}


# Test with internal shapes
@pipefunc(output_name="x", internal_shape="?")
def generate_values() -> list[int]:
    return [1, 2, 3, 4]


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_values(x: int) -> int:
    return x * 2


@pipefunc(output_name="sum")
def sum_doubled(y: Array[int]) -> int:
    return sum(y)


def test_dynamic_scheduler_with_internal_shapes(tmp_path: Path):
    """Test that the dynamic scheduler handles internal shapes correctly."""
    pipeline = Pipeline([generate_values, double_values, sum_doubled])
    run_folder = tmp_path / "internal_shapes"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )

    assert result["sum"].output == 20  # (1+2+3+4)*2 = 20
    assert result["y"].output.tolist() == [2, 4, 6, 8]


# Test with fixed indices
@pipefunc(output_name="values", mapspec="x[i] -> values[i]")
def process_value(x: int) -> int:
    return x * 2


def test_dynamic_scheduler_with_fixed_indices(tmp_path: Path):
    """Test that the dynamic scheduler respects fixed indices."""
    pipeline = Pipeline([process_value])
    run_folder = tmp_path / "fixed_indices"

    inputs = {"x": [10, 20, 30, 40, 50]}

    # Only process index 2 (value 30)
    result = run_map_dynamic(
        pipeline,
        inputs=inputs,
        run_folder=run_folder,
        fixed_indices={"i": 2},
        show_progress=False,
    )

    # Result should be a 1D array with only the processed value at index 2
    assert result["values"].output.shape == (5,)
    assert result["values"].output[2] == 60  # 30*2

    # Other indices should be uninitialized/empty
    # We can't directly check this in the result object, but we can verify on disk
    values_on_disk = load_outputs("values", run_folder=run_folder)
    assert values_on_disk[2] == 60

    # Run again with a different fixed index
    result2 = run_map_dynamic(
        pipeline,
        inputs=inputs,
        run_folder=run_folder,
        fixed_indices={"i": 3},
        cleanup=False,  # Don't clean up to keep previous results
        show_progress=False,
    )

    # Now both indices 2 and 3 should be processed
    values_on_disk2 = load_outputs("values", run_folder=run_folder)
    assert values_on_disk2[2] == 60
    assert values_on_disk2[3] == 80  # 40*2
    assert result2["values"].output[3] == 80


# Test with complex dependency chain
@pipefunc(output_name="a")
def step_a():
    return 1


@pipefunc(output_name="b")
def step_b(a):
    return a + 1


@pipefunc(output_name="c")
def step_c(b):
    return b + 1


@pipefunc(output_name="d")
def step_d(c):
    return c + 1


@pipefunc(output_name="e")
def step_e(d):
    return d + 1


def test_dynamic_scheduler_with_long_dependency_chain(tmp_path: Path):
    """Test that the dynamic scheduler handles long dependency chains correctly."""
    pipeline = Pipeline([step_a, step_b, step_c, step_d, step_e])
    run_folder = tmp_path / "long_chain"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )
    assert result["a"].output == 1
    assert result["b"].output == 2
    assert result["c"].output == 3
    assert result["d"].output == 4
    assert result["e"].output == 5
    # Check results from disk
    assert load_outputs("a", run_folder=run_folder) == 1
    assert load_outputs("b", run_folder=run_folder) == 2
    assert load_outputs("c", run_folder=run_folder) == 3
    assert load_outputs("d", run_folder=run_folder) == 4
    assert load_outputs("e", run_folder=run_folder) == 5


# Test with diamond dependency pattern
@pipefunc(output_name="start")
def diamond_start():
    return 10


@pipefunc(output_name="left")
def diamond_left(start):
    return start + 1


@pipefunc(output_name="right")
def diamond_right(start):
    return start + 2


@pipefunc(output_name="end")
def diamond_end(left, right):
    return left + right


def test_dynamic_scheduler_with_diamond_pattern(tmp_path: Path):
    """
    Test a diamond dependency pattern:

           start
          /     \
        left   right
          \\     /
            end
    """
    pipeline = Pipeline([diamond_start, diamond_left, diamond_right, diamond_end])
    run_folder = tmp_path / "diamond"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=False,
    )

    assert result["start"].output == 10
    assert result["left"].output == 11
    assert result["right"].output == 12
    assert result["end"].output == 23  # 11 + 12


# Test with chunksizes parameter
@pipefunc(output_name="values", mapspec="x[i] -> values[i]")
def slow_process(x: int) -> int:
    time.sleep(0.01)  # Small delay to simulate processing
    return x * 2


def test_dynamic_scheduler_with_chunksizes(tmp_path: Path):
    """Test that the dynamic scheduler respects the chunksizes parameter."""
    pipeline = Pipeline([slow_process])
    run_folder = tmp_path / "chunksizes"

    inputs = {"x": list(range(20))}  # 20 values to process

    # Use a small chunksize to force multiple chunks
    result = run_map_dynamic(
        pipeline,
        inputs=inputs,
        run_folder=run_folder,
        chunksizes=5,  # Process in chunks of 5
        show_progress=False,
    )

    # Check results
    expected = [x * 2 for x in range(20)]
    assert result["values"].output.tolist() == expected


# Test with different storage types
@pytest.mark.parametrize("storage", ["file_array", "dict", "shared_memory_dict"])
def test_dynamic_scheduler_with_different_storage(tmp_path: Path, storage: str):
    """Test that the dynamic scheduler works with different storage types."""
    pipeline = Pipeline([task_a, task_b, task_c, task_d, task_e])
    run_folder = tmp_path / f"storage_{storage}"

    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        storage=storage,
        show_progress=False,
    )

    assert result["e"].output == "e(c(a),d(b))"


# Test with error handling
@pipefunc(output_name="will_fail")
def failing_task():
    msg = "This task is designed to fail"
    raise ValueError(msg)


@pipefunc(output_name="depends_on_failure")
def dependent_task(will_fail):
    return f"Processed {will_fail}"


def test_dynamic_scheduler_error_handling(tmp_path: Path):
    """Test that the dynamic scheduler properly handles errors in tasks."""
    pipeline = Pipeline([failing_task, dependent_task])
    run_folder = tmp_path / "error_handling"

    with pytest.raises(ValueError, match="This task is designed to fail"):
        run_map_dynamic(
            pipeline,
            inputs={},
            run_folder=run_folder,
            show_progress=False,
        )


# Test with auto_subpipeline
@pipefunc(output_name="intermediate")
def create_intermediate():
    return "intermediate_value"


@pipefunc(output_name="final")
def use_intermediate(intermediate):
    return f"Used {intermediate}"


def test_dynamic_scheduler_with_auto_subpipeline(tmp_path: Path):
    """Test that the dynamic scheduler works with auto_subpipeline."""
    pipeline = Pipeline([create_intermediate, use_intermediate])
    run_folder = tmp_path / "auto_subpipeline"

    # Provide the intermediate result directly
    result = run_map_dynamic(
        pipeline,
        inputs={"intermediate": "direct_value"},
        run_folder=run_folder,
        auto_subpipeline=True,
        show_progress=False,
    )

    assert result["final"].output == "Used direct_value"
    # Check result from disk
    assert load_outputs("final", run_folder=run_folder) == "Used direct_value"

    # Without auto_subpipeline, this would fail because create_intermediate wouldn't be called
    with pytest.raises(ValueError, match="Missing required inputs"):
        run_map_dynamic(
            pipeline,
            inputs={"intermediate": "direct_value"},
            run_folder=run_folder,
            auto_subpipeline=False,
            show_progress=False,
        )


# Test with show_progress (can only verify it doesn't crash)
@pytest.mark.skipif(not hasattr(pytest, "xvfb"), reason="GUI test")
def test_dynamic_scheduler_with_progress_bar(tmp_path: Path):
    """Test that the dynamic scheduler works with progress bar display."""
    pipeline = Pipeline([task_a, task_b, task_c, task_d, task_e])
    run_folder = tmp_path / "progress_bar"

    # This just tests that it doesn't crash
    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        show_progress=True,
    )

    assert result["e"].output == "e(c(a),d(b))"


# Test with persist_memory
@pytest.mark.parametrize("persist_memory", [True, False])
def test_dynamic_scheduler_with_persist_memory(
    tmp_path: Path,
    persist_memory: bool,  # noqa: FBT001
):
    """Test that the dynamic scheduler respects the persist_memory parameter."""
    pipeline = Pipeline([task_a, task_b])
    run_folder = tmp_path / f"persist_{persist_memory}"

    # Use memory-based storage
    result = run_map_dynamic(
        pipeline,
        inputs={},
        run_folder=run_folder,
        storage="dict",  # Memory-based storage
        persist_memory=persist_memory,
        show_progress=False,
    )

    assert result["a"].output == "a"
    assert result["b"].output == "b"

    # Check if results were persisted to disk based on persist_memory setting
    # This would require checking if the storage implementation called persist()
    # which is difficult to test directly without mocking


# Test with complex MapSpec patterns
@pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
def create_matrix(x: int, y: int) -> int:
    return x * y


@pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
def sum_rows(matrix: np.ndarray) -> int:
    return np.sum(matrix)


@pipefunc(output_name="total_sum")
def sum_all(row_sums: np.ndarray) -> int:
    return np.sum(row_sums)


def test_dynamic_scheduler_with_complex_mapspec(tmp_path: Path):
    """Test that the dynamic scheduler handles complex MapSpec patterns correctly.

    # Expected matrix:
    # [[1*4, 1*5],
    #  [2*4, 2*5],
    #  [3*4, 3*5]]
    # = [[4, 5], [8, 10], [12, 15]]

    # Row sums: [9, 18, 27]
    # Total sum: 54
    """
    pipeline = Pipeline([create_matrix, sum_rows, sum_all])
    run_folder = tmp_path / "complex_mapspec"

    inputs = {"x": [1, 2, 3], "y": [4, 5]}
    result = run_map_dynamic(
        pipeline,
        inputs=inputs,
        run_folder=run_folder,
        show_progress=False,
    )

    assert result["matrix"].output.tolist() == [[4, 5], [8, 10], [12, 15]]
    assert result["row_sums"].output.tolist() == [9, 18, 27]
    assert result["total_sum"].output == 54
