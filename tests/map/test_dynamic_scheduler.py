# test_dynamic_scheduler.py

import time
from pathlib import Path

from pipefunc import Pipeline, pipefunc
from pipefunc.map._run_dynamic import run_map_dynamic


@pipefunc(output_name="slow")
def slow_task():
    # Simulate a slow CPU-bound task
    time.sleep(2)
    return "slow"


@pipefunc(output_name="quick")
def quick_task():
    # Simulate a fast CPU-bound task
    time.sleep(0.5)
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

    Since slow_task sleeps for 2 seconds and quick_task for 0.5 seconds,
    if they were executed sequentially the total time would be >2.5 seconds.
    With parallel dynamic scheduling, the total time should be close to the slow branch (≈2 seconds + overhead).
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
    # The total elapsed time should be close to 2 seconds (the slow branch) rather than 2.5 seconds.
    # Allowing for some overhead, we assert elapsed < 2.5 seconds.
    assert elapsed < 2.5, f"Elapsed time was {elapsed:.2f} seconds, expected less than 2.5 seconds"
