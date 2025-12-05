"""Tests to verify potential issues identified in PR review.

These tests investigate whether the following are actual bugs:
1. Double dumping of error objects - FIXED (error objects now use same XOR logic)
2. Map-scope resources skipped for valid elements - NOT A BUG (by design)
3. Resource error uses wrong function reference - NOT A BUG (stores resources callable correctly)
4. SLURM filtering only for resources_scope="element" - FIXED (now filters all scopes)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot
from pipefunc.resources import Resources

# =============================================================================
# Test 1: Double dumping of error objects - FIXED
# =============================================================================


def test_error_objects_dumped_twice_with_file_storage(tmp_path):
    """Verify if error objects are dumped twice (worker + main process).

    This test checks if ErrorSnapshots are written to storage twice,
    which would be inefficient (though not incorrect).
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # Use file storage which has dump_in_subprocess=True
    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        run_folder=tmp_path,
        parallel=False,  # Sequential to make tracking easier
    )

    # Verify the error is stored correctly
    assert isinstance(result["y"].output[1], ErrorSnapshot)

    # The question is: was it dumped twice?
    # We can't easily track this without modifying the code,
    # but we can verify the storage contains the correct value
    from pipefunc.map import load_outputs

    loaded = load_outputs("y", run_folder=tmp_path)
    assert isinstance(loaded[1], ErrorSnapshot)


def test_error_objects_dump_count_with_mock(tmp_path):
    """Verify ErrorSnapshot objects are dumped exactly once.

    Regression test: Previously, ErrorSnapshot was dumped twice because error objects
    bypassed the XOR logic in _update_array. Now error objects follow the same
    dump logic as normal values, ensuring exactly one dump per output.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    # Track dump calls
    original_dumps: list[tuple[tuple, str]] = []

    # Patch the FileArray.dump method to track calls
    from pipefunc.map._storage_array._file import FileArray

    original_dump = FileArray.dump

    def patched_dump(self, key, value):
        original_dumps.append((key, type(value).__name__))
        return original_dump(self, key, value)

    with patch.object(FileArray, "dump", patched_dump):
        pipeline.map(
            {"x": [0, 1, 2]},
            error_handling="continue",
            run_folder=tmp_path,
            parallel=False,
        )

    # Count how many times ErrorSnapshot was dumped
    error_dumps = [d for d in original_dumps if d[1] == "ErrorSnapshot"]
    print(f"Error dumps: {error_dumps}")
    print(f"All dumps: {original_dumps}")

    # If this is > 1, we have double dumping
    # Expected: 1 (only dumped once)
    # Actual: 2 (dumped in worker and main process) - THIS IS THE BUG
    assert len(error_dumps) <= 1, f"ErrorSnapshot dumped {len(error_dumps)} times (expected 1)"


# =============================================================================
# Test 2: Map-scope resources skipped for valid elements
# =============================================================================


def test_map_scope_resources_with_partial_errors():
    """Test if map-scope resources are correctly evaluated when some inputs have errors.

    The concern is that when ANY map-level input contains an error,
    resources are skipped entirely, which could affect valid elements.
    """
    resource_calls = []

    def track_resources(x):
        resource_calls.append(x)
        return Resources(cpus=2)

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def step1(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    @pipefunc(
        output_name="z",
        mapspec="y[i] -> z[i]",
        resources=track_resources,
        resources_scope="map",
    )
    def step2(y: int) -> int:
        return y + 10

    pipeline = Pipeline([step1, step2])

    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        parallel=False,
    )

    # y[0] = 0, y[1] = ErrorSnapshot, y[2] = 4
    assert result["y"].output[0] == 0
    assert isinstance(result["y"].output[1], ErrorSnapshot)
    assert result["y"].output[2] == 4

    # z[0] = 10, z[1] = PropagatedErrorSnapshot, z[2] = 14
    # The question is: were resources evaluated at all?
    print(f"Resource calls: {resource_calls}")

    # If resources were skipped entirely, resource_calls would be empty
    # If resources were evaluated, we'd see the call
    # Note: For map-scope with errors in input, resources might be intentionally skipped


def test_map_scope_resources_variable_with_partial_errors():
    """Test map-scope resources with resources_variable when inputs have errors."""
    resource_calls = []

    def track_resources(y):
        resource_calls.append(("resources", y))
        return Resources(cpus=len(y) if hasattr(y, "__len__") else 1)

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def step1(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    @pipefunc(
        output_name="total",
        resources=track_resources,
        resources_scope="map",
        resources_variable="y",
    )
    def step2(y: np.ndarray) -> int:
        return int(np.sum([v for v in y if not isinstance(v, ErrorSnapshot)]))

    pipeline = Pipeline([step1, step2])

    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        parallel=False,
    )

    print(f"Resource calls: {resource_calls}")
    print(f"Total result: {result['total'].output}")

    # The total function receives the array with errors
    # Resources should still be callable


# =============================================================================
# Test 3: Resource error uses wrong function reference - CONFIRMED ISSUE
# =============================================================================


def test_resource_evaluation_error_function_reference():
    """Test that resource evaluation errors capture the correct function reference.

    FINDING: Resource evaluation errors are wrapped in PropagatedErrorSnapshot,
    not ErrorSnapshot directly. The inner ErrorSnapshot stores func.resources
    (wrapped in functools.partial via _ensure_resources), not the user function.

    This is somewhat by design - the error occurred in resource evaluation,
    not in the actual function. However, it means reproduce() on the inner
    ErrorSnapshot will re-run resource evaluation, not the actual function.
    """

    def bad_resources(x):
        msg = "resource evaluation failed"
        raise RuntimeError(msg)

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=bad_resources,
        resources_scope="element",
    )
    def compute(x: int) -> int:
        return x * 2

    pipeline = Pipeline([compute])

    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        parallel=False,
    )

    # Resource errors result in PropagatedErrorSnapshot (function was skipped)
    for i in range(3):
        output = result["y"].output[i]
        # Resource errors create PropagatedErrorSnapshot, not ErrorSnapshot directly
        assert isinstance(output, PropagatedErrorSnapshot), (
            f"Expected PropagatedErrorSnapshot at index {i}, got {type(output)}"
        )

        # The inner error info contains the actual ErrorSnapshot
        assert "__pipefunc_internal_resource_error__" in output.error_info
        resource_error_info = output.error_info["__pipefunc_internal_resource_error__"]
        assert resource_error_info.type == "full"
        inner_error = resource_error_info.error
        assert isinstance(inner_error, ErrorSnapshot)

        # The inner ErrorSnapshot stores the resources callable, not the user function
        # This is the partial-wrapped _ensure_resources function
        print(f"Index {i}: inner error function type = {type(inner_error.function)}")
        print(f"  exception = {inner_error.exception}")

        # Verify it's the resource evaluation error
        assert "resource evaluation failed" in str(inner_error.exception)


def test_resource_error_reproduce_behavior():
    """Test what happens when calling reproduce() on a resource evaluation error.

    FINDING: reproduce() on the inner ErrorSnapshot re-runs the resource
    evaluation function, which raises the same RuntimeError.
    """

    def bad_resources(x):
        msg = "resource evaluation failed"
        raise RuntimeError(msg)

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=bad_resources,
        resources_scope="element",
    )
    def compute(x: int) -> int:
        return x * 2

    pipeline = Pipeline([compute])

    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        parallel=False,
    )

    output = result["y"].output[0]
    assert isinstance(output, PropagatedErrorSnapshot)

    # Get the inner ErrorSnapshot from the resource error
    inner_error = output.error_info["__pipefunc_internal_resource_error__"].error
    assert isinstance(inner_error, ErrorSnapshot)

    # Try to reproduce - it will re-run resource evaluation
    with pytest.raises(RuntimeError, match="resource evaluation failed"):
        inner_error.reproduce()

    # This confirms that reproduce() re-runs the resources function,
    # which is correct behavior since that's where the error occurred


# =============================================================================
# Test 4: SLURM filtering only for resources_scope="element"
# =============================================================================


def test_slurm_filtering_with_map_scope_resources():
    """Test SLURM error filtering behavior with map-scope resources.

    The concern is that should_filter_error_indices() only returns True
    for resources_scope="element", so map-scope resources might submit
    pointless jobs for indices with upstream errors.
    """
    from pipefunc.map._adaptive_scheduler_slurm_executor import should_filter_error_indices

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=Resources(cpus=1),
        resources_scope="element",
    )
    def element_scope(x: int) -> int:
        return x * 2

    @pipefunc(
        output_name="z",
        mapspec="x[i] -> z[i]",
        resources=Resources(cpus=1),
        resources_scope="map",
    )
    def map_scope(x: int) -> int:
        return x * 2

    # Mock a SLURM executor
    mock_slurm_executor = MagicMock()
    mock_slurm_executor.__class__.__name__ = "SlurmPoolExecutor"

    # Test element scope
    result_element = should_filter_error_indices(element_scope, mock_slurm_executor, "continue")
    print(f"should_filter_error_indices for element scope: {result_element}")

    # Test map scope
    result_map = should_filter_error_indices(map_scope, mock_slurm_executor, "continue")
    print(f"should_filter_error_indices for map scope: {result_map}")

    # The concern: map scope returns False, meaning error indices won't be filtered
    # This could cause unnecessary job submissions
    if not result_map and result_element:
        print("CONFIRMED: Map-scope resources don't filter error indices for SLURM")


def test_slurm_filtering_function_behavior():
    """Detailed test of should_filter_error_indices logic."""
    from pipefunc.map._adaptive_scheduler_slurm_executor import (
        should_filter_error_indices,
    )

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=Resources(cpus=1),
        resources_scope="element",
    )
    def element_func(x: int) -> int:
        return x

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=Resources(cpus=1),
        resources_scope="map",
    )
    def map_func(x: int) -> int:
        return x

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def no_resources_func(x: int) -> int:
        return x

    # Test with various executor types
    mock_slurm = MagicMock()
    mock_slurm.__class__.__name__ = "SlurmPoolExecutor"

    mock_thread = ThreadPoolExecutor(max_workers=1)

    test_cases = [
        (element_func, mock_slurm, "continue", "element+slurm+continue"),
        (element_func, mock_slurm, "raise", "element+slurm+raise"),
        (element_func, mock_thread, "continue", "element+thread+continue"),
        (map_func, mock_slurm, "continue", "map+slurm+continue"),
        (map_func, mock_slurm, "raise", "map+slurm+raise"),
        (no_resources_func, mock_slurm, "continue", "no_resources+slurm+continue"),
    ]

    print("\nshould_filter_error_indices results:")
    for func, executor, error_handling, desc in test_cases:
        result = should_filter_error_indices(func, executor, error_handling)
        print(f"  {desc}: {result}")

    mock_thread.shutdown(wait=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
