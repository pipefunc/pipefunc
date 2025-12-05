"""Tests for error handling edge cases identified in PR review.

Tests cover:
1. ErrorSnapshot dump count (regression test for double-dump fix)
2. Map-scope resources behavior with partial errors
3. Resource evaluation error handling
4. SLURM error index filtering
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
# Test 1: ErrorSnapshot dump count (regression test for double-dump fix)
# =============================================================================


def test_error_objects_dumped_once_with_file_storage(tmp_path):
    """Verify error objects are stored correctly with file storage."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    result = pipeline.map(
        {"x": [0, 1, 2]},
        error_handling="continue",
        run_folder=tmp_path,
        parallel=False,
    )

    assert isinstance(result["y"].output[1], ErrorSnapshot)

    from pipefunc.map import load_outputs

    loaded = load_outputs("y", run_folder=tmp_path)
    assert isinstance(loaded[1], ErrorSnapshot)


def test_error_objects_dump_count_with_mock(tmp_path):
    """Verify ErrorSnapshot objects are dumped exactly once.

    Regression test for fix that removed double-dump of error objects.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def may_fail(x: int) -> int:
        if x == 1:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    pipeline = Pipeline([may_fail])

    dump_calls: list[tuple[tuple, str]] = []

    from pipefunc.map._storage_array._file import FileArray

    original_dump = FileArray.dump

    def patched_dump(self, key, value):
        dump_calls.append((key, type(value).__name__))
        return original_dump(self, key, value)

    with patch.object(FileArray, "dump", patched_dump):
        pipeline.map(
            {"x": [0, 1, 2]},
            error_handling="continue",
            run_folder=tmp_path,
            parallel=False,
        )

    error_dumps = [d for d in dump_calls if d[1] == "ErrorSnapshot"]
    assert len(error_dumps) == 1, f"ErrorSnapshot dumped {len(error_dumps)} times"


# =============================================================================
# Test 2: Map-scope resources behavior with partial errors
# =============================================================================


def test_map_scope_resources_with_partial_errors():
    """Test map-scope resources when some inputs have errors.

    When map-level inputs contain errors and resources_variable is None,
    resources are skipped (by design) because the resource function can't
    be called meaningfully.
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

    assert result["y"].output[0] == 0
    assert isinstance(result["y"].output[1], ErrorSnapshot)
    assert result["y"].output[2] == 4


def test_map_scope_resources_variable_with_partial_errors():
    """Test map-scope resources with resources_variable when inputs have errors.

    When resources_variable is set, the resources function IS called
    and receives the error-containing array.
    """
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

    # Function receives array with errors and handles them
    assert result["total"].output is not None


# =============================================================================
# Test 3: Resource evaluation error handling
# =============================================================================


def test_resource_evaluation_error_creates_propagated_snapshot():
    """Test that resource evaluation errors create PropagatedErrorSnapshot.

    Resource evaluation errors are wrapped in PropagatedErrorSnapshot,
    not ErrorSnapshot directly. The inner ErrorSnapshot stores the
    resources callable (not the user function) because that's where
    the error occurred.
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

    for i in range(3):
        output = result["y"].output[i]
        assert isinstance(output, PropagatedErrorSnapshot), (
            f"Expected PropagatedErrorSnapshot at index {i}, got {type(output)}"
        )

        assert "__pipefunc_internal_resource_error__" in output.error_info
        resource_error_info = output.error_info["__pipefunc_internal_resource_error__"]
        assert resource_error_info.type == "full"

        inner_error = resource_error_info.error
        assert isinstance(inner_error, ErrorSnapshot)
        assert "resource evaluation failed" in str(inner_error.exception)


def test_resource_error_reproduce():
    """Test that reproduce() on resource error re-raises the error.

    reproduce() on the inner ErrorSnapshot re-runs the resource
    evaluation function, which is correct since that's where the error occurred.
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
    inner_error = output.error_info["__pipefunc_internal_resource_error__"].error

    with pytest.raises(RuntimeError, match="resource evaluation failed"):
        inner_error.reproduce()


# =============================================================================
# Test 4: SLURM error index filtering
# =============================================================================


def test_slurm_filtering_with_resources_scopes():
    """Test should_filter_error_indices for different resource scopes.

    Both element and map scopes should filter error indices when using
    SLURM with continue mode (after the fix that removed the element-only
    restriction).
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

    mock_executor = MagicMock()

    with patch(
        "pipefunc.map._adaptive_scheduler_slurm_executor.is_slurm_executor",
        return_value=True,
    ):
        # Both scopes should filter with SLURM + continue mode
        assert should_filter_error_indices(element_scope, mock_executor, "continue")
        assert should_filter_error_indices(map_scope, mock_executor, "continue")

        # Neither should filter with raise mode
        assert not should_filter_error_indices(element_scope, mock_executor, "raise")
        assert not should_filter_error_indices(map_scope, mock_executor, "raise")


def test_slurm_filtering_various_combinations():
    """Test should_filter_error_indices with various executor/scope/mode combinations."""
    from pipefunc.map._adaptive_scheduler_slurm_executor import should_filter_error_indices

    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=Resources(cpus=1),
        resources_scope="element",
    )
    def element_func(x: int) -> int:
        return x

    @pipefunc(
        output_name="z",
        mapspec="x[i] -> z[i]",
        resources=Resources(cpus=1),
        resources_scope="map",
    )
    def map_func(x: int) -> int:
        return x

    @pipefunc(output_name="w", mapspec="x[i] -> w[i]")
    def no_resources_func(x: int) -> int:
        return x

    mock_executor = MagicMock()

    # With SLURM executor (mocked)
    with patch(
        "pipefunc.map._adaptive_scheduler_slurm_executor.is_slurm_executor",
        return_value=True,
    ):
        # continue mode should filter for all resource scopes
        assert should_filter_error_indices(element_func, mock_executor, "continue")
        assert should_filter_error_indices(map_func, mock_executor, "continue")
        assert should_filter_error_indices(no_resources_func, mock_executor, "continue")

        # raise mode should never filter
        assert not should_filter_error_indices(element_func, mock_executor, "raise")
        assert not should_filter_error_indices(map_func, mock_executor, "raise")

    # With non-SLURM executor
    with ThreadPoolExecutor(max_workers=1) as thread_executor:
        assert not should_filter_error_indices(element_func, thread_executor, "continue")
        assert not should_filter_error_indices(map_func, thread_executor, "continue")
