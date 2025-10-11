"""Additional tests to achieve 100% coverage for error handling code."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipefunc._error_handling import (
    ErrorInfo,
    check_for_error_inputs,
    cloudpickle_function_state,
    cloudunpickle_function_state,
    create_propagated_error,
    handle_error_inputs,
)
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot


def test_error_info_creation():
    """Test ErrorInfo dataclass creation methods."""
    # Test creating ErrorInfo for full error
    error = ErrorSnapshot(
        function=lambda x: x,
        exception=ValueError("test"),
        args=(1,),
        kwargs={},
    )
    error_info = ErrorInfo.from_full_error(error)
    assert error_info.type == "full"
    assert error_info.error == error
    assert error_info.shape is None
    assert error_info.error_indices is None
    assert error_info.error_count is None

    # Test creating ErrorInfo for partial error
    shape = (3, 3)
    indices = (np.array([0, 2]), np.array([1, 2]))
    count = 2
    error_info = ErrorInfo.from_partial_error(shape, indices, count)
    assert error_info.type == "partial"
    assert error_info.error is None
    assert error_info.shape == shape
    assert len(error_info.error_indices) == 2
    assert error_info.error_count == count


def test_create_propagated_error_default_reason():
    """Test create_propagated_error with default reason (no errors)."""
    # This tests the case where error_info is empty, triggering default_reason
    error_info = {}
    func = lambda x: x * 2  # noqa: E731
    kwargs = {"x": 5}

    propagated = create_propagated_error(
        error_info,
        func,
        kwargs,
        default_reason="custom_default",
    )

    assert propagated.reason == "custom_default"
    assert propagated.skipped_function == func
    assert propagated.attempted_kwargs == kwargs


def test_handle_error_inputs_with_raise_mode():
    """Test handle_error_inputs returns None when error_handling is 'raise'."""
    kwargs = {"x": 5}
    func = lambda x: x * 2  # noqa: E731

    error_info = check_for_error_inputs(kwargs)
    result = handle_error_inputs(kwargs, func, "raise", error_info=error_info)
    assert result is None


def test_cloudpickle_functions_no_function_key():
    """Test cloudpickle functions when function_key is not in state."""
    state = {"other_key": "value"}

    # Test cloudpickle_function_state with missing key
    pickled = cloudpickle_function_state(state, "missing_key")
    assert pickled == state
    assert "missing_key" not in pickled

    # Test cloudunpickle_function_state with missing key
    unpickled = cloudunpickle_function_state(state.copy(), "missing_key")
    assert unpickled == state


def test_error_snapshot_methods():
    """Test ErrorSnapshot methods not covered by existing tests."""

    # Create an ErrorSnapshot
    def failing_func(x: int) -> int:
        msg = f"Cannot process {x}"
        raise ValueError(msg)

    try:
        failing_func(42)
    except ValueError as e:
        error = ErrorSnapshot(
            function=failing_func,
            exception=e,
            args=(42,),
            kwargs={},
        )

    # Test string representation
    str_repr = str(error)
    assert "ErrorSnapshot:" in str_repr
    assert "Cannot process 42" in str_repr
    assert "failing_func" in str_repr

    # Test save and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Save to file
        error.save_to_file(temp_path)
        assert temp_path.exists()

        # Load from file
        loaded_error = ErrorSnapshot.load_from_file(temp_path)
        assert isinstance(loaded_error, ErrorSnapshot)
        assert str(loaded_error.exception) == "Cannot process 42"
        assert loaded_error.args == (42,)

        # Test reproduce - should raise the same error
        with pytest.raises(ValueError, match="Cannot process 42"):
            loaded_error.reproduce()
    finally:
        temp_path.unlink()


def test_propagated_error_snapshot_methods():
    """Test PropagatedErrorSnapshot methods not covered by existing tests."""
    # Create error info with a PropagatedErrorSnapshot inside
    original_error = ErrorSnapshot(
        function=lambda x: x,
        exception=ValueError("original"),
        args=(1,),
        kwargs={},
    )

    # Create a propagated error that contains another propagated error
    inner_propagated = PropagatedErrorSnapshot(
        error_info={"x": ErrorInfo.from_full_error(original_error)},
        skipped_function=lambda x: x * 2,
        reason="input_is_error",
        attempted_kwargs={"y": 10},
    )

    # Create outer propagated error
    outer_propagated = PropagatedErrorSnapshot(
        error_info={"z": ErrorInfo.from_full_error(inner_propagated)},
        skipped_function=lambda z: z + 5,
        reason="input_is_error",
        attempted_kwargs={"w": 20},
    )

    # Test get_root_causes with nested PropagatedErrorSnapshot
    root_causes = outer_propagated.get_root_causes()
    assert len(root_causes) == 1
    assert root_causes[0] == original_error

    # Test string representation
    str_repr = str(outer_propagated)
    assert "PropagatedErrorSnapshot:" in str_repr
    assert "was skipped" in str_repr
    assert "z (complete failure)" in str_repr


def test_propagated_error_snapshot_pickle_unpickle():
    """Test pickling and unpickling of PropagatedErrorSnapshot with various error types."""
    # Create a complex PropagatedErrorSnapshot with both full and partial errors
    original_error = ErrorSnapshot(
        function=lambda x: x,
        exception=ValueError("test error"),
        args=(1,),
        kwargs={},
    )

    error_info = {
        "full_param": ErrorInfo.from_full_error(original_error),
        "partial_param": ErrorInfo.from_partial_error(
            shape=(2, 2),
            error_indices=(np.array([0, 1]), np.array([0, 1])),
            error_count=2,
        ),
    }

    def skipped_func(full_param, partial_param):
        return full_param + partial_param

    propagated = PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=skipped_func,
        reason="array_contains_errors",
        attempted_kwargs={"other": 42},
    )

    # Test pickling and unpickling
    import cloudpickle

    pickled = cloudpickle.dumps(propagated)
    unpickled = cloudpickle.loads(pickled)

    # Verify unpickled object
    assert isinstance(unpickled, PropagatedErrorSnapshot)
    assert unpickled.reason == "array_contains_errors"
    assert unpickled.attempted_kwargs == {"other": 42}
    assert len(unpickled.error_info) == 2
    assert "full_param" in unpickled.error_info
    assert "partial_param" in unpickled.error_info

    # Check full error was preserved
    full_info = unpickled.error_info["full_param"]
    assert full_info.type == "full"
    assert isinstance(full_info.error, ErrorSnapshot)
    assert str(full_info.error.exception) == "test error"

    # Check partial error was preserved
    partial_info = unpickled.error_info["partial_param"]
    assert partial_info.type == "partial"
    assert partial_info.shape == (2, 2)
    assert partial_info.error_count == 2
