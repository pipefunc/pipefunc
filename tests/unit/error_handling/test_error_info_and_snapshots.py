"""Additional tests to achieve 100% coverage for error handling code."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pytest

from pipefunc._error_handling import (
    ErrorInfo,
    cloudpickle_function_state,
    cloudunpickle_function_state,
    create_propagated_error,
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


def test_create_propagated_error_reason_selection():
    """create_propagated_error picks canonical reason based on error_info."""
    # Full error -> input_is_error
    func = lambda x: x * 2  # noqa: E731
    full_err = ErrorSnapshot(function=func, exception=ValueError("oops"), args=(), kwargs={})
    err_info_full = {"x": ErrorInfo.from_full_error(full_err)}
    propagated_full = create_propagated_error(err_info_full, func, {"x": 5})
    assert propagated_full.reason == "input_is_error"
    assert propagated_full.skipped_function == func
    # attempted_kwargs excludes keys that are errors
    assert propagated_full.attempted_kwargs == {}

    # Partial error -> array_contains_errors
    err_info_partial = {
        "y": ErrorInfo.from_partial_error((3,), (np.array([1]),), 1),
    }
    propagated_partial = create_propagated_error(
        err_info_partial,
        func,
        {"y": np.array([1, 2, 3]), "q": 7},
    )
    assert propagated_partial.reason == "array_contains_errors"
    assert propagated_partial.skipped_function == func
    # attempted_kwargs keeps non-error kwargs
    assert propagated_partial.attempted_kwargs == {"q": 7}


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


def test_error_snapshot_loads_legacy_pickle(tmp_path, monkeypatch):
    """Regression: load snapshots pickled before function serialization change."""

    def failing() -> None:
        message = "legacy boom"
        raise RuntimeError(message)

    try:
        failing()
    except RuntimeError as exc:
        legacy_snapshot = ErrorSnapshot(
            function=failing,
            exception=exc,
            args=(),
            kwargs={},
        )

    def legacy_getstate(self: ErrorSnapshot):  # type: ignore[no-redef]
        return self.__dict__.copy()

    monkeypatch.setattr(ErrorSnapshot, "__getstate__", legacy_getstate, raising=False)

    legacy_path = tmp_path / "legacy_error.pkl"
    with legacy_path.open("wb") as fh:
        cloudpickle.dump(legacy_snapshot, fh)

    loaded = ErrorSnapshot.load_from_file(legacy_path)
    assert isinstance(loaded, ErrorSnapshot)
    assert loaded.function.__name__ == failing.__name__
    assert str(loaded.exception) == "legacy boom"
    with pytest.raises(RuntimeError, match="legacy boom"):
        loaded.reproduce()


def test_propagated_error_snapshot_loads_legacy_pickle(tmp_path, monkeypatch):
    """Regression: load propagated snapshots containing direct ErrorSnapshots."""

    def exploding(x: int) -> int:
        message = f"bad {x}"
        raise ValueError(message)

    try:
        exploding(1)
    except ValueError as exc:
        nested_snapshot = ErrorSnapshot(
            function=exploding,
            exception=exc,
            args=(1,),
            kwargs={},
        )

    error_info = {"p": ErrorInfo.from_full_error(nested_snapshot)}

    def skipped(p):
        return p

    propagated = PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=skipped,
        reason="input_is_error",
        attempted_kwargs={},
    )

    def legacy_pickle_error_info(self, info):  # type: ignore[no-redef]
        legacy = {}
        for param, entry in info.items():
            legacy[param] = {
                "type": entry.type,
                "shape": entry.shape,
                "error_indices": entry.error_indices,
                "error_count": entry.error_count,
            }
            if entry.type == "full" and entry.error is not None:
                legacy[param]["error"] = entry.error
        return legacy

    monkeypatch.setattr(
        PropagatedErrorSnapshot,
        "_pickle_error_info",
        legacy_pickle_error_info,
        raising=False,
    )

    propagated_path = tmp_path / "legacy_propagated.pkl"
    with propagated_path.open("wb") as fh:
        cloudpickle.dump(propagated, fh)

    with propagated_path.open("rb") as fh:
        loaded = cloudpickle.load(fh)
    assert isinstance(loaded, PropagatedErrorSnapshot)
    info = loaded.error_info["p"]
    assert info.type == "full"
    assert isinstance(info.error, ErrorSnapshot)
    assert str(info.error.exception) == "bad 1"


def test_scan_inputs_for_errors_0d_object_array():
    """Test that scan_inputs_for_errors handles 0-D object arrays correctly."""
    from pipefunc._error_handling import scan_inputs_for_errors

    # Test 0-D array with non-error value - should return empty dict
    arr = np.array(42, dtype=object)
    result = scan_inputs_for_errors({"x": arr})
    assert result == {}

    # Test 0-D array containing ErrorSnapshot - should detect it as full error
    error = ErrorSnapshot(
        function=lambda x: x,
        exception=ValueError("test error"),
        args=(1,),
        kwargs={},
    )
    arr_with_error = np.array(error, dtype=object)
    result = scan_inputs_for_errors({"x": arr_with_error})
    assert "x" in result
    assert result["x"].type == "full"
    assert result["x"].error == error

    # Test 0-D array containing PropagatedErrorSnapshot
    propagated = PropagatedErrorSnapshot(
        error_info={"y": ErrorInfo.from_full_error(error)},
        skipped_function=lambda x: x,
        reason="input_is_error",
        attempted_kwargs={},
    )
    arr_with_propagated = np.array(propagated, dtype=object)
    result = scan_inputs_for_errors({"x": arr_with_propagated})
    assert "x" in result
    assert result["x"].type == "full"
    assert result["x"].error == propagated


def test_scan_inputs_for_errors_0d_numeric_array():
    """Test that scan_inputs_for_errors skips 0-D numeric arrays."""
    from pipefunc._error_handling import scan_inputs_for_errors

    # Numeric 0-D arrays should be skipped (cannot contain errors)
    arr = np.array(42, dtype=int)
    result = scan_inputs_for_errors({"x": arr})
    assert result == {}

    arr_float = np.array(3.14, dtype=float)
    result = scan_inputs_for_errors({"x": arr_float})
    assert result == {}
