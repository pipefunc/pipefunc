"""Final tests to achieve 100% coverage for exceptions.py."""

from __future__ import annotations

import numpy as np

from pipefunc._error_handling import ErrorInfo
from pipefunc.exceptions import PropagatedErrorSnapshot


def test_propagated_error_snapshot_partial_error_string():
    """Test string representation of PropagatedErrorSnapshot with partial errors."""
    # Create error info with partial error
    error_info = {
        "matrix": ErrorInfo.from_partial_error(
            shape=(3, 3),
            error_indices=(np.array([0, 2]), np.array([1, 2])),
            error_count=2,
        ),
    }

    def matrix_func(matrix):
        return matrix.sum()

    propagated = PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=matrix_func,
        reason="array_contains_errors",
        attempted_kwargs={},
    )

    # Test string representation includes partial error info
    str_repr = str(propagated)
    assert "PropagatedErrorSnapshot:" in str_repr
    assert "matrix_func" in str_repr
    assert "was skipped" in str_repr
    assert "matrix (2 errors in array)" in str_repr  # This tests line 149


def test_propagated_error_get_root_causes_partial():
    """Test get_root_causes with partial errors (currently returns empty)."""
    # Create error info with partial error
    error_info = {
        "data": ErrorInfo.from_partial_error(
            shape=(5,),
            error_indices=(np.array([2, 4]),),
            error_count=2,
        ),
    }

    propagated = PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=lambda x: x,
        reason="array_contains_errors",
        attempted_kwargs={},
    )

    # get_root_causes returns empty list for partial errors
    # because we don't store the actual error objects
    root_causes = propagated.get_root_causes()
    assert root_causes == []  # This tests lines 135-138
