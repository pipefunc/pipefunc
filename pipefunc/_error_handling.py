"""Common error handling utilities for pipefunc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle
import numpy as np

from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ErrorInfo:
    """Information about errors in a parameter."""

    type: Literal["full", "partial"]
    error: ErrorSnapshot | PropagatedErrorSnapshot | None = None
    shape: tuple[int, ...] | None = None
    error_indices: tuple[np.ndarray, ...] | None = None
    error_count: int | None = None

    @classmethod
    def from_full_error(cls, error: ErrorSnapshot | PropagatedErrorSnapshot) -> ErrorInfo:
        """Create ErrorInfo for a parameter that is entirely an error."""
        return cls(type="full", error=error)

    @classmethod
    def from_partial_error(
        cls,
        shape: tuple[int, ...],
        error_indices: tuple[np.ndarray, ...],
        error_count: int,
    ) -> ErrorInfo:
        """Create ErrorInfo for an array containing some errors."""
        return cls(
            type="partial",
            shape=shape,
            error_indices=error_indices,
            error_count=error_count,
        )


def scan_inputs_for_errors(kwargs: dict[str, Any]) -> dict[str, ErrorInfo]:
    """Check if any input contains ErrorSnapshot objects.

    Parameters
    ----------
    kwargs
        The keyword arguments to check for errors.

    Returns
    -------
        A dictionary mapping parameter names to ErrorInfo objects.
        Empty if no errors are found.

    """
    from pipefunc.map._storage_array._base import StorageBase

    error_info = {}
    for param_name, value in kwargs.items():
        if isinstance(value, (ErrorSnapshot, PropagatedErrorSnapshot)):
            # Input parameter is itself an error
            error_info[param_name] = ErrorInfo.from_full_error(value)
            continue

        array: np.ndarray | np.ma.MaskedArray | None = None
        if isinstance(value, StorageBase):
            dtype = getattr(value, "dtype", None)
            if dtype is not None and dtype is not object:
                continue  # Numeric/bytes storages cannot hold ErrorSnapshot objects
            array = value.to_array()
        elif isinstance(value, np.ndarray) and value.dtype == object:
            array = value

        if array is not None:
            if array.dtype != object:
                continue  # Skip cheap scan for non-object arrays

            # Handle 0-D arrays specially (np.where doesn't work on 0-D arrays)
            if array.ndim == 0:
                item = array.item()
                if isinstance(item, (ErrorSnapshot, PropagatedErrorSnapshot)):
                    error_info[param_name] = ErrorInfo.from_full_error(item)
                continue

            # Check if array contains any ErrorSnapshot objects
            # Using np.fromiter is generally faster than list comprehension for flat iteration
            error_mask = np.fromiter(
                (isinstance(v, (ErrorSnapshot, PropagatedErrorSnapshot)) for v in array.flat),
                dtype=bool,
                count=array.size,
            )
            if error_mask.any():
                error_info[param_name] = ErrorInfo.from_partial_error(
                    shape=array.shape,
                    error_indices=np.where(error_mask.reshape(array.shape)),
                    error_count=int(error_mask.sum()),
                )

    return error_info


def create_propagated_error(
    error_info: dict[str, ErrorInfo],
    skipped_function: Callable[..., Any],
    kwargs: dict[str, Any],
) -> PropagatedErrorSnapshot:
    """Create a PropagatedErrorSnapshot from error information.

    Parameters
    ----------
    error_info
        Dictionary mapping parameter names to ErrorInfo objects.
    skipped_function
        The function that was skipped due to errors.
    kwargs
        All keyword arguments passed to the function.

    Returns
    -------
        A PropagatedErrorSnapshot instance.

    """
    # Determine the reason based on error types
    if any(info.type == "full" for info in error_info.values()):
        reason = "input_is_error"
    else:
        reason = "array_contains_errors"

    return PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=skipped_function,
        reason=reason,  # type: ignore[arg-type]
        attempted_kwargs={k: v for k, v in kwargs.items() if k not in error_info},
    )


def cloudpickle_function_state(state: dict[str, Any], function_key: str) -> dict[str, Any]:
    """Prepare state dict for pickling by serializing function references.

    Parameters
    ----------
    state
        The state dictionary to prepare.
    function_key
        The key in the state dict containing the function reference.

    Returns
    -------
        Modified state dict with function serialized using cloudpickle.

    """
    pickled_state = state.copy()
    if function_key in pickled_state:
        pickled_state[function_key] = cloudpickle.dumps(pickled_state[function_key])
    return pickled_state


def cloudunpickle_function_state(state: dict[str, Any], function_key: str) -> dict[str, Any]:
    """Restore state dict from pickling by deserializing function references.

    Parameters
    ----------
    state
        The pickled state dictionary.
    function_key
        The key in the state dict containing the serialized function.

    Returns
    -------
        State dict with function deserialized from cloudpickle.

    """
    if function_key in state:
        value = state[function_key]
        # NOTE: Starting with v0.88 we persist raw bytes here;
        # pre-v0.88 pickles still contain the live callable, so only deserialize if
        # the stored value is bytes.
        if isinstance(value, bytes):
            state[function_key] = cloudpickle.loads(value)
    return state
