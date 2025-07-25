"""Common error handling utilities for pipefunc."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable


def check_for_error_inputs(
    kwargs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Check if any input contains ErrorSnapshot objects.

    Parameters
    ----------
    kwargs
        The keyword arguments to check for errors.

    Returns
    -------
        A dictionary mapping parameter names to error information.
        Empty if no errors are found.

    """
    error_info = {}
    for param_name, value in kwargs.items():
        if isinstance(value, (ErrorSnapshot, PropagatedErrorSnapshot)):
            # Input parameter is itself an error
            error_info[param_name] = {"type": "full", "error": value}
        elif isinstance(value, np.ndarray) and value.dtype == object:
            # Check if array contains any ErrorSnapshot objects
            error_mask = np.array(
                [isinstance(v, (ErrorSnapshot, PropagatedErrorSnapshot)) for v in value.flat],
            )
            if error_mask.any():
                error_info[param_name] = {
                    "type": "partial",
                    "shape": value.shape,
                    "error_indices": np.where(error_mask.reshape(value.shape)),  # type: ignore[dict-item]
                    "error_count": error_mask.sum(),
                }
    return error_info


def create_propagated_error(
    error_info: dict[str, dict[str, Any]],
    skipped_function: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    default_reason: str = "input_contains_errors",
) -> PropagatedErrorSnapshot:
    """Create a PropagatedErrorSnapshot from error information.

    Parameters
    ----------
    error_info
        Dictionary mapping parameter names to error details.
    skipped_function
        The function that was skipped due to errors.
    kwargs
        All keyword arguments passed to the function.
    default_reason
        Default reason to use when not specified.

    Returns
    -------
        A PropagatedErrorSnapshot instance.

    """
    # Determine the reason based on error types
    if any(info["type"] == "full" for info in error_info.values()):
        reason = "input_is_error"
    elif any(info["type"] == "partial" for info in error_info.values()):
        reason = "array_contains_errors"
    else:
        reason = default_reason

    return PropagatedErrorSnapshot(
        error_info=error_info,
        skipped_function=skipped_function,
        reason=reason,
        attempted_kwargs={k: v for k, v in kwargs.items() if k not in error_info},
    )


def handle_error_inputs(
    kwargs: dict[str, Any],
    func: Callable[..., Any],
    error_handling: str,
) -> PropagatedErrorSnapshot | None:
    """Check for error inputs and create PropagatedErrorSnapshot if needed.

    Parameters
    ----------
    kwargs
        The keyword arguments to check.
    func
        The function that would be called.
    error_handling
        The error handling mode ("raise" or "continue").

    Returns
    -------
        PropagatedErrorSnapshot if errors found and error_handling is "continue",
        None otherwise.

    """
    if error_handling != "continue":
        return None

    error_info = check_for_error_inputs(kwargs)
    if error_info:
        return create_propagated_error(error_info, func, kwargs)
    return None


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
        state[function_key] = cloudpickle.loads(state[function_key])
    return state
