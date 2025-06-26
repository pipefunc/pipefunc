"""pipefunc utility functions, may import things unlike `_utils.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._utils import handle_error
from .exceptions import ErrorSnapshot

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


def handle_pipefunc_error(
    e: Exception,
    func: PipeFunc,
    kwargs: dict[str, Any],
    return_error: bool = False,  # noqa: FBT002
) -> ErrorSnapshot | None:
    """Handle an error that occurred while executing a PipeFunc."""
    if return_error:
        return ErrorSnapshot(function=func.func, exception=e, args=(), kwargs=kwargs)
    handle_error(e, func, kwargs)
    return None
