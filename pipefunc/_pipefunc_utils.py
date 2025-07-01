"""pipefunc utility functions, may import things unlike `_utils.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipefunc.exceptions import ErrorSnapshot, PipeFuncError

from ._utils import handle_error

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


def handle_pipefunc_error(
    e: Exception,
    func: PipeFunc,
    kwargs: dict[str, Any],
) -> None:
    """Handle an error that occurred while executing a PipeFunc."""
    if isinstance(e, PipeFuncError):
        func.error_snapshot = ErrorSnapshot(
            func.func,
            e.original_exception,
            args=(),
            kwargs=kwargs,
            **e.metadata,
        )
        e = e.original_exception
    return handle_error(e, func, kwargs)
