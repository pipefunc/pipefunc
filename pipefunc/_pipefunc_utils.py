"""pipefunc utility functions, may import things unlike `_utils.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._utils import handle_error

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


def handle_pipefunc_error(
    e: Exception,
    func: PipeFunc,
    kwargs: dict[str, Any],
) -> None:
    """Handle an error that occurred while executing a PipeFunc."""
    return handle_error(e, func, kwargs)
