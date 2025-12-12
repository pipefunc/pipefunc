"""pipefunc utility functions, may import things unlike `_utils.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pipefunc._utils import is_installed
from pipefunc.exceptions import ErrorSnapshot

from ._utils import handle_error

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


def handle_pipefunc_error(
    e: Exception,
    func: PipeFunc,
    kwargs: dict[str, Any],
    error_handling: Literal["raise", "continue"] = "raise",
) -> ErrorSnapshot | None:
    """Handle an error that occurred while executing a PipeFunc."""
    renamed_kwargs = func._rename_to_native(kwargs)
    snapshot = ErrorSnapshot(func.func, e, args=(), kwargs=renamed_kwargs)
    func.error_snapshot = snapshot
    if error_handling == "continue":
        return snapshot
    if is_installed("rich"):
        import rich

        rich.print(
            "\n💥 [bold red]Error snapshot attached![/bold red]\n"
            " Use [yellow]`pipeline.error_snapshot`[/yellow] to debug:\n"
            " [dim][yellow]`.reproduce()`[/yellow], [yellow]`.kwargs`[/yellow], "
            "[yellow]`.save_to_file()`[/yellow], [yellow]`.function`[/yellow], "
            "or just [yellow]`print()`[/yellow] it.[/dim]\n"
            " [dim italic]↓ Scroll down to see the full traceback.[/dim italic]",
        )

    handle_error(e, func, kwargs)
    return None  # pragma: no cover
