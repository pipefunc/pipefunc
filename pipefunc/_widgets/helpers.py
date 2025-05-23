# Module should not import ipywidgets or optional dependencies

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from pipefunc._utils import is_running_in_ipynb

if TYPE_CHECKING:
    import asyncio

    from pipefunc._widgets.async_status_widget import AsyncTaskStatusWidget

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None


def maybe_async_task_status_widget(task: asyncio.Task) -> AsyncTaskStatusWidget | None:
    """Create an AsyncTaskStatusWidget and attach it to a task."""
    if not is_running_in_ipynb():
        return None
    if not has_ipywidgets:  # pragma: no cover
        print(
            "⚠️ `pipeline.map_async` provides task status visualization with ipywidgets."
            " Install with: pip install ipywidgets",
        )
        return None
    from pipefunc._widgets.async_status_widget import AsyncTaskStatusWidget

    widget = AsyncTaskStatusWidget()
    widget.attach_task(task)
    return widget
