from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pipefunc._utils import at_least_tuple, is_installed, is_running_in_ipynb, requires

from ._shapes import shape_is_resolved
from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    from concurrent.futures import Future

    from pipefunc import PipeFunc
    from pipefunc._widgets.progress import ProgressTracker
    from pipefunc._widgets.progress_rich import RichProgressTracker

    from ._result import StoreType


@dataclass
class Status:
    """A class to keep track of the progress of a function."""

    n_total: int | None
    n_in_progress: int = 0
    n_completed: int = 0
    n_failed: int = 0
    start_time: float | None = None
    end_time: float | None = None

    @property
    def n_left(self) -> int:
        return self.n_total - self.n_attempted  # type: ignore[operator]

    def mark_in_progress(self, *, n: int = 1) -> None:
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.n_in_progress += n

    def mark_complete(
        self,
        future: Future | None = None,
        *,
        n: int = 1,
    ) -> None:
        self.n_in_progress -= n
        if future is not None and future.exception() is not None:
            self.n_failed += n
        else:
            self.n_completed += n

        if self.n_total is not None and self.n_attempted >= self.n_total:
            self.end_time = time.monotonic()

    @property
    def progress(self) -> float:
        if self.n_total is None:
            return 0.0
        if self.n_total == 0:
            return 1.0
        return self.n_attempted / self.n_total

    @property
    def n_attempted(self) -> int:
        return self.n_completed + self.n_failed

    def elapsed_time(self) -> float:
        if self.start_time is None:  # Happens when n_total is 0
            return 0.0
        if self.end_time is None:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time


def init_tracker(
    store: dict[str, StoreType],
    functions: list[PipeFunc],
    show_progress: bool | None,
    in_async: bool,  # noqa: FBT001
    implementation: Literal["rich", "ipywidgets"] = "rich",
) -> ProgressTracker | RichProgressTracker | None:
    if show_progress is None:
        show_progress = is_running_in_ipynb() and is_installed("ipywidgets")
    if not show_progress:
        return None
    if implementation == "rich":
        requires("ipywidgets", reason="show_progress", extras="ipywidgets")
        from pipefunc._widgets.progress_rich import RichProgressTracker
    elif implementation == "ipywidgets":
        from pipefunc._widgets.progress import ProgressTracker

    progress = {}
    for func in functions:
        name, *_ = at_least_tuple(func.output_name)  # if multiple, the have equal size
        s = store[name]
        if isinstance(s, StorageBase):
            if shape_is_resolved(s.shape):  # noqa: SIM108
                size = s.size
            else:
                size = None  # Defer size calculation until shape is resolved
        else:
            size = 1
        progress[func.output_name] = Status(n_total=size)
    if implementation == "rich":
        return RichProgressTracker(progress, None, display=False, in_async=in_async)
    if implementation == "ipywidgets":
        return ProgressTracker(progress, None, display=False, in_async=in_async)
    msg = f"Invalid implementation: {implementation}"
    raise ValueError(msg)
