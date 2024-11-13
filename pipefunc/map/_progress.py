from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple, requires

from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    from pipefunc import PipeFunc
    from pipefunc._widgets import ProgressTracker

    from ._result import StoreType


@dataclass
class Status:
    """A class to keep track of the progress of a function."""

    n_total: int
    n_in_progress: int = 0
    n_completed: int = 0
    n_failed: int = 0
    start_time: float | None = None
    end_time: float | None = None

    @property
    def n_left(self) -> int:
        return self.n_total - self.n_completed - self.n_failed

    def mark_in_progress(self) -> None:
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.n_in_progress += 1

    def mark_complete(self, _: Any = None) -> None:  # needs arg to be used as callback
        self.n_in_progress -= 1
        self.n_completed += 1
        if self.n_completed == self.n_total:
            self.end_time = time.monotonic()

    @property
    def progress(self) -> float:
        return self.n_completed / self.n_total

    def elapsed_time(self) -> float:
        assert self.start_time is not None
        if self.end_time is None:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time


def init_tracker(
    store: dict[str, StoreType],
    functions: list[PipeFunc],
    show_progress: bool,  # noqa: FBT001
    in_async: bool,  # noqa: FBT001
) -> ProgressTracker | None:
    if not show_progress:
        return None
    requires("ipywidgets", reason="show_progress", extras="ipywidgets")
    from pipefunc._widgets import ProgressTracker

    progress = {}
    for func in functions:
        name, *_ = at_least_tuple(func.output_name)  # if multiple, the have equal size
        s = store[name]
        size = s.size if isinstance(s, StorageBase) else 1
        progress[func.output_name] = Status(n_total=size)
    return ProgressTracker(progress, None, display=False, in_async=in_async)
