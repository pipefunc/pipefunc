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
    from pipefunc._widgets.progress_headless import HeadlessProgressTracker
    from pipefunc._widgets.progress_ipywidgets import IPyWidgetsProgressTracker
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

    def remaining_time(self, *, elapsed_time: float | None = None) -> float | None:
        if elapsed_time is None:  # pragma: no cover
            elapsed_time = self.elapsed_time()
        if elapsed_time == 0:
            return None
        progress = self.progress
        if progress == 0:
            return None
        return (1.0 - progress) * (elapsed_time / progress)


def _progress_tracker_implementation(
    show_progress: Literal[True, "rich", "ipywidgets", "headless"] | None,
) -> Literal["rich", "ipywidgets", "headless"] | None:
    if isinstance(show_progress, str):
        return show_progress
    if show_progress is True:
        if is_running_in_ipynb() and is_installed("ipywidgets"):  # pragma: no cover
            return "ipywidgets"
        if is_installed("rich"):
            return "rich"
        msg = "No progress bar implementation found. Please install 'ipywidgets' or 'rich'."  # pragma: no cover
        raise ModuleNotFoundError(msg)  # pragma: no cover
    if (
        show_progress is None and is_running_in_ipynb() and is_installed("ipywidgets")
    ):  # pragma: no cover
        return "ipywidgets"

    return None  # pragma: no cover


def init_tracker(
    store: dict[str, StoreType],
    functions: list[PipeFunc],
    show_progress: bool | Literal["rich", "ipywidgets", "headless"] | None,
    in_async: bool,
) -> IPyWidgetsProgressTracker | RichProgressTracker | HeadlessProgressTracker | None:
    if show_progress is False:
        return None
    implementation = _progress_tracker_implementation(show_progress)
    if implementation == "rich":
        requires("rich", reason="show_progress", extras="rich")
        from pipefunc._widgets.progress_rich import RichProgressTracker as ProgressTracker
    elif implementation == "ipywidgets":
        requires("ipywidgets", reason="show_progress", extras="ipywidgets")
        from pipefunc._widgets.progress_ipywidgets import (  # type: ignore[assignment]
            IPyWidgetsProgressTracker as ProgressTracker,
        )
    elif implementation == "headless":
        from pipefunc._widgets.progress_headless import (  # type: ignore[assignment]
            HeadlessProgressTracker as ProgressTracker,
        )
    else:
        return None

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
    return ProgressTracker(progress, None, in_async=in_async)
