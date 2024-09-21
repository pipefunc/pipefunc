from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import TypeAlias

from pipefunc._utils import prod

_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]


@dataclass
class Status:
    n_total: int
    n_current: int = 0
    n_start: int = 0
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    def increment(self) -> None:
        self.n_current += 1
        if self.n_current == self.n_total:
            self.end_time = time.monotonic()

    def elapsed_time(self) -> float:
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time

    def estimated_remaining_time(self) -> float:
        elapsed = self.elapsed_time()
        progress = self.n_current - self.n_start
        if progress <= 0:
            return float("inf")
        remaining_work = self.n_total - self.n_current
        time_per_unit = elapsed / progress
        return remaining_work * time_per_unit

    def estimated_remaining_datetime(self) -> datetime.datetime:
        return datetime.datetime.now() + datetime.timedelta(seconds=self.estimated_remaining_time())  # noqa: DTZ005


class ProgressTracker:
    def __init__(self, output_names: set[str], shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]) -> None:
        self.data = {}
        for output_name in output_names:
            n_total = prod(shapes[output_name]) if output_name in shapes else 1
            self.data[output_name] = Status(n_total=n_total)

    def increment(self, output_name: str) -> None:
        self.data[output_name].increment()
