from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipefunc._utils import requires, table

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing import Any as Self
    from types import TracebackType


@dataclass
class ResourceStats:
    """A class for storing execution statistics for a function."""

    num_executions: int = 0
    average: float = 0.0
    variance: float = 0.0
    max: float = 0.0

    def update(self, execution_time: float) -> None:
        """Update the execution statistics with a new execution time.

        Parameters
        ----------
        execution_time
            The execution time of the new function call.

        """
        self.num_executions += 1
        delta = execution_time - self.average
        self.average += delta / self.num_executions
        delta2 = execution_time - self.average
        self.variance += delta * delta2
        self.max = max(self.max, execution_time)  # Update maximum time

    @property
    def std(self) -> float:
        """Compute the standard deviation of the execution times.

        Returns
        -------
            The standard deviation of the execution times.

        """
        if self.num_executions < 2:  # noqa: PLR2004
            return 0.0
        return (self.variance / (self.num_executions - 1)) ** 0.5

    def __repr__(self) -> str:
        """Return a string representation of the execution statistics."""
        return f"ResourceStats(num_executions={self.num_executions}, average={self.average:.4e}, max={self.max:.4e}, std={self.std:.4e})"


@dataclass(frozen=True, slots=True)
class ProfilingStats:
    """A class for storing execution statistics."""

    cpu: ResourceStats = field(default_factory=ResourceStats)
    memory: ResourceStats = field(default_factory=ResourceStats)
    time: ResourceStats = field(default_factory=ResourceStats)


class ResourceProfiler:
    """A class for profiling the resource usage of a process.

    Parameters
    ----------
    pid
        The process ID for which resource profiling will be performed.
    stats
        The ProfilingStats instance in which the profiling data will be stored.
    interval
        The time interval between resource measurements, in
        seconds (default is 0.1).

    """

    def __init__(self, pid: int, stats: ProfilingStats, *, interval: float = 10) -> None:
        """Initialize the ResourceProfiler instance."""
        self.pid = pid
        self.stats = stats
        self.interval = interval
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.execution_time: float | None = None
        self.start_time: float | None = None

    def __enter__(self) -> Self:
        """Enter, start the measurement thread, and return the profiler instance.

        Returns
        -------
            The profiler instance.

        """
        self.thread = threading.Thread(target=self.measure_resources)
        assert self.thread is not None
        self.thread.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager and stop the measurement thread.

        Parameters
        ----------
        exc_type
            The exception type, if an exception occurred, otherwise None.
        exc_value
            The exception instance, if an exception occurred, otherwise None.
        traceback
            A traceback object, if an exception occurred, otherwise None.

        """
        assert self.start_time is not None
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        total_time = time.perf_counter() - self.start_time
        self.stats.time.update(total_time)

    def measure_resources(self) -> None:
        """Measure resource usage (CPU and memory) for the specified process."""
        requires("psutil", reason="profile", extras="profiling")
        import psutil

        process = psutil.Process(self.pid)
        while not self.stop_event.is_set():  # pragma: no cover
            try:
                mem_info = process.memory_info()
                memory = mem_info.rss
                cpu_percent = process.cpu_percent()
            except psutil.NoSuchProcess:
                break

            self.stats.memory.update(memory)
            self.stats.cpu.update(cpu_percent)
            self.stop_event.wait(self.interval)


def print_profiling_stats(profiling_stats: dict[str, ProfilingStats]) -> None:
    """Print the resource usage report for each function in ``profiling_stats``."""
    headers = [
        "Function",
        "Avg CPU Usage (%)",
        "Max Memory Usage (MB)",
        "Avg Time (s)",
        "Total Time (%)",
        "Number of Calls",
    ]
    rows = []
    for func_name, stats in profiling_stats.items():
        row = [
            func_name,
            f"{stats.cpu.average:.2f}",
            f"{stats.memory.max / (1024 * 1024):.2f}",
            f"{stats.time.average:.2e}",
            stats.time.average * stats.time.num_executions,
            stats.time.num_executions,
        ]
        rows.append(row)

    total_time = sum(row[4] for row in rows)  # type: ignore[misc]
    if total_time > 0:
        for row in rows:
            row[4] = f"{row[4] / total_time * 100:.2f}"  # type: ignore[operator]

    print("Resource Usage Report:")
    print(table(rows, headers))
