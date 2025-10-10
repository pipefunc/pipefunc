"""Define error-related classes for `pipefunc`."""

from __future__ import annotations

import datetime
import getpass
import os
import platform
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cloudpickle

from pipefunc._utils import get_local_ip

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class UnusedParametersError(ValueError):
    """Exception raised when unused parameters are provided to a function."""


def _timestamp() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


@dataclass
class ErrorSnapshot:
    """A snapshot that represents an error in a function call."""

    function: Callable[..., Any]
    exception: Exception
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    traceback: str = field(init=False)
    timestamp: str = field(default_factory=_timestamp)
    user: str = field(default_factory=getpass.getuser)
    machine: str = field(default_factory=platform.node)
    ip_address: str = field(default_factory=get_local_ip)
    current_directory: str = field(default_factory=os.getcwd)

    def __post_init__(self) -> None:
        """Initialize the error snapshot with a formatted traceback."""
        tb = traceback.format_exception(
            type(self.exception),
            self.exception,
            self.exception.__traceback__,
        )
        self.traceback = "".join(tb)

    def __str__(self) -> str:
        """Return a string representation of the error snapshot."""
        args_repr = ", ".join(repr(a) for a in self.args)
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        func_name = f"{self.function.__module__}.{self.function.__qualname__}"

        return (
            "ErrorSnapshot:\n"
            "--------------\n"
            f"- 🛠 Function: {func_name}\n"
            f"- 🚨 Exception type: {type(self.exception).__name__}\n"
            f"- 💥 Exception message: {self.exception}\n"
            f"- 📋 Args: ({args_repr})\n"
            f"- 🗂 Kwargs: {{{kwargs_repr}}}\n"
            f"- 🕒 Timestamp: {self.timestamp}\n"
            f"- 👤 User: {self.user}\n"
            f"- 💻 Machine: {self.machine}\n"
            f"- 📡 IP Address: {self.ip_address}\n"
            f"- 📂 Current Directory: {self.current_directory}\n"
            "\n"
            "🔁 Reproduce the error by calling `error_snapshot.reproduce()`.\n"
            "📄 Or see the full stored traceback using `error_snapshot.traceback`.\n"
            "🔍 Inspect `error_snapshot.args` and `error_snapshot.kwargs`.\n"
            "💾 Or save the error to a file using `error_snapshot.save_to_file(filename)`"
            " and load it using `ErrorSnapshot.load_from_file(filename)`."
        )

    def reproduce(self) -> Any | None:
        """Attempt to recreate the error by calling the function with stored arguments."""
        return self.function(*self.args, **self.kwargs)

    def save_to_file(self, filename: str | Path) -> None:
        """Save the error snapshot to a file using cloudpickle."""
        with open(filename, "wb") as f:  # noqa: PTH123
            cloudpickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str | Path) -> ErrorSnapshot:
        """Load an error snapshot from a file using cloudpickle."""
        with open(filename, "rb") as f:  # noqa: PTH123
            return cloudpickle.load(f)

    def _ipython_display_(self) -> None:  # pragma: no cover
        from IPython.display import HTML, display

        display(HTML(f"<pre>{self}</pre>"))
