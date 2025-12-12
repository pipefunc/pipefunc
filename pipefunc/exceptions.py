"""Define error-related classes for `pipefunc`."""

from __future__ import annotations

import datetime
import getpass
import os
import platform
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import cloudpickle

from pipefunc._utils import get_local_ip

Reason = Literal["input_is_error", "array_contains_errors"]

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pipefunc._error_handling import ErrorInfo


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

    def __repr__(self) -> str:
        """Return a concise representation for use in arrays and containers."""
        func_name = getattr(self.function, "__name__", "?")
        exc_type = type(self.exception).__name__
        return f"ErrorSnapshot({func_name!r}, {exc_type}: {self.exception})"

    def __str__(self) -> str:
        """Return a detailed string representation of the error snapshot."""
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

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickling to handle function references using cloudpickle."""
        from pipefunc._error_handling import cloudpickle_function_state

        return cloudpickle_function_state(self.__dict__.copy(), "function")

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickling to restore function references."""
        from pipefunc._error_handling import cloudunpickle_function_state

        self.__dict__.update(cloudunpickle_function_state(state, "function"))


@dataclass
class PropagatedErrorSnapshot:
    """Represents a function that was skipped due to upstream errors."""

    error_info: dict[str, ErrorInfo]  # parameter -> error details
    skipped_function: Callable[..., Any]
    reason: Reason  # normalized reason label
    attempted_kwargs: dict[str, Any]  # kwargs that were not errors
    timestamp: str = field(default_factory=_timestamp)

    def __repr__(self) -> str:
        """Return a concise representation for use in arrays and containers."""
        func_name = getattr(self.skipped_function, "__name__", str(self.skipped_function))
        return f"PropagatedErrorSnapshot({func_name!r}, reason={self.reason!r})"

    def __str__(self) -> str:
        """Return a detailed string representation of the propagated error snapshot."""
        func_name = getattr(self.skipped_function, "__name__", str(self.skipped_function))
        error_summary = []
        for param, info in self.error_info.items():
            if info.type == "full":
                error_summary.append(f"{param} (complete failure)")
            else:
                error_summary.append(f"{param} ({info.error_count} errors in array)")

        return (
            f"PropagatedErrorSnapshot: Function '{func_name}' was skipped\n"
            f"Reason: {self.reason}\n"
            f"Errors in: {', '.join(error_summary)}"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickling to handle function references using cloudpickle."""
        from pipefunc._error_handling import cloudpickle_function_state

        state = cloudpickle_function_state(self.__dict__.copy(), "skipped_function")
        # Also handle nested ErrorSnapshots in error_info
        state["error_info"] = self._pickle_error_info(self.error_info)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickling to restore function references."""
        from pipefunc._error_handling import cloudunpickle_function_state

        state = cloudunpickle_function_state(state, "skipped_function")
        # Restore error_info
        state["error_info"] = self._unpickle_error_info(state["error_info"])
        self.__dict__.update(state)

    def _pickle_error_info(
        self,
        error_info: dict[str, ErrorInfo],
    ) -> dict[str, dict[str, Any]]:
        """Helper to pickle error_info dict that may contain ErrorSnapshots."""
        pickled_info = {}
        for param, info in error_info.items():
            # Convert ErrorInfo to dict for pickling
            info_dict = {
                "type": info.type,
                "shape": info.shape,
                "error_indices": info.error_indices,
                "error_count": info.error_count,
            }
            if info.type == "full" and info.error is not None:
                # The error might be an ErrorSnapshot or PropagatedErrorSnapshot
                # Let their own __getstate__ handle it
                info_dict["error"] = cloudpickle.dumps(info.error)
            pickled_info[param] = info_dict
        return pickled_info

    def _unpickle_error_info(
        self,
        pickled_info: dict[str, dict[str, Any]],
    ) -> dict[str, ErrorInfo]:
        """Helper to unpickle error_info dict."""
        from pipefunc._error_handling import ErrorInfo

        error_info = {}
        for param, info_dict in pickled_info.items():
            if info_dict["type"] == "full" and "error" in info_dict:
                serialized_error = info_dict["error"]
                if isinstance(serialized_error, bytes):
                    error = cloudpickle.loads(serialized_error)
                else:
                    # NOTE: v0.88 stores serialized bytes, while
                    # older snapshots stored the ErrorSnapshot directly.
                    error = serialized_error
                error_info[param] = ErrorInfo.from_full_error(error)
            else:
                error_info[param] = ErrorInfo(
                    type=info_dict["type"],
                    shape=info_dict.get("shape"),
                    error_indices=info_dict.get("error_indices"),
                    error_count=info_dict.get("error_count"),
                )
        return error_info

    def get_root_causes(self) -> list[ErrorSnapshot]:
        """Extract all original ErrorSnapshot objects (for full-error inputs).

        For array-containing errors (reductions), this currently returns an
        empty list. Downstream code can still rely on `reason` and
        `error_info` metadata to understand which parameters contained errors.
        """
        root_causes: list[ErrorSnapshot] = []
        for info in self.error_info.values():
            if info.type == "full" and info.error is not None:
                if isinstance(info.error, PropagatedErrorSnapshot):
                    root_causes.extend(info.error.get_root_causes())
                elif isinstance(info.error, ErrorSnapshot):
                    root_causes.append(info.error)
        return root_causes
