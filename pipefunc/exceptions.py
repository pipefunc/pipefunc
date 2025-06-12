"""Custom exceptions for the pipefunc package."""

from __future__ import annotations

import functools
from typing import Any

import cloudpickle


class UnusedParametersError(ValueError):
    """Exception raised when unused parameters are provided to a function."""


class PipeFuncError(Exception):
    """Wrapper exception that has the original exception and some data."""

    def __init__(
        self,
        original_exception: Exception,
        data: Any,
    ) -> None:
        self.original_exception = original_exception
        self._data_bytes = cloudpickle.dumps(data)

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return self.original_exception.__str__()

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        return self.original_exception.__repr__()

    @functools.cached_property
    def data(self) -> Any:
        """Return the data that caused the error."""
        return cloudpickle.loads(self._data_bytes)
