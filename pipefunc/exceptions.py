"""Custom exceptions for the pipefunc package."""


class UnusedParametersError(ValueError):
    """Exception raised when unused parameters are provided to a function."""
