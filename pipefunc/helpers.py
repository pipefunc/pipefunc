"""Provides `pipefunc.helpers` module with various tools."""

import inspect
from collections.abc import Callable
from typing import Any


class _ReturnsKwargs:
    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Returns keyword arguments it receives as a dictionary."""
        return kwargs


def collect_kwargs(
    parameters: tuple[str, ...],
    *,
    annotations: tuple[type, ...] | None = None,
    function_name: str = "call",
) -> Callable[..., dict[str, Any]]:
    """Returns a callable with a signature as specified in ``parameters`` which returns a dict.

    Parameters
    ----------
    parameters
        Tuple of names, these names will be used for the function parameters.
    annotations
        Optionally, provide type-annotations for the ``parameters``. Must be
        the same length as ``parameters`` or ``None``.
    function_name
        The ``__name__`` that is assigned to the returned callable.

    Returns
    -------
        Callable that returns the parameters in a dictionary.

    Examples
    --------
    This creates `def yolo(a: int, b: list[int]) -> dict[str, Any]`:
    >>> f = collect_kwargs(("a", "b"), annotations=(int, list[int]), function_name="yolo")
    >>> f(a=1, b=2)
    {"a": 1, "b": 2}

    """
    cls = _ReturnsKwargs()
    sig = inspect.signature(cls.__call__)
    if annotations is None:
        annotations = (inspect.Parameter.empty,) * len(parameters)
    elif len(parameters) != len(annotations):
        msg = f"`parameters` and `annotations` should have equal length ({len(parameters)}!={len(annotations)})"
        raise ValueError(msg)
    new_params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
        for name, annotation in zip(parameters, annotations)
    ]
    new_sig = sig.replace(parameters=new_params)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = new_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return cls(**bound.arguments)

    _wrapped.__signature__ = new_sig  # type: ignore[attr-defined]
    _wrapped.__name__ = function_name
    return _wrapped
