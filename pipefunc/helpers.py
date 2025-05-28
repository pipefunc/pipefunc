"""Provides `pipefunc.helpers` module with various tools."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import dump, load
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FileArray",  # To keep in the same namespace as FileValue
    "FileValue",
    "collect_kwargs",
    "get_attribute_factory",
]


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
    This creates ``def yolo(a: int, b: list[int]) -> dict[str, Any]``:

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


def get_attribute_factory(
    attribute_name: str,
    parameter_name: str,
    parameter_annotation: type = inspect.Parameter.empty,
    return_annotation: type = inspect.Parameter.empty,
    function_name: str = "get_attribute",
) -> Callable[[Any], Any]:
    """Returns a callable that retrieves an attribute from its input parameter.

    Parameters
    ----------
    attribute_name
        The name of the attribute to access.
    parameter_name
        The name of the input parameter.
    parameter_annotation
        Optional, type annotation for the input parameter.
    return_annotation
        Optional, type annotation for the return value.
    function_name
        The ``__name__`` that is assigned to the returned callable.

    Returns
    -------
        Callable that returns an attribute of its input parameter.

    Examples
    --------
    This creates ``def get_data(obj: MyClass) -> int``:

    >>> class MyClass:
    ...     def __init__(self, data: int) -> None:
    ...         self.data = data
    >>> f = get_attribute_factory("data", parameter_name="obj", parameter_annotation=MyClass, return_annotation=int, function_name="get_data")
    >>> f(MyClass(data=123))
    123

    """
    param = inspect.Parameter(
        parameter_name,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=parameter_annotation,
    )
    sig = inspect.Signature(parameters=[param], return_annotation=return_annotation)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        obj = bound.arguments[parameter_name]
        return getattr(obj, attribute_name)

    _wrapped.__signature__ = sig  # type: ignore[attr-defined]
    _wrapped.__name__ = function_name
    return _wrapped


class FileValue:
    """A reference to a value stored in a file.

    This class provides a way to store and load values from files, which is useful
    for passing large objects between processes without serializing them directly.

    Parameters
    ----------
    path
        Path to the file containing the serialized value.

    Examples
    --------
    >>> ref = FileValue.from_data([1, 2, 3], Path("data.pkl"))
    >>> ref.load()
    [1, 2, 3]

    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).absolute()

    def load(self) -> Any:
        """Load the stored data from disk."""
        return load(self.path)

    @classmethod
    def from_data(cls, data: Any, path: Path) -> FileValue:
        """Serializes data to the given file path and returns a FileValue to it.

        This is useful for preparing a single large, non-iterable object
        for use with `pipeline.map` in distributed environments.
        The object is stored once on disk, and the lightweight FileValue
        can be passed to tasks, which then load the data on demand.

        Parameters
        ----------
        data
            The Python object to serialize and store.
        path
            The full file path (including filename) where the data will be stored.
            This path must be accessible by all worker nodes if used in
            a distributed setting.

        Returns
        -------
        FileValue
            A new FileValue instance pointing to the stored data.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(data, path)
        return cls(path=path)
