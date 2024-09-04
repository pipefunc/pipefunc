from __future__ import annotations

import contextlib
import functools
import inspect
import math
import operator
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def load(path: Path, *, cache: bool = False) -> Any:
    """Load a cloudpickled object from a path.

    If ``cache`` is ``True``, the object will be cached in memory.
    """
    if cache:
        cache_key = _get_cache_key(path)
        return _cached_load(cache_key)

    with path.open("rb") as f:
        return cloudpickle.load(f)


def dump(obj: Any, path: Path) -> None:
    """Dump an object to a path using cloudpickle."""
    with path.open("wb") as f:
        cloudpickle.dump(obj, f)


def _get_cache_key(path: Path) -> tuple:
    """Generate a cache key based on the path, file modification time, and file size."""
    resolved_path = path.resolve()
    stats = resolved_path.stat()
    return (str(resolved_path), stats.st_mtime, stats.st_size)


@functools.lru_cache(maxsize=128)
def _cached_load(cache_key: tuple) -> Any:
    """Load a cloudpickled object using a cache key."""
    path = Path(cache_key[0])
    return load(path, cache=False)


def format_kwargs(kwargs: dict[str, Any]) -> str:
    """Format kwargs as a string."""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def format_args(args: tuple) -> str:
    """Format args as a string."""
    return ", ".join(repr(arg) for arg in args)


def format_function_call(func_name: str, args: tuple, kwargs: dict[str, Any]) -> str:
    """Format a function call as a string."""
    if args and kwargs:
        return f"{func_name}({format_args(args)}, {format_kwargs(kwargs)})"
    if args:
        return f"{func_name}({format_args(args)})"
    if kwargs:
        return f"{func_name}({format_kwargs(kwargs)})"
    return f"{func_name}()"


def handle_error(e: Exception, func: Callable, kwargs: dict[str, Any]) -> None:
    """Handle an error that occurred while executing a function."""
    call_str = format_function_call(func.__name__, (), kwargs)
    msg = f"Error occurred while executing function `{call_str}`."
    if sys.version_info <= (3, 11):  # pragma: no cover
        raise type(e)(e.args[0] + msg) from e
    e.add_note(msg)
    raise  # noqa: PLE0704


def prod(iterable: Iterable[int]) -> int:
    """Return the product of an iterable."""
    return functools.reduce(operator.mul, iterable, 1)


def _is_equal(a: Any, b: Any) -> bool | None:  # noqa: PLR0911
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return equal_dicts(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, set):
        return a == b
    if isinstance(a, float | np.floating):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    if isinstance(a, str):
        return a == b
    if isinstance(a, list | tuple):
        if len(a) != len(b):  # type: ignore[arg-type]
            return False
        return all(_is_equal(x, y) for x, y in zip(a, b))
    return a == b


def equal_dicts(d1: dict[str, Any], d2: dict[str, Any], *, verbose: bool = False) -> bool | None:
    """Check if two dictionaries are equal.

    Returns True if the dictionaries are equal, False if they are not equal,
    and None if there are errors comparing keys and values.
    """
    if len(d1) != len(d2):
        if verbose:
            print(f"Not equal lengths: `{len(d1)} != {len(d2)}`")
        return False

    if d1.keys() != d2.keys():
        if verbose:
            print(f"Not equal keys: `{d1.keys()} != {d2.keys()}`")
        return False

    errors = []
    for k, v1 in d1.items():
        v2 = d2[k]
        try:
            equal = _is_equal(v1, v2)
        except Exception:  # noqa: BLE001
            errors.append((k, v1, v2))
        else:
            if not equal:
                if verbose:
                    print(f"Not equal `{k}`: `{v1} != {v2}`")
                return False
    if errors:
        warnings.warn(f"Errors comparing keys and values: {errors}", stacklevel=3)
        return None
    return True


def _format_table_row(row: list[str], widths: list[int], seperator: str = " | ") -> str:
    """Format a row of the table with specified column widths."""
    return seperator.join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row))


def table(rows: list[Any], headers: list[str]) -> str:
    """Create a printable table from a list of rows and headers."""
    column_widths = [len(header) for header in headers]
    for row in rows:
        for i, x in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(x)))

    separator_line = [w * "-" for w in column_widths]
    table_rows = [
        _format_table_row(separator_line, column_widths, seperator="-+-"),
        _format_table_row(headers, column_widths),
        _format_table_row(["-" * width for width in column_widths], column_widths),
    ]
    for row in rows:
        table_rows.append(_format_table_row(row, column_widths))  # noqa: PERF401
    table_rows.append(_format_table_row(separator_line, column_widths, seperator="-+-"))

    return "\n".join(table_rows)


def clear_cached_properties(obj: object, until_type: type | None = None) -> None:
    """Clear all `functools.cached_property`s from an object and its super types."""
    cls = type(obj)
    if until_type is None:
        until_type = cls
    while True:
        for k, v in cls.__dict__.items():
            if isinstance(v, functools.cached_property):
                with contextlib.suppress(AttributeError):
                    delattr(obj, k)
        if cls is object or cls is until_type or cls.__base__ is None:
            break
        cls = cls.__base__


def assert_complete_kwargs(
    kwargs: dict[str, Any],
    function: Callable[..., Any],
    skip: set[str] | None = None,
) -> None:
    """Validate that the kwargs contain all kwargs for a function."""
    valid_kwargs = set(inspect.signature(function).parameters.keys())
    if skip is not None:
        valid_kwargs -= set(skip)
    missing = valid_kwargs - set(kwargs)
    assert not missing, f"Missing required kwargs: {missing}"
