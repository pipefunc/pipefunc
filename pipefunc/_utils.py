from __future__ import annotations

import contextlib
import functools
import importlib.util
import inspect
import logging
import math
import operator
import socket
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, TypeVar, get_args

import cloudpickle
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    import pydantic
    from griffe import DocstringSection


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if sys.version_info < (3, 11):  # pragma: no cover
        original_msg = e.args[0] if e.args else ""
        raise type(e)(original_msg + msg) from e
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


def get_local_ip() -> str:
    try:
        # Create a socket to connect to a remote host
        # This helps in getting the network interface's IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # This does not actually connect to '8.8.8.8', it is simply used to find the local IP
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001  # pragma: no cover
        return "unknown"


def is_running_in_ipynb() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
    except NameError:  # pragma: no cover
        return False  # Probably standard Python interpreter


def is_installed(package: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(package) is not None


def requires(*packages: str, reason: str = "", extras: str | None = None) -> None:
    """Check if a package is installed, raise an ImportError if not."""
    conda_name_mapping = {"graphviz": "python-graphviz", "graphviz_anywidget": "graphviz-anywidget"}

    for package in packages:
        if is_installed(package):
            continue
        conda_package = conda_name_mapping.get(package, package)
        error_message = f"The '{package}' package is required"
        if reason:
            error_message += f" for {reason}"
        error_message += ".\n"
        error_message += "Please install it using one of the following methods:\n"
        if extras:
            error_message += f'- pip install "pipefunc[{extras}]"\n'
        error_message += f"- pip install {package}\n"
        error_message += f"- conda install -c conda-forge {conda_package}"
        raise ImportError(error_message)


def is_min_version(package: str, version: str) -> bool:
    """Check if a package is at least a given version."""
    import importlib.metadata

    installed_version = importlib.metadata.version(package)
    installed_major, installed_minor, installed_patch, *_ = installed_version.split(".")
    major, minor, patch, *_ = version.split(".")
    if installed_major < major:
        return False
    if installed_major == major and installed_minor < minor:
        return False
    if installed_major == major and installed_minor == minor and installed_patch < patch:  # noqa: SIM103
        return False
    return True


def is_pydantic_base_model(x: Any) -> TypeGuard[type[pydantic.BaseModel]]:
    if not is_imported("pydantic"):  # pragma: no cover
        return False
    if not inspect.isclass(x):
        return False
    import pydantic

    return issubclass(x, pydantic.BaseModel)


T = TypeVar("T")


def first(x: T | tuple[T, ...]) -> T:
    if isinstance(x, tuple):  # pragma: no cover
        return x[0]
    return x


def is_imported(package: str) -> bool:
    """Check if a package is imported."""
    return package in sys.modules


def get_ncores(ex: Executor) -> int:
    """Return the maximum number of cores that an executor can use."""
    if isinstance(ex, ProcessPoolExecutor | ThreadPoolExecutor):
        return ex._max_workers  # type: ignore[union-attr]
    if is_imported("ipyparallel"):  # pragma: no cover
        import ipyparallel

        if isinstance(ex, ipyparallel.client.view.ViewExecutor):
            return len(ex.view)
    if is_imported("loky"):  # pragma: no cover
        import loky

        if isinstance(ex, loky.reusable_executor._ReusablePoolExecutor):
            return ex._max_workers
    if is_imported("distributed"):  # pragma: no cover
        import distributed

        if isinstance(ex, distributed.cfexecutor.ClientExecutor):
            return sum(n for n in ex._client.ncores().values())
    if is_imported("mpi4py"):  # pragma: no cover
        import mpi4py.futures

        if isinstance(ex, mpi4py.futures.MPIPoolExecutor):
            ex.bootup()  # wait until all workers are up and running
            return ex._pool.size  # not public API!
    if is_imported("adaptive_scheduler"):
        import adaptive_scheduler

        if isinstance(ex, adaptive_scheduler.SlurmExecutor):
            # This could be better but since there is `cores`, `cores_per_node`,
            # and `nodes`; and they can be `None`, we just return 1 for now.
            return 1
    msg = f"Cannot get number of cores for {ex.__class__}"
    raise TypeError(msg)


@contextlib.contextmanager
def temporarily_disable_logger(logger_name: str) -> Generator[None, None, None]:
    """Temporarily disable a logger within a context manager scope.

    Upon entering, disables the specified logger.
    Upon exiting, restores the logger to its original enabled/disabled state.

    Parameters
    ----------
    logger_name
        Name of the logger to temporarily disable

    Examples
    --------
    >>> with temporarily_disable_logger("my_logger"):
    ...     # Logger is disabled here
    ...     perform_noisy_operation()
    ... # Logger is restored to original state here

    """
    logger = logging.getLogger(name=logger_name)
    original_state = logger.disabled
    try:
        logger.disabled = True
        yield
    finally:
        logger.disabled = original_state


DocstringStyle = Literal["google", "numpy", "sphinx", "auto"]


def _parse_docstring_sections(
    docstring: str,
    docstring_parser: DocstringStyle,
) -> list[DocstringSection]:
    requires("griffe", reason="extracting docstrings", extras="autodoc")
    from griffe import Docstring, Parser

    options = get_args(DocstringStyle)
    if docstring_parser == "auto":
        # Poor man's "auto" parser selection because griffe has this as a paid feature
        # https://mkdocstrings.github.io/griffe-autodocstringstyle/insiders/
        results = [
            _parse_docstring_sections(docstring, parser)  # type: ignore[arg-type]
            for parser in options
            if parser != "auto"
        ]
        return max(results, key=len)

    if docstring_parser not in options:
        msg = f"Invalid docstring parser: {docstring_parser}, must be one of {', '.join(options)}"
        raise ValueError(msg)

    parser = Parser(docstring_parser)
    with temporarily_disable_logger("griffe"):
        return Docstring(docstring).parse(parser)


@dataclass
class DocstringInfo:
    """A class to store a function's docstring and its extracted parameter docstrings."""

    description: str | None
    parameters: dict[str, str]
    returns: str | None


def parse_function_docstring(
    func: Callable[..., Any],
    docstring_parser: DocstringStyle = "auto",
) -> DocstringInfo:
    """Parse a function's docstring into structured components.

    Extracts the main description, parameter descriptions, and return description
    from a function's docstring. Supports Google, NumPy, and standard Python
    docstring formats using the `griffe` library for parsing.

    Parameters
    ----------
    func
        The function whose docstring should be parsed.
    docstring_parser
        The docstring style to use for parsing. Can be 'google', 'numpy',
        'sphinx', or 'auto' to automatically detect the style.

    Returns
    -------
        A structured representation of the docstring containing the main description,
        parameter descriptions, and return description.

    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return DocstringInfo(None, {}, None)

    parameters: dict[str, str] = {}
    returns: list[str] = []
    description: list[str] = []
    sections = _parse_docstring_sections(docstring, docstring_parser)
    for section in sections:
        if section.kind.name == "parameters":
            for parameter in section.value:
                parameters[parameter.name] = parameter.description
        if section.kind.name == "returns":
            for return_value in section.value:
                if return_value.description or return_value.annotation:
                    # If numpy style without types, the description is the annotation
                    value = return_value.description or return_value.annotation.lstrip()
                    returns.append(value)
        if section.kind.name == "text":
            description.append(section.value)

    return DocstringInfo("\n".join(description), parameters, "\n".join(returns))


def is_classmethod(func: Callable) -> bool:
    """Check if a function is a classmethod."""
    return inspect.ismethod(func) and func.__self__ is not None
