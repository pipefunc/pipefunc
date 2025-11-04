"""Provides `pipefunc.helpers` module with various tools."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple, dump, is_running_in_ipynb, load, requires
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ipywidgets import Widget

    from pipefunc import PipeFunc
    from pipefunc._widgets.output_tabs import OutputTabs
    from pipefunc.map._result import ResultDict
    from pipefunc.map._run import AsyncMap

__all__ = [
    "FileArray",  # To keep in the same namespace as FileValue
    "FileValue",
    "chain",
    "collect_kwargs",
    "gather_maps",
    "get_attribute_factory",
    "launch_maps",
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


def _setup_automatic_tab_updates(index_output: int, tabs: OutputTabs, async_map: AsyncMap) -> None:
    def create_callback() -> Callable[[asyncio.Task[ResultDict]], None]:
        def callback(task: asyncio.Task[ResultDict]) -> None:
            if task.exception() is not None:
                tabs.set_tab_status(index_output, "failed")
            else:
                tabs.set_tab_status(index_output, "completed")

        return callback

    # Set initial status to running and add callbacks
    tabs.set_tab_status(index_output, "running")
    async_map.task.add_done_callback(create_callback())


async def gather_maps(
    *async_maps: AsyncMap,
    max_concurrent: int = 1,
    max_completed_tabs: int | None = None,
    _tabs: OutputTabs | None = None,
) -> list[ResultDict]:
    """Run AsyncMap objects with a limit on simultaneous executions.

    Parameters
    ----------
    async_maps
        `AsyncMap` objects created with ``pipeline.map_async(..., start=False)``.
    max_concurrent
        Maximum number of concurrent jobs
    max_completed_tabs
        Maximum number of completed tabs to show. If ``None``, all completed tabs
        are shown. Only used if ``display_widgets=True``.

    Returns
    -------
        List of results from each AsyncMap's task

    """
    _validate_async_maps(async_maps)
    for async_map in async_maps:
        if async_map._task is not None:
            msg = "`pipeline.map_async(..., start=False)` must be called before `launch_maps`."
            raise RuntimeError(msg)

    if _tabs is None:  # Prefer to get from the caller (in sync context), otherwise create it here
        _tabs = _maybe_output_tabs(async_maps, max_completed_tabs)
    else:
        _tabs._max_completed_tabs = max_completed_tabs

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(index: int, async_map: AsyncMap) -> ResultDict:
        async with semaphore:
            if _tabs is not None and async_map._display_widgets:
                # Cannot use output_context here, because it is not thread-safe
                # See https://github.com/jupyter-widgets/ipywidgets/issues/3993
                from pipefunc._widgets.progress_ipywidgets import IPyWidgetsProgressTracker

                # Disable `display` on the first call to `start`
                async_map._display_widgets = False
                async_map.start()
                widgets = []
                if async_map.status_widget is not None:  # pragma: no cover
                    widgets.append(async_map.status_widget.widget)
                if isinstance(async_map.progress, IPyWidgetsProgressTracker):
                    widgets.append(async_map.progress._style())
                    widgets.append(async_map.progress._widgets)
                elif async_map.progress is not None:  # pragma: no cover
                    msg = "Only `show_progress='ipywidgets'` is supported in this tab widget."
                    widgets.append(msg)
                if async_map.multi_run_manager is not None:  # pragma: no cover
                    widgets.append(async_map.multi_run_manager.info())
                for widget in widgets:
                    _register_widget(widget)
                    _tabs.output(index).append_display_data(widget)
                if widgets:
                    _tabs.show_output(index)
                _setup_automatic_tab_updates(index, _tabs, async_map)
            else:
                async_map.start()

            return await async_map.task

    tasks = [run_with_semaphore(index, async_map) for index, async_map in enumerate(async_maps)]
    return await asyncio.gather(*tasks)


def _maybe_output_tabs(
    async_maps: Sequence[AsyncMap],
    max_completed_tabs: int | None,
) -> OutputTabs | None:
    display_widgets = any(async_map._display_widgets for async_map in async_maps)
    has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
    if has_ipywidgets and display_widgets and is_running_in_ipynb():
        requires("ipywidgets", reason="tab_widget=True", extras="widgets")
        from pipefunc._widgets.output_tabs import OutputTabs

        if max_completed_tabs and os.environ.get("VSCODE_PID") is not None:  # pragma: no cover
            warnings.warn(
                "`max_completed_tabs` is buggy in VS Code Jupyter notebook environment.",
                stacklevel=2,
            )

        tabs = OutputTabs(len(async_maps), max_completed_tabs)
        tabs.display()
        return tabs
    return None


def _register_widget(widget: Widget) -> None:  # pragma: no cover
    """Register widget in VS Code to work around widget rendering bug.

    This is a workaround for VS Code Jupyter notebook environment where
    widgets created and immediately used in append_display_data() without
    being displayed first don't get properly registered in the widget state.

    See: https://github.com/microsoft/vscode-jupyter/issues/16739
    """
    if os.environ.get("VSCODE_PID") is None:
        return

    from IPython.display import display
    from ipywidgets import Output

    with Output():
        display(widget)


def launch_maps(
    *async_maps: AsyncMap,
    max_concurrent: int = 1,
    max_completed_tabs: int | None = None,
) -> asyncio.Task[list[ResultDict]]:
    """Launch a collection of map operations to run concurrently in the background.

    This is a user-friendly, non-blocking wrapper around ``gather_maps``.
    It immediately returns an ``asyncio.Task`` object, which can be awaited
    later to retrieve the results. This is ideal for use in interactive
    environments like Jupyter notebooks.

    Parameters
    ----------
    async_maps
        `AsyncMap` objects created with ``pipeline.map_async(..., start=False)``.
    max_concurrent
        Maximum number of map operations to run at the same time.
    max_completed_tabs
        Maximum number of completed tabs to show. If ``None``, all completed tabs
        are shown. Only used if ``display_widgets=True``.

    Returns
    -------
    asyncio.Task
        A task handle representing the background execution of the maps.
        ``await`` this task to get the list of results.

    Examples
    --------
    >>> # In a Jupyter notebook cell:
    >>> task = launch_maps(runners, max_concurrent=2)

    >>> # In a later cell:
    >>> results = await task
    >>> print("Computation finished!")

    """
    _validate_async_maps(async_maps)
    tabs = _maybe_output_tabs(async_maps, max_completed_tabs)
    coro = gather_maps(
        *async_maps,
        max_concurrent=max_concurrent,
        max_completed_tabs=max_completed_tabs,
        _tabs=tabs,
    )
    return asyncio.create_task(coro)


def chain(
    functions: Sequence[PipeFunc | Callable],
    *,
    copy: bool = True,
) -> list[PipeFunc]:
    """Return a new list of PipeFuncs connected linearly by applying minimal renames.

    The i+1-th function's first parameter is renamed to the i-th function's output name,
    creating a linear data flow. Other parameters (including additional inputs) are untouched.

    Parameters
    ----------
    functions
        Sequence of PipeFuncs (or callables). Callables are wrapped as PipeFuncs with
        ``output_name=f.__name__``.
    copy
        If True (default), return copies of the input PipeFuncs; original instances are
        not modified.

    Returns
    -------
    list[PipeFunc]
        New PipeFunc objects with renames applied so the data flows linearly.

    Notes
    -----
    - If a downstream function already has an *unbound* parameter matching an upstream
      output name, no rename is applied (prefer existing matches).
    - When no explicit match exists, the first parameter is renamed to the upstream output.
      The first parameter must not be bound; if it is, a ValueError is raised.
    - If a function has zero parameters (and is not the first in the chain), a ValueError
      is raised.

    """
    from pipefunc import PipeFunc as _PipeFunc  # local import to avoid cyclic in typing

    if not functions:
        msg = "chain requires at least one function"
        raise ValueError(msg)

    # Normalize to PipeFunc instances
    pfs: list[_PipeFunc] = []
    for f in functions:
        pf = (
            f
            if isinstance(f, _PipeFunc)
            else _PipeFunc(f, output_name=getattr(f, "__name__", "output"))
        )
        pfs.append(pf.copy() if copy else pf)

    # Nothing to connect if only one
    if len(pfs) == 1:
        return pfs

    # Apply renames to connect each pair
    upstream = pfs[0]
    for downstream in pfs[1:]:
        upstream_outputs = at_least_tuple(upstream.output_name)
        free_params = [p for p in downstream.parameters if p not in downstream.bound]

        # Prefer existing matches among free parameters
        if any(name in free_params for name in upstream_outputs):
            upstream = downstream
            continue

        # No explicit match - validate and rename first parameter
        if not downstream.parameters:
            msg = f"Function {downstream} has no parameters to receive upstream value."
            raise ValueError(msg)

        if not free_params:
            msg = f"All parameters of {downstream} are bound; cannot auto-select input parameter."
            raise ValueError(msg)

        # Require first parameter to be non-bound for auto-selection
        first_param = downstream.parameters[0]
        if first_param in downstream.bound:
            upstream_out = upstream_outputs[0]
            msg = (
                f"chain: First parameter '{first_param}' of {downstream.output_name} is bound.\n"
                f"Solution: Either reorder parameters to put the data-flow parameter first,\n"
                f"or rename a parameter to '{upstream_out}' to create an explicit match."
            )
            raise ValueError(msg)

        # Rename first parameter to upstream output
        desired_name = upstream_outputs[0]
        downstream.update_renames({first_param: desired_name}, update_from="current")

        upstream = downstream

    return pfs


def _validate_async_maps(async_maps: Sequence[AsyncMap]) -> None:
    caller_name = inspect.stack()[1].function
    _validate_async_maps_length(async_maps, caller_name)
    _validate_unique_run_folders(async_maps, caller_name)
    _validate_slurm_executor_names(async_maps, caller_name)


def _validate_async_maps_length(async_maps: Sequence[AsyncMap], caller_name: str) -> None:
    if len(async_maps) == 0:
        msg = f"`{caller_name}` requires at least one `AsyncMap` object."
        raise ValueError(msg)
    if len(async_maps) == 1 and isinstance(async_maps[0], tuple | list):
        msg = (
            f"It seems you passed a list or tuple of `AsyncMap` objects as a single argument to `{caller_name}`. "
            "Instead, you should unpack the sequence into individual arguments. "
            f"For example, use `{caller_name}(*my_async_maps)` instead of `{caller_name}(my_async_maps)`."
        )
        raise ValueError(msg)


def _validate_unique_run_folders(async_maps: Sequence[AsyncMap], caller_name: str) -> None:
    run_folders = [
        am.run_info.run_folder for am in async_maps if am.run_info.run_folder is not None
    ]
    if len(run_folders) != len(set(run_folders)):
        msg = (
            f"All `run_folder`s must be unique among the provided `AsyncMap` objects in `{caller_name}` "
            "unless they are None."
        )
        raise ValueError(msg)


def _validate_slurm_executor_names(async_maps: Sequence[AsyncMap], caller_name: str) -> None:
    from pipefunc.map._adaptive_scheduler_slurm_executor import is_slurm_executor

    cnt: Counter[str] = Counter()
    for am in async_maps:
        executors = am._prepared.executor
        if executors is None:
            continue
        for v in executors.values():
            if is_slurm_executor(v):
                cnt[v.name] += 1
    violations = [name for name, count in cnt.items() if count > 1]
    if violations:
        msg = (
            f"All `map_async`s provided to `{caller_name}` that use a `SlurmExecutor`"
            " must have instances with a unique `name`."
            f" Currently, the following names are used multiple times: {violations}."
            " Use `SlurmExecutor(name=...)` to set a unique name."
        )
        raise ValueError(msg)
