"""Generator-based ScanFunc implementation for more Pythonic iteration patterns."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from pipefunc._pipefunc import PipeFunc

if TYPE_CHECKING:
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._mapspec import MapSpec
    from pipefunc.map._types import ShapeTuple
    from pipefunc.resources import Resources

T = TypeVar("T", bound=Callable[..., Any])


class ScanIterFunc(PipeFunc[T]):
    """A PipeFunc subclass for generator-based iterative execution.

    ScanIterFunc enables natural Python generator patterns for iterative
    algorithms. Instead of the complex carry dict and tuple returns,
    you can use regular Python generators with yield statements.

    Parameters
    ----------
    func
        A generator function that yields outputs for each iteration.
        The function receives all inputs directly (no signature transformation).
    output_name
        The identifier for the output of the scan operation.
    xs
        Optional name of the parameter containing the list/array to iterate over.
        If not provided, the generator should handle iteration internally.
    return_final_only
        If True, only return the final yielded value instead of collecting all.
    output_picker
        Function to pick specific outputs, inherited from PipeFunc.
    renames
        Parameter renames, inherited from PipeFunc.
    defaults
        Default parameter values, inherited from PipeFunc.
    bound
        Bound parameters, inherited from PipeFunc.
    profile
        Whether to profile execution, inherited from PipeFunc.
    debug
        Whether to print debug info, inherited from PipeFunc.
    print_error
        Whether to print errors, inherited from PipeFunc.
    cache
        Whether to cache results, inherited from PipeFunc.
    mapspec
        MapSpec for the scan operation, inherited from PipeFunc.
    internal_shape
        Shape of internal arrays, inherited from PipeFunc.
    post_execution_hook
        Hook called after execution, inherited from PipeFunc.
    resources
        Resource requirements, inherited from PipeFunc.
    resources_variable
        Variable name for resources, inherited from PipeFunc.
    resources_scope
        Scope for resources, inherited from PipeFunc.
    scope
        Parameter scope, inherited from PipeFunc.
    variant
        Variant specification, inherited from PipeFunc.

    Examples
    --------
    >>> @PipeFunc.scan_iter(output_name="cumsum")
    ... def accumulator(values: list[int], total: int = 0):
    ...     for x in values:
    ...         total += x
    ...         yield total

    """

    def __init__(
        self,
        func: T,
        output_name: OUTPUT_TYPE,
        *,
        xs: str | None = None,
        return_final_only: bool = False,
        output_picker: Callable[[Any, str], Any] | None = None,
        renames: dict[str, str] | None = None,
        defaults: dict[str, Any] | None = None,
        bound: dict[str, Any] | None = None,
        profile: bool = False,
        debug: bool = False,
        print_error: bool = True,
        cache: bool = False,
        mapspec: str | MapSpec | None = None,
        internal_shape: int | Literal["?"] | ShapeTuple | None = None,
        post_execution_hook: Callable[[PipeFunc, Any, dict[str, Any]], None] | None = None,
        resources: dict
        | Resources
        | Callable[[dict[str, Any]], Resources | dict[str, Any]]
        | None = None,
        resources_variable: str | None = None,
        resources_scope: Literal["map", "element"] = "map",
        scope: str | None = None,
        variant: str | dict[str | None, str] | None = None,
    ) -> None:
        """Initialize a ScanIterFunc instance."""
        # Store scan-specific attributes
        self.xs = xs
        self.return_final_only = return_final_only
        self._generator_func = func

        # Create wrapper function
        wrapper_func = self._create_wrapper()

        # Initialize parent PipeFunc
        super().__init__(
            func=wrapper_func,  # type: ignore[arg-type]
            output_name=output_name,
            output_picker=output_picker,
            renames=renames,
            defaults=defaults,
            bound=bound,
            profile=profile,
            debug=debug,
            print_error=print_error,
            cache=cache,
            mapspec=mapspec,
            internal_shape=internal_shape,
            post_execution_hook=post_execution_hook,
            resources=resources,
            resources_variable=resources_variable,
            resources_scope=resources_scope,
            scope=scope,
            variant=variant,
        )

    def _create_wrapper(self) -> Callable[..., Any]:
        """Create the wrapper function that collects generator results."""

        @functools.wraps(self._generator_func)
        def wrapper(**kwargs: Any) -> Any:
            # If xs is specified, extract it from kwargs
            if self.xs is not None and self.xs not in kwargs:
                msg = f"Required parameter '{self.xs}' not provided"
                raise ValueError(msg)
                # For compatibility, we keep xs in kwargs
                # The generator function should handle it appropriately

            # Call the generator function
            gen = self._generator_func(**kwargs)

            # Ensure we got a generator
            if not inspect.isgenerator(gen):
                msg = f"Function {self._generator_func.__name__} must be a generator (use yield)"
                raise TypeError(msg)

            # Collect results
            results = []
            last_value = None

            for value in gen:
                last_value = value
                if not self.return_final_only:
                    results.append(value)

            # Return appropriate result
            if self.return_final_only:
                return last_value

            if not results:
                return np.array([])

            # Convert to numpy array if all results are numeric
            if all(isinstance(r, (int, float)) for r in results):
                return np.array(results)

            return results

        # Preserve the original function's signature
        wrapper.__signature__ = inspect.signature(self._generator_func)  # type: ignore[attr-defined]

        return wrapper

    @property
    def generator_func(self) -> Callable[..., Generator]:
        """Access the underlying generator function for testing/debugging."""
        return self._generator_func

    def copy(self, **update: Any) -> ScanIterFunc:
        """Create a copy of the ScanIterFunc instance."""
        # Get current configuration
        kwargs = {
            "func": self._generator_func,
            "output_name": self._output_name,
            "xs": self.xs,
            "return_final_only": self.return_final_only,
            "output_picker": self._output_picker,
            "renames": self._renames,
            "defaults": self._defaults,
            "bound": self._bound,
            "profile": self._profile,
            "debug": self.debug,
            "print_error": self.print_error,
            "cache": self.cache,
            "mapspec": self.mapspec,
            "internal_shape": self.internal_shape,
            "post_execution_hook": self.post_execution_hook,
            "resources": self.resources,
            "resources_variable": self.resources_variable,
            "resources_scope": self.resources_scope,
            "scope": None,  # Let new instance handle scope
            "variant": self.variant,
        }

        # Apply updates
        kwargs.update(update)

        # Create new instance
        return ScanIterFunc(**kwargs)  # type: ignore[arg-type]


def scan_iter(
    output_name: OUTPUT_TYPE,
    *,
    xs: str | None = None,
    return_final_only: bool = False,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    bound: dict[str, Any] | None = None,
    profile: bool = False,
    debug: bool = False,
    print_error: bool = True,
    cache: bool = False,
    mapspec: str | MapSpec | None = None,
    internal_shape: int | Literal["?"] | ShapeTuple | None = None,
    post_execution_hook: Callable[[PipeFunc, Any, dict[str, Any]], None] | None = None,
    resources: dict
    | Resources
    | Callable[[dict[str, Any]], Resources | dict[str, Any]]
    | None = None,
    resources_variable: str | None = None,
    resources_scope: Literal["map", "element"] = "map",
    scope: str | None = None,
    variant: str | dict[str | None, str] | None = None,
) -> Callable[[Callable[..., Generator]], ScanIterFunc]:
    """Decorator to create a ScanIterFunc from a generator function.

    This decorator enables natural Python generator patterns for iterative
    algorithms, avoiding the complexity of carry dicts and tuple returns.

    Parameters
    ----------
    output_name
        The identifier for the output of the scan operation.
    xs
        Optional name of the parameter containing the list/array to iterate over.
    return_final_only
        If True, only return the final yielded value.
    output_picker
        Function to pick specific outputs from return value.
    renames
        Mapping of parameter names to rename in the function signature.
    defaults
        Default values for parameters.
    bound
        Parameters bound to specific values.
    profile
        Whether to profile execution time.
    debug
        Whether to print debug information.
    print_error
        Whether to print error messages.
    cache
        Whether to cache function results.
    mapspec
        MapSpec specification for array processing.
    internal_shape
        Shape specification for internal arrays.
    post_execution_hook
        Hook function called after execution.
    resources
        Resource requirements for execution.
    resources_variable
        Variable name for resources in function signature.
    resources_scope
        Scope for resource allocation (map or element level).
    scope
        Parameter scope specification.
    variant
        Variant specification for conditional execution.

    Returns
    -------
    decorator
        A decorator that creates a ScanIterFunc instance.

    Examples
    --------
    >>> @scan_iter(output_name="cumsum")
    ... def accumulator(values: list[int], total: int = 0):
    ...     for x in values:
    ...         total += x
    ...         yield total

    >>> @scan_iter(output_name="trajectory", return_final_only=True)
    ... def simulate(time_steps: np.ndarray, y0: float = 1.0, dt: float = 0.1):
    ...     y = y0
    ...     for t in time_steps:
    ...         y = y - y * dt  # Simple decay
    ...         yield {"t": t, "y": y}

    """

    def decorator(func: Callable[..., Generator]) -> ScanIterFunc:
        return ScanIterFunc(
            func=func,
            output_name=output_name,
            xs=xs,
            return_final_only=return_final_only,
            output_picker=output_picker,
            renames=renames,
            defaults=defaults,
            bound=bound,
            profile=profile,
            debug=debug,
            print_error=print_error,
            cache=cache,
            mapspec=mapspec,
            internal_shape=internal_shape,
            post_execution_hook=post_execution_hook,
            resources=resources,
            resources_variable=resources_variable,
            resources_scope=resources_scope,
            scope=scope,
            variant=variant,
        )

    return decorator
