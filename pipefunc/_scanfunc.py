"""ScanFunc implementation for iterative execution with feedback loops."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from weakref import WeakSet

import cloudpickle
import numpy as np

from pipefunc._pipefunc import PipeFunc

if TYPE_CHECKING:
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._mapspec import MapSpec
    from pipefunc.map._types import ShapeTuple
    from pipefunc.resources import Resources

T = TypeVar("T", bound=Callable[..., Any])


class ScanFunc(PipeFunc[T]):
    """A PipeFunc subclass for iterative execution with feedback loops.

    ScanFunc enables iterative algorithms where the output of one iteration
    becomes part of the input for the next iteration. This is similar to
    jax.lax.scan and allows for implementing algorithms like:
    - Optimization routines (gradient descent, genetic algorithms)
    - Time-stepping methods (Runge-Kutta, finite differences)
    - Iterative solvers
    - Sequential processing with state

    Parameters
    ----------
    func
        The function to iterate. Should return a tuple of (carry, output) where:
        - carry: dict that will be merged with kwargs for next iteration
        - output: the output value for this iteration (can be None)
    output_name
        The identifier for the output of the scan operation.
    xs
        The name of the parameter containing the list/array to iterate over.
        Can be provided as input to pipeline.run/map or from preceding pipefuncs.
    return_intermediate
        Whether to return intermediate results. If True (default), returns
        an array of all outputs. If False, returns only the final carry dict.
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

    """

    def __init__(
        self,
        func: T,
        output_name: OUTPUT_TYPE,
        xs: str,
        *,
        return_intermediate: bool = True,
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
        """Initialize a ScanFunc instance."""
        # Store scan-specific attributes
        self.xs = xs
        self.return_intermediate = return_intermediate
        self._scan_func = func

        # Analyze the scan function signature
        self._scan_signature = inspect.signature(func)
        scan_params = list(self._scan_signature.parameters.values())
        if not scan_params:
            msg = "Scan function must have at least one parameter"
            raise ValueError(msg)

        # First parameter becomes 'x' in scan iterations
        self._x_param = scan_params[0]
        self._x_param_name = self._x_param.name

        # Remaining parameters are carry parameters
        self._carry_params = scan_params[1:]
        self._carry_param_names = [p.name for p in self._carry_params]

        # Create the wrapper function signature and implementation
        wrapper_func = self._create_scan_wrapper()

        # Initialize parent PipeFunc with wrapper function
        super().__init__(  # type: ignore[misc]
            func=wrapper_func,
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

    def _create_scan_wrapper(self) -> Callable[..., Any]:
        """Create the wrapper function with proper signature transformation."""
        # Create new signature: remove x param, add xs param with carry params
        wrapper_params = []

        # Add carry parameters (all except first)
        wrapper_params.extend(self._carry_params)

        # Add xs parameter
        xs_param = inspect.Parameter(
            self.xs,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=list,
        )
        wrapper_params.append(xs_param)

        # Create the wrapper signature
        self._wrapper_signature = inspect.Signature(wrapper_params)

        # Create wrapper function
        def scan_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute_scan(*args, **kwargs)

        # Set function metadata
        scan_wrapper.__name__ = self._scan_func.__name__
        scan_wrapper.__doc__ = self._scan_func.__doc__
        # Note: Setting __signature__ at runtime for inspect.signature() to work correctly
        scan_wrapper.__signature__ = self._wrapper_signature

        return scan_wrapper

    @functools.cached_property
    def original_parameters(self) -> dict[str, inspect.Parameter]:
        """Return the scan wrapper parameters."""
        return dict(self._wrapper_signature.parameters)

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the parameter names for the scan function."""
        # This is what pipefunc uses to know what parameters the function needs
        return tuple(self.original_parameters.keys())

    def _execute_scan(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the scan operation."""
        # Bind arguments to signature for proper parameter handling
        bound = self._wrapper_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bound_kwargs = dict(bound.arguments)

        # Extract xs from bound arguments
        xs_key = self._renames.get(self.xs, self.xs)
        if xs_key not in bound_kwargs:
            msg = f"Required parameter '{xs_key}' (xs) not provided"
            raise ValueError(msg)

        xs_values = bound_kwargs.pop(xs_key)

        # Initialize carry with remaining bound arguments
        carry = dict(bound_kwargs)

        # Track intermediate results if needed
        intermediate_results = []

        # Iterate over xs
        for x in xs_values:
            # Prepare kwargs for this iteration by copying carry
            iter_kwargs = carry.copy()

            # Map parameter names back to original if renamed
            original_kwargs = {}
            for key, value in iter_kwargs.items():
                original_key = self._inverse_renames.get(key, key)
                original_kwargs[original_key] = value

            # Add current x value using the first parameter name
            original_kwargs[self._x_param_name] = x

            # Call original scan function
            result = self._scan_func(**original_kwargs)

            if not isinstance(result, tuple) or len(result) != 2:  # noqa: PLR2004
                msg = f"Scan function must return tuple of (carry, output), got {type(result)}"
                raise ValueError(msg)

            new_carry, output = result

            if not isinstance(new_carry, dict):
                msg = f"Carry must be a dict, got {type(new_carry)}"
                raise TypeError(msg)

            # Update carry for next iteration
            # Apply renames to carry keys if needed
            renamed_carry: dict[str, Any] = {}
            for key, value in new_carry.items():
                renamed_key = self._renames.get(key, key)
                renamed_carry[renamed_key] = value
            carry.update(renamed_carry)

            # Store intermediate result if needed
            if self.return_intermediate and output is not None:
                intermediate_results.append(output)

        # Store final carry for later access
        self._last_carry = carry

        # Return results based on return_intermediate flag
        if self.return_intermediate:
            if intermediate_results:
                # Convert to appropriate array type
                return np.array(intermediate_results)
            return np.array([])
        # Return final carry
        return carry

    @property
    def carry(self) -> dict[str, Any] | None:
        """Get the final carry dict from the last execution.

        Returns None if the function hasn't been executed yet or
        if return_intermediate is True.
        """
        if hasattr(self, "_last_carry"):
            return self._last_carry
        return None

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickling to avoid circular references."""
        # Build state manually to avoid the wrapper function that has a closure on self
        state = {
            # Core PipeFunc attributes (copy from parent without the problematic func)
            "_output_name": self._output_name,
            "debug": self.debug,
            "print_error": self.print_error,
            "cache": self.cache,
            "mapspec": self.mapspec,
            "internal_shape": self.internal_shape,
            "post_execution_hook": self.post_execution_hook,
            "_output_picker": self._output_picker,
            "_profile": self._profile,
            "_renames": self._renames,
            "_defaults": self._defaults,
            "_bound": self._bound,
            "resources": self.resources,
            "resources_variable": self.resources_variable,
            "resources_scope": self.resources_scope,
            "variant": self.variant,
            # ScanFunc specific attributes
            "xs": self.xs,
            "return_intermediate": self.return_intermediate,
            "_scan_func": cloudpickle.dumps(self._scan_func),
            "_scan_signature": self._scan_signature,
            "_x_param_name": self._x_param_name,
            "_carry_param_names": self._carry_param_names,
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickling to restore the scan wrapper."""
        # Restore scan-specific attributes
        self.xs = state["xs"]
        self.return_intermediate = state["return_intermediate"]
        self._scan_func = cloudpickle.loads(state["_scan_func"])
        self._scan_signature = state["_scan_signature"]
        self._x_param_name = state["_x_param_name"]
        self._carry_param_names = state["_carry_param_names"]

        # Restore core PipeFunc attributes manually
        self._output_name = state["_output_name"]
        self.debug = state["debug"]
        self.print_error = state["print_error"]
        self.cache = state["cache"]
        self.mapspec = state["mapspec"]
        self.internal_shape = state["internal_shape"]
        self.post_execution_hook = state["post_execution_hook"]
        self._output_picker = state["_output_picker"]
        self._profile = state["_profile"]
        self._renames = state["_renames"]
        self._defaults = state["_defaults"]
        self._bound = state["_bound"]
        self.resources = state["resources"]
        self.resources_variable = state["resources_variable"]
        self.resources_scope = state["resources_scope"]
        self.variant = state["variant"]

        # Reconstruct parameter analysis from signature
        scan_params = list(self._scan_signature.parameters.values())
        self._x_param = scan_params[0]
        self._carry_params = scan_params[1:]

        # Recreate the wrapper function
        wrapper_func = self._create_scan_wrapper()
        self.func = wrapper_func

        # Initialize other PipeFunc attributes that might not be in state
        self._pipelines = WeakSet()
        self.profiling_stats = None
        self.error_snapshot = None
        self.__name__ = wrapper_func.__name__

        # Initialize inverse renames
        self._inverse_renames = {v: k for k, v in self._renames.items()}

    def copy(self, **update: Any) -> ScanFunc:
        """Create a copy of the ScanFunc instance, preserving scan-specific attributes."""
        # Get the scan-specific arguments
        scan_kwargs = {
            "func": self._scan_func,  # Use original scan function, not wrapper
            "output_name": self._output_name,
            "xs": self.xs,
            "return_intermediate": self.return_intermediate,
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
            "scope": None,  # Let the new instance handle scope
            "variant": self.variant,
        }

        # Apply any updates
        scan_kwargs.update(update)

        # Create new ScanFunc instance
        return ScanFunc(**scan_kwargs)  # type: ignore[misc]


def scan(
    output_name: OUTPUT_TYPE,
    xs: str,
    *,
    return_intermediate: bool = True,
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
) -> Callable[[Callable[..., tuple[dict[str, Any], Any]]], ScanFunc]:
    """Decorator to create a ScanFunc from a function.

    The decorated function should return a tuple of (carry, output) where:
    - carry: dict that will be merged with kwargs for next iteration
    - output: the output value for this iteration (can be None)

    Parameters
    ----------
    output_name
        The identifier for the output of the scan operation.
    xs
        The name of the parameter containing the list/array to iterate over.
    return_intermediate
        Whether to return intermediate results. If True (default), returns
        an array of all outputs. If False, returns only the final carry dict.
    **kwargs
        Additional parameters passed to ScanFunc constructor.

    Returns
    -------
    decorator
        A decorator that creates a ScanFunc instance.

    Examples
    --------
    >>> @scan(output_name="trajectory", xs="time_steps")
    ... def simulate(t: float, y: float = 1.0, dt: float = 0.1) -> tuple[dict[str, Any], float]:
    ...     y_next = y - y * dt  # Simple decay
    ...     carry = {"y": y_next}
    ...     return carry, y_next

    """

    def decorator(func: Callable[..., tuple[dict[str, Any], Any]]) -> ScanFunc:
        return ScanFunc(
            func=func,
            output_name=output_name,
            xs=xs,
            return_intermediate=return_intermediate,
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
