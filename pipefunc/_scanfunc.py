"""ScanFunc implementation for iterative execution with feedback loops."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import cloudpickle

from pipefunc._pipefunc import PipeFunc
from pipefunc.map._mapspec import MapSpec

if TYPE_CHECKING:
    from pipefunc._pipeline._types import OUTPUT_TYPE
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
        self._x_param_name = list(inspect.signature(func).parameters.keys())[0]

        # Fix the parameters to include xs instead of x first
        self._update_parameters_for_scan()

        # Store a reference to the ScanFunc instance
        # This will be set after super().__init__
        self._scanfunc_instance = None

        # Create a wrapper function with the correct signature dynamically
        sig = inspect.Signature(list(self._scan_parameters.values()))

        # Build parameter string for exec - need to order parameters correctly
        # First params without defaults, then params with defaults
        param_parts_no_default = []
        param_parts_with_default = []
        var_positional = None
        var_keyword = None

        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                var_positional = f"*{param.name}"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                var_keyword = f"**{param.name}"
            elif param.default == inspect.Parameter.empty:
                param_parts_no_default.append(param.name)
            else:
                # Use repr for the default value
                param_parts_with_default.append(f"{param.name}={param.default!r}")

        # Combine in correct order
        all_parts = param_parts_no_default + param_parts_with_default
        if var_positional:
            all_parts.append(var_positional)
        if var_keyword:
            all_parts.append(var_keyword)

        params_str = ", ".join(all_parts)

        # Build kwargs dict for calling _execute_scan
        kwargs_parts = []
        for param in sig.parameters.values():
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                kwargs_parts.append(f"'{param.name}': {param.name}")
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # Include the **kwargs in the call
                kwargs_parts.append(f"**{param.name}")

        kwargs_str = "{" + ", ".join(kwargs_parts) + "}"

        # Create the wrapper function dynamically
        func_code = f"""
def scan_wrapper({params_str}):
    if hasattr(scan_wrapper, '_scanfunc_instance') and scan_wrapper._scanfunc_instance:
        return scan_wrapper._scanfunc_instance._execute_scan(**{kwargs_str})
    raise RuntimeError("ScanFunc instance not set")
"""

        # Execute the code to create the function
        local_vars = {}
        exec(func_code, {}, local_vars)
        scan_wrapper = local_vars["scan_wrapper"]

        # Copy function metadata
        scan_wrapper.__name__ = func.__name__
        scan_wrapper.__doc__ = func.__doc__
        scan_wrapper.__signature__ = sig

        dummy_func = scan_wrapper

        # Initialize parent PipeFunc with dummy function
        super().__init__(
            func=dummy_func,
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

        # Now set the reference to self in the wrapper function
        dummy_func._scanfunc_instance = self

    def _update_parameters_for_scan(self):
        """Update the function parameters to use xs instead of x."""
        # Override the original_parameters cached property
        original_sig = inspect.signature(self._scan_func)
        params = list(original_sig.parameters.values())

        # Find and remove the first parameter (x)
        x_param_idx = None
        for i, param in enumerate(params):
            if param.name == self._x_param_name:
                x_param_idx = i
                break

        if x_param_idx is not None:
            params.pop(x_param_idx)

        # Add xs as a parameter (make sure it's before any VAR_KEYWORD)
        xs_param = inspect.Parameter(
            self.xs,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=list,
            default=inspect.Parameter.empty,
        )

        # Find where to insert xs (before any VAR_KEYWORD)
        insert_idx = len(params)
        for i, param in enumerate(params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                insert_idx = i
                break

        params.insert(insert_idx, xs_param)

        # Create new parameters dict preserving order
        self._scan_parameters = {p.name: p for p in params}

    @functools.cached_property
    def original_parameters(self) -> dict[str, inspect.Parameter]:
        """Return the original parameters with xs instead of x."""
        return self._scan_parameters.copy()

    def _execute_scan(self, **kwargs: Any) -> Any:
        """Execute the scan operation."""
        # Extract xs from kwargs
        xs_key = self._renames.get(self.xs, self.xs)
        if xs_key not in kwargs:
            raise ValueError(f"Required parameter '{xs_key}' (xs) not provided")

        xs_values = kwargs.pop(xs_key)

        # Initialize carry with remaining kwargs
        carry = {k: v for k, v in kwargs.items()}

        # Track intermediate results if needed
        intermediate_results = []

        # Iterate over xs
        for x in xs_values:
            # Prepare kwargs for this iteration
            iter_kwargs = carry.copy()

            # Map parameter names back to original if renamed
            original_kwargs = {}
            for key, value in iter_kwargs.items():
                original_key = self._inverse_renames.get(key, key)
                original_kwargs[original_key] = value

            # Add current x value using the first parameter name
            original_kwargs[self._x_param_name] = x

            # Call original function
            result = self._scan_func(**original_kwargs)

            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    f"Scan function must return tuple of (carry, output), got {type(result)}",
                )

            new_carry, output = result

            if not isinstance(new_carry, dict):
                raise ValueError(
                    f"Carry must be a dict, got {type(new_carry)}",
                )

            # Update carry for next iteration
            # Apply renames to carry keys
            renamed_carry = {}
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
                import numpy as np

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

    def __getstate__(self):
        """Custom pickling to avoid circular references."""
        # Get all instance variables except the problematic ones
        state = {k: v for k, v in self.__dict__.items() if k not in ("func", "_pipelines")}

        # Create a simple placeholder function for pickling
        def placeholder_func(**kwargs):
            raise RuntimeError("This is a placeholder function")

        placeholder_func.__name__ = self._scan_func.__name__
        placeholder_func.__doc__ = self._scan_func.__doc__

        # Store the placeholder function
        state["func"] = cloudpickle.dumps(placeholder_func)

        # Store scan-specific attributes explicitly
        state["_scan_func"] = self._scan_func
        state["_x_param_name"] = self._x_param_name
        state["xs"] = self.xs
        state["return_intermediate"] = self.return_intermediate
        state["_scan_parameters"] = self._scan_parameters

        # Resources need special handling
        state["resources"] = (
            cloudpickle.dumps(self.resources) if self.resources is not None else None
        )

        return state

    def __setstate__(self, state):
        """Custom unpickling to restore the scan wrapper."""
        # Extract scan-specific attributes
        self._scan_func = state.pop("_scan_func")
        self._x_param_name = state.pop("_x_param_name")
        self.xs = state.pop("xs")
        self.return_intermediate = state.pop("return_intermediate")
        self._scan_parameters = state.pop("_scan_parameters")

        # Create a wrapper function with the correct signature dynamically
        sig = inspect.Signature(list(self._scan_parameters.values()))

        # Build parameter string for exec - need to order parameters correctly
        # First params without defaults, then params with defaults
        param_parts_no_default = []
        param_parts_with_default = []
        var_positional = None
        var_keyword = None

        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                var_positional = f"*{param.name}"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                var_keyword = f"**{param.name}"
            elif param.default == inspect.Parameter.empty:
                param_parts_no_default.append(param.name)
            else:
                # Use repr for the default value
                param_parts_with_default.append(f"{param.name}={param.default!r}")

        # Combine in correct order
        all_parts = param_parts_no_default + param_parts_with_default
        if var_positional:
            all_parts.append(var_positional)
        if var_keyword:
            all_parts.append(var_keyword)

        params_str = ", ".join(all_parts)

        # Build kwargs dict for calling _execute_scan
        kwargs_parts = []
        for param in sig.parameters.values():
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                kwargs_parts.append(f"'{param.name}': {param.name}")
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # Include the **kwargs in the call
                kwargs_parts.append(f"**{param.name}")

        kwargs_str = "{" + ", ".join(kwargs_parts) + "}"

        # Create the wrapper function dynamically
        func_code = f"""
def scan_wrapper({params_str}):
    if hasattr(scan_wrapper, '_scanfunc_instance') and scan_wrapper._scanfunc_instance:
        return scan_wrapper._scanfunc_instance._execute_scan(**{kwargs_str})
    raise RuntimeError("ScanFunc instance not set")
"""

        # Execute the code to create the function
        local_vars = {}
        exec(func_code, {}, local_vars)
        scan_wrapper = local_vars["scan_wrapper"]

        # Copy function metadata
        scan_wrapper.__name__ = self._scan_func.__name__
        scan_wrapper.__doc__ = self._scan_func.__doc__
        scan_wrapper.__signature__ = sig

        # Store the function in state for parent class
        state["func"] = cloudpickle.dumps(scan_wrapper)

        # Call parent __setstate__
        super().__setstate__(state)

        # Set the reference to self in the wrapper
        self.func._scanfunc_instance = self


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
