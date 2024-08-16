from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable

from adaptive import Learner1D, Learner2D, LearnerND

from pipefunc._utils import at_least_tuple
from pipefunc.map.adaptive import _validate_adaptive

if TYPE_CHECKING:
    from pipefunc import Pipeline


def _adaptive_wrapper(
    _adaptive_value: float | tuple[float, ...],
    pipeline: Pipeline,
    kwargs: dict[str, Any],
    adaptive_dimensions: tuple[str, ...],
    adaptive_output: str,
    output_name: str,
) -> float:
    values: tuple[float, ...] = at_least_tuple(_adaptive_value)
    kwargs_ = kwargs.copy()
    for dim, val in zip(adaptive_dimensions, values):
        kwargs_[dim] = val
    results = pipeline.run(output_name, kwargs=kwargs_, full_output=True)
    return results[adaptive_output]


def to_adaptive_learner(
    pipeline: Pipeline,
    output_name: str,
    kwargs: dict[str, Any],
    adaptive_dimensions: dict[str, tuple[float, float]],
    adaptive_output: str | None = None,
    loss_function: Callable[..., Any] | None = None,
) -> Learner1D | Learner2D | LearnerND:
    """Create an adaptive learner in 1D, 2D, or ND from a ``pipeline.run`.

    Parameters
    ----------
    pipeline
        The pipeline to create the learner from.
    output_name
        The output to calculate in the pipeline.
    kwargs
        The kwargs to the pipeline, as passed to `pipeline.run`. Should not
        contain the adaptive dimensions.
    adaptive_dimensions
        A dictionary mapping the adaptive dimensions to their bounds.
        If the length of the dictionary is 1, a `adaptive.Learner1D` is created.
        If the length is 2, a `adaptive.Learner2D` is created.
        If the length is 3 or more, a `adaptive.LearnerND` is created.
    adaptive_output
        The output to adapt to. If not provided, the `output_name` is used.
    loss_function
        The loss function to use for the adaptive learner.
        The ``loss_per_interval`` argument for `adaptive.Learner1D`,
        the ``loss_per_triangle`` argument for `adaptive.Learner2D`, and
        the ``loss_per_simplex`` argument for `adaptive.LearnerND`.
        If not provided, the default loss function is used.

    Returns
    -------
        A `Learner1D`, `Learner2D`, or `LearnerND` object.

    """
    _validate_adaptive(pipeline, kwargs, adaptive_dimensions)
    dims, bounds = zip(*adaptive_dimensions.items())
    function = functools.partial(
        _adaptive_wrapper,
        pipeline=pipeline,
        kwargs=kwargs,
        adaptive_dimensions=dims,
        adaptive_output=adaptive_output or output_name,
        output_name=output_name,
    )
    n = len(adaptive_dimensions)
    if n == 1:
        return Learner1D(function, bounds[0], loss_per_interval=loss_function)
    if n == 2:  # noqa: PLR2004
        return Learner2D(function, bounds, loss_per_triangle=loss_function)
    return LearnerND(function, bounds, loss_per_simplex=loss_function)
