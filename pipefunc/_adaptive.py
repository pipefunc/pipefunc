from __future__ import annotations

import functools
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypeAlias

from adaptive import DataSaver, Learner1D, Learner2D, LearnerND
from adaptive.types import Int

from pipefunc._utils import at_least_tuple
from pipefunc.map.adaptive import _validate_adaptive

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipefunc import Pipeline

PointType: TypeAlias = tuple[Int, Any]


def _adaptive_wrapper(
    _adaptive_value: float | tuple[float, ...],
    *,
    pipeline: Pipeline,
    kwargs: dict[str, Any],
    adaptive_dimensions: tuple[str, ...],
    adaptive_output: str,
    output_name: str,
    full_output: bool = False,
) -> float | dict[str, Any]:
    values: tuple[float, ...] = at_least_tuple(_adaptive_value)
    kwargs_ = kwargs.copy()
    for dim, val in zip(adaptive_dimensions, values):
        kwargs_[dim] = val
    results = pipeline.run(output_name, kwargs=kwargs_, full_output=True)
    return results if full_output else results[adaptive_output]


def to_adaptive_learner(
    pipeline: Pipeline,
    output_name: str,
    kwargs: dict[str, Any],
    adaptive_dimensions: dict[str, tuple[float, float]],
    adaptive_output: str | None = None,
    loss_function: Callable[..., Any] | None = None,
    *,
    full_output: bool = False,
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
    full_output
        If True, the full output of the pipeline is stored in the learner.
        In this case, the learner is wrapped in a `adaptive.DataSaver`.
        To access the full output, use the `extra_data` attribute of the learner.
        To access the ``Learner{1,2,N}D``, use the ``learner`` attribute of the `DataSaver`.

    Returns
    -------
        A `Learner1D`, `Learner2D`, or `LearnerND` object.

    """
    _validate_adaptive(pipeline, kwargs, adaptive_dimensions)
    if adaptive_output is None:
        adaptive_output = output_name
    if output_name != adaptive_output and not full_output:
        msg = "If `adaptive_output != output_name`, `full_output` must be True."
        raise ValueError(msg)
    dims, bounds = zip(*adaptive_dimensions.items())
    function = functools.partial(
        _adaptive_wrapper,
        pipeline=pipeline,
        kwargs=kwargs,
        adaptive_dimensions=dims,
        adaptive_output=adaptive_output,
        output_name=output_name,
        full_output=full_output,
    )
    n = len(adaptive_dimensions)
    if n == 1:
        learner = Learner1D(function, bounds[0], loss_per_interval=loss_function)
    if n == 2:  # noqa: PLR2004
        learner = Learner2D(function, bounds, loss_per_triangle=loss_function)
    else:
        learner = LearnerND(function, bounds, loss_per_simplex=loss_function)
    if not full_output:
        return learner
    return DataSaver(learner, arg_picker=itemgetter(adaptive_output))
