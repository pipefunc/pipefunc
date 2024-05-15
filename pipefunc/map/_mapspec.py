# This file is part of the pipefunc package.
# Originally, it is based on code from the `aiida-dynamic-workflows` package.
# Its license can be found in the LICENSE file in this folder.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def shape_to_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute strides for a multidimensional array given its shape.

    Parameters
    ----------
    shape
        The dimensions of the array.

    Returns
    -------
    The strides for each dimension, where each stride is the product of
    subsequent dimension sizes.

    """
    strides = []
    for i in range(len(shape)):
        product = 1
        for j in range(i + 1, len(shape)):
            product *= shape[j]
        strides.append(product)
    return tuple(strides)


@dataclass(frozen=True)
class ArraySpec:
    """Specification for a named array, with some axes indexed by named indices."""

    name: str
    axes: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            msg = f"Array name '{self.name}' is not a valid Python identifier"
            raise ValueError(msg)
        for i in self.axes:
            if not (i is None or i.isidentifier()):
                msg = f"Index name '{i}' is not a valid Python identifier"
                raise ValueError(msg)

    def __str__(self) -> str:
        indices = (":" if x is None else x for x in self.axes)
        return f"{self.name}[{', '.join(indices)}]"

    @property
    def indices(self) -> tuple[str, ...]:
        """Return the names of the indices for this array spec."""
        return tuple(x for x in self.axes if x is not None)

    @property
    def rank(self) -> int:
        """Return the rank of this array spec."""
        return len(self.axes)

    def validate(self, shape: tuple[int, ...]) -> None:
        """Raise an exception if 'shape' is not compatible with this array spec."""
        if len(shape) != self.rank:
            msg = f"Expecting array of rank {self.rank}, but got array of shape {shape}"
            raise ValueError(msg)


@dataclass(frozen=True)
class MapSpec:
    """Specification for how to map input axes to output axes.

    Examples
    --------
    >>> mapped = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    >>> partial_reduction = MapSpec.from_string("a[i, :], b[:, k] -> q[i, k]")

    """

    inputs: tuple[ArraySpec, ...]
    outputs: tuple[ArraySpec, ...]

    def __post_init__(self) -> None:
        if any(x is None for x in self.outputs[0].axes):
            msg = "Output array must have all axes indexed (no ':')."
            raise ValueError(msg)

        if not all(x.indices == self.outputs[0].indices for x in self.outputs[1:]):
            msg = "All output arrays must have identical indices."
            raise ValueError(msg)

        output_indices = set(self.outputs[0].indices)
        input_indices: set[str] = {index for x in self.inputs for index in x.indices}

        if extra_indices := output_indices - input_indices:
            msg = f"Output array has indices that do not appear in the input: {extra_indices}"
            raise ValueError(msg)
        if unused_indices := input_indices - output_indices:
            msg = f"Input array have indices that do not appear in the output: {unused_indices}"
            raise ValueError(msg)

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the parameter names of this mapspec."""
        return tuple(x.name for x in self.inputs)

    @property
    def indices(self) -> tuple[str, ...]:
        """Return the index names for this MapSpec."""
        return self.outputs[0].indices  # All outputs have the same indices

    def shape(self, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """Return the shape of the output of this MapSpec.

        Parameters
        ----------
        shapes
            Shapes of the inputs, keyed by name.

        """
        input_names = {x.name for x in self.inputs}

        if extra_names := set(shapes.keys()) - input_names:
            msg = f"Got extra array {extra_names} that are not accepted by this map."
            raise ValueError(msg)
        if missing_names := input_names - set(shapes.keys()):
            msg = f"Inputs expected by this map were not provided: {missing_names}"
            raise ValueError(msg)

        # Each individual array is of the appropriate rank
        for x in self.inputs:
            x.validate(shapes[x.name])

        # Shapes match between array sharing a named index

        def get_dim(array: ArraySpec, index: str) -> int:
            axis = array.axes.index(index)
            return shapes[array.name][axis]

        shape = []
        for index in self.outputs[0].indices:  # All outputs have the same indices
            relevant_arrays = [x for x in self.inputs if index in x.indices]
            dim, *rest = (get_dim(x, index) for x in relevant_arrays)
            if any(dim != x for x in rest):
                msg = f"Dimension mismatch for arrays {relevant_arrays} along {index} axis."
                raise ValueError(msg)
            shape.append(dim)

        return tuple(shape)

    def output_key(self, shape: tuple[int, ...], linear_index: int) -> tuple[int, ...]:
        """Return a key used for indexing the output of this map.

        Parameters
        ----------
        shape
            The shape of the map output.
        linear_index
            The index of the element for which to return the key.

        Examples
        --------
        >>> spec = MapSpec.from_string("x[i, j], y[j, :, k] -> z[i, j, k]")
        >>> spec.output_key((5, 2, 3), 23)
        (3, 1, 2)

        """
        if len(shape) != len(self.indices):
            msg = f"Expected a shape of length {len(self.indices)}, got {shape}"
            raise ValueError(msg)
        return tuple(
            (linear_index // stride) % dim for stride, dim in zip(shape_to_strides(shape), shape)
        )

    def input_keys(
        self,
        shape: tuple[int, ...],
        linear_index: int,
    ) -> dict[str, tuple[slice | int, ...]]:
        """Return keys for indexing inputs of this map.

        Parameters
        ----------
        shape
            The shape of the map output.
        linear_index
            The index of the element for which to return the keys.

        Examples
        --------
        >>> spec = MapSpec("x[i, j], y[j, :, k] -> z[i, j, k]")
        >>> spec.input_keys((5, 2, 3), 23)
        {'x': (3, 1), 'y': (1, slice(None, None, None), 2)}

        """
        output_key = self.output_key(shape, linear_index)
        ids = dict(zip(self.indices, output_key))
        return {
            x.name: tuple(slice(None) if ax is None else ids[ax] for ax in x.axes)
            for x in self.inputs
        }

    def __str__(self) -> str:
        inputs = ", ".join(map(str, self.inputs))
        outputs = ", ".join(map(str, self.outputs))
        return f"{inputs} -> {outputs}"

    @classmethod
    def from_string(cls: type[MapSpec], expr: str) -> MapSpec:
        """Construct an MapSpec from a string."""
        try:
            in_, out_ = expr.split("->")
        except ValueError:
            msg = f"Expected expression of form 'a -> b', but got '{expr}''"
            raise ValueError(msg)  # noqa: B904, TRY200

        inputs = _parse_indexed_arrays(in_)
        outputs = _parse_indexed_arrays(out_)

        return cls(inputs, outputs)

    def to_string(self) -> str:
        """Return a faithful representation of a MapSpec as a string."""
        return str(self)


def _parse_index_string(index_string: str) -> tuple[str | None, ...]:
    indices = (idx.strip() for idx in index_string.split(","))
    return tuple(i if i != ":" else None for i in indices)


def _parse_indexed_arrays(expr: str) -> tuple[ArraySpec, ...]:
    if "[" not in expr or "]" not in expr:
        msg = (
            f"Invalid expression '{expr.strip()}'. Expected an expression that includes "
            "array indices in square brackets. For example, 'a[i]' or 'b[i, j]'. "
            "Please check your syntax and try again."
        )
        raise ValueError(msg)
    array_pattern = r"(\w+?)\[(.+?)\]"
    return tuple(
        ArraySpec(name, _parse_index_string(indices))
        for name, indices in re.findall(array_pattern, expr)
    )


def array_mask(x: npt.NDArray | list) -> npt.NDArray[np.bool_]:
    """Return the mask applied to 'x', depending on its type.

    Parameters
    ----------
    x
        The input for which to create a mask. If 'x' has a 'mask' attribute, it is returned;
        otherwise, a mask of False values is created for the input.

    Returns
    -------
    A boolean array where each element is False, indicating no masking by default.

    Raises
    ------
    TypeError
        If 'x' is not a list or numpy.ndarray and does not have a 'mask' attribute.

    Examples
    --------
    >>> array_mask([1, 2, 3])
    array([False, False, False])

    >>> array_mask(np.array([1, 2, 3]))
    array([False, False, False])

    """
    if hasattr(x, "mask"):
        return x.mask
    if isinstance(x, list):
        return np.full((len(x),), fill_value=False)
    if isinstance(x, np.ndarray):
        return np.full(x.shape, fill_value=False)
    msg = f"No array mask defined for type {type(x)}"
    raise TypeError(msg)


def array_shape(x: npt.NDArray | list) -> tuple[int, ...]:
    """Return the shape of 'x'.

    Parameters
    ----------
    x
        The input for which to determine the shape. If 'x' has a 'shape' attribute, it is returned;
        otherwise, the length of 'x' is returned if 'x' is a list.

    Raises
    ------
    TypeError
        If 'x' is not a list or numpy.ndarray and does not have a 'shape' attribute.

    Returns
    -------
    The shape of 'x' as a tuple of integers.

    """
    if hasattr(x, "shape"):
        return tuple(map(int, x.shape))
    if isinstance(x, list):
        return (len(x),)
    msg = f"No array shape defined for type {type(x)}"
    raise TypeError(msg)


def expected_mask(mapspec: MapSpec, inputs: dict[str, Any]) -> npt.NDArray[np.bool_]:
    kwarg_shapes = {k: array_shape(v) for k, v in inputs.items()}
    kwarg_masks = {k: array_mask(v) for k, v in inputs.items()}
    map_shape = mapspec.shape(kwarg_shapes)
    map_size = np.prod(map_shape)

    def is_masked(i: int) -> bool:
        return any(kwarg_masks[k][v] for k, v in mapspec.input_keys(map_shape, i).items())

    return np.array([is_masked(x) for x in range(map_size)]).reshape(map_shape)


def num_tasks_from_mask(mask: npt.NDArray[np.bool_]) -> int:
    """Return the number of tasks that will be executed given a mask."""
    return np.sum(~mask)  # type: ignore[return-value]


def num_tasks(kwargs: dict[str, Any], mapspec: str | MapSpec) -> int:
    """Return the number of tasks."""
    if isinstance(mapspec, str):
        mapspec = MapSpec.from_string(mapspec)
    mapped_kwargs = {k: v for k, v in kwargs.items() if k in mapspec.parameters}
    mask = expected_mask(mapspec, mapped_kwargs)
    return num_tasks_from_mask(mask)
