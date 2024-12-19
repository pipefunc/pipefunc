# This file is part of the pipefunc package.
# Originally, it is based on code from the `aiida-dynamic-workflows` package.
# Its license can be found in the LICENSE file in this folder.
# See `git diff 98a1736 pipefunc/map/_mapspec.py` for the changes made.

from __future__ import annotations

import functools
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ._types import ShapeDict, ShapeTuple


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


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Specification for a named array, with some axes indexed by named indices."""

    name: str
    axes: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if "." in self.name:
            scope, name = self.name.split(".", 1)
            if not (scope.isidentifier() and name.isidentifier()):
                msg = (
                    f"Array name '{self.name}' is not a valid Python identifier."
                    " Both the scope and parameter name must be valid identifiers."
                )
                raise ValueError(msg)
        elif not self.name.isidentifier():
            msg = f"Array name '{self.name}' is not a valid Python identifier"
            raise ValueError(msg)
        for i in self.axes:
            if not (i is None or i.isidentifier()):
                msg = f"Index name '{i}' is not a valid Python identifier."
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

    def validate(self, shape: ShapeTuple) -> None:
        """Raise an exception if 'shape' is not compatible with this array spec."""
        if len(shape) != self.rank:
            msg = (
                f"Expecting array of rank {self.rank}, but got array of shape {shape} for `{self}`."
            )
            raise ValueError(msg)

    def add_axes(self, *axis: str | None) -> ArraySpec:
        """Return a new ArraySpec with additional axes."""
        # check for no duplicate axes
        if any(ax in self.axes for ax in axis if ax is not None):
            msg = f"Duplicate axes are not allowed: {axis}"
            raise ValueError(msg)
        return ArraySpec(self.name, self.axes + axis)


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
    _is_generated: bool = False

    def __post_init__(self) -> None:
        if any(x is None for x in self.outputs[0].axes):
            msg = "Output array must have all axes indexed (no ':')."
            raise ValueError(msg)

        if not all(x.indices == self.outputs[0].indices for x in self.outputs[1:]):
            msg = "All output arrays must have identical indices."
            raise ValueError(msg)

        output_indices = set(self.outputs[0].indices)
        input_indices: set[str] = {index for x in self.inputs for index in x.indices}

        if unused_indices := input_indices - output_indices:
            msg = f"Input array have indices that do not appear in the output: {unused_indices}"
            raise ValueError(msg)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the parameter names of this mapspec."""
        return tuple(x.name for x in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the names of the output arrays."""
        return tuple(x.name for x in self.outputs)

    @property
    def output_indices(self) -> tuple[str, ...]:
        """Return the index names of the output array."""
        return self.outputs[0].indices  # All outputs have the same indices

    @functools.cached_property
    def external_indices(self) -> tuple[str, ...]:
        """Output indices that are shared with the input indices."""
        return tuple(n for n in self.output_indices if n in self.input_indices)

    @property
    def input_indices(self) -> set[str]:
        """Return the index names of the input arrays."""
        return {index for x in self.inputs for index in x.indices}

    def shape(
        self,
        input_shapes: ShapeDict,
        internal_shapes: ShapeDict | None = None,
    ) -> tuple[ShapeTuple, tuple[bool, ...]]:
        """Return the shape of the output of this MapSpec.

        Parameters
        ----------
        input_shapes
            Shapes of the inputs, keyed by name.
        internal_shapes
            Shapes of the outputs, keyed by name. Provide this only if the output
            has an axis not shared with any input.

        """
        input_names = set(self.input_names)
        _validate_shapes(input_names, input_shapes, self.inputs, internal_shapes, self.output_names)

        internal_shapes = internal_shapes or {}
        shape: list[int | Literal["?"]] = []
        mask = []
        internal_shape_index = 0
        output = self.outputs[0]  # All outputs have the same shape
        for index in output.axes:
            assert isinstance(index, str)
            relevant_arrays = [x for x in self.inputs if index in x.indices]
            if relevant_arrays:
                dim = _get_common_dim(relevant_arrays, index, input_shapes)
                shape.append(dim)
                mask.append(True)
            else:
                dim = _get_output_dim(output, internal_shapes, internal_shape_index)
                shape.append(dim)
                mask.append(False)
                internal_shape_index += 1
        return tuple(shape), tuple(mask)

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
        if len(shape) != len(self.input_indices):
            msg = f"Expected a shape of length {len(self.input_indices)}, got {shape}"
            raise ValueError(msg)
        return _shape_to_key(shape, linear_index)

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
        if len(shape) != len(self.external_indices):
            msg = f"Expected a shape of length {len(self.external_indices)}, got {shape}"
            raise ValueError(msg)
        key = _shape_to_key(shape, linear_index)
        ids = dict(zip(self.external_indices, key))
        return {
            x.name: tuple(slice(None) if ax is None else ids[ax] for ax in x.axes)
            for x in self.inputs
        }

    def __str__(self) -> str:
        inputs = ", ".join(map(str, self.inputs)) if self.inputs else "..."
        outputs = ", ".join(map(str, self.outputs))
        return f"{inputs} -> {outputs}"

    @classmethod
    def from_string(cls: type[MapSpec], expr: str) -> MapSpec:
        """Construct an MapSpec from a string."""
        try:
            in_, out_ = expr.split("->")
        except ValueError:
            msg = f"Expected expression of form 'a -> b', but got '{expr}''"
            raise ValueError(msg)  # noqa: B904

        inputs = _parse_indexed_arrays(in_)
        outputs = _parse_indexed_arrays(out_)

        return cls(inputs, outputs)

    def to_string(self) -> str:
        """Return a faithful representation of a MapSpec as a string."""
        return str(self)

    def add_axes(self, *axis: str | None) -> MapSpec:
        """Return a new MapSpec with additional axes."""
        return MapSpec(
            tuple(x.add_axes(*axis) for x in self.inputs),
            tuple(x.add_axes(*axis) for x in self.outputs),
        )

    def rename(self, renames: dict[str, str]) -> MapSpec:
        """Return a new renamed MapSpec if any of the names are in 'renames'."""
        if not any(name in renames for name in self.input_names + self.output_names):
            return self

        def _rename(spec: ArraySpec) -> ArraySpec:
            return ArraySpec(renames.get(spec.name, spec.name), spec.axes)

        return MapSpec(tuple(map(_rename, self.inputs)), tuple(map(_rename, self.outputs)))


def _shape_to_key(shape: tuple[int, ...], linear_index: int) -> tuple[int, ...]:
    # Could use np.unravel_index
    return tuple(
        (linear_index // stride) % dim for stride, dim in zip(shape_to_strides(shape), shape)
    )


def _parse_index_string(index_string: str) -> tuple[str | None, ...]:
    indices = (idx.strip() for idx in index_string.split(","))
    return tuple(i if i != ":" else None for i in indices)


def _parse_indexed_arrays(expr: str) -> tuple[ArraySpec, ...]:
    if expr.strip() == "...":
        return ()
    if "[" not in expr or "]" not in expr:
        msg = (
            f"Invalid expression '{expr.strip()}'. Expected an expression that includes "
            "array indices in square brackets. For example, 'a[i]' or 'b[i, j]'. "
            "Please check your syntax and try again."
        )
        raise ValueError(msg)
    array_pattern = r"(\w+(?:\.\w+)?\w*)\[(.+?)\]"
    return tuple(
        ArraySpec(name, _parse_index_string(indices))
        for name, indices in re.findall(array_pattern, expr)
    )


# NOTE: This function is not used in the current implementation!
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
    if isinstance(x, list | range):
        return np.full((len(x),), fill_value=False)
    if isinstance(x, np.ndarray):
        return np.full(x.shape, fill_value=False)
    msg = f"No array mask defined for type {type(x)}"
    raise TypeError(msg)


def array_shape(x: npt.NDArray | list, key: str = "?") -> tuple[int, ...]:
    """Return the shape of 'x'.

    Parameters
    ----------
    x
        The input for which to determine the shape. If 'x' has a 'shape' attribute, it is returned;
        otherwise, the length of 'x' is returned if 'x' is a list.
    key
        The key for which to determine the shape. Only used in error messages.

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
    if isinstance(x, list | range):
        return (len(x),)
    msg = f"No array shape defined for `{key}` of type {type(x)}"
    raise TypeError(msg)


def validate_consistent_axes(mapspecs: list[MapSpec]) -> None:
    """Raise an exception if the axes of the mapspecs are inconsistent."""
    indices: dict[str, set[ArraySpec]] = defaultdict(set)
    for mapspec in mapspecs:
        for spec in mapspec.inputs:
            indices[spec.name].add(spec)
        for spec in mapspec.outputs:
            indices[spec.name].add(spec)

    for name, specs in indices.items():
        specs_str = ", ".join(str(spec) for spec in specs)
        lengths = {len(spec.axes) for spec in specs}
        if len(lengths) > 1:
            msg = (
                f"MapSpec axes for `{name}` are inconsistent: {specs_str}."
                " All axes should have the same length."
            )
            raise ValueError(msg)
        axes: dict[int, str] = {}
        for spec in specs:
            for i, axis in enumerate(spec.axes):
                if axis is not None:
                    if i in axes and axes[i] != axis:
                        msg = (
                            f"MapSpec axes for `{name}` are inconsistent: {specs_str}."
                            " All axes should have the same name at the same index."
                        )
                        raise ValueError(msg)
                    axes[i] = axis


def mapspec_dimensions(mapspecs: list[MapSpec]) -> dict[str, int]:
    """Return the number of dimensions for each array parameter in the pipeline."""
    return {
        arrayspec.name: len(arrayspec.axes)
        for mapspec in mapspecs
        for arrayspec in itertools.chain(mapspec.inputs, mapspec.outputs)
    }


def mapspec_axes(mapspecs: list[MapSpec]) -> dict[str, tuple[str, ...]]:
    """Return the axes for each array parameter in the pipeline."""
    axes: dict[str, dict[int, str]] = defaultdict(dict)
    for mapspec in mapspecs:
        for arrayspec in itertools.chain(mapspec.inputs, mapspec.outputs):
            for i, axis in enumerate(arrayspec.axes):
                if axis is not None:
                    axes[arrayspec.name][i] = axis
    return {name: tuple(dct[i] for i in range(len(dct))) for name, dct in axes.items()}


def _validate_shapes(
    input_names: set[str],
    input_shapes: ShapeDict,
    inputs: tuple[ArraySpec, ...],
    internal_shapes: ShapeDict | None,
    output_names: tuple[str, ...],
) -> None:
    if extra_names := input_shapes.keys() - input_names:
        msg = f"Got extra array {extra_names} that are not accepted by this map."
        raise ValueError(msg)
    if missing_names := input_names - input_shapes.keys():
        msg = f"Inputs expected by this map were not provided: {missing_names}"
        raise ValueError(msg)
    for x in inputs:
        x.validate(input_shapes[x.name])
    if internal_shapes:
        for output_name in internal_shapes:
            if output_name not in output_names:
                msg = f"Internal shape of `{output_name}` is not accepted by this map."
                raise ValueError(msg)


def _get_common_dim(
    arrays: list[ArraySpec],
    index: str,
    input_shapes: ShapeDict,
) -> int | Literal["?"]:
    def _get_dim(array: ArraySpec, index: str) -> int | Literal["?"]:
        axis = array.axes.index(index)
        return input_shapes[array.name][axis]

    dims = [dim for x in arrays if (dim := _get_dim(x, index)) != "?"]
    if not dims:
        return "?"
    dim, *rest = dims
    if any(dim != x for x in rest):
        arrs = ", ".join(x.name for x in arrays)
        msg = f"Dimension mismatch for arrays `{arrs}` along `{index}` axis."
        raise ValueError(msg)
    return dim


def _get_output_dim(
    output: ArraySpec,
    internal_shapes: ShapeDict,
    internal_shape_index: int,
) -> int | Literal["?"]:
    if output.name in internal_shapes:
        if internal_shape_index >= len(internal_shapes[output.name]):
            msg = f"Internal shape for '{output.name}' is too short."
            raise ValueError(msg)
        dim = internal_shapes[output.name][internal_shape_index]
        if not (isinstance(dim, int) or dim == "?"):
            msg = f"Internal shape for '{output.name}' must be a tuple of integers or '?'."
            raise TypeError(msg)
        return dim
    # Infer that the dimension is unknown
    return "?"


def _trace_dependencies(
    output_name: str,
    mapspec_mapping: dict[str, MapSpec],
) -> dict[str, tuple[str, ...]]:
    dependencies: defaultdict[str, set[str]] = defaultdict(set)
    mapspec = mapspec_mapping[output_name]
    for input_spec in mapspec.inputs:
        for axis in input_spec.axes:
            if axis is not None:
                if input_spec.name in mapspec_mapping:
                    nested_dependencies = _trace_dependencies(input_spec.name, mapspec_mapping)
                    if axis in nested_dependencies:
                        dependencies[axis].update(nested_dependencies[axis])
                else:
                    dependencies[axis].add(input_spec.name)
    return {axis: tuple(sorted(inputs)) for axis, inputs in dependencies.items()}


def trace_dependencies(mapspecs: list[MapSpec]) -> dict[str, dict[str, tuple[str, ...]]]:
    mapspec_mapping = {
        output_name: mapspec
        for mapspec in mapspecs
        for output_name in mapspec.output_names
        if mapspec.inputs
    }

    # Go from {output: {axis: list[input]}} to {output: {input: set[axis]}}
    deps = {name: _trace_dependencies(name, mapspec_mapping) for name in mapspec_mapping}
    reordered: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for output_name, dct in deps.items():
        for index, input_names in dct.items():
            for input_name in input_names:
                reordered[output_name][input_name].add(index)

    axes = mapspec_axes(mapspecs)

    def order_like_mapspec_axes(name: str, axes_set: set[str]) -> tuple[str, ...]:
        return tuple(i for i in axes[name] if i in axes_set)

    return {
        output_name: {name: order_like_mapspec_axes(name, axs) for name, axs in dct.items()}
        for output_name, dct in reordered.items()
    }
