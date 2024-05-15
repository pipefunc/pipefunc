# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import annotations

from dataclasses import dataclass
import functools
import re

from .array import _make_strides


@dataclass(frozen=True)
class ArraySpec:
    """Specification for a named array, with some axes indexed by named indices."""

    name: str
    axes: tuple[str | None]

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(
                f"Array name '{self.name}' is not a valid Python identifier"
            )
        for i in self.axes:
            if not (i is None or i.isidentifier()):
                raise ValueError(f"Index name '{i}' is not a valid Python identifier")

    def __str__(self) -> str:
        indices = [":" if x is None else x for x in self.axes]
        return f"{self.name}[{', '.join(indices)}]"

    @property
    def indices(self) -> tuple[str]:
        """Return the names of the indices for this array spec."""
        return tuple(x for x in self.axes if x is not None)

    @property
    def rank(self) -> int:
        """Return the rank of this array spec."""
        return len(self.axes)

    def validate(self, shape: tuple[int, ...]):
        """Raise an exception if 'shape' is not compatible with this array spec."""
        if len(shape) != self.rank:
            raise ValueError(
                f"Expecting array of rank {self.rank}, but got array of shape {shape}"
            )


@dataclass(frozen=True)
class MapSpec:
    """Specification for how to map input axes to output axes.

    Examples
    --------
    >>> mapped = MapSpec.from_string("a[i, j], b[i, j], c[k] -> q[i, j, k]")
    >>> partial_reduction = MapSpec.from_string("a[i, :], b[:, k] -> q[i, k]")
    """

    inputs: tuple[ArraySpec]
    output: ArraySpec

    def __post_init__(self):
        if any(x is None for x in self.output.axes):
            raise ValueError("Output array must have all axes indexed (no ':').")

        output_indices = set(self.output.indices)
        input_indices = functools.reduce(
            set.union, (x.indices for x in self.inputs), set()
        )

        if extra_indices := output_indices - input_indices:
            raise ValueError(
                "Output array has indices that do not appear "
                f"in the input: {extra_indices}"
            )
        if unused_indices := input_indices - output_indices:
            raise ValueError(
                "Input array have indices that do not appear "
                f"in the output: {unused_indices}"
            )

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the parameter names of this mapspec."""
        return tuple(x.name for x in self.inputs)

    @property
    def indices(self) -> tuple[str, ...]:
        """Return the index names for this MapSpec."""
        return self.output.indices

    def shape(self, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """Return the shape of the output of this MapSpec.

        Parameters
        ----------
        shapes
            Shapes of the inputs, keyed by name.
        """
        input_names = {x.name for x in self.inputs}

        if extra_names := set(shapes.keys()) - input_names:
            raise ValueError(
                f"Got extra array {extra_names} that are not accepted by this map."
            )
        if missing_names := input_names - set(shapes.keys()):
            raise ValueError(
                f"Inputs expected by this map were not provided: {missing_names}"
            )

        # Each individual array is of the appropriate rank
        for x in self.inputs:
            x.validate(shapes[x.name])

        # Shapes match between array sharing a named index

        def get_dim(array, index):
            axis = array.axes.index(index)
            return shapes[array.name][axis]

        shape = []
        for index in self.output.indices:
            relevant_arrays = [x for x in self.inputs if index in x.indices]
            dim, *rest = (get_dim(x, index) for x in relevant_arrays)
            if any(dim != x for x in rest):
                raise ValueError(
                    f"Dimension mismatch for arrays {relevant_arrays} "
                    f"along {index} axis."
                )
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
            raise ValueError(
                f"Expected a shape of length {len(self.indices)}, got {shape}"
            )
        return tuple(
            (linear_index // stride) % dim
            for stride, dim in zip(_make_strides(shape), shape)
        )

    def input_keys(
        self,
        shape: tuple[int, ...],
        linear_index: int,
    ) -> dict[str, tuple[slice | int]]:
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
        if len(output_key) != len(self.indices):
            raise ValueError(
                f"Expected a key of shape {len(self.indices)}, got {output_key}"
            )
        ids = dict(zip(self.indices, output_key))
        return {
            x.name: tuple(slice(None) if ax is None else ids[ax] for ax in x.axes)
            for x in self.inputs
        }

    def __str__(self) -> str:
        return f"{', '.join(map(str, self.inputs))} -> {self.output}"

    @classmethod
    def from_string(cls, expr):
        """Construct an MapSpec from a string."""
        try:
            in_, out_ = expr.split("->")
        except ValueError:
            raise ValueError(f"Expected expression of form 'a -> b', but got '{expr}''")

        inputs = _parse_indexed_arrays(in_)
        outputs = _parse_indexed_arrays(out_)
        if len(outputs) != 1:
            raise ValueError(f"Expected a single output, but got {len(outputs)}")
        (output,) = outputs

        return cls(inputs, output)

    def to_string(self) -> str:
        """Return a faithful representation of a MapSpec as a string."""
        return str(self)


def _parse_index_string(index_string) -> list[str | None]:
    indices = [idx.strip() for idx in index_string.split(",")]
    return [i if i != ":" else None for i in indices]


def _parse_indexed_arrays(expr) -> list[ArraySpec]:
    array_pattern = r"(\w+?)\[(.+?)\]"
    return [
        ArraySpec(name, _parse_index_string(indices))
        for name, indices in re.findall(array_pattern, expr)
    ]
