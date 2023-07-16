from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Generator, Sequence

import networkx as nx

if TYPE_CHECKING:
    from pipefunc import Pipeline, PipelineFunction


def _at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


class Sweep:
    """Create a sweep of a pipeline.

    Considering certain dimensions to be linked together (zipped) rather than
    forming a full Cartesian product. If 'dims' is not provided, a Cartesian
    product is formed.

    Parameters
    ----------
    items
        A dictionary where the key is the name of the dimension and the
        value is a sequence of values for that dimension.
    dims
        A list of tuples. Each tuple contains names of dimensions that are
        linked together. If not provided, a Cartesian product is formed.
    exclude
        A function that takes a dictionary of dimension values and returns
        True if the combination should be excluded from the sweep.

    Returns
    -------
    list
        A list of dictionaries, each representing a specific combination
        of dimension values.

    Examples
    --------
    >>> items = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    >>> Sweep(items)
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6}, {'a': 1, 'b': 4, 'c': 5}, {'a': 1, 'b': 4, 'c': 6},
     {'a': 2, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 6}, {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

    Which is equivalent to the following:

    >>> Sweep(items, dims=["a", "b", "c"]).list()
    >>> Sweep(items, dims=[("a",), ("b",), ("c",)]).list()

    Or zip together dimensions:

    >>> Sweep(items, dims=[('a', 'b'), ('c',)]).list()
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}].list()
    """

    def __init__(
        self,
        items: dict[str, Sequence[Any]],
        dims: list[Any] | None = None,
        exclude: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.items = items
        self.dims = dims
        self.exclude = exclude

    def generate(self) -> Generator[dict[str, Any], None, None]:
        if self.dims is None or set(self.dims) == self.items.keys():
            # Create the full Cartesian product if no dimensions are provided.
            names = self.items.keys()
            vals = self.items.values()
            for res in product(*vals):
                combination = dict(zip(names, res))
                if self.exclude is None or not self.exclude(combination):
                    yield combination
        else:
            # Otherwise, create a product considering the provided dimensions.
            product_parts = []
            for dim_group in self.dims:
                dims = _at_least_tuple(dim_group)
                dim_seqs = [self.items[dim] for dim in dims]
                product_parts.append([dict(zip(dims, res)) for res in zip(*dim_seqs)])
            for combo in product(*product_parts):
                combination = {k: v for item in combo for k, v in item.items()}
                if self.exclude is None or not self.exclude(combination):
                    yield combination

    def list(self) -> list[dict[str, Any]]:  # noqa: A003
        """Return the sweep as a list."""
        return list(self.generate())

    def filtered_sweep(self, keys: Iterable[str]) -> Sweep:
        """Return the sweep as a list, but only include the specified keys in each dictionary, and remove duplicates."""
        dims: list[str | tuple[str, ...]]
        if self.dims is None or set(self.dims) == self.items.keys():
            dims = [k for k in self.items if k in keys]
        else:
            dims = []
            for dim_group in self.dims:
                if isinstance(dim_group, str) and dim_group in keys:
                    dims.append(dim_group)
                elif isinstance(dim_group, tuple):
                    _dims = tuple(k for k in dim_group if k in keys)
                    if _dims:
                        if len(_dims) == 1:
                            dims.append(_dims[0])
                        else:
                            dims.append(_dims)
        return Sweep(self.items, dims=dims, exclude=self.exclude)

    def __len__(self) -> int:
        """Return the number of unique combinations in the sweep."""
        if self.exclude is not None:
            return len(self.list())
        if self.dims is None or set(self.dims) == self.items.keys():
            # Full Cartesian product; simply multiply together lengths of each dimension
            total_length = 1
            for value in self.items.values():
                total_length *= len(value)
            return total_length
        # Otherwise, calculate lengths considering the provided dimensions.
        total_length = 1
        for dim_group in self.dims:
            dims = _at_least_tuple(dim_group)
            group_length = len(self.items[dims[0]])
            total_length *= group_length
        return total_length

    def __add__(self, other: Sweep) -> MultiSweep:
        """Combine this Sweep with another one, creating a MultiSweep."""
        if not isinstance(other, Sweep):
            msg = "Other object must be a Sweep or a MultiSweep instance."
            raise TypeError(msg)
        return MultiSweep(self, other)

    def add(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        return other + self


class MultiSweep(Sweep):
    """A class to concatenate multiple sweeps into one.

    Parameters
    ----------
    *sweeps
        Sweep objects to be concatenated together.

    Returns
    -------
    MultiSweep
        A MultiSweep object containing the concatenated sweeps.

    Examples
    --------
    >>> sweep1 = Sweep({'a': [1, 2], 'b': [3, 4]})
    >>> sweep2 = Sweep({'x': [5, 6], 'y': [7, 8]})
    >>> multi_sweep = MultiSweep(sweep1, sweep2)
    """

    def __init__(self, *sweeps: Sweep) -> None:
        super().__init__(items={})
        self.sweeps = list(sweeps)

    def generate(self) -> Generator[dict[str, Any], None, None]:
        for sweep in self.sweeps:
            yield from sweep.generate()

    def __len__(self) -> int:
        return sum(len(sweep) for sweep in self.sweeps)

    def list(self) -> list[dict[str, Any]]:  # noqa: A003
        """Return the sweep as a list."""
        return list(self.generate())

    def filtered_sweep(self, keys: Iterable[str]) -> MultiSweep:
        """Return a new MultiSweep, but only include the specified keys in each dictionary, and remove duplicates."""
        filtered_sweeps = [sweep.filtered_sweep(keys) for sweep in self.sweeps]
        return MultiSweep(*filtered_sweeps)

    def __add__(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        if not isinstance(other, Sweep):
            msg = "Other object must be a Sweep or a MultiSweep instance."
            raise TypeError(msg)
        return self.add(other)

    def add(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        if isinstance(other, MultiSweep):
            self.sweeps.extend(other.sweeps)
        else:
            self.sweeps.append(other)
        return self


def generate_sweep(
    items: dict[str, Sequence[Any]],
    dims: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Create a sweep of a pipeline.

    Considering certain dimensions to be linked together (zipped) rather than
    forming a full Cartesian product. If 'dims' is not provided, a Cartesian
    product is formed.

    Parameters
    ----------
    items
        A dictionary where the key is the name of the dimension and the
        value is a sequence of values for that dimension.

    dims
        A list of tuples. Each tuple contains names of dimensions that are
        linked together. If not provided, a Cartesian product is formed.

    Returns
    -------
    list
        A list of dictionaries, each representing a specific combination
        of dimension values.

    Examples
    --------
    >>> items = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    >>> generate_sweep(items)
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6}, {'a': 1, 'b': 4, 'c': 5}, {'a': 1, 'b': 4, 'c': 6},
     {'a': 2, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 6}, {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

    Which is equivalent to the following:

    >>> generate_sweep(items, dims=["a", "b", "c"])
    >>> generate_sweep(items, dims=[("a",), ("b",), ("c",)])

    Or zip together dimensions:

    >>> generate_sweep(items, dims=[('a', 'b'), ('c',)])
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]
    """
    return Sweep(items, dims).list()


def count_sweep(
    output_name: str,
    sweep: list[dict[str, Any]] | Sweep,
    pipeline: Pipeline,
    *,
    use_pandas: bool = False,
) -> dict[str | tuple[str, ...], dict[tuple[Any, ...], int]]:
    """Count the number of times each argument combination is used.

    Useful for determining which functions to execute first or which
    functions to cache.

    Parameters
    ----------
    output_name
        The name of the output to count the argument combinations for.
    sweep
        A list of dictionaries, each representing a values for which
        the function is called. Or a Sweep object.
    pipeline
        The pipeline to count the argument combinations for.
    use_pandas
        Whether to use pandas to create the counts. Note that this is
        slower than the default method for sweeps <≈ 1e6.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the arguments and the
        values are dictionaries where the keys are the argument combinations
        and the values are the number of times that combination is used.
    """
    if isinstance(sweep, Sweep):
        # TODO: we can likely special case this to be faster.
        sweep = sweep.list()  # type: ignore[assignment]
    assert isinstance(sweep, Iterable)
    counts: dict[str | tuple[str, ...], dict[tuple[Any, ...], int]] = {}
    deps = pipeline.func_dependencies(output_name)
    for _output_name in deps:
        arg_combination = pipeline.arg_combinations(_output_name, root_args_only=True)
        assert isinstance(arg_combination, tuple)
        if use_pandas:
            import pandas as pd

            df = pd.DataFrame(list(sweep))
            cols = list(arg_combination)
            counts[_output_name] = df[cols].groupby(cols).size().to_dict()
        else:
            _cnt: dict[tuple[Any, ...], int] = {}
            for combo in sweep:
                key = tuple(combo[arg] for arg in arg_combination)
                _cnt[key] = _cnt.get(key, 0) + 1
            counts[_output_name] = _cnt
    return dict(counts)


def set_cache_for_sweep(
    output_name: str,
    pipeline: Pipeline,
    sweep: list[dict[str, Any]],
    min_executions: int = 2,
    *,
    verbose: bool = False,
) -> None:
    """Set the cache for a sweep of a pipeline."""
    # Disable for the output node
    pipeline.node_mapping[output_name].cache = False  # type: ignore[union-attr]
    cnt = count_sweep(output_name, sweep, pipeline)
    max_executions = {k: max(v.values()) for k, v in cnt.items()}
    for _output_name, n in max_executions.items():
        enable_cache = n >= min_executions
        func = pipeline.node_mapping[_output_name]
        if verbose:
            print(f"Setting cache for '{_output_name}' to {enable_cache} (n={n})")
        func.cache = enable_cache  # type: ignore[union-attr]


def get_precalculation_order(
    pipeline: Pipeline,
    counts: dict[str | tuple[str, ...], dict[tuple[Any, ...], int]],
    min_executions: int = 2,
) -> list[PipelineFunction]:
    """Determine the order in which functions in a pipeline should be precalculated and cached.

    The order is determined by the topological dependencies of the functions
    and the count of their executions in the context of a parameter sweep.
    Only functions that are executed multiple times (as specified by `min_executions`)
    are included in the precalculation order.

    Parameters
    ----------
    pipeline
        The pipeline of functions.
    counts
        A dictionary mapping function output names to dictionaries of
        parameter combinations and their counts in the pipeline.
    min_executions
        The minimum number of times a function must be used in the pipeline
        for it to be included in the precalculation order. Defaults to 2.

    Returns
    -------
    list[PipelineFunction]
        The ordered list of functions to be precalculated and cached.
    """

    def key_func(node: PipelineFunction) -> int:
        return -sum(counts[node.output_name].values())

    m = pipeline.node_mapping
    # Get nodes with counts ≥min_executions
    nodes_with_counts = [
        m[node]
        for node, count_dict in counts.items()
        if any(val >= min_executions for val in count_dict.values())
    ]
    # Create a subgraph with only the nodes with sufficient counts
    subgraph = pipeline.graph.subgraph(nodes_with_counts)
    # Return the ordered list of nodes
    return list(nx.lexicographical_topological_sort(subgraph, key=key_func))
