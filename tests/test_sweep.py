from itertools import product

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._sweep import (
    MultiSweep,
    Sweep,
    count_sweep,
    generate_sweep,
    set_cache_for_sweep,
)


@pytest.fixture()
def pipeline():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    return Pipeline([f1, f2, f3], debug=True, profile=True)


def test_generate_sweep_no_dims():
    items = {"a": [1, 2], "b": [3, 4]}
    expected_result = [dict(zip(items.keys(), res)) for res in product(*items.values())]
    assert generate_sweep(items) == expected_result


def test_generate_sweep_with_dims_zipped():
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    dims = [("a", "b"), "c"]
    expected_result = [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]
    assert generate_sweep(items, dims=dims) == expected_result


def test_generate_sweep_with_dims_separate():
    items = {"a": [1, 2], "b": [3, 4]}
    dims = ["a", "b"]
    expected_result = [dict(zip(items.keys(), res)) for res in product(*items.values())]
    assert generate_sweep(items, dims=dims) == generate_sweep(items) == expected_result


def test_generate_sweep():
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    assert generate_sweep(items) == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 3, "c": 6},
        {"a": 1, "b": 4, "c": 5},
        {"a": 1, "b": 4, "c": 6},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]
    assert generate_sweep(items, dims=[("a", "b"), ("c",)]) == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]


@pytest.mark.parametrize("use_pandas", [True, False])
def test_count_sweep(pipeline, use_pandas):
    sweep = [
        {"a": 1, "b": 2, "x": 3},
        {"a": 1, "b": 2, "x": 3},
        {"a": 2, "b": 3, "x": 4},
    ]
    output_name = "e"
    expected_result = {"c": {(1, 2): 2, (2, 3): 1}, "d": {(1, 2, 3): 2, (2, 3, 4): 1}}
    assert (
        count_sweep(output_name, sweep, pipeline, use_pandas=use_pandas)
        == expected_result
    )


def test_set_cache_for_sweep(pipeline):
    sweep = [
        {"a": 1, "b": 2, "x": 3},
        {"a": 1, "b": 2, "x": 3},
        {"a": 2, "b": 3, "x": 4},
    ]
    output_name = "e"
    set_cache_for_sweep(output_name, pipeline, sweep, verbose=True)
    assert pipeline.node_mapping["c"].cache is True
    assert pipeline.node_mapping["d"].cache is True
    assert pipeline.node_mapping["e"].cache is False


def test_sweep_with_exclude():
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}

    def exclude(combination):
        return combination["a"] == 1 and combination["b"] == 3

    # Test with zipped dims
    dims = [("a", "b"), ("c",)]
    sweep = Sweep(items, dims, exclude)
    expected_result = [{"a": 2, "b": 4, "c": 5}, {"a": 2, "b": 4, "c": 6}]
    actual_result = sweep.list()
    assert actual_result == expected_result

    # Test with separate dims
    sweep = Sweep(items, exclude=exclude)
    expected_result = [
        {"a": 1, "b": 4, "c": 5},
        {"a": 1, "b": 4, "c": 6},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]
    actual_result = sweep.list()
    assert actual_result == expected_result


def test_filtered_sweep():
    combos = {
        "a": [1, 2],
        "b": [1, 2],
        "c": [1, 2],
        "d": [1, 2],
    }

    sweep = Sweep(combos)
    filtered = sweep.filtered_sweep(("a", "b", "c"))
    assert (
        len(sweep) == len(sweep.list()) == len(filtered.list()) * 2 == len(filtered) * 2
    )

    sweep = Sweep(combos, dims=[("a", "b"), "c", "d"])
    filtered = sweep.filtered_sweep(("a", "c", "d"))
    assert len(sweep.list()) == len(filtered.list()) == len(sweep) == len(filtered)
    assert filtered.dims == ["a", "c", "d"]
    first = next(filtered.generate())
    assert "b" not in first

    filtered = sweep.filtered_sweep(("a", "b", "c"))
    assert len(sweep.list()) == len(filtered.list()) * 2


def test_multi_sweep():
    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]})
    sweep2 = Sweep({"x": [5, 6], "y": [7, 8]})
    multi_sweep = MultiSweep(sweep1, sweep2)
    expected_result = sweep1.list() + sweep2.list()
    assert multi_sweep.list() == expected_result
    assert len(multi_sweep) == 8


def test_sweep_add():
    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]})
    sweep2 = Sweep({"x": [5, 6], "y": [7, 8]})
    multi_sweep = sweep1 + sweep2
    assert isinstance(multi_sweep, MultiSweep)
    expected_result = sweep1.list() + sweep2.list()
    assert multi_sweep.list() == expected_result
    assert len(multi_sweep) == 8


def test_multi_sweep_add():
    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]})
    sweep2 = Sweep({"x": [5, 6], "y": [7, 8]})
    sweep3 = Sweep({"z": [9, 10]})
    multi_sweep = sweep1 + sweep2 + sweep3
    expected_result = sweep1.list() + sweep2.list() + sweep3.list()
    assert multi_sweep.list() == expected_result
    assert len(multi_sweep) == 10
