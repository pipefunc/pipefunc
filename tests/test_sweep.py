from __future__ import annotations

import importlib.util
from itertools import product

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.sweep import MultiSweep, Sweep, count_sweep, generate_sweep, set_cache_for_sweep

has_pandas = importlib.util.find_spec("pandas") is not None


@pytest.fixture
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

    return Pipeline([f1, f2, f3], debug=True)


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


@pytest.mark.skipif(not has_pandas, reason="pandas not installed")
@pytest.mark.parametrize("use_pandas", [True, False])
def test_count_sweep(pipeline, use_pandas):
    sweep = [
        {"a": 1, "b": 2, "x": 3},
        {"a": 1, "b": 2, "x": 3},
        {"a": 2, "b": 3, "x": 4},
    ]
    output_name = "e"
    expected_result = {"c": {(1, 2): 2, (2, 3): 1}, "d": {(1, 2, 3): 2, (2, 3, 4): 1}}
    assert count_sweep(output_name, sweep, pipeline, use_pandas=use_pandas) == expected_result

    sweep = Sweep({"a": [1, 2], "b": [3, 4], "x": [5, 6]})
    expected_result = {
        "c": {(1, 3): 2, (1, 4): 2, (2, 3): 2, (2, 4): 2},
        "d": {
            (1, 3, 5): 1,
            (1, 3, 6): 1,
            (1, 4, 5): 1,
            (1, 4, 6): 1,
            (2, 3, 5): 1,
            (2, 3, 6): 1,
            (2, 4, 5): 1,
            (2, 4, 6): 1,
        },
    }
    assert count_sweep(output_name, sweep, pipeline, use_pandas=use_pandas) == expected_result


def test_set_cache_for_sweep(pipeline):
    sweep = [
        {"a": 1, "b": 2, "x": 3},
        {"a": 1, "b": 2, "x": 3},
        {"a": 2, "b": 3, "x": 4},
    ]
    output_name = "e"
    set_cache_for_sweep(output_name, pipeline, sweep, verbose=True)
    assert pipeline["c"].cache is True
    assert pipeline["d"].cache is True
    assert pipeline["e"].cache is False


def test_sweep_with_exclude():
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}

    def exclude(combination):
        return combination["a"] == 1 and combination["b"] == 3

    # Test with zipped dims
    dims = [("a", "b"), ("c",)]
    sweep = Sweep(items, dims, exclude)
    expected_result = [{"a": 2, "b": 4, "c": 5}, {"a": 2, "b": 4, "c": 6}]
    assert len(sweep) == 2
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
    assert len(sweep) == 6


def test_filtered_sweep():
    combos = {
        "a": [1, 2],
        "b": [1, 2],
        "c": [1, 2],
        "d": [1, 2],
    }

    sweep = Sweep(combos)
    filtered = sweep.filtered_sweep(("a", "b", "c"))
    assert len(sweep) == len(sweep.list()) == len(filtered.list()) * 2 == len(filtered) * 2

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
    multi_sweep2 = sweep1 + sweep2
    multi_sweep3 = sweep1.combine(sweep2)
    expected_result = sweep1.list() + sweep2.list()
    assert multi_sweep.list() == multi_sweep2.list() == expected_result
    assert multi_sweep.list() == multi_sweep3.list() == expected_result
    assert len(multi_sweep) == len(multi_sweep2) == 8
    assert len(multi_sweep) == len(multi_sweep3) == 8
    double = multi_sweep.combine(multi_sweep)
    assert len(double) == 16


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


def test_constants() -> None:
    items = {"a": [1, 2], "b": [3, 4]}
    sweep1 = Sweep(items, constants={"c": 5})  # type: ignore[arg-type]
    assert sweep1.list() == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 4, "c": 5},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 4, "c": 5},
    ]
    sweep2 = Sweep(items, constants={"c": 6})  # type: ignore[arg-type]
    assert sweep2.list() == [
        {"a": 1, "b": 3, "c": 6},
        {"a": 1, "b": 4, "c": 6},
        {"a": 2, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 6},
    ]
    assert MultiSweep(sweep1, sweep2).list() == sweep1.list() + sweep2.list()
    sweep3 = Sweep(items, dims=[("a",), ("b",)], constants={"c": 5})  # type: ignore[arg-type]
    assert sweep3.list() == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 4, "c": 5},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 4, "c": 5},
    ]

    sweep4 = Sweep(items, constants={"a": 9000})  # type: ignore[arg-type]
    assert sweep4.list() == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_derivers() -> None:
    items = {"a": [1, 2], "b": [3, 4]}

    def double_a(combo):
        return combo["a"] * 2

    def square_b(combo):
        return combo["b"] ** 2

    derivers = {
        "a": double_a,
        "b": square_b,
        "c": lambda combo: combo["a"] + combo["b"],
    }

    sweep = Sweep(items, derivers=derivers)  # type: ignore[arg-type]
    expected = [
        {"a": 2, "b": 9, "c": 11},
        {"a": 2, "b": 16, "c": 18},
        {"a": 4, "b": 9, "c": 13},
        {"a": 4, "b": 16, "c": 20},
    ]
    assert sweep.list() == expected

    sweep1 = Sweep(items, dims=[("a",), ("b",)], derivers=derivers)  # type: ignore[arg-type]
    assert sweep1.list() == expected

    sweep2 = Sweep(items, dims=[("a",), ("b",)])  # type: ignore[arg-type]
    sweep3 = sweep2.add_derivers(**derivers)
    assert sweep3.list() == expected


def test_filtered_sweep_with_derivers() -> None:
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}

    def multiply_ab(combo):
        return combo["a"] * combo["b"]

    derivers = {
        "d": multiply_ab,
    }

    sweep = Sweep(items, derivers=derivers)  # type: ignore[arg-type]
    filtered_sweep = sweep.filtered_sweep(("a", "d"))

    assert filtered_sweep.list() == [
        {"a": 1, "d": 3},
        {"a": 1, "d": 4},
        {"a": 2, "d": 6},
        {"a": 2, "d": 8},
    ]

    sweep = Sweep(
        items,  # type: ignore[arg-type]
        dims=[("a", "b"), ("c",)],
        derivers=derivers,
    )
    filtered_sweep = sweep.filtered_sweep(("b", "d"))

    assert filtered_sweep.list() == [{"b": 3, "d": 3}, {"b": 4, "d": 8}]

    multi = sweep.filtered_sweep(("b",)) + sweep.filtered_sweep(("d",))
    assert multi.list() == [{"b": 3}, {"b": 4}, {"d": 3}, {"d": 8}]
    assert multi.filtered_sweep(("b",)).list() == [{"b": 3}, {"b": 4}]

    # Test with unhashable items
    sweep = Sweep(
        items={"a": [[1], [2]], "b": [[3], [4]]},  # type: ignore[arg-type]
        derivers={"c": lambda combo: combo["a"][0] * 2},
    )
    assert sweep.list() == [  # check normal sweep
        {"a": [1], "b": [3], "c": 2},
        {"a": [1], "b": [4], "c": 2},
        {"a": [2], "b": [3], "c": 4},
        {"a": [2], "b": [4], "c": 4},
    ]
    with pytest.raises(TypeError, match="All items must be hashable"):
        sweep.filtered_sweep(("a", "c"))

    assert sweep.filtered_sweep(("c",)).list() == [{"c": 2}, {"c": 4}]


def test_filtered_sweep_without_derivers() -> None:
    items = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}

    sweep = Sweep(items)  # type: ignore[arg-type]
    filtered_sweep = sweep.filtered_sweep(("a", "b"))

    assert filtered_sweep.list() == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]

    sweep = Sweep(
        items,  # type: ignore[arg-type]
        dims=[("a", "b"), ("c",)],
    )
    filtered_sweep = sweep.filtered_sweep(("a", "c"))

    assert filtered_sweep.list() == [
        {"a": 1, "c": 5},
        {"a": 1, "c": 6},
        {"a": 2, "c": 5},
        {"a": 2, "c": 6},
    ]


def test_sweep_product() -> None:
    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]})
    sweep2 = Sweep({"c": [5, 6]})
    sweep3 = sweep1.product(sweep2)

    assert sweep3.list() == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 3, "c": 6},
        {"a": 1, "b": 4, "c": 5},
        {"a": 1, "b": 4, "c": 6},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]

    sweep2 = Sweep({"c": [5, 6], "d": [1]}, dims=[("c", "d")])
    sweep3 = sweep1.product(sweep2)
    assert sweep3.list() == [
        {"a": 1, "b": 3, "c": 5, "d": 1},
        {"a": 1, "b": 3, "c": 6, "d": 1},
        {"a": 1, "b": 4, "c": 5, "d": 1},
        {"a": 1, "b": 4, "c": 6, "d": 1},
        {"a": 2, "b": 3, "c": 5, "d": 1},
        {"a": 2, "b": 3, "c": 6, "d": 1},
        {"a": 2, "b": 4, "c": 5, "d": 1},
        {"a": 2, "b": 4, "c": 6, "d": 1},
    ]


def test_sweep_product_with_dims() -> None:
    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]}, dims=[("a", "b")])
    sweep2 = Sweep({"c": [5, 6]}, dims=[("c",)])
    sweep3 = sweep1.product(sweep2)
    assert sweep1.list() == [
        {"a": 1, "b": 3},
        {"a": 2, "b": 4},
    ]
    assert sweep3.list() == [
        {"a": 1, "b": 3, "c": 5},
        {"a": 1, "b": 3, "c": 6},
        {"a": 2, "b": 4, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]

    # double product
    assert sweep3.product(Sweep({"d": [7]})).list() == [
        {"a": 1, "b": 3, "c": 5, "d": 7},
        {"a": 1, "b": 3, "c": 6, "d": 7},
        {"a": 2, "b": 4, "c": 5, "d": 7},
        {"a": 2, "b": 4, "c": 6, "d": 7},
    ]


def test_sweep_product_with_exclude() -> None:
    def exclude1(combo):
        return combo["a"] == 1 and combo["b"] == 3

    def exclude2(combo):
        return combo["c"] == 6

    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]}, exclude=exclude1)
    assert sweep1.list() == [
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    sweep2 = Sweep({"c": [5, 6]}, exclude=exclude2)
    assert sweep2.list() == [{"c": 5}]

    sweep3 = sweep1.product(sweep2)

    assert sweep3.list() == [
        {"a": 1, "b": 4, "c": 5},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 4, "c": 5},
    ]

    sweep1 = Sweep({"a": [1, 2], "b": [3, 4]}, exclude=exclude1)
    sweep2 = Sweep({"c": [5, 6]}, exclude=None, constants={"x": 1})
    sweep3 = sweep1.product(sweep2)
    assert sweep3.list() == [
        {"a": 1, "b": 4, "c": 5, "x": 1},
        {"a": 1, "b": 4, "c": 6, "x": 1},
        {"a": 2, "b": 3, "c": 5, "x": 1},
        {"a": 2, "b": 3, "c": 6, "x": 1},
        {"a": 2, "b": 4, "c": 5, "x": 1},
        {"a": 2, "b": 4, "c": 6, "x": 1},
    ]


def test_sweep_product_with_derivers() -> None:
    sweep1 = Sweep({"a": [1, 2]}, derivers={"x": lambda combo: combo["a"] * 10})
    sweep2 = Sweep({"b": [3, 4]}, derivers={"y": lambda combo: combo["b"] * 20})
    sweep3 = sweep1.product(sweep2)

    assert sweep3.list() == [
        {"a": 1, "b": 3, "x": 10, "y": 60},
        {"a": 1, "b": 4, "x": 10, "y": 80},
        {"a": 2, "b": 3, "x": 20, "y": 60},
        {"a": 2, "b": 4, "x": 20, "y": 80},
    ]


def test_sweep_product_with_constants() -> None:
    sweep1 = Sweep({"a": [1, 2]}, constants={"x": 10})
    sweep2 = Sweep({"b": [3, 4]}, constants={"y": 20})
    sweep3 = sweep1.product(sweep2)

    assert sweep3.list() == [
        {"a": 1, "b": 3, "x": 10, "y": 20},
        {"a": 1, "b": 4, "x": 10, "y": 20},
        {"a": 2, "b": 3, "x": 10, "y": 20},
        {"a": 2, "b": 4, "x": 10, "y": 20},
    ]


def test_empty_filtered_sweep() -> None:
    assert Sweep({}).list() == []

    sweep = Sweep({"a": [1, 2], "b": [3, 4]})
    filtered_sweep = sweep.filtered_sweep(("x",))
    assert filtered_sweep.list() == []


def test_exception_dims_length() -> None:
    sweep = Sweep({"a": [1, 2], "b": [3], "c": [4]}, dims=[("a", "b"), ("c",)])
    with pytest.raises(
        ValueError,
        match="Dimension 'b' has a different length than the other dimensions",
    ):
        sweep.list()
