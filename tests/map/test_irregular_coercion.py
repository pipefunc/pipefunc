from __future__ import annotations

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="vals", mapspec="n[i] -> vals[i, j*]")
def generate_masked_array(n: int) -> np.ma.MaskedArray:
    arr = np.ma.masked_all(3, dtype=int)
    if n > 0:
        arr[0] = n
    return arr


@pipefunc(output_name="vals2", mapspec="n[i] -> vals2[i, j*]")
def generate_list_with_masked_constants(n: int) -> list:
    if n == 0:
        return [np.ma.masked]
    if n == 1:
        return [np.ma.masked, 1]
    return [0, 1, np.ma.masked]


@pipefunc(output_name="vals3", mapspec="n[i] -> vals3[i, j*, k*]")
def generate_nested_structures(n: int) -> list:
    if n == 0:
        return []
    if n == 1:
        return [[np.ma.masked]]
    return [[0, np.ma.masked], [np.ma.masked, 1]]


@pipefunc(output_name="scalar", mapspec="n[i] -> scalar[i, j*]")
def generate_numpy_scalar(n: int) -> np.ndarray:
    return np.array(n)


@pipefunc(output_name="masked_scalar", mapspec="n[i] -> masked_scalar[i, j*]")
def generate_masked_array_scalar(n: int) -> np.ma.MaskedArray:
    return np.ma.array(n, mask=n % 2 == 0)


@pipefunc(output_name="simple", mapspec="n[i] -> simple[i, j*]")
def generate_simple_value(n: int) -> int:
    return n


@pipefunc(output_name="too_long", mapspec="n[i] -> too_long[i, j*]")
def generate_too_long(n: int) -> list[int]:
    return list(range(n + 1))


@pipefunc(output_name="mixed", mapspec="n[i] -> mixed[i, j*, k*]")
def generate_mixed_structure(n: int) -> list:
    return [np.ma.masked, [n]]


def test_coerce_irregular_masked_array_output() -> None:
    pipeline = Pipeline([generate_masked_array])
    result = pipeline.map(
        {"n": [0, 1, 2]},
        internal_shapes={"vals": (3,)},
        storage="dict",
        parallel=False,
    )
    output = result["vals"].output
    assert isinstance(output, np.ma.MaskedArray)
    assert output.shape == (3, 3)
    assert output[1, 0] == 1
    assert np.ma.is_masked(output[0, 1])


def test_coerce_irregular_masked_constants() -> None:
    pipeline = Pipeline([generate_list_with_masked_constants])
    result = pipeline.map(
        {"n": [0, 1, 2]},
        internal_shapes={"vals2": (3,)},
        storage="dict",
        parallel=False,
    )
    output = result["vals2"].output
    assert isinstance(output, np.ma.MaskedArray)
    assert output.shape == (3, 3)
    assert np.ma.is_masked(output[0, 0])
    assert output[1, 1] == 1
    assert np.ma.is_masked(output[1, 0])


def test_coerce_irregular_nested_structures() -> None:
    pipeline = Pipeline([generate_nested_structures])
    result = pipeline.map(
        {"n": [0, 1, 2]},
        internal_shapes={"vals3": (2, 2)},
        storage="dict",
        parallel=False,
    )
    output = result["vals3"].output
    assert isinstance(output, np.ma.MaskedArray)
    assert output.shape == (3, 2, 2)
    assert output[2, 0, 0] == 0
    assert np.ma.is_masked(output[2, 0, 1])


def test_coerce_irregular_numpy_scalar_branch() -> None:
    pipeline = Pipeline([generate_numpy_scalar])
    result = pipeline.map(
        {"n": [1, 2]},
        internal_shapes={"scalar": (1,)},
        storage="dict",
        parallel=False,
    )
    output = result["scalar"].output
    assert output[0, 0] == 1
    assert output[1, 0] == 2


def test_coerce_irregular_masked_array_scalar_branch() -> None:
    pipeline = Pipeline([generate_masked_array_scalar])
    result = pipeline.map(
        {"n": [1, 2]},
        internal_shapes={"masked_scalar": (1,)},
        storage="dict",
        parallel=False,
    )
    output = result["masked_scalar"].output
    assert output[0, 0] == 1
    assert np.ma.is_masked(output[1, 0])


def test_coerce_irregular_simple_value_branch() -> None:
    pipeline = Pipeline([generate_simple_value])
    result = pipeline.map(
        {"n": [5]},
        internal_shapes={"simple": (1,)},
        storage="dict",
        parallel=False,
    )
    output = result["simple"].output
    assert output[0, 0] == 5


def test_coerce_irregular_raises_on_excess() -> None:
    pipeline = Pipeline([generate_too_long])
    with pytest.raises(ValueError, match="exceeds internal_shape at axis 0"):
        pipeline.map(
            {"n": [1]},
            internal_shapes={"too_long": (1,)},
            storage="dict",
            parallel=False,
        )


def test_coerce_irregular_masked_sentinel_branch() -> None:
    pipeline = Pipeline([generate_mixed_structure])
    result = pipeline.map(
        {"n": [1]},
        internal_shapes={"mixed": (2, 1)},
        storage="dict",
        parallel=False,
    )
    output = result["mixed"].output
    assert np.ma.is_masked(output[0, 0, 0])
    assert output[0, 1, 0] == 1
