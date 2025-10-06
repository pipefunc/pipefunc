from __future__ import annotations

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map import DictArray


def test_masked_then_value_then_masked_in_irregular_slice() -> None:
    @pipefunc(output_name="seed")
    def seed() -> list[int]:
        return [10]

    @pipefunc(output_name="data", mapspec="seed[i] -> data[i, j*]")
    def data(seed: int) -> np.ma.MaskedArray:
        arr = np.ma.masked_all(3, dtype=int)
        arr[1] = seed + 1
        return arr

    pipeline = Pipeline([seed, data])

    results = pipeline.map(
        inputs={},
        internal_shapes={"data": (3,)},
        storage="dict",
        parallel=False,
        cleanup=True,
    )
    store = results["data"].store
    assert isinstance(store, DictArray)
    masked_slice = store[(0, slice(None))]
    assert masked_slice.mask.tolist() == [True, False, True]
