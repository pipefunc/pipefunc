"""Integration tests for input error scanning via public APIs."""

from __future__ import annotations

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map import StorageBase


class CountingStorage(StorageBase):
    """Minimal StorageBase implementation for feeding map inputs."""

    storage_id = "counting"
    requires_serialization = False

    def __init__(
        self,
        array: np.ndarray,
        *,
        dtype_attr: object | None,
        allow_to_array: bool,
    ) -> None:
        self.array = array
        self.shape = array.shape
        self.internal_shape = ()
        self.shape_mask = (True,) * len(self.shape)
        self.dtype = dtype_attr
        self.allow_to_array = allow_to_array
        self.to_array_calls = 0

    def get_from_index(self, index: int) -> object:
        return self.array.flat[index]

    def has_index(self, index: int) -> bool:
        return index < self.array.size

    def __getitem__(self, key: tuple[int | slice, ...]) -> object:
        return self.array[key]

    def to_array(self, *, splat_internal: bool | None = None) -> np.ma.core.MaskedArray:
        self.to_array_calls += 1
        if not self.allow_to_array:
            msg = "to_array should not be called for non-object storages"
            raise AssertionError(msg)
        _ = splat_internal  # unused but kept for signature parity
        mask = np.zeros(self.array.shape, dtype=bool)
        return np.ma.MaskedArray(self.array, mask=mask, dtype=self.array.dtype)

    @property
    def mask(self) -> np.ma.core.MaskedArray:  # type: ignore[override]
        shape = tuple(int(s) for s in self.shape)
        mask = np.zeros(shape, dtype=bool)
        return np.ma.MaskedArray(mask, mask=mask, dtype=bool)

    def mask_linear(self) -> list[bool]:
        return [False] * self.array.size

    def dump(self, key: tuple[int | slice, ...], value: object) -> None:
        raise NotImplementedError

    @property
    def dump_in_subprocess(self) -> bool:
        return False


def _pipeline_for_scan_test() -> Pipeline:
    @pipefunc(
        output_name="y",
        mapspec="x[i] -> y[i]",
        resources=lambda _kw: {},  # callable map-scope resources triggers scanning
    )
    def double(x: float) -> float:
        return 2 * x

    return Pipeline([double])


def test_scan_inputs_skips_to_array_when_dtype_is_not_object() -> None:
    storage = CountingStorage(np.array([1.0, 2.0]), dtype_attr=float, allow_to_array=False)
    pipeline = _pipeline_for_scan_test()

    result = pipeline.map({"x": storage}, error_handling="continue", parallel=False)

    output = result["y"].output
    assert np.allclose(output, np.array([2.0, 4.0], dtype=float))
    assert storage.to_array_calls == 0


def test_scan_inputs_calls_to_array_for_storage_without_dtype_hint() -> None:
    storage = CountingStorage(np.array([1.0, 2.0]), dtype_attr=None, allow_to_array=True)
    pipeline = _pipeline_for_scan_test()

    result = pipeline.map({"x": storage}, error_handling="continue", parallel=False)

    output = result["y"].output
    assert np.allclose(output, np.array([2.0, 4.0], dtype=float))
    assert storage.to_array_calls == 1
