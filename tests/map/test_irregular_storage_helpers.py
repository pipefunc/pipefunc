from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import _IrregularSkipContext
from pipefunc.map._storage_array._base import StorageBase, infer_irregular_length
from pipefunc.map._storage_array._dict import DictArray
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from pathlib import Path


class MinimalStorage(StorageBase):
    """Lightweight ``StorageBase`` implementation for unit testing."""

    storage_id = "minimal"
    requires_serialization = False

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (1,),
        internal_shape: tuple[int, ...] = (),
        shape_mask: tuple[bool, ...] = (True,),
        irregular: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.internal_shape = tuple(internal_shape)
        self.shape_mask = tuple(shape_mask)
        self.irregular = irregular
        self._store: dict[int, Any] = {}

    def get_from_index(self, index: int) -> Any:
        return self._store[index]

    def has_index(self, index: int) -> bool:
        return index in self._store

    def __getitem__(self, key: tuple[int | slice, ...]) -> Any:  # pragma: no cover - not needed
        raise NotImplementedError

    def to_array(
        self,
        *,
        splat_internal: bool | None = None,
    ) -> np.ma.core.MaskedArray:  # pragma: no cover
        raise NotImplementedError

    @property
    def mask(self) -> np.ma.core.MaskedArray:  # pragma: no cover - not exercised here
        data = np.zeros(tuple(int(s) for s in self.shape), dtype=bool)
        return np.ma.array(data, mask=data)

    def mask_linear(self) -> list[bool]:
        return [False] * max(1, math.prod(int(s) for s in self.shape))

    def dump(self, key: tuple[int | slice, ...], value: Any) -> None:
        index = key[0]
        if isinstance(index, slice):  # pragma: no cover - defensive
            msg = "Slice indexing unsupported in MinimalStorage.dump"
            raise TypeError(msg)
        self._store[int(index)] = value

    @property
    def dump_in_subprocess(self) -> bool:
        return False


class CacheStorage(MinimalStorage):
    def __init__(self) -> None:
        super().__init__(shape=(1,), internal_shape=(1,), shape_mask=(True, False), irregular=True)
        self.calls = 0

    def _compute_irregular_extent(self, external_index: tuple[int, ...]) -> tuple[int, ...] | None:  # noqa: ARG002
        self.calls += 1
        return (1,)


class NoneExtentStorage(MinimalStorage):
    def __init__(self) -> None:
        super().__init__(shape=(1,), internal_shape=(1,), shape_mask=(True, False), irregular=True)

    def _compute_irregular_extent(self, external_index: tuple[int, ...]) -> tuple[int, ...] | None:  # noqa: ARG002
        return None


class RecordingStorage(CacheStorage):
    def __init__(self, masked: bool) -> None:
        super().__init__()
        self._masked = masked
        self.received: list[tuple[int | slice, ...]] = []

    def is_element_masked(self, key: tuple[int | slice, ...]) -> bool:
        self.received.append(key)
        return self._masked


def test_storage_base_irregular_extent_defaults() -> None:
    base = MinimalStorage()
    assert base.irregular_extent((0,)) is None
    assert not base.is_element_masked((0,))

    no_internal = MinimalStorage(irregular=True)
    assert no_internal.irregular_extent((0,)) is None

    none_extent = NoneExtentStorage()
    assert none_extent.irregular_extent((0,)) is None

    cache_storage = CacheStorage()
    assert cache_storage.irregular_extent((0,)) == (1,)
    assert cache_storage.calls == 1
    assert cache_storage.irregular_extent((0,)) == (1,)
    assert cache_storage.calls == 2
    assert not cache_storage.is_element_masked((0, 0))


def test_infer_length_variants() -> None:
    masked_constant = np.ma.masked
    masked_data = np.ma.array([1, 2], mask=np.ma.nomask)
    masked_with_gap = np.ma.array([1, np.ma.masked, 3, 4], mask=[False, True, False, False])
    all_masked = np.ma.array([np.ma.masked, np.ma.masked], mask=[True, True])

    class NoLen:
        pass

    for infer in (infer_irregular_length,):
        assert infer(None) == 0
        assert infer(masked_constant) == 0
        assert infer(masked_data) == 2
        assert infer(masked_with_gap) == 4
        assert infer(all_masked) == 0
        assert infer(NoLen()) == 1


def test_dictarray_irregular_extent_and_mask_operations() -> None:
    arr = DictArray(
        folder=None,
        shape=(4,),
        internal_shape=(4,),
        shape_mask=(True, False),
        irregular=True,
    )
    arr._dict[(0,)] = [0, 1]
    arr._dict[(1,)] = np.ma.array([0, 1, np.ma.masked, 3], mask=[False, False, True, False])
    arr._dict[(2,)] = object()
    arr._dict[(3,)] = np.ma.masked

    assert arr.irregular_extent((0,)) == (2,)
    assert arr.irregular_extent((0,)) == (2,)
    assert arr.irregular_extent((1,)) == (4,)
    assert arr.irregular_extent((2,)) == (1,)
    assert arr.irregular_extent((3,)) == (0,)
    assert arr.irregular_extent((99,)) == (0,)

    assert not arr.is_element_masked((0, 1))
    assert arr.is_element_masked((0, 2))
    assert not arr.is_element_masked((1, 1))
    assert arr.is_element_masked((2, 1))
    assert not arr.is_element_masked((0, slice(None)))


def test_filearray_irregular_extent_and_mask_operations(tmp_path: Path) -> None:
    arr = FileArray(
        tmp_path,
        shape=(3,),
        internal_shape=(4,),
        shape_mask=(True, False),
        irregular=True,
    )
    arr.dump((0,), [0, 1, 2])
    arr.dump((1,), np.ma.array([0, 1, np.ma.masked, 3], mask=[False, False, True, False]))
    arr.dump((2,), object())

    assert arr.irregular_extent((0,)) == (3,)
    assert arr.irregular_extent((0,)) == (3,)
    assert arr.irregular_extent((1,)) == (4,)
    assert arr.irregular_extent((5,)) == (0,)

    assert not arr.is_element_masked((0, 1))
    assert arr.is_element_masked((0, 3))
    assert not arr.is_element_masked((1, 1))
    assert arr.is_element_masked((2, 1))
    assert not arr.is_element_masked((0, slice(None)))


def test_irregular_skip_context_variants(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="n[i] -> x[i, j*]")
    def generate_values(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j*] -> y[i, j*]")
    def echo(x: int) -> int:
        return x

    pipeline = Pipeline([generate_values, echo])
    results = pipeline.map(
        inputs={"n": [1, 3, 0]},
        internal_shapes={"x": (5,), "y": (5,)},
        run_folder=tmp_path,
        parallel=False,
        storage="dict",
    )

    x_store = results["x"].store
    y_store = results["y"].store
    assert isinstance(x_store, StorageBase)
    assert isinstance(y_store, StorageBase)
    shape = tuple(int(dim) if isinstance(dim, int) else 0 for dim in y_store.shape)
    mask = tuple(bool(flag) for flag in y_store.shape_mask)
    spec = MapSpec.from_string("x[i, j*] -> y[i, j*]")
    func_stub = cast("PipeFunc", SimpleNamespace(mapspec=spec))
    ctx = _IrregularSkipContext(func_stub, {"x": x_store}, shape, mask)
    assert ctx.enabled
    assert ctx.should_skip(3)
    assert not ctx.should_skip(6)

    ctx_no_mapspec = _IrregularSkipContext(
        cast("PipeFunc", SimpleNamespace(mapspec=None)),
        {},
        shape,
        mask,
    )
    assert not ctx_no_mapspec.enabled
    assert not ctx_no_mapspec.should_skip(0)

    regular_spec = MapSpec.from_string("x[i, j] -> y[i, j]")
    ctx_regular = _IrregularSkipContext(
        cast("PipeFunc", SimpleNamespace(mapspec=regular_spec)),
        {"x": x_store},
        shape,
        mask,
    )
    assert not ctx_regular.enabled

    ctx_no_storage = _IrregularSkipContext(
        func_stub,
        {"x": object()},
        shape,
        mask,
    )
    assert not ctx_no_storage.enabled

    spec_short = MapSpec.from_string("x[i, j*] -> y[i, j*]")

    def short_input_keys(
        self: MapSpec,
        shape: tuple[int, ...],
        index: int,
    ) -> dict[str, tuple[int, ...]]:
        return {"x": (0,)}

    object.__setattr__(spec_short, "input_keys", MethodType(short_input_keys, spec_short))
    short_storage = RecordingStorage(masked=False)
    ctx_short_tuple = _IrregularSkipContext(
        cast("PipeFunc", SimpleNamespace(mapspec=spec_short)),
        {"x": short_storage},
        shape,
        mask,
    )
    assert ctx_short_tuple.enabled
    assert not ctx_short_tuple.should_skip(0)

    spec_slice = MapSpec.from_string("x[j*] -> y[j*]")

    def slice_input_keys(
        self: MapSpec,
        shape: tuple[int, ...],
        index: int,
    ) -> dict[str, tuple[int | slice, ...]]:
        return {"x": (slice(None),)}

    object.__setattr__(spec_slice, "input_keys", MethodType(slice_input_keys, spec_slice))
    slice_storage = RecordingStorage(masked=False)
    ctx_slice = _IrregularSkipContext(
        cast("PipeFunc", SimpleNamespace(mapspec=spec_slice)),
        {"x": slice_storage},
        (5,),
        (True,),
    )
    assert ctx_slice.enabled
    assert not ctx_slice.should_skip(0)
    assert slice_storage.received == []


def test_dictarray_is_element_masked_guards() -> None:
    arr_plain = DictArray(folder=None, shape=(2,), irregular=False)
    assert not arr_plain.is_element_masked((0,))

    arr_irregular = DictArray(
        folder=None,
        shape=(1,),
        internal_shape=(3,),
        shape_mask=(True, False),
        irregular=True,
    )
    assert not arr_irregular.is_element_masked((0, slice(None)))


def test_filearray_is_element_masked_guards(tmp_path: Path) -> None:
    arr_plain = FileArray(tmp_path / "plain", shape=(1,), irregular=False)
    assert not arr_plain.is_element_masked((0,))

    arr_irregular = FileArray(
        tmp_path / "irregular",
        shape=(1,),
        internal_shape=(4,),
        shape_mask=(True, False),
        irregular=True,
    )
    assert not arr_irregular.is_element_masked((0, slice(None)))


def test_skip_context_disabled_cases() -> None:
    storage = MinimalStorage(shape=(1,), irregular=True)
    storage.internal_shape = (1,)
    func_single = cast("PipeFunc", SimpleNamespace(mapspec=MapSpec.from_string("x[i*] -> y[i*]")))
    ctx_unknown = _IrregularSkipContext(func_single, {"x": storage}, ("?",), (True,))
    assert not ctx_unknown.enabled

    func_multi = cast(
        "PipeFunc",
        SimpleNamespace(mapspec=MapSpec.from_string("x[i, j*] -> y[i, j*, k*]")),
    )
    ctx_multi = _IrregularSkipContext(func_multi, {"x": storage}, (5, 5), (True, True))
    assert not ctx_multi.enabled
