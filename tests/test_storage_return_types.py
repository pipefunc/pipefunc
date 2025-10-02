from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc.map._storage_array._dict import DictArray
from pipefunc.map._storage_array._file import FileArray

if TYPE_CHECKING:
    from pathlib import Path


def test_dictarray_regular_returns_plain_sequence() -> None:
    arr = DictArray(folder=None, shape=(1,), irregular=False)
    arr.dump((0,), [1, 2])
    assert isinstance(arr[(0,)], list)
    out = arr.to_array(splat_internal=False)
    assert isinstance(out, np.ma.MaskedArray)
    assert np.all(out.mask == np.array([False]))


def test_dictarray_irregular_returns_masked_array() -> None:
    arr = DictArray(
        folder=None,
        shape=(1,),
        internal_shape=(3,),
        shape_mask=(True, False),
        irregular=True,
    )
    arr.dump((0,), [1, 2, 3])
    assert not isinstance(arr[(0, 0)], list)
    out = arr.to_array(splat_internal=True)
    assert isinstance(out, np.ma.MaskedArray)


def test_dictarray_irregular_getitem_returns_masked_array() -> None:
    arr = DictArray(
        folder=None,
        shape=(1,),
        irregular=True,
    )
    arr.dump((0,), [1, 2])
    result = arr[(0,)]
    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_equal(result.data, np.array([1, 2], dtype=object))


def test_dictarray_irregular_to_array_skips_missing_entries() -> None:
    arr = DictArray(
        folder=None,
        shape=(1,),
        internal_shape=(3,),
        shape_mask=(True, False),
        irregular=True,
    )
    arr.dump((0,), [42])
    array = arr.to_array(splat_internal=True)
    assert np.ma.is_masked(array[0, 1])
    assert np.ma.is_masked(array[0, 2])


def test_filearray_regular_returns_plain_sequence(tmp_path: Path) -> None:
    arr = FileArray(folder=tmp_path, shape=(1,), irregular=False)
    arr.dump((0,), [1, 2])
    assert isinstance(arr[(0,)], list)
    out = arr.to_array(splat_internal=False)
    assert isinstance(out, np.ma.MaskedArray)
    assert np.all(out.mask == np.array([False]))


def test_filearray_irregular_returns_masked_array(tmp_path: Path) -> None:
    arr = FileArray(
        folder=tmp_path,
        shape=(1,),
        internal_shape=(3,),
        shape_mask=(True, False),
        irregular=True,
    )
    arr.dump((0,), [1, 2, 3])
    assert not isinstance(arr[(0, 0)], list)
    out = arr.to_array(splat_internal=True)
    assert isinstance(out, np.ma.MaskedArray)


def test_filearray_irregular_getitem_root_returns_masked(tmp_path: Path) -> None:
    arr = FileArray(folder=tmp_path, shape=(1,), irregular=True)
    arr.dump((0,), value=True)
    result = arr[(0,)]
    assert isinstance(result, np.ma.MaskedArray)
    assert result.item() is True


@pytest.mark.parametrize(
    "storage_factory",
    [
        pytest.param("dict", id="dict"),
        pytest.param("file", id="file"),
    ],
)
def test_irregular_slice_returns_masked_array(storage_factory: str, tmp_path: Path) -> None:
    if storage_factory == "dict":
        arr: DictArray | FileArray = DictArray(
            folder=None,
            shape=(1,),
            internal_shape=(3,),
            shape_mask=(True, False),
            irregular=True,
        )
    else:
        arr = FileArray(
            folder=tmp_path,
            shape=(1,),
            internal_shape=(3,),
            shape_mask=(True, False),
            irregular=True,
        )

    arr.dump((0,), [1, 2])
    result = arr[(slice(None), slice(None))]
    assert isinstance(result, np.ma.MaskedArray)
    mask = np.ma.getmaskarray(result)
    assert mask.shape == (1, 3)
    assert mask[0, 2]
