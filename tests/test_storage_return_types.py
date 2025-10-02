from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
