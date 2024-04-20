import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipefunc._filearray import (
    FileBasedObjectArray,
    _load_all,
    dump,
    load,
)


def test_load_and_dump(tmp_path):
    obj = {"a": 1, "b": [2, 3]}
    file_path = tmp_path / "test.pickle"
    dump(obj, file_path)
    loaded_obj = load(file_path)
    assert loaded_obj == obj


def test_file_based_object_array_init():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileBasedObjectArray(folder, shape)
        assert arr.folder == folder
        assert arr.shape == shape
        assert arr.strides == (12, 4, 1)
        assert arr.filename_template == "__{:d}__.pickle"


def test_file_based_object_array_properties():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileBasedObjectArray(folder, shape)
        assert arr.size == 24
        assert arr.rank == 3


def test_file_based_object_array_normalize_key():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileBasedObjectArray(folder, shape)
        assert arr._normalize_key((1, 2, 3)) == (1, 2, 3)
        with pytest.raises(IndexError):
            arr._normalize_key((1, 2, 3, 4))
        with pytest.raises(NotImplementedError):
            arr._normalize_key((slice(None), 1, 2))
        with pytest.raises(IndexError):
            arr._normalize_key((1, 2, 10))


def test_file_based_object_array_index_to_file():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileBasedObjectArray(folder, shape)
        assert arr._index_to_file(0) == folder / "__0__.pickle"
        assert arr._index_to_file(23) == folder / "__23__.pickle"


def test_file_based_object_array_key_to_file():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileBasedObjectArray(folder, shape)
        assert arr._key_to_file((0, 0, 0)) == folder / "__0__.pickle"
        assert arr._key_to_file((1, 2, 3)) == folder / "__23__.pickle"


def test_file_based_object_array_files():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileBasedObjectArray(folder, shape)
        files = list(arr._files())
        assert len(files) == 6
        assert files[0] == folder / "__0__.pickle"
        assert files[-1] == folder / "__5__.pickle"


def test_file_based_object_array_getitem():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileBasedObjectArray(folder, shape)
        dump({"a": 1}, arr._key_to_file((0, 0)))
        dump({"b": 2}, arr._key_to_file((1, 2)))
        assert arr[0, 0] == {"a": 1}
        assert arr[1, 2] == {"b": 2}
        assert arr[0, 1] is np.ma.masked
        with pytest.raises(NotImplementedError):
            arr[0:1, 0]


def test_file_based_object_array_to_array():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileBasedObjectArray(folder, shape)
        dump({"a": 1}, arr._key_to_file((0, 0)))
        dump({"b": 2}, arr._key_to_file((1, 2)))
        result = arr.to_array()
        assert result.shape == (2, 3)
        assert result.dtype == object
        assert result[0, 0] == {"a": 1}
        assert result[1, 2] == {"b": 2}
        assert result[0, 1] is np.ma.masked
        assert result[1, 0] is np.ma.masked


def test_file_based_object_array_dump():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileBasedObjectArray(folder, shape)
        arr.dump((0, 0), {"a": 1})
        arr.dump((1, 2), {"b": 2})
        assert load(arr._key_to_file((0, 0))) == {"a": 1}
        assert load(arr._key_to_file((1, 2))) == {"b": 2}
        with pytest.raises(NotImplementedError):
            arr.dump((slice(0, 1), 0), {"c": 3})


def test_load_all(tmp_path):
    file1 = tmp_path / "file1.pickle"
    file2 = tmp_path / "file2.pickle"
    file3 = tmp_path / "file3.pickle"  # Non-existent file
    dump({"a": 1}, file1)
    dump({"b": 2}, file2)
    result = _load_all([file1, file2, file3])
    assert result == [{"a": 1}, {"b": 2}, None]
