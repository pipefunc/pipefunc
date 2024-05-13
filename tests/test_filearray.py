import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipefunc._filearray import FileArray, _load_all, dump, load


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
        arr = FileArray(folder, shape)
        assert arr.folder == folder
        assert arr.shape == shape
        assert arr.strides == (12, 4, 1)
        assert arr.filename_template == "__{:d}__.pickle"


def test_file_based_object_array_properties():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)
        assert arr.size == 24
        assert arr.rank == 3


def test_file_based_object_array_normalize_key():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)
        assert arr._normalize_key((1, 2, 3)) == (1, 2, 3)
        assert arr._normalize_key((slice(None), 1, 2)) == (slice(None), 1, 2)
        with pytest.raises(IndexError):
            arr._normalize_key((1, 2, 3, 4))
        with pytest.raises(IndexError):
            arr._normalize_key((1, 2, 10))


def test_file_based_object_array_index_to_file():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)
        assert arr._index_to_file(0) == folder / "__0__.pickle"
        assert arr._index_to_file(23) == folder / "__23__.pickle"


def test_file_based_object_array_key_to_file():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)
        assert arr._key_to_file((0, 0, 0)) == folder / "__0__.pickle"
        assert arr._key_to_file((1, 2, 3)) == folder / "__23__.pickle"


def test_file_based_object_array_files():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileArray(folder, shape)
        files = list(arr._files())
        assert len(files) == 6
        assert files[0] == folder / "__0__.pickle"
        assert files[-1] == folder / "__5__.pickle"


def test_file_based_object_array_getitem():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileArray(folder, shape)
        arr.dump((0, 0), {"a": 1})
        arr.dump((1, 2), {"b": 2})
        assert arr[0, 0] == {"a": 1}
        assert arr[1, 2] == {"b": 2}
        assert arr[0, 1] is np.ma.masked
        assert arr[0:1, 0] == {"a": 1}


def test_file_based_object_array_to_array():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3)
        arr = FileArray(folder, shape)
        arr.dump((0, 0), {"a": 1})
        arr.dump((1, 2), {"b": 2})
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
        arr = FileArray(folder, shape)
        arr.dump((0, 0), {"a": 1})
        arr.dump((1, 2), {"b": 2})
        assert load(arr._key_to_file((0, 0))) == {"a": 1}
        assert load(arr._key_to_file((1, 2))) == {"b": 2}
        arr.dump((slice(0, 1), 0), {"c": 3})
        assert load(arr._key_to_file((0, 0))) == {"c": 3}


def test_load_all(tmp_path):
    file1 = tmp_path / "file1.pickle"
    file2 = tmp_path / "file2.pickle"
    file3 = tmp_path / "file3.pickle"  # Non-existent file
    dump({"a": 1}, file1)
    dump({"b": 2}, file2)
    result = _load_all([file1, file2, file3])
    assert result == [{"a": 1}, {"b": 2}, None]


def test_file_array_getitem_with_slicing():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)
        arr.dump((0, 0, 0), {"a": 1})
        arr.dump((0, 1, 0), {"b": 2})
        arr.dump((1, 0, 0), {"c": 3})
        arr.dump((1, 1, 0), {"d": 4})

        # Test slicing along a single axis
        result = arr[0, :, 0]
        assert result.shape == (3,)
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}
        assert result[2] is np.ma.masked

        # Test slicing along multiple axes
        result = arr[1, 0:2, 0]
        assert result.shape == (2,)
        assert result[0] == {"c": 3}
        assert result[1] == {"d": 4}

        # Test slicing with step
        result = arr[0, ::2, 0]
        assert result.shape == (2,)
        assert result[0] == {"a": 1}
        assert result[1] is np.ma.masked


def test_file_array_dump_with_slicing():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)

        # Test dumping with slicing along a single axis
        arr.dump((0, slice(None), 0), {"a": 1})
        assert load(arr._key_to_file((0, 0, 0))) == {"a": 1}
        assert load(arr._key_to_file((0, 1, 0))) == {"a": 1}
        assert load(arr._key_to_file((0, 2, 0))) == {"a": 1}

        # Test dumping with slicing along multiple axes
        arr.dump((slice(None), 0, 0), {"b": 2})
        assert load(arr._key_to_file((0, 0, 0))) == {"b": 2}
        assert load(arr._key_to_file((1, 0, 0))) == {"b": 2}

        # Test dumping with step
        arr.dump((0, slice(None, None, 2), 0), {"c": 3})
        assert load(arr._key_to_file((0, 0, 0))) == {"c": 3}
        assert load(arr._key_to_file((0, 2, 0))) == {"c": 3}


def test_file_array_slice_indices():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)

        key = (0, slice(None), slice(1, 3))
        indices = arr._slice_indices(key)
        assert indices == [range(1), range(3), range(1, 3)]

        key = (slice(None), 1, slice(None, None, 2))
        indices = arr._slice_indices(key)
        assert indices == [range(2), range(1, 2), range(0, 4, 2)]


def test_file_array_normalize_key_with_slicing():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4)
        arr = FileArray(folder, shape)

        key = (0, slice(None), 1)
        normalized_key = arr._normalize_key(key)
        assert normalized_key == (0, slice(None, None, None), 1)

        key = (slice(None), -1, slice(1, None, 2))
        normalized_key = arr._normalize_key(key)
        assert normalized_key == (slice(None, None, None), 2, slice(1, None, 2))

        with pytest.raises(IndexError):
            arr._normalize_key((0, slice(None), 10))


def test_high_dim_with_slicing():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (2, 3, 4, 5)
        arr = FileArray(folder, shape)
        np_arr = np.zeros(shape, dtype=object)
        np_arr[:] = np.ma.masked
        keys = [
            (0, 0, slice(None), 0),
            (0, 1, slice(None), 0),
            (1, 0, slice(None), 0),
            (1, 1, slice(None), 0),
        ]
        values = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
        for k, v in zip(keys, values):
            np_arr[k] = v
            arr.dump(k, v)

        assert (arr.to_array() == np_arr).all()

        # Test slicing along a single axis
        result = arr[0, :, :, 0]
        assert result.shape == (3, 4), result.shape
        assert result[0, 0] == {"a": 1}
        assert result[1, 0] == {"b": 2}
        assert result[2, 0] is np.ma.masked

        # Test slicing along multiple axes
        result = arr[1, 0:2, :, 0]
        assert result.shape == (2, 4)
        assert result[0, 0] == {"c": 3}
        assert result[1, 0] == {"d": 4}

        # Test slicing with step
        result = arr[0, ::2, :, 0]
        result_arr = np_arr[0, ::2, :, 0]
        assert result.shape == (2, 4)
        assert result[0, 0] == result_arr[0, 0]
        assert result[1, 0] is np.ma.masked
        assert result[0, 1] == result_arr[0, 1]
        assert result[1, 1] is np.ma.masked


def test_slice_indices_with_step_size():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (5, 6, 7)
        arr = FileArray(folder, shape)

        # Test case 1: Slice indices with step size along one axis
        key = (slice(0, 5, 2), 1, 2)
        indices = arr._slice_indices(key)
        assert indices == [range(0, 5, 2), range(1, 2), range(2, 3)]

        # Test case 2: Slice indices with step size along multiple axes
        key = (slice(1, 4, 2), slice(2, 5, 2), slice(3, 7, 3))
        indices = arr._slice_indices(key)
        assert indices == [range(1, 4, 2), range(2, 5, 2), range(3, 7, 3)]

        # Test case 3: Slice indices with step size and negative indices
        key = (slice(4, 1, -2), slice(5, 2, -2), slice(6, 1, -3))
        indices = arr._slice_indices(key)
        assert indices == [range(4, 1, -2), range(5, 2, -2), range(6, 1, -3)]


def test_key_to_file():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (3, 4, 5)
        arr = FileArray(folder, shape)

        key = (1, 2, 3)
        file_path = arr._key_to_file(key)
        assert file_path == (folder / "__33__.pickle")

        key = (0, 1, 2)
        file_path = arr._key_to_file(key)
        assert file_path == (folder / "__7__.pickle")


def test_sliced_arange():
    with tempfile.TemporaryDirectory() as tempdir:
        folder = Path(tempdir)
        shape = (3, 4, 5)
        arr = FileArray(folder, shape)
        np_arr = np.arange(np.prod(shape)).reshape(shape)
        for key in np.ndindex(shape):
            arr.dump(key, np_arr[key])

        assert (arr[:, :, :] == np_arr[:, :, :]).all()
        assert (arr[:, ::2, :] == np_arr[:, ::2, :]).all()
        assert (arr[:, ::3, :] == np_arr[:, ::3, :]).all()
        assert (arr[:, ::-1, :] == np_arr[:, ::-1, :]).all()
        assert (arr[:, ::-1, ::2] == np_arr[:, ::-1, ::2]).all()
        assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
        assert (arr[1:, ::-1, ::2] == np_arr[1:, ::-1, ::2]).all()
        assert (arr[1:, ::-1, ::-1] == np_arr[1:, ::-1, ::-1]).all()
        assert (arr[1:, ::-1, -1] == np_arr[1:, ::-1, -1]).all()
        assert (arr[2, ::-1, -1] == np_arr[2, ::-1, -1]).all()
        assert (arr[:1, :1, -1] == np_arr[:1, :1, -1]).all()
        assert (arr[:1, :1, 5:1:-1] == np_arr[:1, :1, 5:1:-1]).all()
