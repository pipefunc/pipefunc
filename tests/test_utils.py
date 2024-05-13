import cloudpickle
import pytest

from pipefunc._utils import _cached_load, load


@pytest.fixture(autouse=True)  # Automatically use in all tests
def _clear_cache() -> None:
    _cached_load.cache_clear()  # Clear the cache before each test


def create_temp_file(tmp_path, data, filename="test.pickle"):
    file_path = tmp_path / filename
    with file_path.open("wb") as f:
        cloudpickle.dump(data, f)
    return file_path


def test_cached_load_hits_cache(tmp_path):
    assert _cached_load.cache_info().hits == 0
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # First load to populate the cache
    result1 = load(file_path, cache=True)
    # Second load should hit the cache
    result2 = load(file_path, cache=True)

    # Check that the function returns the correct data
    assert result1 == data
    assert result1 == result2
    # Ensure the file was not loaded again from disk
    assert _cached_load.cache_info().hits == 1


def test_cached_load_invalidates_on_file_change(tmp_path):
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # First load to populate the cache
    result1 = load(file_path, cache=True)
    assert _cached_load.cache_info().misses == 1

    # Modify the file
    with file_path.open("wb") as f:
        data_modified = {"key": "new value"}
        cloudpickle.dump(data_modified, f)

    # Second load should not hit the cache due to file modification
    result2 = load(file_path, cache=True)

    # Check that the results are correct
    assert result1 == {"key": "value"}
    assert result2 == {"key": "new value"}

    # Ensure the file was loaded again from disk
    assert _cached_load.cache_info().misses == 2
    assert _cached_load.cache_info().hits == 0


@pytest.mark.parametrize("modify_size", [True, False])
def test_cache_invalidation_on_file_size_change(tmp_path, modify_size):
    data = {"key": "value"}
    file_path = create_temp_file(tmp_path, data)

    # Load once to cache
    result1 = load(file_path, cache=True)

    # Modify file and optionally change its size
    with file_path.open("wb") as f:
        if modify_size:
            data_modified = {"key": "new value", "extra": "data to increase size"}
        else:
            data_modified = {"key": "new value"}
        cloudpickle.dump(data_modified, f)

    # Second load should reload because of file size change or content change
    result2 = load(file_path, cache=True)

    # Ensure data is as expected
    assert result1 != result2
    # Check cache was invalidated (i.e., miss occurred)
    assert _cached_load.cache_info().misses == 2
