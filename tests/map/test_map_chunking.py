import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from pipefunc.map._run import _get_optimal_chunk_size, get_ncores


def test_get_optimal_chunk_size_small_input() -> None:
    """Test that very small inputs return chunk size of 1."""
    with ProcessPoolExecutor() as ex:
        n_cores = get_ncores(ex)
        assert _get_optimal_chunk_size(total_items=n_cores, executor=ex) == 1
        assert _get_optimal_chunk_size(total_items=n_cores * 2 - 1, executor=ex) == 1


def test_get_optimal_chunk_size_medium_input() -> None:
    """Test chunk sizes for medium-sized inputs."""
    with ProcessPoolExecutor() as ex:
        n_cores = get_ncores(ex)
        # For 1000 items, 8 cores, 20 chunks per worker
        # chunk_size should be ceil(1000 / (8 * 20)) = ceil(6.25) = 7
        expected = -(-1000 // (n_cores * 20))  # ceiling division
        assert _get_optimal_chunk_size(total_items=1000, executor=ex) == expected


def test_get_optimal_chunk_size_large_input() -> None:
    """Test chunk sizes for large inputs."""
    with ProcessPoolExecutor() as ex:
        n_cores = get_ncores(ex)
        # For 1M items, should give substantial chunks
        expected = -(-1_000_000 // (n_cores * 20))  # ceiling division
        assert _get_optimal_chunk_size(total_items=1_000_000, executor=ex) == expected


def test_get_optimal_chunk_size_custom_min_chunks() -> None:
    """Test that custom min_chunks_per_worker affects chunk size."""
    with ProcessPoolExecutor() as ex:
        # With default min_chunks=20
        default_chunk = _get_optimal_chunk_size(total_items=1000, executor=ex)

        # With min_chunks=10 (should give larger chunks)
        large_chunk = _get_optimal_chunk_size(
            total_items=1000,
            min_chunks_per_worker=10,
            executor=ex,
        )

        # With min_chunks=40 (should give smaller chunks)
        small_chunk = _get_optimal_chunk_size(
            total_items=1000,
            min_chunks_per_worker=40,
            executor=ex,
        )

        assert small_chunk < default_chunk < large_chunk


def test_get_optimal_chunk_size_edge_cases() -> None:
    """Test edge cases and invalid inputs."""
    with ProcessPoolExecutor() as ex:
        # Zero items should return 1
        assert _get_optimal_chunk_size(total_items=0, executor=ex) == 1

        # Negative items should return 1
        assert _get_optimal_chunk_size(total_items=-10, executor=ex) == 1

        # Very large numbers shouldn't cause overflow
        huge = 10**9
        assert _get_optimal_chunk_size(total_items=huge, executor=ex) > 0


@pytest.mark.parametrize(
    ("total_items", "min_chunks", "expected_minimum"),
    [
        (1000, 20, 1),  # Should always return at least 1
        (1000, 1000, 1),  # Even with many min_chunks, should return at least 1
    ],
)
def test_get_optimal_chunk_size_minimum_bounds(total_items, min_chunks, expected_minimum):
    """Test that chunk size is never less than 1."""
    with ProcessPoolExecutor() as ex:
        result = _get_optimal_chunk_size(
            total_items,
            min_chunks_per_worker=min_chunks,
            executor=ex,
        )
        assert result >= expected_minimum


def test_get_optimal_chunk_size_reasonable_defaults() -> None:
    """Test that default parameters give reasonable chunk sizes."""
    with ProcessPoolExecutor() as ex:
        n_cores = get_ncores(ex)
        result = _get_optimal_chunk_size(total_items=10000, executor=ex)

        # Check that we get a reasonable number of total chunks
        total_chunks = 10000 / result
        assert n_cores * 10 <= total_chunks <= n_cores * 30  # reasonable range


def test_get_ncores_process_pool() -> None:
    """Test getting core count from ProcessPoolExecutor."""
    with ProcessPoolExecutor(max_workers=4) as ex:
        assert get_ncores(ex) == 4


def test_get_ncores_thread_pool() -> None:
    """Test getting core count from ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=4) as ex:
        assert get_ncores(ex) == 4


def test_get_ncores_invalid() -> None:
    """Test that invalid executor raises TypeError."""

    class DummyExecutor:
        pass

    with pytest.raises(TypeError, match="Cannot get number of cores for"):
        get_ncores(DummyExecutor())  # type: ignore[arg-type]

    with pytest.warns(UserWarning, match="Automatic chunksize calculation failed"):
        n = _get_optimal_chunk_size(1000, DummyExecutor())  # type: ignore[arg-type]
    assert n == math.ceil(1000 / 20) == 50
