import time

from pipefunc.map._progress import Status


def test_elapsed_time():
    status = Status(n_total=100)
    time.sleep(0.1)
    elapsed = status.elapsed_time()
    assert 0.1 <= elapsed < 0.2


def test_estimated_completion_time_initial():
    status = Status(n_total=100)
    assert status.estimated_completion_time() == float("inf")


def test_estimated_completion_time_during_progress():
    status = Status(n_total=100, n_start=0)
    status.n_current = 50

    # Simulating elapsed time by manually setting start_time to an earlier point
    status.start_time -= 5  # Assume 5 seconds have elapsed

    estimated_time = status.estimated_completion_time()
    assert 4 < estimated_time < 6


def test_estimated_completion_time_almost_complete():
    status = Status(n_total=100, n_start=0)
    status.n_current = 99

    # Simulating elapsed time by manually setting start_time to an earlier point
    status.start_time -= 5  # Assume 5 seconds have elapsed

    estimated_time = status.estimated_completion_time()
    assert 0 < estimated_time < 1


def test_estimated_completion_time_complete():
    status = Status(n_total=100, n_current=100)
    assert abs(status.estimated_completion_time()) < 1e-12
