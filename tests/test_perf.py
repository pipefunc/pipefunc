from __future__ import annotations

from pipefunc._profile import ResourceStats


def test_resource_stats_initialization():
    stats = ResourceStats()
    assert stats.num_executions == 0
    assert stats.average == 0.0
    assert stats.variance == 0.0
    assert stats.max == 0.0
    assert stats.std == 0.0


def test_resource_stats_update():
    stats = ResourceStats()
    execution_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    for time in execution_times:
        stats.update(time)

    assert stats.num_executions == len(execution_times)
    assert stats.average == sum(execution_times) / len(execution_times)
    assert stats.max == max(execution_times)


def test_resource_stats_std():
    stats = ResourceStats()
    execution_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    for time in execution_times:
        stats.update(time)

    variance = sum((xi - stats.average) ** 2 for xi in execution_times) / (len(execution_times) - 1)
    std_dev = variance**0.5

    assert stats.std == std_dev


def test_resource_stats_repr():
    stats = ResourceStats()
    execution_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    for time in execution_times:
        stats.update(time)

    assert repr(stats) == (
        f"ResourceStats(num_executions={stats.num_executions}, "
        f"average={stats.average:.4e}, max={stats.max:.4e}, std={stats.std:.4e})"
    )
