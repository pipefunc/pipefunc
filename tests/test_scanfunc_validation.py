"""Cross-validation tests between current ScanFunc and new scan_iter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc


class TestScanFuncValidation:
    """Validate that scan_iter produces the same results as scan."""

    def test_simple_accumulator_equivalence(self):
        """Test simple accumulator produces same results."""

        # Current design
        @PipeFunc.scan(output_name="scan_result", xs="values")
        def accumulator_current(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        # New design
        @PipeFunc.scan_iter(output_name="iter_result")
        def accumulator_iter(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        values = [1, 2, 3, 4, 5]

        pipeline1 = Pipeline([accumulator_current])
        result1 = pipeline1.run("scan_result", kwargs={"values": values})

        pipeline2 = Pipeline([accumulator_iter])
        result2 = pipeline2.run("iter_result", kwargs={"values": values})

        assert np.array_equal(result1, result2)

    def test_fibonacci_equivalence(self):
        """Test Fibonacci sequence produces same results."""

        # Current design
        @PipeFunc.scan(output_name="scan_fib", xs="n_steps")
        def fib_current(x: int, a: int = 0, b: int = 1) -> tuple[dict[str, Any], int]:
            next_val = a + b
            carry = {"a": b, "b": next_val}
            return carry, next_val

        # New design
        @PipeFunc.scan_iter(output_name="iter_fib")
        def fib_iter(n_steps: list[int], a: int = 0, b: int = 1):
            for _ in n_steps:
                a, b = b, a + b
                yield b

        n_steps = list(range(10))

        pipeline1 = Pipeline([fib_current])
        result1 = pipeline1.run("scan_fib", kwargs={"n_steps": n_steps})

        pipeline2 = Pipeline([fib_iter])
        result2 = pipeline2.run("iter_fib", kwargs={"n_steps": n_steps})

        assert np.array_equal(result1, result2)

    def test_complex_state_equivalence(self):
        """Test complex state tracking produces same results."""

        # Current design
        @PipeFunc.scan(output_name="scan_stats", xs="values")
        def stats_current(
            x: float,
            count: int = 0,
            sum_val: float = 0.0,
            sum_sq: float = 0.0,
        ) -> tuple[dict[str, Any], dict[str, float]]:
            new_count = count + 1
            new_sum = sum_val + x
            new_sum_sq = sum_sq + x**2

            carry = {"count": new_count, "sum_val": new_sum, "sum_sq": new_sum_sq}
            stats = {
                "count": new_count,
                "mean": new_sum / new_count,
                "var": (new_sum_sq / new_count) - (new_sum / new_count) ** 2,
            }
            return carry, stats

        # New design
        @PipeFunc.scan_iter(output_name="iter_stats")
        def stats_iter(
            values: list[float],
            count: int = 0,
            sum_val: float = 0.0,
            sum_sq: float = 0.0,
        ):
            for x in values:
                count += 1
                sum_val += x
                sum_sq += x**2

                yield {
                    "count": count,
                    "mean": sum_val / count,
                    "var": (sum_sq / count) - (sum_val / count) ** 2,
                }

        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pipeline1 = Pipeline([stats_current])
        result1 = pipeline1.run("scan_stats", kwargs={"values": values})

        pipeline2 = Pipeline([stats_iter])
        result2 = pipeline2.run("iter_stats", kwargs={"values": values})

        # Compare each dict
        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1["count"] == r2["count"]
            assert r1["mean"] == pytest.approx(r2["mean"])
            assert r1["var"] == pytest.approx(r2["var"])

    def test_final_only_equivalence(self):
        """Test return_intermediate=False equivalence."""

        # Current design
        @PipeFunc.scan(output_name="scan_final", xs="values", return_intermediate=False)
        def sum_final_current(x: int, total: int = 0) -> tuple[dict[str, Any], None]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, None

        # New design
        @PipeFunc.scan_iter(output_name="iter_final", return_final_only=True)
        def sum_final_iter(values: list[int], total: int = 0):
            for x in values:
                total += x
            yield {"total": total}

        values = [1, 2, 3, 4, 5]

        pipeline1 = Pipeline([sum_final_current])
        result1 = pipeline1.run("scan_final", kwargs={"values": values})

        pipeline2 = Pipeline([sum_final_iter])
        result2 = pipeline2.run("iter_final", kwargs={"values": values})

        assert result1 == result2  # Both should be {"total": 15}

    def test_parallel_execution_equivalence(self):
        """Test parallel execution with pipeline.map produces same results."""

        # Generate values function (same for both)
        @pipefunc(output_name="values", mapspec="batch_id[i] -> values[i]")
        def generate_values(batch_id: int) -> list[int]:
            start = batch_id * 3 + 1
            return [start, start + 1, start + 2]

        # Current design
        @PipeFunc.scan(
            output_name="scan_cumsum",
            xs="values",
            mapspec="values[i] -> scan_cumsum[i]",
        )
        def cumsum_current(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        # New design
        @PipeFunc.scan_iter(output_name="iter_cumsum", mapspec="values[i] -> iter_cumsum[i]")
        def cumsum_iter(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        inputs = {"batch_id": [0, 1, 2]}

        pipeline1 = Pipeline([generate_values, cumsum_current])
        results1 = pipeline1.map(inputs, run_folder="test_validation_scan", parallel=False)

        pipeline2 = Pipeline([generate_values, cumsum_iter])
        results2 = pipeline2.map(inputs, run_folder="test_validation_iter", parallel=False)

        # Compare results
        for i in range(3):
            assert np.array_equal(
                results1["scan_cumsum"].output[i],
                results2["iter_cumsum"].output[i],
            )

    @pytest.mark.skip(reason="Current ScanFunc rename behavior is complex, revisit later")
    def test_renames_equivalence(self):
        """Test parameter renames work the same."""
        # TODO: The current ScanFunc's signature transformation makes rename
        # behavior complex. The wrapper function has different parameters than
        # the original function, which interacts with renames in non-obvious ways.
        # This needs deeper investigation to ensure proper compatibility.

    def test_bound_parameters_equivalence(self):
        """Test bound parameters work the same."""

        # Current design
        @PipeFunc.scan(
            output_name="scan_bound",
            xs="values",
            bound={"scale": 2.0},
            defaults={"offset": 1.0},
        )
        def scaled_current(
            x: int,
            total: float = 0.0,
            scale: float = 1.0,
            offset: float = 0.0,
        ) -> tuple[dict[str, Any], float]:
            new_total = total + scale * x + offset
            carry = {"total": new_total}
            return carry, new_total

        # New design
        @PipeFunc.scan_iter(
            output_name="iter_bound",
            bound={"scale": 2.0},
            defaults={"offset": 1.0},
        )
        def scaled_iter(
            values: list[int],
            total: float = 0.0,
            scale: float = 1.0,
            offset: float = 0.0,
        ):
            for x in values:
                total += scale * x + offset
                yield total

        values = [1, 2, 3]

        pipeline1 = Pipeline([scaled_current])
        result1 = pipeline1.run("scan_bound", kwargs={"values": values})

        pipeline2 = Pipeline([scaled_iter])
        result2 = pipeline2.run("iter_bound", kwargs={"values": values})

        # With scale=2.0, offset=1.0:
        # (0 + 2*1 + 1) = 3, (3 + 2*2 + 1) = 8, (8 + 2*3 + 1) = 15
        expected = np.array([3.0, 8.0, 15.0])
        assert np.array_equal(result1, expected)
        assert np.array_equal(result2, expected)

    def test_empty_sequence_equivalence(self):
        """Test empty sequences behave the same."""

        # Current design
        @PipeFunc.scan(output_name="scan_empty", xs="values")
        def empty_current(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        # New design
        @PipeFunc.scan_iter(output_name="iter_empty")
        def empty_iter(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        values = []

        pipeline1 = Pipeline([empty_current])
        result1 = pipeline1.run("scan_empty", kwargs={"values": values})

        pipeline2 = Pipeline([empty_iter])
        result2 = pipeline2.run("iter_empty", kwargs={"values": values})

        # Both should return empty arrays
        assert len(result1) == 0
        assert len(result2) == 0
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
