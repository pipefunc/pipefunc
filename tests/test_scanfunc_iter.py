"""Test suite for generator-based ScanIterFunc."""

from __future__ import annotations

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._scanfunc_iter import ScanIterFunc


class TestScanIterFunc:
    """Test generator-based scan functionality."""

    def test_basic_accumulator(self):
        """Test basic accumulation with generator."""

        @PipeFunc.scan_iter(output_name="cumsum")
        def accumulator(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        assert isinstance(accumulator, ScanIterFunc)

        # Test with simple list
        values = [1, 2, 3, 4, 5]
        pipeline = Pipeline([accumulator])
        result = pipeline.run("cumsum", kwargs={"values": values})

        # Should return array of accumulated values
        expected = np.array([1, 3, 6, 10, 15])
        assert np.array_equal(result, expected)

    def test_fibonacci_generator(self):
        """Test Fibonacci sequence with generator."""

        @PipeFunc.scan_iter(output_name="fibonacci")
        def fib_gen(n_steps: int, a: int = 0, b: int = 1):
            for _ in range(n_steps):
                a, b = b, a + b
                yield b

        pipeline = Pipeline([fib_gen])
        result = pipeline.run("fibonacci", kwargs={"n_steps": 10})

        # First 10 Fibonacci numbers after [0, 1]
        expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        assert list(result) == expected

    def test_return_final_only(self):
        """Test returning only final value."""

        @PipeFunc.scan_iter(output_name="final_sum", return_final_only=True)
        def sum_final(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        values = [1, 2, 3, 4, 5]
        pipeline = Pipeline([sum_final])
        result = pipeline.run("final_sum", kwargs={"values": values})

        # Should return only final sum
        assert result == 15

    def test_complex_state_dict(self):
        """Test generator with complex state as dict."""

        @PipeFunc.scan_iter(output_name="trajectory")
        def particle_sim(time_steps: list[float], x: float = 0.0, v: float = 1.0, dt: float = 0.1):
            for t in time_steps:
                x += v * dt
                v *= 0.99  # damping
                yield {"t": t, "x": x, "v": v}

        time_steps = [0.0, 0.1, 0.2, 0.3]
        pipeline = Pipeline([particle_sim])
        result = pipeline.run("trajectory", kwargs={"time_steps": time_steps, "dt": 0.1})

        # Check it's a list of dicts
        assert isinstance(result, list)
        assert len(result) == 4
        assert all("t" in r and "x" in r and "v" in r for r in result)

        # Check values make sense
        assert result[0]["t"] == 0.0
        assert result[0]["x"] == pytest.approx(0.1)  # x + v*dt = 0 + 1*0.1
        assert result[-1]["t"] == 0.3

    def test_early_stopping(self):
        """Test generator with early stopping."""

        @PipeFunc.scan_iter(output_name="converged")
        def optimize(max_iters: int, x0: float = 5.0, tol: float = 1e-6):
            x = x0
            for i in range(max_iters):
                gradient = 2 * x  # gradient of x^2
                x = x - 0.1 * gradient

                yield {"iter": i, "x": x, "gradient": gradient}

                if abs(gradient) < tol:
                    break  # Early stopping!

        pipeline = Pipeline([optimize])
        result = pipeline.run("converged", kwargs={"max_iters": 100, "tol": 1e-4})

        # Should stop early
        assert len(result) < 100
        assert abs(result[-1]["gradient"]) < 1e-4

    def test_empty_generator(self):
        """Test generator that yields nothing."""

        @PipeFunc.scan_iter(output_name="empty")
        def empty_gen(n: int):
            if n < 0:
                return
            yield from range(n)

        pipeline = Pipeline([empty_gen])
        result = pipeline.run("empty", kwargs={"n": 0})

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_with_xs_parameter(self):
        """Test generator with explicit xs parameter."""

        @PipeFunc.scan_iter(output_name="doubled", xs="values")
        def double_values(values: list[int]):
            # The values parameter is treated as xs
            for x in values:
                yield x * 2

        pipeline = Pipeline([double_values])
        result = pipeline.run("doubled", kwargs={"values": [1, 2, 3, 4]})

        expected = np.array([2, 4, 6, 8])
        assert np.array_equal(result, expected)

    def test_generator_with_pipeline_map(self):
        """Test generator integrated with pipeline.map."""

        @pipefunc(output_name="values", mapspec="batch_id[i] -> values[i]")
        def generate_values(batch_id: int) -> list[int]:
            start = batch_id * 3 + 1
            return [start, start + 1, start + 2]

        @PipeFunc.scan_iter(output_name="cumsum", mapspec="values[i] -> cumsum[i]")
        def cumulative_sum(values: list[int], total: int = 0):
            for x in values:
                total += x
                yield total

        inputs = {"batch_id": [0, 1, 2]}
        pipeline = Pipeline([generate_values, cumulative_sum])
        results = pipeline.map(inputs, run_folder="test_scan_iter_map", parallel=False)

        # Check results
        expected = [
            [1, 3, 6],  # batch 0: cumsum of [1, 2, 3]
            [4, 9, 15],  # batch 1: cumsum of [4, 5, 6]
            [7, 15, 24],  # batch 2: cumsum of [7, 8, 9]
        ]

        for i, cumsum in enumerate(results["cumsum"].output):
            assert np.array_equal(cumsum, expected[i])

    def test_non_generator_error(self):
        """Test error when function is not a generator."""

        @PipeFunc.scan_iter(output_name="bad")
        def not_a_generator(values: list[int]):
            # This is not a generator!
            return sum(values)

        pipeline = Pipeline([not_a_generator])

        with pytest.raises(TypeError, match="must be a generator"):
            pipeline.run("bad", kwargs={"values": [1, 2, 3]})

    def test_generator_with_renames(self):
        """Test generator with parameter renames."""

        @PipeFunc.scan_iter(output_name="result", renames={"data": "values"})
        def renamed_gen(data: list[int]):
            total = 0
            for x in data:
                total += x
                yield total

        pipeline = Pipeline([renamed_gen])
        result = pipeline.run("result", kwargs={"values": [1, 2, 3]})

        expected = np.array([1, 3, 6])
        assert np.array_equal(result, expected)

    def test_generator_with_bounds(self):
        """Test generator with bound parameters."""

        @PipeFunc.scan_iter(output_name="scaled", bound={"scale": 2.0})
        def scaled_sum(values: list[int], scale: float = 1.0):
            total = 0.0
            for x in values:
                total += x * scale
                yield total

        pipeline = Pipeline([scaled_sum])
        result = pipeline.run("scaled", kwargs={"values": [1, 2, 3]})

        # With scale=2.0: [2, 6, 12]
        expected = np.array([2, 6, 12])
        assert np.array_equal(result, expected)

    def test_generator_return_value(self):
        """Test generator with return statement."""

        @PipeFunc.scan_iter(output_name="with_return")
        def gen_with_return(n: int):
            total = 0
            for i in range(n):
                total += i
                yield total
            return total  # Return final value

        pipeline = Pipeline([gen_with_return])
        result = pipeline.run("with_return", kwargs={"n": 5})

        # Should include values from yield, and return value
        # 0, 0+1=1, 1+2=3, 3+3=6, 6+4=10
        expected = np.array([0, 1, 3, 6, 10])
        assert np.array_equal(result, expected)

    def test_mixed_type_yields(self):
        """Test generator yielding mixed types."""

        @PipeFunc.scan_iter(output_name="mixed")
        def mixed_types(n: int):
            yield 42
            yield "hello"
            yield {"key": "value"}
            yield [1, 2, 3]

        pipeline = Pipeline([mixed_types])
        result = pipeline.run("mixed", kwargs={"n": 4})

        # Should return list (not numpy array) for mixed types
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0] == 42
        assert result[1] == "hello"
        assert result[2] == {"key": "value"}
        assert result[3] == [1, 2, 3]

    def test_copy_method(self):
        """Test copying a ScanIterFunc."""

        @PipeFunc.scan_iter(output_name="original", return_final_only=False)
        def original(values: list[int]):
            for x in values:
                yield x * 2

        # Make a copy with different settings
        copied = original.copy(output_name="copied", return_final_only=True)

        assert copied._output_name == "copied"
        assert copied.return_final_only is True
        assert original.return_final_only is False

        # Test they work differently
        pipeline1 = Pipeline([original])
        pipeline2 = Pipeline([copied])

        values = [1, 2, 3]
        result1 = pipeline1.run("original", kwargs={"values": values})
        result2 = pipeline2.run("copied", kwargs={"values": values})

        assert np.array_equal(result1, [2, 4, 6])
        assert result2 == 6  # Only final value

    def test_generator_with_defaults(self):
        """Test generator with custom defaults."""

        @PipeFunc.scan_iter(output_name="with_defaults", defaults={"offset": 10, "scale": 2})
        def custom_defaults(values: list[int], offset: int = 0, scale: int = 1):
            for x in values:
                yield x * scale + offset

        pipeline = Pipeline([custom_defaults])
        result = pipeline.run("with_defaults", kwargs={"values": [1, 2, 3]})

        # With offset=10, scale=2: [12, 14, 16]
        expected = np.array([12, 14, 16])
        assert np.array_equal(result, expected)
