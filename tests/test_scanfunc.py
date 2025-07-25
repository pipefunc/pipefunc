"""Test suite for ScanFunc - iterative execution with feedback loops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._scanfunc import ScanFunc


# Module-level functions for multiprocessing tests
@pipefunc(output_name="values", mapspec="batch_id[i] -> values[i]")
def generate_values(batch_id: int) -> list[int]:
    """Generate values based on batch_id for testing."""
    start = batch_id * 3 + 1
    return [start, start + 1, start + 2]


@PipeFunc.scan(output_name="cumsum", xs="values", mapspec="values[i] -> cumsum[i]")
def cumulative_sum(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    """Cumulative sum scan function for testing."""
    new_total = total + x
    carry = {"total": new_total}
    return carry, new_total


class TestScanFunc:
    """Test basic ScanFunc functionality."""

    def test_basic_scan_decorator(self):
        """Test basic scan functionality with decorator."""

        # Simple accumulator that adds x to the carry value
        @PipeFunc.scan(output_name="result", xs="values")
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        assert isinstance(accumulator, ScanFunc)

        # Test with simple list
        values = [1, 2, 3, 4, 5]
        pipeline = Pipeline([accumulator])
        result = pipeline.run("result", kwargs={"values": values})

        # Should return array of accumulated values [1, 3, 6, 10, 15]
        expected = [1, 3, 6, 10, 15]
        assert np.array_equal(result, expected)

    def test_scan_with_multiple_carry_values(self):
        """Test scan with multiple values in carry dict."""

        @PipeFunc.scan(output_name="fibonacci", xs="n_steps")
        def fib_step(x: int, a: int = 0, b: int = 1) -> tuple[dict[str, Any], int]:
            next_val = a + b
            carry = {"a": b, "b": next_val}
            return carry, next_val

        # Generate first 10 Fibonacci numbers
        n_steps = list(range(10))
        pipeline = Pipeline([fib_step])
        result = pipeline.run("fibonacci", kwargs={"n_steps": n_steps})

        # First few Fibonacci numbers after initial [0, 1]
        expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        assert np.array_equal(result, expected)

    def test_scan_without_intermediate_results(self):
        """Test scan that only returns final carry, no intermediate results."""

        @PipeFunc.scan(output_name="final_sum", xs="values", return_intermediate=False)
        def sum_only_final(x: int, total: int = 0) -> tuple[dict[str, Any], None]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, None

        values = [1, 2, 3, 4, 5]
        pipeline = Pipeline([sum_only_final])
        result = pipeline.run("final_sum", kwargs={"values": values})

        # Should return only the final sum
        assert result == {"total": 15}

    def test_scan_with_pipeline_map(self):
        """Test scan integrated with pipeline.map for parallel execution."""
        # Multiple batches
        inputs = {
            "batch_id": [0, 1, 2],  # This will generate [1,2,3], [4,5,6], [7,8,9]
        }

        pipeline = Pipeline([generate_values, cumulative_sum])
        results = pipeline.map(inputs, run_folder="test_scan_map", parallel=False)

        # Each batch should have its own cumulative sum
        expected = [
            [1, 3, 6],  # batch 0: cumsum of [1, 2, 3]
            [4, 9, 15],  # batch 1: cumsum of [4, 5, 6]
            [7, 15, 24],  # batch 2: cumsum of [7, 8, 9]
        ]

        output = results["cumsum"].output
        # Convert array of arrays to list of lists for comparison
        output_list = [arr.tolist() for arr in output]
        assert output_list == expected

    def test_nested_pipefunc_scan(self):
        """Test scan with NestedPipeFunc for complex iterations."""

        @pipefunc(output_name="k1")
        def calc_k1(y: float, t: float, dt: float) -> float:
            return -y * dt  # Simple differential equation dy/dt = -y

        @pipefunc(output_name="k2")
        def calc_k2(y: float, k1: float, t: float, dt: float) -> float:
            return -(y + 0.5 * k1) * dt

        @pipefunc(output_name="y_next")
        def rk2_step(y: float, k1: float, k2: float) -> float:
            return y + k2  # Simplified RK2

        # Create nested pipeline for RK2 step
        rk2_pipeline = Pipeline([calc_k1, calc_k2, rk2_step])

        @rk2_pipeline.nest_funcs_scan(
            output_name="trajectory",
            xs="time_steps",
            output_nodes={"y_next"},
        )
        def rk2_scan(t: float, y: float = 1.0, dt: float = 0.1) -> tuple[dict[str, Any], float]:
            # This will be replaced by the nested pipeline execution
            return {}, 0.0

        time_steps = np.linspace(0, 1, 11)
        pipeline = Pipeline([rk2_scan])
        result = pipeline.run("trajectory", kwargs={"time_steps": time_steps, "dt": 0.1})

        # Should decay exponentially
        assert len(result) == 11
        assert result[0] > result[-1]  # Decaying
        assert all(result[i] > result[i + 1] for i in range(len(result) - 1))  # Monotonic

    def test_scan_error_handling(self):
        """Test error handling in scan functions."""

        @PipeFunc.scan(output_name="result", xs="values")
        def failing_scan(x: int, count: int = 0) -> tuple[dict[str, Any], int]:
            if x > 5:
                msg = f"Value {x} is too large!"
                raise ValueError(msg)
            carry = {"count": count + 1}
            return carry, count + 1

        values = [1, 2, 3, 10, 4]  # 10 will cause an error
        pipeline = Pipeline([failing_scan])

        with pytest.raises(ValueError, match="Value 10 is too large!"):
            pipeline.run("result", kwargs={"values": values})

    def test_scan_with_bound_and_defaults(self):
        """Test scan with bound parameters and defaults."""

        @PipeFunc.scan(
            output_name="scaled_sum",
            xs="values",
            bound={"scale": 2.0},
            defaults={"offset": 1.0},
        )
        def scaled_accumulator(
            x: int,
            total: float = 0.0,
            scale: float = 1.0,
            offset: float = 0.0,
        ) -> tuple[dict[str, Any], float]:
            new_total = total + scale * x + offset
            carry = {"total": new_total}
            return carry, new_total

        values = [1, 2, 3]
        pipeline = Pipeline([scaled_accumulator])

        # scale is bound to 2.0, offset defaults to 1.0
        result = pipeline.run("scaled_sum", kwargs={"values": values})
        # (0 + 2*1 + 1) = 3, (3 + 2*2 + 1) = 8, (8 + 2*3 + 1) = 15
        expected = [3, 8, 15]
        assert np.array_equal(result, expected)

    def test_scan_with_resources(self):
        """Test scan with resource requirements."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            resources={"cpus": 2, "memory": "4GB"},
        )
        def resource_scan(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        assert resource_scan.resources.cpus == 2
        assert resource_scan.resources.memory == "4GB"


class TestScanFuncEdgeCases:
    """Test edge cases and error conditions for ScanFunc."""

    def test_scan_function_no_parameters(self):
        """Test error when scan function has no parameters."""
        with pytest.raises(ValueError, match="Scan function must have at least one parameter"):

            @PipeFunc.scan(output_name="result", xs="values")
            def no_params() -> tuple[dict[str, Any], int]:
                return {}, 0

    def test_scan_missing_xs_parameter(self):
        """Test error when xs parameter is not provided."""

        @PipeFunc.scan(output_name="result", xs="missing_values")
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        pipeline = Pipeline([accumulator])
        with pytest.raises(ValueError, match="Missing value for argument `missing_values`"):
            pipeline.run("result", kwargs={"values": [1, 2, 3]})

    def test_scan_invalid_return_format(self):
        """Test error when scan function doesn't return tuple of length 2."""

        @PipeFunc.scan(output_name="result", xs="values")
        def bad_return(x: int) -> int:  # Wrong return type
            return x

        pipeline = Pipeline([bad_return])
        with pytest.raises(
            ValueError,
            match="Scan function must return tuple of \\(carry, output\\)",
        ):
            pipeline.run("result", kwargs={"values": [1, 2, 3]})

    def test_scan_invalid_carry_type(self):
        """Test error when carry is not a dict."""

        @PipeFunc.scan(output_name="result", xs="values")
        def bad_carry(x: int) -> tuple[int, int]:  # Carry should be dict
            return x, x  # First element should be dict

        pipeline = Pipeline([bad_carry])
        with pytest.raises(TypeError, match="Carry must be a dict"):
            pipeline.run("result", kwargs={"values": [1, 2, 3]})

    def test_scan_empty_values_list(self):
        """Test scan with empty values list."""

        @PipeFunc.scan(output_name="result", xs="values")
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        pipeline = Pipeline([accumulator])
        result = pipeline.run("result", kwargs={"values": []})

        # Should return empty array for empty input
        assert np.array_equal(result, np.array([]))

    def test_scan_empty_values_no_intermediate(self):
        """Test scan with empty values and return_intermediate=False."""

        @PipeFunc.scan(output_name="result", xs="values", return_intermediate=False)
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        pipeline = Pipeline([accumulator])
        result = pipeline.run("result", kwargs={"values": []})

        # Should return initial carry dict with defaults when no iterations happen
        assert result == {"total": 0}

    def test_scan_with_none_outputs(self):
        """Test scan where outputs are None (no intermediate tracking)."""

        @PipeFunc.scan(output_name="result", xs="values")
        def none_output(x: int, count: int = 0) -> tuple[dict[str, Any], None]:
            carry = {"count": count + 1}
            return carry, None  # Output is None

        pipeline = Pipeline([none_output])
        result = pipeline.run("result", kwargs={"values": [1, 2, 3]})

        # Should return empty array when all outputs are None
        assert np.array_equal(result, np.array([]))

    def test_carry_property_before_execution(self):
        """Test carry property when function hasn't been executed."""

        @PipeFunc.scan(output_name="result", xs="values", return_intermediate=False)
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        # Before execution, carry should be None
        assert accumulator.carry is None

    def test_carry_property_after_execution(self):
        """Test carry property after function execution."""

        @PipeFunc.scan(output_name="result", xs="values", return_intermediate=False)
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        # Execute the scan function directly to test carry property
        result = accumulator._execute_scan(values=[1, 2, 3])

        # After execution, carry should contain final values
        assert accumulator.carry == {"total": 6}
        assert result == {"total": 6}

    def test_scan_with_parameter_renames(self):
        """Test scan with parameter renames."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            renames={"values": "vals"},
        )
        def renamed_scan(x: int, accumulator: int = 0) -> tuple[dict[str, Any], int]:
            new_acc = accumulator + x
            carry = {"accumulator": new_acc}
            return carry, new_acc

        pipeline = Pipeline([renamed_scan])
        result = pipeline.run("result", kwargs={"vals": [1, 2, 3]})

        expected = [1, 3, 6]  # 0+1, 1+2, 3+3
        assert np.array_equal(result, expected)

    def test_scanfunc_copy_method(self):
        """Test the copy method preserves ScanFunc attributes."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            return_intermediate=False,
            defaults={"initial": 5},
        )
        def original_scan(x: int, initial: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"initial": initial + x}
            return carry, initial + x

        # Create a copy with updated parameters
        copied_scan = original_scan.copy(
            output_name="copied_result",
            return_intermediate=True,
        )

        # Verify copy preserves original scan function but updates parameters
        assert copied_scan.xs == original_scan.xs
        assert copied_scan._scan_func == original_scan._scan_func
        assert copied_scan.output_name == "copied_result"
        assert copied_scan.return_intermediate is True
        assert original_scan.return_intermediate is False

    def test_scanfunc_original_parameters_property(self):
        """Test the original_parameters property."""

        @PipeFunc.scan(output_name="result", xs="values")
        def test_scan(x: int, total: int = 0, scale: float = 1.0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x * scale}
            return carry, int(total + x * scale)

        params = test_scan.original_parameters

        # Should contain all parameters except x (which becomes xs)
        assert "total" in params
        assert "scale" in params
        assert "values" in params  # xs parameter
        assert "x" not in params  # x parameter is transformed

    def test_scanfunc_parameters_property(self):
        """Test the parameters property."""

        @PipeFunc.scan(output_name="result", xs="values")
        def test_scan(x: int, total: int = 0, scale: float = 1.0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x * scale}
            return carry, int(total + x * scale)

        params = test_scan.parameters

        # Should return tuple of parameter names
        assert isinstance(params, tuple)
        assert "total" in params
        assert "scale" in params
        assert "values" in params  # xs parameter


class TestScanFuncPickling:
    """Test custom pickling and unpickling functionality."""

    def test_scanfunc_pickle_roundtrip(self):
        """Test that ScanFunc can be pickled and unpickled correctly."""
        import pickle

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            defaults={"initial": 5},
            bound={"scale": 2.0},
        )
        def pickleable_scan(
            x: int,
            initial: int = 0,
            scale: float = 1.0,
        ) -> tuple[dict[str, Any], int]:
            result_val = initial + int(x * scale)
            carry = {"initial": result_val}
            return carry, result_val

        # Test that it works before pickling
        pipeline = Pipeline([pickleable_scan])
        original_result = pipeline.run("result", kwargs={"values": [1, 2, 3]})

        # Pickle and unpickle
        pickled_data = pickle.dumps(pickleable_scan)
        unpickled_scan = pickle.loads(pickled_data)  # noqa: S301

        # Test that unpickled version works the same way
        pipeline2 = Pipeline([unpickled_scan])
        unpickled_result = pipeline2.run("result", kwargs={"values": [1, 2, 3]})

        # Results should be identical
        assert np.array_equal(original_result, unpickled_result)

        # Verify attributes are preserved
        assert unpickled_scan.xs == pickleable_scan.xs
        assert unpickled_scan.return_intermediate == pickleable_scan.return_intermediate
        assert unpickled_scan.output_name == pickleable_scan.output_name

    def test_scanfunc_cloudpickle_compatibility(self):
        """Test that ScanFunc works with cloudpickle (used in multiprocessing)."""
        import cloudpickle

        @PipeFunc.scan(output_name="result", xs="values")
        def cloudpickleable_scan(x: int, counter: int = 0) -> tuple[dict[str, Any], int]:
            new_counter = counter + 1
            carry = {"counter": new_counter}
            return carry, x * new_counter

        # Test cloudpickle roundtrip
        pickled_data = cloudpickle.dumps(cloudpickleable_scan)
        unpickled_scan = cloudpickle.loads(pickled_data)

        # Verify it still works
        pipeline = Pipeline([unpickled_scan])
        result = pipeline.run("result", kwargs={"values": [5, 10, 15]})

        expected = [5, 20, 45]  # 5*1, 10*2, 15*3
        assert np.array_equal(result, expected)


class TestScanFuncIntegration:
    """Test ScanFunc integration with other pipefunc features."""

    def test_scan_with_variant(self):
        """Test scan with variant specification."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            variant="fast",
        )
        def variant_scan(x: int, multiplier: int = 1) -> tuple[dict[str, Any], int]:
            carry = {"multiplier": multiplier + 1}
            return carry, x * multiplier

        assert variant_scan.variant == {None: "fast"}

        pipeline = Pipeline([variant_scan])
        result = pipeline.run("result", kwargs={"values": [2, 3, 4]})

        expected = [2, 6, 12]  # 2*1, 3*2, 4*3
        assert np.array_equal(result, expected)

    def test_scan_with_debug_mode(self):
        """Test scan with debug mode enabled."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            debug=True,
        )
        def debug_scan(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        assert debug_scan.debug is True

        pipeline = Pipeline([debug_scan])
        result = pipeline.run("result", kwargs={"values": [1, 2, 3]})

        expected = [1, 3, 6]
        assert np.array_equal(result, expected)

    def test_scan_decorator_standalone(self):
        """Test the standalone scan decorator function."""
        from pipefunc._scanfunc import scan

        @scan(output_name="standalone_result", xs="items")
        def standalone_scan(item: str, message: str = "") -> tuple[dict[str, Any], str]:
            new_message = message + item
            carry = {"message": new_message}
            return carry, new_message

        # Verify it creates a ScanFunc instance
        assert isinstance(standalone_scan, ScanFunc)
        assert standalone_scan.xs == "items"
        assert standalone_scan.output_name == "standalone_result"

        pipeline = Pipeline([standalone_scan])
        result = pipeline.run("standalone_result", kwargs={"items": ["a", "b", "c"]})

        expected = ["a", "ab", "abc"]
        assert np.array_equal(result, expected)

    def test_scan_with_mapspec_integration(self):
        """Test scan with mapspec for array handling."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            mapspec="values[i] -> result[i]",
        )
        def mapspec_scan(x: int, accumulator: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"accumulator": accumulator + x}
            return carry, accumulator + x

        assert str(mapspec_scan.mapspec) == "values[i] -> result[i]"

    def test_multiple_scanfunc_in_pipeline(self):
        """Test multiple ScanFunc instances in the same pipeline."""

        @PipeFunc.scan(output_name="cumsum", xs="values")
        def cumulative_sum_step(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        @pipefunc(output_name="doubled_values")
        def double_values(cumsum) -> np.ndarray:
            return cumsum * 2

        @PipeFunc.scan(output_name="cumsum_doubled", xs="doubled_values")
        def scan_doubled(x: int, running_sum: int = 0) -> tuple[dict[str, Any], int]:
            new_sum = running_sum + x
            carry = {"running_sum": new_sum}
            return carry, new_sum

        pipeline = Pipeline([cumulative_sum_step, double_values, scan_doubled])
        result = pipeline.run("cumsum_doubled", kwargs={"values": [1, 2, 3]})

        # First scan: [1, 3, 6], Double: [2, 6, 12], Second scan: [2, 8, 20]
        expected = [2, 8, 20]
        assert np.array_equal(result, expected)


class TestScanFuncFullCoverage:
    """Additional tests to achieve 100% test coverage."""

    def test_scan_parameter_missing_direct_execution(self):
        """Test error when xs parameter is missing during direct execution."""

        @PipeFunc.scan(output_name="result", xs="missing_values")
        def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        # Test direct execution with incorrect parameter name
        with pytest.raises(
            ValueError,
            match="Required parameter 'missing_values' \\(xs\\) not provided",
        ):
            accumulator._execute_scan(wrong_param=[1, 2, 3], total=0)

    def test_scan_empty_intermediate_explicit(self):
        """Test scan where all outputs are None during iteration."""

        @PipeFunc.scan(output_name="result", xs="values")
        def none_outputs(x: int, count: int = 0) -> tuple[dict[str, Any], None]:
            carry = {"count": count + 1}
            return carry, None

        # Execute directly to test the empty intermediate results path
        result = none_outputs._execute_scan(values=[1, 2, 3])

        # Should return empty array when all outputs are None
        assert np.array_equal(result, np.array([]))
        # But carry should still be tracked
        assert none_outputs.carry == {"count": 3}

    def test_scan_carry_property_unexecuted(self):
        """Test carry property when function hasn't been executed."""

        @PipeFunc.scan(output_name="result", xs="values")
        def unexecuted_scan(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"total": total + x}
            return carry, total + x

        # Should return None when not executed
        assert unexecuted_scan.carry is None

    def test_scan_invalid_tuple_length_direct(self):
        """Test error when scan function returns tuple with wrong length."""

        @PipeFunc.scan(output_name="result", xs="values")
        def bad_tuple_length(x: int) -> tuple[dict[str, Any], int, str]:  # 3 elements instead of 2
            return {"x": x}, x, "extra"

        # Test direct execution
        with pytest.raises(
            ValueError,
            match="Scan function must return tuple of \\(carry, output\\)",
        ):
            bad_tuple_length._execute_scan(values=[1])

    def test_scan_invalid_carry_type_direct(self):
        """Test error when carry is not a dict during direct execution."""

        @PipeFunc.scan(output_name="result", xs="values")
        def bad_carry_type(x: int) -> tuple[list, int]:  # Carry should be dict, not list
            return [x], x

        # Test direct execution
        with pytest.raises(TypeError, match="Carry must be a dict"):
            bad_carry_type._execute_scan(values=[1])

    def test_scan_with_renames_carry_update(self):
        """Test that renames work correctly in carry updates."""

        @PipeFunc.scan(
            output_name="result",
            xs="values",
            renames={"accumulator": "acc"},
        )
        def renamed_carry_scan(x: int, accumulator: int = 0) -> tuple[dict[str, Any], int]:
            new_acc = accumulator + x
            # Return using original parameter name
            carry = {"accumulator": new_acc}
            return carry, new_acc

        # Execute directly to test rename handling in carry
        result = renamed_carry_scan._execute_scan(values=[1, 2, 3], accumulator=10)

        expected = [11, 13, 16]  # 10+1, 11+2, 13+3
        assert np.array_equal(result, expected)
        # Final carry should use renamed key
        assert renamed_carry_scan.carry == {"acc": 16}
