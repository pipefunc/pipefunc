"""Test suite for ScanFunc - iterative execution with feedback loops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._scanfunc import ScanFunc


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

        # First create a function that produces values for each batch
        @pipefunc(output_name="values", mapspec="batch_id[i] -> values[i]")
        def generate_values(batch_id: int) -> list[int]:
            # Generate values based on batch_id
            start = batch_id * 3 + 1
            return [start, start + 1, start + 2]

        @PipeFunc.scan(output_name="cumsum", xs="values", mapspec="values[i] -> cumsum[i]")
        def cumulative_sum(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
            new_total = total + x
            carry = {"total": new_total}
            return carry, new_total

        # Multiple batches
        inputs = {
            "batch_id": [0, 1, 2],  # This will generate [1,2,3], [4,5,6], [7,8,9]
        }

        pipeline = Pipeline([generate_values, cumulative_sum])
        results = pipeline.map(inputs, run_folder="test_scan_map")

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
            pass

        time_steps = np.linspace(0, 1, 11)
        pipeline = Pipeline([rk2_scan])
        result = pipeline.run("trajectory", kwargs={"time_steps": time_steps, "dt": 0.1})

        # Should decay exponentially
        assert len(result) == 11
        assert result[0] > result[-1]  # Decaying
        assert all(result[i] > result[i + 1] for i in range(len(result) - 1))  # Monotonic

    def test_scan_with_varying_xs_shapes(self):
        """Test scan with xs that has varying shapes from mapspec."""

        @pipefunc(output_name="param_grid", mapspec="alpha[i], beta[j] -> param_grid[i, j]")
        def create_params(alpha: float, beta: float) -> dict[str, float]:
            return {"alpha": alpha, "beta": beta}

        @PipeFunc.scan(output_name="optimization_path", xs="param_grid")
        def optimizer_step(
            params: dict[str, float],
            best_loss: float = float("inf"),
            best_params: dict[str, float] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            # Simple optimization: minimize (alpha - 1)^2 + (beta - 2)^2
            loss = (params["alpha"] - 1) ** 2 + (params["beta"] - 2) ** 2

            if loss < best_loss:
                carry = {"best_loss": loss, "best_params": params}
                intermediate = {"params": params, "loss": loss, "improved": True}
            else:
                carry = {"best_loss": best_loss, "best_params": best_params}
                intermediate = {"params": params, "loss": loss, "improved": False}

            return carry, intermediate

        pipeline = Pipeline([create_params, optimizer_step])
        inputs = {
            "alpha": [0.5, 1.0, 1.5],
            "beta": [1.5, 2.0, 2.5],
        }

        results = pipeline.map(inputs, run_folder="test_optimizer")
        opt_path = results["optimization_path"].output

        # Should have tried 9 parameter combinations
        assert len(opt_path) == 9
        # Best should be close to (1, 2)
        final_carry = results["optimization_path"].carry
        assert final_carry["best_params"]["alpha"] == 1.0
        assert final_carry["best_params"]["beta"] == 2.0

    def test_scan_error_handling(self):
        """Test error handling in scan functions."""

        @PipeFunc.scan(output_name="result", xs="values")
        def failing_scan(x: int, count: int = 0) -> tuple[dict[str, Any], int]:
            if x > 5:
                raise ValueError(f"Value {x} is too large!")
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

    def test_scan_dag_unrolling(self):
        """Test that scan properly unrolls into DAG for visualization."""

        @PipeFunc.scan(output_name="result", xs="values")
        def simple_scan(x: int, acc: int = 0) -> tuple[dict[str, Any], int]:
            carry = {"acc": acc + x}
            return carry, acc + x

        pipeline = Pipeline([simple_scan])

        # When visualizing with known xs length, should show unrolled structure
        values = [1, 2, 3]
        unrolled = pipeline._unroll_scan_for_visualization(
            scan_func=simple_scan,
            xs_length=len(values),
        )

        # Should have 3 nodes for the 3 iterations
        assert len(unrolled.nodes) == 3
        # Each node should depend on the previous
        assert unrolled.nodes[1].dependencies == {unrolled.nodes[0].name}
        assert unrolled.nodes[2].dependencies == {unrolled.nodes[1].name}

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
