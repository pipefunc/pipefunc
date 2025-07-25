---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# ScanFunc: Iterative Execution with Feedback Loops

> Build iterative algorithms with state management similar to `jax.lax.scan`

:::{note}
A new generator-based approach `scan_iter` is now available that provides a more Pythonic API. See the [scan_iter documentation](scan-iter.md) or the [comparison guide](scanfunc-comparison.md) to learn more.
:::

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

ScanFunc enables iterative algorithms where the output of one iteration becomes part of the input for the next iteration. This is particularly useful for:

- **Optimization routines** (gradient descent, genetic algorithms)
- **Time-stepping methods** (Runge-Kutta, finite differences)
- **Iterative solvers** (Newton's method, fixed-point iterations)
- **Sequential processing** with persistent state
- **Simulation loops** with evolving parameters

:::{note}
ScanFunc is similar to `jax.lax.scan` but integrates seamlessly with pipefunc's pipeline system, including parallel execution via `pipeline.map`.
:::

---

## Basic Usage

### Simple Accumulator

The most basic use case is an accumulator that maintains running state:

```{code-cell} ipython3
from typing import Any
from pipefunc import PipeFunc, Pipeline

@PipeFunc.scan(output_name="cumsum", xs="values")
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    """Accumulate values by adding each to the running total."""
    new_total = total + x
    carry = {"total": new_total}
    return carry, new_total

# Create pipeline and run
values = [1, 2, 3, 4, 5]
pipeline = Pipeline([accumulator])
result = pipeline.run("cumsum", kwargs={"values": values})
print(f"Cumulative sum: {result}")
# Output: [1 3 6 10 15]
```

### Key Concepts

1. **Function Signature**: The first parameter (`x`) receives each element from `xs`. Remaining parameters are carry state.
2. **Return Value**: Must return `(carry_dict, output)` where:
   - `carry_dict`: State to pass to next iteration
   - `output`: Value to collect (or `None` if not needed)
3. **Initial State**: Carry parameters use their default values for the first iteration

---

## Advanced Examples

### Fibonacci Sequence

Generate Fibonacci numbers using dual carry values:

```{code-cell} ipython3
@PipeFunc.scan(output_name="fibonacci", xs="n_steps")
def fib_step(x: int, a: int = 0, b: int = 1) -> tuple[dict[str, Any], int]:
    """Generate next Fibonacci number."""
    next_val = a + b
    carry = {"a": b, "b": next_val}
    return carry, next_val

# Generate first 10 Fibonacci numbers
n_steps = list(range(10))
pipeline = Pipeline([fib_step])
result = pipeline.run("fibonacci", kwargs={"n_steps": n_steps})
print(f"Fibonacci: {result}")
# Output: [1 2 3 5 8 13 21 34 55 89]
```

### Final Result Only

Sometimes you only want the final state, not intermediate results:

```{code-cell} ipython3
@PipeFunc.scan(output_name="final_sum", xs="values", return_intermediate=False)
def sum_only_final(x: int, total: int = 0) -> tuple[dict[str, Any], None]:
    """Sum all values, return only final total."""
    new_total = total + x
    carry = {"total": new_total}
    return carry, None  # No intermediate output

values = [1, 2, 3, 4, 5]
pipeline = Pipeline([sum_only_final])
result = pipeline.run("final_sum", kwargs={"values": values})
print(f"Final sum: {result}")
# Output: {'total': 15}
```

---

## Real-World Applications

### Particle Simulation

Simulate a particle moving through time with evolving position and velocity:

```{code-cell} ipython3
import numpy as np

@PipeFunc.scan(output_name="trajectory", xs="time_steps")
def simulate_particle(
    t: float,
    x: float = 0.0,
    v: float = 1.0,
    dt: float = 0.1
) -> tuple[dict[str, Any], dict[str, float]]:
    """Simulate particle motion with simple physics."""
    # Update position and velocity
    new_x = x + v * dt
    new_v = v * 0.99  # Add slight damping

    carry = {"x": new_x, "v": new_v}
    state = {"t": t, "x": new_x, "v": new_v}
    return carry, state

# Run simulation
time_steps = np.linspace(0, 2, 21)
pipeline = Pipeline([simulate_particle])
trajectory = pipeline.run("trajectory", kwargs={
    "time_steps": time_steps,
    "dt": 0.1,
    "x": 0.0,
    "v": 2.0
})

# Extract positions for plotting
positions = [state["x"] for state in trajectory]
print(f"Final position: {positions[-1]:.2f}")
```

### Optimization Algorithm

Implement gradient descent to minimize a function:

```{code-cell} ipython3
@PipeFunc.scan(output_name="optimization_path", xs="iterations")
def gradient_descent(
    iteration: int,
    x: float = 5.0,
    learning_rate: float = 0.1
) -> tuple[dict[str, Any], dict[str, float]]:
    """Minimize f(x) = x^2 using gradient descent."""
    # Compute gradient of f(x) = x^2
    gradient = 2 * x

    # Update parameter
    new_x = x - learning_rate * gradient

    # Track state
    carry = {"x": new_x}
    state = {
        "iteration": iteration,
        "x": new_x,
        "gradient": gradient,
        "loss": new_x**2
    }
    return carry, state

# Run optimization
iterations = list(range(20))
pipeline = Pipeline([gradient_descent])
path = pipeline.run("optimization_path", kwargs={
    "iterations": iterations,
    "learning_rate": 0.2
})

print(f"Initial x: {path[0]['x']:.3f}")
print(f"Final x: {path[-1]['x']:.3f}")
print(f"Final loss: {path[-1]['loss']:.6f}")
```

---

## Parallel Execution

ScanFunc integrates seamlessly with `pipeline.map` for parallel execution across multiple parameter sets:

```{code-cell} ipython3
from pipefunc import pipefunc

# Generate different input sequences
@pipefunc(output_name="values", mapspec="batch_id[i] -> values[i]")
def generate_values(batch_id: int) -> list[int]:
    """Generate values for each batch."""
    start = batch_id * 3 + 1
    return [start, start + 1, start + 2]

# Scan function with mapspec
@PipeFunc.scan(output_name="cumsum", xs="values", mapspec="values[i] -> cumsum[i]")
def cumulative_sum(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    """Cumulative sum for each batch."""
    new_total = total + x
    carry = {"total": new_total}
    return carry, new_total

# Run in parallel across multiple batches
inputs = {"batch_id": [0, 1, 2]}
pipeline = Pipeline([generate_values, cumulative_sum])
results = pipeline.map(inputs, run_folder="scan_parallel", parallel=False)

print("Results per batch:")
for i, cumsum in enumerate(results["cumsum"].output):
    print(f"Batch {i}: {cumsum}")
# Output:
# Batch 0: [1 3 6]
# Batch 1: [4 9 15]
# Batch 2: [7 15 24]
```

---

## Nested Pipeline Scans

For complex iteration bodies, use `Pipeline.nest_funcs_scan` to execute entire pipelines per iteration:

```{code-cell} ipython3
# Define individual computation steps
@pipefunc(output_name="k1")
def calc_k1(y: float, t: float, dt: float) -> float:
    """First stage of Runge-Kutta method."""
    return -y * dt  # Simple ODE: dy/dt = -y

@pipefunc(output_name="k2")
def calc_k2(y: float, k1: float, t: float, dt: float) -> float:
    """Second stage of Runge-Kutta method."""
    return -(y + 0.5 * k1) * dt

@pipefunc(output_name="y_next")
def rk2_step(y: float, k1: float, k2: float) -> float:
    """Complete RK2 step."""
    return y + k2

# Create nested pipeline
rk2_pipeline = Pipeline([calc_k1, calc_k2, rk2_step])

# Use entire pipeline as scan body
@rk2_pipeline.nest_funcs_scan(
    output_name="trajectory",
    xs="time_steps",
    output_nodes={"y_next"},  # Extract y_next from pipeline
)
def rk2_scan(t: float, y: float = 1.0, dt: float = 0.1) -> tuple[dict[str, Any], float]:
    """This function body is replaced by the nested pipeline."""
    return {}, 0.0

# Run the nested scan
time_steps = np.linspace(0, 1, 11)
pipeline = Pipeline([rk2_scan])
result = pipeline.run("trajectory", kwargs={"time_steps": time_steps, "dt": 0.1})

print(f"Initial value: {result[0]:.3f}")
print(f"Final value: {result[-1]:.3f}")
print(f"Exponential decay: {np.exp(-1.0):.3f}")  # Expected analytical result
```

---

## Configuration Options

### Bound Parameters and Defaults

Control scan behavior with bound parameters and custom defaults:

```{code-cell} ipython3
@PipeFunc.scan(
    output_name="scaled_sum",
    xs="values",
    bound={"scale": 2.0},        # Fix scale parameter
    defaults={"offset": 1.0},    # Custom default for offset
)
def scaled_accumulator(
    x: int,
    total: float = 0.0,
    scale: float = 1.0,     # Will be bound to 2.0
    offset: float = 0.0,    # Will default to 1.0
) -> tuple[dict[str, Any], float]:
    """Accumulator with scaling and offset."""
    new_total = total + scale * x + offset
    carry = {"total": new_total}
    return carry, new_total

values = [1, 2, 3]
pipeline = Pipeline([scaled_accumulator])
result = pipeline.run("scaled_sum", kwargs={"values": values})
print(f"Scaled accumulation: {result}")
# With scale=2.0, offset=1.0:
# (0 + 2*1 + 1) = 3, (3 + 2*2 + 1) = 8, (8 + 2*3 + 1) = 15
```

### Resource Requirements

Specify computational resources for scan operations:

```{code-cell} ipython3
@PipeFunc.scan(
    output_name="heavy_computation",
    xs="data_chunks",
    resources={"cpus": 4, "memory": "8GB"},
)
def process_chunks(
    chunk: list,
    processed_count: int = 0
) -> tuple[dict[str, Any], int]:
    """Process data chunks with high resource requirements."""
    # Simulate heavy computation
    result = sum(chunk)
    carry = {"processed_count": processed_count + len(chunk)}
    return carry, result

# Resources are automatically used by schedulers
print(f"Resource requirements: {process_chunks.resources}")
```

---

## Accessing Scan State

### Carry Property

Access the final carry state after execution:

```{code-cell} ipython3
@PipeFunc.scan(output_name="result", xs="values", return_intermediate=False)
def stateful_scan(x: int, state: dict = None) -> tuple[dict[str, Any], None]:
    """Scan that maintains complex state."""
    if state is None:
        state = {"count": 0, "sum": 0, "max": float("-inf")}

    new_state = {
        "count": state["count"] + 1,
        "sum": state["sum"] + x,
        "max": max(state["max"], x)
    }
    return {"state": new_state}, None

# Execute scan
values = [3, 1, 4, 1, 5, 9]
result = stateful_scan._execute_scan(values=values)

# Access final state
final_state = stateful_scan.carry["state"]
print(f"Final state: {final_state}")
# Output: {'count': 6, 'sum': 23, 'max': 9}
```

---

## Error Handling

ScanFunc provides clear error messages for common issues:

```{code-cell} ipython3
:tags: [raises-exception]

# Example: Invalid return format
@PipeFunc.scan(output_name="bad_scan", xs="values")
def invalid_return(x: int) -> int:  # Should return tuple!
    return x

pipeline = Pipeline([invalid_return])
try:
    pipeline.run("bad_scan", kwargs={"values": [1, 2, 3]})
except ValueError as e:
    print(f"Error: {e}")
```

```{code-cell} ipython3
:tags: [raises-exception]

# Example: Invalid carry type
@PipeFunc.scan(output_name="bad_carry", xs="values")
def invalid_carry(x: int) -> tuple[list, int]:  # Carry must be dict!
    return [x], x

pipeline = Pipeline([invalid_carry])
try:
    pipeline.run("bad_carry", kwargs={"values": [1]})
except TypeError as e:
    print(f"Error: {e}")
```

---

## Best Practices

### 1. Initialize Carry State Properly

Use meaningful default values for carry parameters:

```{code-cell} ipython3
@PipeFunc.scan(output_name="statistics", xs="values")
def compute_stats(
    x: float,
    count: int = 0,
    mean: float = 0.0,
    m2: float = 0.0  # For variance calculation
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute running statistics using Welford's algorithm."""
    new_count = count + 1
    delta = x - mean
    new_mean = mean + delta / new_count
    delta2 = x - new_mean
    new_m2 = m2 + delta * delta2

    carry = {"count": new_count, "mean": new_mean, "m2": new_m2}
    stats = {
        "count": new_count,
        "mean": new_mean,
        "variance": new_m2 / new_count if new_count > 1 else 0.0
    }
    return carry, stats
```

### 2. Handle Edge Cases

Consider empty input sequences and boundary conditions:

```{code-cell} ipython3
@PipeFunc.scan(output_name="robust_scan", xs="values")
def robust_accumulator(
    x: float,
    total: float = 0.0,
    count: int = 0
) -> tuple[dict[str, Any], dict[str, float]]:
    """Robust accumulator that handles edge cases."""
    new_total = total + x
    new_count = count + 1

    carry = {"total": new_total, "count": new_count}
    result = {
        "total": new_total,
        "count": new_count,
        "average": new_total / new_count if new_count > 0 else 0.0
    }
    return carry, result

# Test with empty sequence
empty_result = robust_accumulator._execute_scan(values=[])
print(f"Empty result: {empty_result}")  # Returns empty array
```

### 3. Use Type Hints

Provide clear type hints for better code documentation and IDE support:

```{code-cell} ipython3
from typing import Dict, List, Tuple, Optional

@PipeFunc.scan(output_name="typed_scan", xs="items")
def well_typed_scan(
    item: str,
    history: Optional[List[str]] = None,
    metadata: Optional[Dict[str, int]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Well-typed scan function with clear interfaces."""
    if history is None:
        history = []
    if metadata is None:
        metadata = {"processed": 0}

    new_history = history + [item]
    new_metadata = {"processed": metadata["processed"] + 1}

    carry = {"history": new_history, "metadata": new_metadata}
    output = {
        "item": item,
        "position": len(new_history),
        "total_processed": new_metadata["processed"]
    }
    return carry, output
```

---

## Comparison with Alternatives

### vs. Regular Loops

```python
# Traditional approach
def traditional_cumsum(values):
    result = []
    total = 0
    for x in values:
        total += x
        result.append(total)
    return result

# ScanFunc approach
@PipeFunc.scan(output_name="cumsum", xs="values")
def scan_cumsum(x, total=0):
    new_total = total + x
    return {"total": new_total}, new_total
```

**ScanFunc advantages:**
- Pipeline integration and parallel execution
- Automatic state management and persistence
- Resource allocation and scheduling support
- Clean separation of iteration logic from pipeline setup

### vs. jax.lax.scan

```python
# JAX scan
def jax_cumsum(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

# ScanFunc equivalent
@PipeFunc.scan(output_name="cumsum", xs="values")
def pipefunc_cumsum(x, total=0):
    new_total = total + x
    return {"total": new_total}, new_total
```

**Key differences:**
- ScanFunc uses dict-based carry for named state
- Full integration with pipefunc ecosystem
- Built-in parallel execution support
- No JAX dependency required

---

## Summary

ScanFunc provides a powerful abstraction for iterative algorithms within pipefunc:

- **Clean API**: Natural function definition with automatic state management
- **Pipeline Integration**: Works seamlessly with existing pipefunc features
- **Parallel Execution**: Built-in support for `pipeline.map`
- **Flexible Configuration**: Bound parameters, defaults, and resource management
- **Nested Pipelines**: Execute complex multi-step iterations
- **Type Safety**: Full type hint support and runtime validation

Use ScanFunc when you need iterative algorithms with persistent state that integrate naturally with your data processing pipelines.
