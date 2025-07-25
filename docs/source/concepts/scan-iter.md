---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# scan_iter: Generator-Based Iterative Execution

> Build iterative algorithms using natural Python generators

```{contents} ToC
:depth: 2
```

`scan_iter` provides a Pythonic way to create iterative algorithms using generator functions. It offers the same capabilities as `ScanFunc` but with a cleaner, more intuitive API.

:::{note}
`scan_iter` is the recommended approach for new code. For comparison with the current `scan` approach, see the [comparison guide](scanfunc-comparison.md).
:::

---

## Basic Usage

### Your First Generator Scan

The simplest use case is an accumulator using a generator:

```{code-cell} ipython3
from pipefunc import PipeFunc, Pipeline

@PipeFunc.scan_iter(output_name="cumsum")
def accumulator(values: list[int], total: int = 0):
    """Accumulate values using a generator."""
    for x in values:
        total += x
        yield total

# Create pipeline and run
values = [1, 2, 3, 4, 5]
pipeline = Pipeline([accumulator])
result = pipeline.run("cumsum", kwargs={"values": values})
print(f"Cumulative sum: {result}")
# Output: [1 3 6 10 15]
```

### Key Concepts

1. **Natural Python Generators**: Use `yield` to produce outputs
2. **Direct Parameters**: Function parameters match exactly what you pass
3. **State as Variables**: Use regular Python variables for state
4. **No Magic**: What you write is what runs

---

## Core Examples

### Fibonacci Generator

Generate Fibonacci numbers with clean generator syntax:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="fibonacci")
def fib_generator(n: int, a: int = 0, b: int = 1):
    """Generate Fibonacci numbers."""
    for _ in range(n):
        a, b = b, a + b
        yield b

pipeline = Pipeline([fib_generator])
result = pipeline.run("fibonacci", kwargs={"n": 10})
print(f"First 10 Fibonacci numbers: {list(result)}")
# Output: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
```

### Complex State with Objects

Track complex state using dictionaries or objects:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="trajectory")
def particle_simulation(time_steps: list[float], x: float = 0.0, v: float = 1.0, dt: float = 0.1):
    """Simulate particle motion with damping."""
    for t in time_steps:
        # Update position and velocity
        x += v * dt
        v *= 0.99  # Apply damping

        # Yield current state
        yield {"time": t, "position": x, "velocity": v}

import numpy as np
time_steps = np.linspace(0, 2, 11)
pipeline = Pipeline([particle_simulation])
trajectory = pipeline.run("trajectory", kwargs={"time_steps": time_steps, "dt": 0.2})

# Show first and last states
print(f"Initial state: {trajectory[0]}")
print(f"Final state: {trajectory[-1]}")
```

### Return Final Value Only

Sometimes you only need the final result:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="final_product", return_final_only=True)
def product_calculator(values: list[float]):
    """Calculate product of all values."""
    result = 1.0
    for x in values:
        result *= x
        yield result  # Still yield each step

# Only the final value is returned
values = [2, 3, 4, 5]
pipeline = Pipeline([product_calculator])
result = pipeline.run("final_product", kwargs={"values": values})
print(f"Product: {result}")  # Output: 120.0
```

---

## Advanced Patterns

### Early Stopping

Implement algorithms with convergence criteria:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="optimization_path")
def gradient_descent(max_iters: int, x0: float = 5.0, lr: float = 0.1, tol: float = 1e-6):
    """Minimize f(x) = x^2 using gradient descent with early stopping."""
    x = x0

    for i in range(max_iters):
        gradient = 2 * x  # Gradient of x^2
        x_new = x - lr * gradient

        yield {
            "iteration": i,
            "x": x_new,
            "gradient": gradient,
            "loss": x_new**2
        }

        # Early stopping condition
        if abs(gradient) < tol:
            print(f"Converged at iteration {i}")
            break

        x = x_new

pipeline = Pipeline([gradient_descent])
path = pipeline.run("optimization_path", kwargs={"max_iters": 100, "lr": 0.3})
print(f"Final x: {path[-1]['x']:.6f}")
print(f"Final loss: {path[-1]['loss']:.9f}")
```

### Using Classes for State

For complex state management, use classes:

```{code-cell} ipython3
from dataclasses import dataclass

@dataclass
class SimulationState:
    position: float = 0.0
    velocity: float = 1.0
    acceleration: float = 0.0
    time: float = 0.0

@PipeFunc.scan_iter(output_name="physics_sim")
def physics_simulation(
    time_steps: np.ndarray,
    initial_state: SimulationState = None,
    dt: float = 0.01,
    force: float = -0.5
):
    """Run physics simulation with proper state management."""
    state = initial_state or SimulationState()

    for t in time_steps:
        # Update physics
        state.acceleration = force - 0.1 * state.velocity  # Force with damping
        state.velocity += state.acceleration * dt
        state.position += state.velocity * dt
        state.time = t

        # Yield snapshot
        yield {
            "t": t,
            "x": state.position,
            "v": state.velocity,
            "a": state.acceleration
        }

# Run simulation
time_steps = np.linspace(0, 5, 51)
pipeline = Pipeline([physics_simulation])
results = pipeline.run("physics_sim", kwargs={
    "time_steps": time_steps,
    "dt": 0.1,
    "force": -0.2
})

print(f"Simulated {len(results)} time steps")
print(f"Final position: {results[-1]['x']:.3f}")
```

### Generator with Mixed Types

Generators can yield different types:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="analysis_results")
def progressive_analysis(data_batches: list[list[float]]):
    """Analyze data progressively with different output types."""
    all_data = []

    for i, batch in enumerate(data_batches):
        all_data.extend(batch)

        if i == 0:
            # First batch: just the data
            yield batch
        elif i == 1:
            # Second batch: basic stats
            yield {
                "mean": np.mean(all_data),
                "std": np.std(all_data)
            }
        else:
            # Later batches: full analysis
            yield {
                "batch": i,
                "cumulative_mean": np.mean(all_data),
                "cumulative_std": np.std(all_data),
                "batch_mean": np.mean(batch),
                "total_points": len(all_data)
            }

# Test with data
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
pipeline = Pipeline([progressive_analysis])
results = pipeline.run("analysis_results", kwargs={"data_batches": data})

for i, result in enumerate(results):
    print(f"Batch {i}: {result}")
```

---

## Parallel Execution

### Basic Parallel Scan

Use `scan_iter` with `pipeline.map` for parallel execution:

```{code-cell} ipython3
from pipefunc import pipefunc

# Generate different sequences for parallel processing
@pipefunc(output_name="sequences", mapspec="seq_id[i] -> sequences[i]")
def generate_sequences(seq_id: int, length: int = 5) -> list[int]:
    """Generate sequences for parallel processing."""
    start = seq_id * 10
    return list(range(start, start + length))

# Scan each sequence independently
@PipeFunc.scan_iter(output_name="products", mapspec="sequences[i] -> products[i]")
def running_product(sequences: list[int], initial: int = 1):
    """Calculate running product for each sequence."""
    product = initial
    for x in sequences:
        product *= x
        yield product

# Run in parallel
inputs = {"seq_id": [0, 1, 2], "length": 5}  # Fixed length for all sequences
pipeline = Pipeline([generate_sequences, running_product])
results = pipeline.map(inputs, run_folder="parallel_scan_iter", parallel=False)

print("Parallel scan results:")
for i, products in enumerate(results["products"].output):
    print(f"Sequence {i}: {list(products)}")
```

### Advanced Parallel Pattern

Process complex data structures in parallel:

```{code-cell} ipython3
# Generate matrix data
@pipefunc(output_name="matrix", mapspec="matrix_id[i] -> matrix[i]")
def create_matrix(matrix_id: int, size: int = 3) -> np.ndarray:
    """Create test matrices."""
    np.random.seed(matrix_id)
    return np.random.rand(size, size)

# Iterative matrix operation
@PipeFunc.scan_iter(output_name="eigenvector", mapspec="matrix[i] -> eigenvector[i]")
def power_iteration(matrix: np.ndarray, max_iters: int = 20, tol: float = 1e-6):
    """Find dominant eigenvector using power iteration."""
    n = matrix.shape[0]
    v = np.ones(n) / np.sqrt(n)  # Initial guess

    for i in range(max_iters):
        v_new = matrix @ v
        v_new = v_new / np.linalg.norm(v_new)

        # Check convergence
        if np.allclose(v, v_new, atol=tol):
            yield v_new
            break

        v = v_new
        yield v

# Run power iteration on multiple matrices
inputs = {"matrix_id": [0, 1, 2], "size": 3}  # Same size for all matrices
pipeline = Pipeline([create_matrix, power_iteration])
results = pipeline.map(inputs, run_folder="parallel_eigen", parallel=False)

print("Dominant eigenvectors:")
for i, eigenvecs in enumerate(results["eigenvector"].output):
    final_eigenvec = list(eigenvecs)[-1]
    print(f"Matrix {i}: {final_eigenvec}")
```

---

## Integration Features

### Using with Bound Parameters

Fix certain parameters while keeping others flexible:

```{code-cell} ipython3
@PipeFunc.scan_iter(
    output_name="scaled_sum",
    bound={"scale": 2.0},      # Fix scale parameter
    defaults={"offset": 10}     # Custom default
)
def scaled_accumulator(values: list[float], scale: float = 1.0, offset: float = 0):
    """Accumulator with fixed scaling."""
    total = offset
    for x in values:
        total += scale * x  # scale is always 2.0
        yield total

pipeline = Pipeline([scaled_accumulator])
result = pipeline.run("scaled_sum", kwargs={"values": [1, 2, 3]})
print(f"Scaled sum: {list(result)}")
# With scale=2.0, offset=10: [12, 16, 22]
```

### Using with Renames

Rename parameters for better pipeline integration:

```{code-cell} ipython3
@PipeFunc.scan_iter(
    output_name="distances",
    renames={"points": "coordinates"}  # External name -> Internal name
)
def calculate_distances(points: list[tuple[float, float]], origin: tuple[float, float] = (0, 0)):
    """Calculate cumulative distances from origin."""
    total_distance = 0.0

    for x, y in points:
        distance = ((x - origin[0])**2 + (y - origin[1])**2)**0.5
        total_distance += distance
        yield {
            "point": (x, y),
            "distance": distance,
            "cumulative": total_distance
        }

# Use external name in pipeline
coords = [(1, 0), (1, 1), (0, 1)]
pipeline = Pipeline([calculate_distances])
result = pipeline.run("distances", kwargs={"coordinates": coords})  # Note: using renamed parameter

for r in result:
    print(f"Point {r['point']}: distance={r['distance']:.2f}, cumulative={r['cumulative']:.2f}")
```

### Resource Specification

Specify computational resources for scan operations:

```{code-cell} ipython3
@PipeFunc.scan_iter(
    output_name="heavy_computation",
    resources={"cpus": 4, "memory": "16GB"}
)
def process_large_datasets(data_chunks: list[np.ndarray], model_size: int = 1000):
    """Process large data chunks with resource requirements."""
    model = np.random.randn(model_size, model_size)

    for i, chunk in enumerate(data_chunks):
        # Simulate heavy computation
        result = chunk @ model
        stats = {
            "chunk_id": i,
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "max": float(np.max(result))
        }
        yield stats

print(f"Resource requirements: {process_large_datasets.resources}")
```

---

## Best Practices

### 1. Use Type Hints

Always provide clear type hints:

```{code-cell} ipython3
from typing import Iterator, Optional

@PipeFunc.scan_iter(output_name="typed_scan")
def well_typed_generator(
    items: list[str],
    prefix: str = "",
    max_length: Optional[int] = None
) -> Iterator[dict[str, str | int]]:
    """Example with proper type hints."""
    processed_count = 0

    for item in items:
        if max_length and len(item) > max_length:
            continue

        processed_count += 1
        yield {
            "original": item,
            "prefixed": f"{prefix}{item}",
            "length": len(item),
            "count": processed_count
        }
```

### 2. Handle Edge Cases

Consider empty inputs and boundary conditions:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="robust_processor")
def robust_generator(data: list[float], window_size: int = 3):
    """Robust generator that handles edge cases."""
    if not data:
        return  # Empty input, no output

    if window_size <= 0:
        raise ValueError("Window size must be positive")

    if len(data) < window_size:
        # Handle case where data is smaller than window
        yield {"window": data, "mean": np.mean(data)}
        return

    # Normal processing
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        yield {
            "position": i,
            "window": window,
            "mean": np.mean(window)
        }

# Test edge cases
pipeline = Pipeline([robust_generator])

# Empty data
result1 = pipeline.run("robust_processor", kwargs={"data": []})
print(f"Empty data result: {list(result1)}")

# Small data
result2 = pipeline.run("robust_processor", kwargs={"data": [1, 2], "window_size": 3})
print(f"Small data result: {list(result2)}")
```

### 3. Document Generator Behavior

Clearly document what your generator yields:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="documented_scan")
def well_documented_generator(
    time_series: list[float],
    threshold: float = 0.5
) -> Iterator[dict[str, float | bool]]:
    """
    Analyze time series data for threshold crossings.

    Parameters
    ----------
    time_series : list[float]
        Input time series data
    threshold : float
        Threshold value for detection

    Yields
    ------
    dict
        Dictionary containing:
        - 'index': Position in time series
        - 'value': Current value
        - 'above_threshold': Whether value exceeds threshold
        - 'crossing': Whether this is a threshold crossing point
    """
    previous_above = False

    for i, value in enumerate(time_series):
        above = value > threshold
        crossing = above != previous_above and i > 0

        yield {
            "index": i,
            "value": value,
            "above_threshold": above,
            "crossing": crossing
        }

        previous_above = above
```

### 4. Consider Memory Usage

For large datasets, yield results incrementally:

```{code-cell} ipython3
@PipeFunc.scan_iter(output_name="memory_efficient")
def process_in_batches(
    huge_dataset: list[int],
    batch_size: int = 1000
):
    """Process large dataset in memory-efficient batches."""
    batch_results = []

    for i, item in enumerate(huge_dataset):
        # Process item
        result = item ** 2  # Simple processing
        batch_results.append(result)

        # Yield batch when full
        if len(batch_results) >= batch_size:
            yield {
                "batch_num": i // batch_size,
                "batch_mean": np.mean(batch_results),
                "batch_size": len(batch_results)
            }
            batch_results = []  # Clear batch

    # Don't forget the last partial batch
    if batch_results:
        yield {
            "batch_num": "final",
            "batch_mean": np.mean(batch_results),
            "batch_size": len(batch_results)
        }
```

---

## Error Handling

### Common Errors and Solutions

```{code-cell} ipython3
:tags: [raises-exception]

# Error: Not a generator
@PipeFunc.scan_iter(output_name="not_generator")
def not_a_generator(values: list[int]):
    return sum(values)  # Should use yield!

pipeline = Pipeline([not_a_generator])
try:
    pipeline.run("not_generator", kwargs={"values": [1, 2, 3]})
except TypeError as e:
    print(f"Error: {e}")
```

```{code-cell} ipython3
# Correct: Use yield
@PipeFunc.scan_iter(output_name="correct_generator")
def correct_generator(values: list[int]):
    total = 0
    for x in values:
        total += x
        yield total  # Now it's a generator!

pipeline = Pipeline([correct_generator])
result = pipeline.run("correct_generator", kwargs={"values": [1, 2, 3]})
print(f"Result: {list(result)}")
```

---

## Summary

`scan_iter` provides a clean, Pythonic way to build iterative algorithms:

### Key Benefits

- **Natural Syntax**: Use Python generators with `yield`
- **Direct Parameters**: No signature transformation
- **Simple State**: Use regular variables, not dicts
- **Type Safety**: Better IDE support and error detection
- **Full Integration**: Works with all pipefunc features

### When to Use

- Building new iterative algorithms
- Converting loops to pipeline operations
- Implementing optimization routines
- Processing sequential data
- Any task requiring state across iterations

### Quick Reference

```python
# Basic pattern
@PipeFunc.scan_iter(output_name="result")
def my_generator(items: list[Any], initial_state: Any = None):
    state = initial_state
    for item in items:
        # Process and update state
        state = process(state, item)
        yield state

# With all options
@PipeFunc.scan_iter(
    output_name="result",
    return_final_only=True,      # Only return last value
    bound={"param": value},      # Fix parameters
    defaults={"param": value},   # Custom defaults
    renames={"old": "new"},      # Rename parameters
    mapspec="spec",              # For parallel execution
    resources={...},             # Resource requirements
)
```

Start using `scan_iter` today for cleaner, more maintainable iterative algorithms!
