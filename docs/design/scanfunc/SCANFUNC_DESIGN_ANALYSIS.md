# ScanFunc Design Analysis and Alternative Strategies

## Current Design Overview

### What is ScanFunc?
ScanFunc is a PipeFunc subclass that enables iterative execution with feedback loops, similar to `jax.lax.scan`. It's designed for:
- Optimization routines (gradient descent, genetic algorithms)
- Time-stepping methods (Runge-Kutta, finite differences)
- Iterative solvers
- Sequential processing with state

### Current Implementation

```python
@PipeFunc.scan(output_name="result", xs="values")
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    new_total = total + x
    carry = {"total": new_total}
    return carry, new_total
```

**Key characteristics:**
1. First parameter (`x`) receives each element from `xs`
2. Other parameters are "carry" state with defaults as initial values
3. Must return `(carry_dict, output)` tuple
4. Signature transformation: `func(x, carry1, carry2)` → `wrapper(carry1, carry2, xs=...)`
5. Complex wrapper machinery to maintain the illusion

## Issues with Current Design

### 1. **Confusing Signature Transformation**
The function you write has a different signature than what gets executed:
- Written: `accumulator(x: int, total: int = 0)`
- Called as: `accumulator(values=[1,2,3])`
- The `x` parameter disappears and `xs` appears

### 2. **Rigid Return Format**
Always requiring `(carry_dict, output)` is unintuitive:
- Forces users to think in terms of "carry" and "output"
- Dict requirement for carry is limiting
- What if output isn't needed for some iterations?

### 3. **Implicit First Parameter Convention**
The first parameter being special is a hidden rule:
- Not obvious from the decorator
- Easy to get wrong
- Requires reading documentation to understand

### 4. **Complex Implementation**
The wrapper machinery is complex:
- Signature manipulation
- Parameter remapping
- Custom pickling to avoid circular references
- Hard to debug when things go wrong

### 5. **Limited State Management**
Dict-based carry has limitations:
- What if state is better represented as a dataclass/tuple/custom object?
- Dict keys must match parameter names (with renames)
- No type safety for carry values

## Alternative Design Strategies

### Strategy 1: Explicit State Class

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar('S')  # State type
T = TypeVar('T')  # Output type

@dataclass
class ScanState(Generic[S]):
    value: S

class ScanFunc(PipeFunc):
    """Explicit state-based scanning."""
    pass

# Usage:
@dataclass
class AccumulatorState:
    total: int = 0

@PipeFunc.scan(output_name="result", xs="values", state_class=AccumulatorState)
def accumulator(x: int, state: AccumulatorState) -> tuple[AccumulatorState, int]:
    new_state = AccumulatorState(total=state.total + x)
    return new_state, state.total + x

# Or even simpler with automatic state wrapping:
@PipeFunc.scan(output_name="result", xs="values")
def accumulator(x: int, total: int = 0) -> tuple[int, int]:
    # Returns (new_total, output)
    return total + x, total + x
```

**Advantages:**
- Type-safe state management
- Clear separation of state and output
- Works with dataclasses, tuples, or any type
- No dict requirement

### Strategy 2: Generator-Based Approach

```python
@PipeFunc.scan_generator(output_name="result", xs="values")
def accumulator(values: list[int], initial_total: int = 0):
    """Generator that yields outputs and maintains internal state."""
    total = initial_total
    for x in values:
        total += x
        yield total

# More complex example:
@PipeFunc.scan_generator(output_name="trajectory")
def particle_simulation(time_steps: np.ndarray, x0: float = 0, v0: float = 1, dt: float = 0.1):
    x, v = x0, v0
    for t in time_steps:
        x += v * dt
        v *= 0.99  # damping
        yield {"t": t, "x": x, "v": v}
```

**Advantages:**
- Natural Python pattern (generators)
- State is implicit in local variables
- No signature transformation
- Easy to understand and debug

### Strategy 3: Explicit Scan Function

```python
class Scan(PipeFunc):
    """Explicit scan function with clear inputs/outputs."""

    def __init__(self, func, output_name, xs, initial_state=None, **kwargs):
        self.scan_func = func
        self.xs = xs
        self.initial_state = initial_state
        super().__init__(self._scan_wrapper, output_name, **kwargs)

    def _scan_wrapper(self, **kwargs):
        xs_values = kwargs[self.xs]
        state = self.initial_state or {}

        outputs = []
        for x in xs_values:
            state, output = self.scan_func(x, state)
            outputs.append(output)

        return outputs

# Usage:
def accumulator_logic(x: int, state: dict) -> tuple[dict, int]:
    total = state.get('total', 0) + x
    return {'total': total}, total

accumulator = Scan(
    accumulator_logic,
    output_name="result",
    xs="values",
    initial_state={'total': 0}
)
```

**Advantages:**
- Explicit construction
- Clear separation of concerns
- No decorator magic
- Easy to extend and customize

### Strategy 4: Functional Approach with Reducers

```python
from functools import reduce

@PipeFunc.reducer(output_name="result", xs="values")
def accumulator(acc: int, x: int) -> int:
    """Simple reducer function."""
    return acc + x

# With intermediate results:
@PipeFunc.scan_reduce(output_name="result", xs="values", keep_intermediate=True)
def accumulator(acc: int, x: int) -> int:
    return acc + x

# With structured state:
@PipeFunc.scan_reduce(output_name="trajectory", xs="time_steps")
def simulate(state: dict, t: float, *, dt: float = 0.1) -> dict:
    return {
        "t": t,
        "x": state.get("x", 0) + state.get("v", 1) * dt,
        "v": state.get("v", 1) * 0.99
    }
```

**Advantages:**
- Familiar reduce/fold pattern
- Simple function signatures
- No tuple return requirement
- Natural accumulation pattern

### Strategy 5: Method Chaining API

```python
# Fluent API for building scans
scan = (
    PipeFunc.scan("result")
    .over("values")
    .with_state(total=0)
    .apply(lambda x, total: total + x)
)

# Or with a more complex example:
trajectory = (
    PipeFunc.scan("trajectory")
    .over("time_steps")
    .with_state(x=0.0, v=1.0)
    .with_params(dt=0.1)
    .apply(lambda t, x, v, dt: {
        "x": x + v * dt,
        "v": v * 0.99,
        "output": {"t": t, "x": x + v * dt, "v": v * 0.99}
    })
    .keep_intermediate()
)
```

**Advantages:**
- Clear, readable API
- Explicit configuration
- No hidden conventions
- Easy to extend with new options

## Recommendation: Hybrid Approach

Based on the analysis, I recommend a hybrid approach that:

1. **Keeps the decorator syntax** (familiar to users)
2. **Removes signature transformation** (less magic)
3. **Supports multiple state types** (flexible)
4. **Makes conventions explicit** (clear)

### Proposed Design:

```python
# Option 1: Simple accumulator with explicit state parameter
@PipeFunc.scan(output_name="result", xs="values", state_param="total")
def accumulator(values: list[int], total: int = 0) -> list[int]:
    """State parameter 'total' is automatically managed."""
    results = []
    for x in values:
        total += x
        results.append(total)
    return results

# Option 2: Generator-based for complex iterations
@PipeFunc.scan_iter(output_name="trajectory", xs="time_steps")
def simulate(time_steps: np.ndarray, x: float = 0, v: float = 1, dt: float = 0.1):
    """Generator that yields outputs."""
    for t in time_steps:
        x += v * dt
        v *= 0.99
        yield {"t": t, "x": x, "v": v}

# Option 3: Explicit state class for type safety
@dataclass
class SimState:
    x: float = 0.0
    v: float = 1.0

@PipeFunc.scan_state(output_name="trajectory", xs="time_steps", state_class=SimState)
def simulate_typed(t: float, state: SimState, dt: float = 0.1) -> tuple[SimState, dict]:
    new_state = SimState(
        x=state.x + state.v * dt,
        v=state.v * 0.99
    )
    output = {"t": t, "x": new_state.x, "v": new_state.v}
    return new_state, output
```

### Benefits of Proposed Design:

1. **No signature transformation** - what you write is what runs
2. **Multiple patterns** - choose the most natural for your use case
3. **Type safety** - optional but available when needed
4. **Clear conventions** - explicit parameter marking
5. **Backward compatible** - can coexist with current design

### Migration Path:

1. Keep current `@PipeFunc.scan` with deprecation warning
2. Introduce new decorators: `@PipeFunc.scan_iter`, `@PipeFunc.scan_state`
3. Update documentation with new patterns
4. Provide migration guide and automated converter
5. Remove old implementation in next major version

## Conclusion

The current ScanFunc design works but has significant usability issues. The proposed alternatives offer:
- More intuitive APIs
- Better type safety
- Clearer mental models
- Easier debugging
- Greater flexibility

The generator-based approach (`scan_iter`) is particularly promising as it:
- Uses familiar Python patterns
- Requires no new concepts
- Is easy to test and debug
- Handles both simple and complex cases well
