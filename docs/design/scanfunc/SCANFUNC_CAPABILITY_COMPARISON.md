# ScanFunc Capability Comparison: Current vs Generator-Based Design

## Original Requirements from Issue #865

### 1. ✅ **Iterative Execution with Feedback Loops**

**Current Design:**
```python
@PipeFunc.scan(output_name="trajectory", xs="time_steps")
def rk4_step(t: float, y: float = 1.0, dt: float = 0.1) -> tuple[dict[str, Any], float]:
    y_next = some_rk4_implementation(y, dt)
    carry = {"y": y_next}  # Feedback to next iteration
    return carry, y_next
```

**Generator Design - YES, it works:**
```python
@PipeFunc.scan_iter(output_name="trajectory")
def rk4_step(time_steps: list[float], y: float = 1.0, dt: float = 0.1):
    for t in time_steps:
        y = some_rk4_implementation(y, dt)  # State persists across iterations
        yield y
```

### 2. ✅ **Optional Intermediate Results**

**Current Design:**
```python
# With intermediate results (default)
@PipeFunc.scan(output_name="result", xs="values", return_intermediate=True)
def func(x, state=0):
    return {"state": state+x}, state+x  # Returns array of all outputs

# Without intermediate results
@PipeFunc.scan(output_name="result", xs="values", return_intermediate=False)
def func(x, state=0):
    return {"state": state+x}, None  # Returns only final carry dict
```

**Generator Design - YES, even more flexible:**
```python
# With intermediate results
@PipeFunc.scan_iter(output_name="result")
def func(values, state=0):
    for x in values:
        state += x
        yield state  # Yield each intermediate

# Without intermediate results - multiple options:
@PipeFunc.scan_iter(output_name="result", return_final_only=True)
def func(values, state=0):
    for x in values:
        state += x
    yield state  # Only yield final

# Or use reduce pattern
@PipeFunc.reduce(output_name="result")
def func(acc, x):
    return acc + x  # Automatically returns final only
```

### 3. ✅ **Integration with pipeline.map**

**Current Design:**
```python
@PipeFunc.scan(output_name="cumsum", xs="values", mapspec="values[i] -> cumsum[i]")
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    new_total = total + x
    return {"total": new_total}, new_total

pipeline.map({"batch_id": [0, 1, 2]})  # Parallel execution
```

**Generator Design - YES, same capability:**
```python
@PipeFunc.scan_iter(output_name="cumsum", mapspec="values[i] -> cumsum[i]")
def accumulator(values: list[int], total: int = 0):
    for x in values:
        total += x
        yield total

pipeline.map({"batch_id": [0, 1, 2]})  # Same parallel execution
```

### 4. ✅ **DAG Unrolling for Visualization**

Both designs can be unrolled into a linear chain of nodes. The generator approach might even be cleaner:

**Current:** Each iteration becomes a node with complex carry dict passing
**Generator:** Each iteration is a simple transformation step

## Additional Capabilities Comparison

### Complex State Management

**Genetic Algorithm Example:**

**Current Design (awkward dict juggling):**
```python
@PipeFunc.scan(output_name="evolution", xs="generations")
def genetic_step(gen: int, population: list = None, fitness: list = None) -> tuple[dict, dict]:
    if population is None:
        population = initialize_population()

    fitness = evaluate_fitness(population)
    new_population = selection_crossover_mutation(population, fitness)

    carry = {"population": new_population, "fitness": fitness}
    stats = {"generation": gen, "best_fitness": max(fitness), "avg_fitness": np.mean(fitness)}
    return carry, stats
```

**Generator Design (natural state management):**
```python
@PipeFunc.scan_iter(output_name="evolution")
def genetic_algorithm(generations: range, population_size: int = 100):
    population = initialize_population(population_size)

    for gen in generations:
        fitness = evaluate_fitness(population)
        population = selection_crossover_mutation(population, fitness)

        yield {
            "generation": gen,
            "best_fitness": max(fitness),
            "avg_fitness": np.mean(fitness),
            "best_individual": population[np.argmax(fitness)]
        }
```

### Nested Pipeline Integration

**Current Design (using nest_funcs_scan):**
```python
rk4_pipeline = Pipeline([calc_k1, calc_k2, calc_k3, calc_k4, rk4_update])

@rk4_pipeline.nest_funcs_scan(
    output_name="trajectory",
    xs="time_steps",
    output_nodes={"y_next"}
)
def rk4_scan(t: float, y: float = 1.0, dt: float = 0.1):
    # This function body is replaced by pipeline
    pass
```

**Generator Design (more explicit and debuggable):**
```python
@PipeFunc.scan_iter(output_name="trajectory")
def rk4_scan(time_steps: list[float], y: float = 1.0, dt: float = 0.1):
    rk4_pipeline = Pipeline([calc_k1, calc_k2, calc_k3, calc_k4, rk4_update])

    for t in time_steps:
        result = rk4_pipeline.run("y_next", kwargs={"t": t, "y": y, "dt": dt})
        y = result  # Update state
        yield {"t": t, "y": y}
```

## Inspection and Debugging

### ❓ Concern: "Does it prevent you from inspecting individual results?"

**Answer: NO, the generator design actually IMPROVES inspection!**

**Current Design - Limited Inspection:**
```python
# Can only access final carry via property after execution
result = scan_func._execute_scan(values=[1, 2, 3])
final_carry = scan_func.carry  # {"total": 6}
# But what happened at each step? Hard to debug!
```

**Generator Design - Full Inspection:**
```python
# Option 1: Step through manually
gen = accumulator.generator_func(values=[1, 2, 3])
first = next(gen)   # 1 - can inspect!
second = next(gen)  # 3 - can inspect!
third = next(gen)   # 6 - can inspect!

# Option 2: Debugging mode
@PipeFunc.scan_iter(output_name="result", debug=True)
def accumulator(values, total=0):
    for i, x in enumerate(values):
        total += x
        print(f"Step {i}: x={x}, total={total}")  # Easy debugging!
        yield total

# Option 3: Capture all states
@PipeFunc.scan_iter(output_name="result")
def accumulator_with_history(values, total=0):
    history = []
    for x in values:
        total += x
        state = {"x": x, "total": total, "timestamp": time.time()}
        history.append(state)
        yield {"current": total, "history": history}
```

## Performance Considerations

**Current Design:**
- Complex wrapper machinery adds overhead
- Dict creation/updating at each iteration
- Parameter remapping costs

**Generator Design:**
- Minimal overhead - just Python's generator protocol
- State in local variables (faster than dict access)
- No parameter remapping needed

## Conclusion

✅ **The generator-based design handles ALL the same use cases as the current design, and actually provides:**

1. **Better debugging** - Can step through iterations
2. **More natural state management** - Use variables, not dicts
3. **Greater flexibility** - Multiple patterns for different needs
4. **Better performance** - Less overhead
5. **Clearer code** - What you write is what runs

The generator approach fully satisfies the requirements from issue #865 while being more Pythonic and easier to use.
