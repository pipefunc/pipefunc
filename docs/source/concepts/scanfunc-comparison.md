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

# ScanFunc vs scan_iter: Comparison Guide

> Compare the current ScanFunc with the new generator-based scan_iter

```{contents} ToC
:depth: 2
```

This guide compares the current `PipeFunc.scan` (ScanFunc) with the new `PipeFunc.scan_iter` (generator-based) approach, helping you understand the differences and choose the right tool for your use case.

---

## Quick Comparison Table

| Feature | `scan` (Current) | `scan_iter` (New) |
|---------|-----------------|-------------------|
| **Syntax** | Complex tuple return | Natural generator |
| **Signature** | Transformed | Direct |
| **State Management** | Dict-based carry | Local variables |
| **Return Format** | `(carry_dict, output)` | `yield output` |
| **Learning Curve** | Steep | Gentle |
| **Code Lines** | More verbose | More concise |
| **Type Safety** | Dict keys | Regular variables |

---

## Side-by-Side Examples

### Simple Accumulator

:::::{grid} 2
:gutter: 3

::::{grid-item}
**Current `scan`**
```python
@PipeFunc.scan(output_name="cumsum", xs="values")
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    new_total = total + x
    carry = {"total": new_total}
    return carry, new_total

# Call with:
pipeline.run("cumsum", kwargs={"values": [1, 2, 3]})
```
::::

::::{grid-item}
**New `scan_iter`**
```python
@PipeFunc.scan_iter(output_name="cumsum")
def accumulator(values: list[int], total: int = 0):
    for x in values:
        total += x
        yield total

# Call with:
pipeline.run("cumsum", kwargs={"values": [1, 2, 3]})
```
::::

:::::

**Key Differences:**
- `scan_iter` uses natural Python generator syntax
- No signature transformation - what you write is what you call
- State is just local variables, not dict keys

### Fibonacci Sequence

:::::{grid} 2
:gutter: 3

::::{grid-item}
**Current `scan`**
```python
@PipeFunc.scan(output_name="fibonacci", xs="n_steps")
def fib_scan(x: int, a: int = 0, b: int = 1) -> tuple[dict[str, Any], int]:
    next_val = a + b
    carry = {"a": b, "b": next_val}
    return carry, next_val

# Call with:
pipeline.run("fibonacci", kwargs={"n_steps": list(range(10))})
```
::::

::::{grid-item}
**New `scan_iter`**
```python
@PipeFunc.scan_iter(output_name="fibonacci")
def fib_iter(n_steps: int, a: int = 0, b: int = 1):
    for _ in range(n_steps):
        a, b = b, a + b
        yield b

# Call with:
pipeline.run("fibonacci", kwargs={"n_steps": 10})
```
::::

:::::

**Key Differences:**
- `scan_iter` can take a simple integer instead of a list
- State updates are natural Python assignments
- No need to manage carry dict keys

### Complex State Tracking

:::::{grid} 2
:gutter: 3

::::{grid-item}
**Current `scan`**
```python
@PipeFunc.scan(output_name="trajectory", xs="time_steps")
def particle_scan(
    t: float,
    x: float = 0.0,
    v: float = 1.0,
    dt: float = 0.1
) -> tuple[dict[str, Any], dict[str, float]]:
    new_x = x + v * dt
    new_v = v * 0.99  # damping

    carry = {"x": new_x, "v": new_v}
    output = {"t": t, "x": new_x, "v": new_v}
    return carry, output
```
::::

::::{grid-item}
**New `scan_iter`**
```python
@PipeFunc.scan_iter(output_name="trajectory")
def particle_iter(
    time_steps: list[float],
    x: float = 0.0,
    v: float = 1.0,
    dt: float = 0.1
):
    for t in time_steps:
        x += v * dt
        v *= 0.99  # damping
        yield {"t": t, "x": x, "v": v}
```
::::

:::::

**Key Differences:**
- `scan_iter` modifies state variables directly
- No need to construct separate carry and output dicts
- More readable and maintainable

### Return Final Only

:::::{grid} 2
:gutter: 3

::::{grid-item}
**Current `scan`**
```python
@PipeFunc.scan(
    output_name="final_sum",
    xs="values",
    return_intermediate=False
)
def sum_final_scan(x: int, total: int = 0) -> tuple[dict[str, Any], None]:
    new_total = total + x
    carry = {"total": new_total}
    return carry, None

# Returns: {"total": 15}
```
::::

::::{grid-item}
**New `scan_iter`**
```python
@PipeFunc.scan_iter(
    output_name="final_sum",
    return_final_only=True
)
def sum_final_iter(values: list[int], total: int = 0):
    for x in values:
        total += x
    yield total

# Returns: 15
```
::::

:::::

**Key Differences:**
- `scan_iter` uses `return_final_only` instead of `return_intermediate`
- Returns the actual value, not wrapped in a dict
- Cleaner syntax for final-value-only operations

---

## Feature Comparison

### Signature Transformation

**Current `scan`**: Transforms function signature
```python
# You write:
def func(x: int, carry1: int = 0) -> tuple[dict[str, Any], int]:
    ...

# But call as:
pipeline.run("output", kwargs={"values": [1, 2, 3]})
# Note: 'x' parameter disappears!
```

**New `scan_iter`**: Direct signature
```python
# You write:
def func(values: list[int], state: int = 0):
    ...

# You call:
pipeline.run("output", kwargs={"values": [1, 2, 3], "state": 0})
# What you see is what you get!
```

### State Management

**Current `scan`**: Dictionary-based
```python
# Must carefully manage dict keys
carry = {"count": count + 1, "sum": sum + x}
# Keys must match parameter names exactly
```

**New `scan_iter`**: Variable-based
```python
# Just use variables naturally
count += 1
sum += x
# No key management needed
```

### Error Handling

**Current `scan`**: Runtime dict key errors
```python
# Easy to make mistakes:
carry = {"total": new_total}  # Oops, should be "sum"
# Error only appears at runtime
```

**New `scan_iter`**: Compile-time variable errors
```python
# Variable errors caught immediately:
totl += x  # NameError: name 'totl' is not defined
# IDE can catch these before running
```

---

## Parallel Execution

Both approaches support parallel execution with `pipeline.map`:

:::::{grid} 2
:gutter: 3

::::{grid-item}
**Current `scan`**
```python
@PipeFunc.scan(
    output_name="cumsum",
    xs="values",
    mapspec="values[i] -> cumsum[i]"
)
def cumsum_scan(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    new_total = total + x
    return {"total": new_total}, new_total
```
::::

::::{grid-item}
**New `scan_iter`**
```python
@PipeFunc.scan_iter(
    output_name="cumsum",
    mapspec="values[i] -> cumsum[i]"
)
def cumsum_iter(values: list[int], total: int = 0):
    for x in values:
        total += x
        yield total
```
::::

:::::

Both work identically with `pipeline.map` for parallel execution.

---

## Migration Guide

### Converting from `scan` to `scan_iter`

1. **Remove tuple return** → Use `yield`
2. **Remove carry dict** → Use local variables
3. **Change `xs` parameter** → Make it the actual parameter
4. **Update decorator** → `@PipeFunc.scan` → `@PipeFunc.scan_iter`

### Example Migration

```python
# Before (scan)
@PipeFunc.scan(output_name="result", xs="items")
def process_scan(item: str, count: int = 0, words: list = None) -> tuple[dict[str, Any], dict]:
    if words is None:
        words = []

    new_count = count + 1
    new_words = words + [item]

    carry = {"count": new_count, "words": new_words}
    output = {"item": item, "count": new_count}
    return carry, output

# After (scan_iter)
@PipeFunc.scan_iter(output_name="result")
def process_iter(items: list[str], count: int = 0, words: list = None):
    if words is None:
        words = []

    for item in items:
        count += 1
        words.append(item)
        yield {"item": item, "count": count}
```

---

## When to Use Which?

### Use `scan` (current) when:
- You have existing code using it
- You need compatibility with current pipefunc versions
- You're following existing team patterns

### Use `scan_iter` (new) when:
- Starting new projects
- You want cleaner, more Pythonic code
- You prefer generator patterns
- You want better IDE support and error detection

---

## Performance Comparison

Both approaches have similar performance characteristics:
- Same parallel execution capabilities
- Same memory efficiency
- Same integration with pipefunc features

The main differences are in developer experience and code clarity.

---

## Summary

The new `scan_iter` provides a more Pythonic and intuitive way to write iterative algorithms:

**Advantages of `scan_iter`:**
- ✅ Natural Python generator syntax
- ✅ No signature transformation magic
- ✅ Direct variable state management
- ✅ Better IDE support and error detection
- ✅ Cleaner, more readable code
- ✅ Easier to test and debug

**Current Compatibility:**
- Both approaches coexist in the codebase
- No breaking changes to existing code
- Gradual migration path available

Choose `scan_iter` for new code to benefit from its cleaner design, while existing `scan` code continues to work without modification.
