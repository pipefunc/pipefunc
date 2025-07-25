# ScanFunc Design Recommendation

## Executive Summary

The current ScanFunc implementation has significant usability issues due to its complex signature transformation and rigid API. I recommend adopting a **generator-based approach** as the primary pattern, with optional type-safe variants for advanced use cases.

## Key Problems with Current Design

1. **Signature Transformation Confusion**
   - Function signature changes between definition and execution
   - First parameter (`x`) disappears, replaced by `xs` parameter
   - Non-intuitive and requires documentation diving

2. **Rigid Return Pattern**
   - Forces `(carry_dict, output)` tuple return
   - Dict requirement for carry is limiting
   - Awkward when output isn't needed

3. **Implementation Complexity**
   - 500+ lines of wrapper machinery
   - Custom pickling to handle circular references
   - Hard to debug and maintain

## Recommended Solution: Generator-Based API

### Primary Pattern: `@PipeFunc.scan_iter`

```python
# Simple, Pythonic, No Magic
@PipeFunc.scan_iter(output_name="cumsum")
def accumulator(values: list[int], total: int = 0):
    """What you write is what you get - no transformation."""
    for x in values:
        total += x
        yield total
```

**Benefits:**
- Natural Python pattern - everyone understands generators
- No signature transformation
- State management via local variables
- Easy to test: just call the function
- 50% less implementation code

### Advanced Pattern: Type-Safe State

For cases requiring explicit state management:

```python
@dataclass
class SimState:
    x: float = 0.0
    v: float = 1.0

@PipeFunc.scan_state(output_name="trajectory", xs="time_steps", state_class=SimState)
def simulate(t: float, state: SimState, dt: float = 0.1) -> tuple[SimState, dict]:
    """Type-safe state management."""
    new_state = SimState(
        x=state.x + state.v * dt,
        v=state.v * 0.99
    )
    return new_state, {"t": t, "x": new_state.x, "v": new_state.v}
```

### Simple Pattern: Reduce Operations

For basic accumulation patterns:

```python
@PipeFunc.reduce(output_name="sum", initial=0)
def add(acc: int, x: int) -> int:
    """Simple reduce pattern."""
    return acc + x
```

## Implementation Strategy

### Phase 1: Parallel Development (Months 1-2)
- Implement `scan_iter` alongside existing `scan`
- No breaking changes
- Gather user feedback

### Phase 2: Documentation & Examples (Month 3)
- Update docs to showcase new patterns
- Provide migration examples
- Create conversion tools

### Phase 3: Deprecation (Months 4-6)
- Add deprecation warnings to old API
- Support both patterns
- Help users migrate

### Phase 4: Cleanup (Month 7+)
- Remove old implementation
- Simplify codebase
- Performance optimizations

## Why This Matters

### Developer Experience
- **Current**: "Why doesn't my function signature match what I call?"
- **Proposed**: "Oh, it's just a generator, I know how those work!"

### Code Clarity
```python
# Current: What's happening here?
@PipeFunc.scan(output_name="result", xs="values")
def func(x: int, state: int = 0) -> tuple[dict[str, Any], int]:
    return {"state": state + x}, state + x

# Proposed: Crystal clear
@PipeFunc.scan_iter(output_name="result")
def func(values: list[int], state: int = 0):
    for x in values:
        state += x
        yield state
```

### Maintenance Burden
- Current: 500+ lines of complex wrapper code
- Proposed: ~200 lines of straightforward implementation

## Conclusion

The current ScanFunc design works but fights against Python's nature. The proposed generator-based approach provides:

1. **Intuitive API** - Works the way Python developers expect
2. **Reduced Complexity** - 50% less code to maintain
3. **Better Debugging** - What you write is what runs
4. **Flexibility** - Multiple patterns for different use cases

**Strong Recommendation**: Adopt the generator-based API as the primary pattern, keeping the current API only for backward compatibility during a transition period.
