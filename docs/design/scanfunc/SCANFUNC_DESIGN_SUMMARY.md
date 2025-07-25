# ScanFunc Design Issues & Recommendations

## 🚨 Critical Issues with Current Design

### 1. **Confusing Signature Magic**
```python
# You write this:
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    ...

# But call it like this:
pipeline.run("result", kwargs={"values": [1, 2, 3]})  # Where did 'x' go? What's 'values'?
```

### 2. **Forced Tuple Return Pattern**
- Must always return `(carry_dict, output)`
- Even when you don't need intermediate outputs
- Dict requirement is rigid and error-prone

### 3. **Hidden First Parameter Convention**
- First parameter is "special" but this isn't obvious
- Easy to get wrong, hard to debug
- Requires documentation diving to understand

### 4. **Complex Implementation**
- 500+ lines of wrapper machinery
- Custom pickling to avoid circular references
- Hard to understand and maintain

## ✅ Recommended Solution: Generator-Based API

### Simple, Pythonic, No Magic:

```python
# Current (confusing):
@PipeFunc.scan(output_name="cumsum", xs="values")
def accumulator(x: int, total: int = 0) -> tuple[dict[str, Any], int]:
    new_total = total + x
    return {"total": new_total}, new_total

# Proposed (intuitive):
@PipeFunc.scan_iter(output_name="cumsum")
def accumulator(values: list[int], total: int = 0):
    for x in values:
        total += x
        yield total
```

### Benefits:
1. **What you write is what runs** - no signature transformation
2. **Natural Python pattern** - everyone knows generators
3. **State is just variables** - no dict juggling
4. **Easy to test** - just call the function directly
5. **50% less code** - simpler implementation

## 🎯 Alternative Patterns for Different Use Cases

### For Type Safety:
```python
@dataclass
class State:
    total: int = 0
    count: int = 0

@PipeFunc.scan_state(output_name="stats", state_class=State)
def compute_stats(x: float, state: State) -> tuple[State, dict]:
    new_state = State(
        total=state.total + x,
        count=state.count + 1
    )
    return new_state, {"mean": new_state.total / new_state.count}
```

### For Simple Reductions:
```python
@PipeFunc.reduce(output_name="sum")
def add(acc: int, x: int) -> int:
    return acc + x
```

### For Builder Pattern:
```python
scan = (
    PipeFunc.scan("trajectory")
    .over("time_steps")
    .with_state(x=0, v=1)
    .apply(lambda t, x, v, dt: {
        "x": x + v * dt,
        "v": v * 0.99
    })
)
```

## 🚀 Migration Strategy

1. **Phase 1**: Add new `scan_iter` alongside existing `scan`
2. **Phase 2**: Deprecation warnings on old API
3. **Phase 3**: Migration tools and guides
4. **Phase 4**: Remove old implementation

## 📊 Comparison Table

| Aspect | Current Design | Proposed Design |
|--------|---------------|-----------------|
| Signature | Transformed | Direct |
| Return Type | Forced tuple | Flexible |
| State Management | Dict only | Any type |
| Mental Model | Complex | Simple |
| Lines of Code | ~500 | ~250 |
| Test Complexity | High | Low |
| Debug Experience | Poor | Excellent |

## 🎬 Bottom Line

The current ScanFunc design works but fights against Python's nature. The proposed generator-based approach:
- Is immediately understandable
- Requires no documentation to use
- Works the way Python developers expect
- Reduces implementation complexity by 50%

**Recommendation**: Adopt the generator-based API as the primary pattern, with optional type-safe variants for advanced use cases.
