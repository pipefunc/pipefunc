# ScanFunc Generator Implementation Plan

## Overview
Implement a new generator-based API for ScanFunc that is more Pythonic and easier to use, while maintaining backward compatibility with the current implementation.

## Implementation Strategy

### Phase 1: Core Implementation (Keep Current Code)
1. Create new branch `scanfunc-generator`
2. Add new classes alongside existing ScanFunc:
   - `ScanIterFunc` - Generator-based scanning
   - `ScanStateFunc` - Type-safe state management
   - `ReduceFunc` - Simple reduce operations
3. Keep all existing code for validation

### Phase 2: API Design

#### Primary Pattern: Generator-Based (`scan_iter`)
```python
@PipeFunc.scan_iter(output_name="result")
def my_scan(values: list[int], state: int = 0):
    for x in values:
        state += x
        yield state
```

**Key features:**
- Natural Python generator pattern
- No signature transformation
- State via local variables
- Easy debugging with print/breakpoints

#### Type-Safe Pattern (`scan_state`)
```python
@dataclass
class State:
    total: int = 0

@PipeFunc.scan_state(output_name="result", xs="values", state_class=State)
def my_scan(x: int, state: State) -> tuple[State, Any]:
    new_state = State(total=state.total + x)
    return new_state, new_state.total
```

**Key features:**
- Explicit state type
- Works with dataclasses, named tuples, etc.
- Type checking support

#### Reduce Pattern (`reduce`)
```python
@PipeFunc.reduce(output_name="sum", initial=0)
def add(acc: int, x: int) -> int:
    return acc + x
```

**Key features:**
- Simple accumulation
- Familiar reduce/fold pattern
- Minimal boilerplate

### Phase 3: Implementation Details

#### ScanIterFunc Class
```python
class ScanIterFunc(PipeFunc):
    def __init__(self, func, output_name, xs=None, return_final_only=False, **kwargs):
        self.generator_func = func
        self.xs = xs
        self.return_final_only = return_final_only

        # Create wrapper that collects generator results
        wrapper = self._create_wrapper()
        super().__init__(wrapper, output_name, **kwargs)
```

**Wrapper logic:**
1. If `xs` provided, extract from kwargs
2. Call generator function
3. Collect results (all or final only)
4. Return appropriate format (array, list, single value)

#### Integration Points
- Must work with `pipeline.run()` and `pipeline.map()`
- Support mapspec for parallel execution
- Handle resources, caching, profiling
- Proper error handling and validation

### Phase 4: Testing Strategy

#### Test Categories
1. **Functional Tests**
   - Basic accumulation
   - Complex state management
   - Early stopping
   - Empty sequences

2. **Integration Tests**
   - Pipeline integration
   - Parallel execution with map
   - Nested pipelines
   - Resource management

3. **Validation Tests**
   - Compare outputs with current ScanFunc
   - Performance benchmarks
   - Memory usage comparison
   - Error handling parity

#### Validation Matrix
| Feature | Current ScanFunc | scan_iter | scan_state | reduce |
|---------|-----------------|-----------|------------|---------|
| Basic accumulation | ✓ | ✓ | ✓ | ✓ |
| Complex state | ✓ | ✓ | ✓ | - |
| Intermediate results | ✓ | ✓ | ✓ | opt |
| Final only | ✓ | ✓ | ✓ | ✓ |
| Pipeline.map | ✓ | ✓ | ✓ | ✓ |
| Early stopping | - | ✓ | - | - |
| Type safety | - | - | ✓ | - |

### Phase 5: Documentation

1. **API Reference**
   - New decorators and their parameters
   - Usage examples
   - When to use which pattern

2. **Migration Guide**
   - Current → scan_iter examples
   - Common patterns
   - Gotchas and solutions

3. **Conceptual Docs**
   - Update concepts/scanfunc.md
   - Add comparison section
   - Performance considerations

### Phase 6: Rollout Plan

1. **Alpha** (Month 1)
   - Implement alongside current
   - Internal testing
   - Gather feedback

2. **Beta** (Month 2-3)
   - Public preview
   - Documentation
   - Migration tools

3. **Stable** (Month 4)
   - Feature parity confirmed
   - Performance validated
   - Migration guide complete

4. **Deprecation** (Month 6)
   - Add warnings to old API
   - Support both patterns

5. **Removal** (Month 12)
   - Remove old implementation
   - Clean up codebase

## Success Criteria

1. **Functionality**
   - All current use cases supported
   - New patterns work as designed
   - No performance regression

2. **Usability**
   - Reduced learning curve
   - Better error messages
   - Improved debugging

3. **Code Quality**
   - 50% less implementation code
   - 100% test coverage
   - Clean, maintainable design

## Risk Mitigation

1. **Backward Compatibility**
   - Keep old code during transition
   - Extensive validation suite
   - Clear migration path

2. **Performance**
   - Benchmark all operations
   - Profile memory usage
   - Optimize hot paths

3. **User Adoption**
   - Clear documentation
   - Migration tools
   - Gradual rollout

## File Structure

```
pipefunc/
├── _scanfunc.py          # Current implementation (keep)
├── _scanfunc_iter.py     # New generator-based
├── _scanfunc_state.py    # Type-safe state
├── _scanfunc_reduce.py   # Reduce pattern
└── _scanfunc_common.py   # Shared utilities

tests/
├── test_scanfunc.py      # Current tests (keep)
├── test_scanfunc_iter.py # New generator tests
└── test_scanfunc_validation.py  # Cross-validation
```

## Next Steps

1. Create branch
2. Implement ScanIterFunc
3. Add basic tests
4. Validate against current implementation
5. Iterate based on findings
