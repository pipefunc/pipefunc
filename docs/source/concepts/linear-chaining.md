---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

# Linear Chaining Helper

`pipefunc.helpers.linear_chain` creates a linear sequence of `PipeFunc`s by applying minimal renames so the output of one function flows into the next. It does not alter or add any `mapspec`; it only adjusts argument names where needed.

```{contents} ToC
:depth: 2
```

## Why use `linear_chain`?

- Avoid boilerplate renames when building simple f1 → f2 → f3 pipelines.
- Keep extra parameters intact: only the “main” input hops along; all other inputs remain as-is.
- Works with multi-output functions (choose which output to forward).
- Honors bound parameters (auto-selects the first non-bound parameter).

## Basic example

```python
from pipefunc import pipefunc, Pipeline
from pipefunc.helpers import linear_chain

@pipefunc("mid1")
def f1(x: int) -> int:
    return x + 1

@pipefunc("mid2")
def f2(input: int, k: int = 2) -> int:
    return input * k

@pipefunc("out")
def f3(value: int) -> int:
    return value - 3

chain = linear_chain([f1, f2, f3])  # renames f2.input->"mid1", f3.value->"mid2"
Pipeline(chain).run("out", kwargs={"x": 2, "k": 3})  # -> 6
```

## When names already match

If a downstream parameter already equals the upstream output name, no rename is applied.

```python
@pipefunc("m1")
def f1(src: int) -> int: return src + 1

@pipefunc("m2b")
def f2b(m1: int, k: int = 5) -> int: return m1 * k

@pipefunc("out")
def f3(value: int) -> int: return value - 3

chain = linear_chain([f1, f2b, f3])  # f2b unchanged
Pipeline(chain).run("out", kwargs={"src": 4, "k": 2})  # -> 7
```

## Multi-output upstream

By default, the first output name is forwarded. You can select a different one.

```python
@pipefunc(("a", "b"))
def split(x: int) -> tuple[int, int]:
    return x, 10 * x

@pipefunc("sink")
def sink(y: int) -> int: return y

# Default -> forwards "a"
Pipeline(linear_chain([split, sink])).run("sink", kwargs={"x": 7})  # -> 7

# Explicitly forward the second output (index 1)
Pipeline(linear_chain([split, sink], select_output=1)).run("sink", kwargs={"x": 7})  # -> 70
```

## Bound parameters are skipped

`linear_chain` auto-selects the first non-bound parameter as the main input when needed.

```python
@pipefunc("m1")
def f1(src: int) -> int: return src + 1

@pipefunc("m2", bound={"skip": 1})
def f2(skip: int, real_input: int) -> int: return real_input + skip

Pipeline(linear_chain([f1, f2])).run("m2", kwargs={"src": 10})  # -> 12
```

## Selecting the downstream parameter

Provide `select_param` to choose which parameter receives the upstream value.

```python
@pipefunc("m1")
def f1(x: int) -> int: return x + 1

@pipefunc("m2")
def f2(aux: int, data: int) -> int: return data * 2

chain = linear_chain([f1, f2], select_param="data")
Pipeline(chain).run("m2", kwargs={"x": 3, "aux": 0})  # -> 8
```

## Plain callables work too

Callables are wrapped as `PipeFunc`s with `output_name=f.__name__`.

```python
def g(z: int) -> int: return z * 2
def h(t: int) -> int: return t + 5

chain = linear_chain([g, h])
Pipeline(chain).run("h", kwargs={"z": 3})  # -> 11
```

## Notes

- No `mapspec` changes: existing mapspecs stay as-is; none are created or removed.
- Middle functions must have at least one parameter to receive the upstream value.
- `select_output` can be a name, index, or a callable `lambda pf: "desired_output"`.
- `copy=True` (default) returns copies; set `copy=False` to modify your originals.
