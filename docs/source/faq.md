# ❓ FAQ: Frequently Asked Questions

## How to handle defaults?

You can provide defaults in

- The original function definition (the normal way)
- The `pipefunc` decorator `@pipefunc(..., defaults={...})`
- Update the defaults of a `PipeFunc` object (a wrapped function) via `PipeFunc.update_defaults({...})`
- Update the defaults of an entire pipeline via `Pipeline.update_defaults({...})`

```python
from pipefunc import pipefunc, Pipeline

@pipefunc(output_name="y", defaults={"x": 2})
def f(a, x):
    return a * x

# update the defaults of the function afterwards
f.update_defaults({"x": 3})

@pipefunc(output_name="z", defaults={"b": 2})  # override `b=1` default
def g(y, b=1):
    return y + b

pipeline = Pipeline([f, g])
pipeline.update_defaults({"a": 1, "b": 3, "x": 1})  # override `b=2` default
print(pipeline.defaults)  # all parameters now have defaults
print(pipeline())  # no arguments required now
```

## How to bind parameters to a fixed value?

Instead of using defaults, you can bind parameters to a fixed value using the `bound` argument.

```python

@pipefunc(output_name="y", bound={"x": 2})  # x is now fixed to 2
def f(a, x):
    return a * x

f(a=3)  # x is fixed to 2
```

## How to handle multiple outputs?
