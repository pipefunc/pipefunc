---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

# Linear Chaining Helper

`pipefunc.helpers.linear_chain` connects a list of functions linearly by applying minimal renames so “output of one becomes input of the next.”
It only renames arguments where needed; it doesn’t change how your functions compute or how batching works.

Use it to compose simple array→array transforms: keep a small toolbox of transforms and pick any subset (in any order) at runtime.

```{code-cell} ipython3
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.helpers import linear_chain

@pipefunc("to_float")
def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)

@pipefunc("normalize")
def normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    rng = np.ptp(img)  # max - min
    return (img - img.min()) / (rng + eps)

@pipefunc("gamma")
def gamma(img: np.ndarray, g: float = 2.2) -> np.ndarray:
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / g)

@pipefunc("threshold")
def threshold(img: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (img > t).astype(np.float32)

# Choose any subset and order at runtime

chain = linear_chain(
    [
        to_float,
        gamma,
        normalize,
        threshold,
        normalize.copy(output_name="output"),  # apply normalize again, with different output name
    ],
)
pipe = Pipeline(chain)

img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
out = pipe.run("output", kwargs={"img": img, "t": 0.4})
out.shape, out.dtype
```

## Multi-output functions

To select a specific output, name your parameter to match it:

```{code-cell} ipython3
@pipefunc(("a", "b"))
def split(x: int) -> tuple[int, int]:
    return x, 10 * x

@pipefunc("sink_b")
def sink_b(b: int) -> int:  # parameter 'b' matches second output
    return b

Pipeline(linear_chain([split, sink_b])).run("sink_b", kwargs={"x": 7})  # -> 70
```

## Bound parameters

When using bound parameters, put the data-flow parameter first:

```{code-cell} ipython3
@pipefunc("m1")
def f1(src: int) -> int: return src + 1

@pipefunc("m2", bound={"skip": 1})
def f2(real_input: int, skip: int) -> int: return real_input + skip

Pipeline(linear_chain([f1, f2])).run("m2", kwargs={"src": 10})  # -> 12
```

## How it works

- Parameter name matches upstream output → uses that match
- Otherwise → renames first parameter to upstream output
- Plain callables auto-wrap as `PipeFunc` with `output_name=f.__name__`
- `copy=True` (default) returns copies; `copy=False` modifies originals

## Batches of arrays

For a batch of arrays, either make each transform vectorized over the batch dimension or use `mapspec` on your functions to define element-wise mapping (e.g., `"img[i] -> out[i]"`). `linear_chain` intentionally does not set or change `mapspec`; declare it on your `@pipefunc(...)` where needed.
