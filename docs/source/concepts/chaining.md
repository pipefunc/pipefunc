---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

# Function Chaining Helper

`pipefunc.helpers.chain` connects functions linearly so the output of one becomes the input of the next.

```{code-cell} ipython3
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.helpers import chain

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

chain = chain(
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

## How it works

- If a parameter name matches an upstream output, uses that match
- Otherwise, renames the first parameter to the upstream output
- Plain callables auto-wrap as `PipeFunc` with `output_name=f.__name__`

## Multi-output functions

By default, the first output is used. To select a different output, name your parameter to match it:

```{code-cell} ipython3
@pipefunc(("a", "b"))
def split(x: int) -> tuple[int, int]:
    return x, 10 * x

@pipefunc("sink_b")
def sink_b(b: int) -> int:  # parameter 'b' matches second output
    return b

Pipeline(chain([split, sink_b])).run("sink_b", kwargs={"x": 7})  # -> 70
```
