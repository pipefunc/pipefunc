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

# Run Adaptive Sweeps in 1D, 2D, 3D, ND

Using Adaptive sweeps instead of regular sweeps can save a lot of time.
Currently, there is no deep integration in `pipefunc` to do adaptive sweeps.
However, we can still do a poor man's version of them.

:::{note}
In the future the idea is to allow a syntax like this:

   ```python
   pipeline.map(inputs={'a': Bound(0, 1), 'b': Bound(0, 1), c=[0, 1, 2]})
   ```

This will turn into a 2D adaptive sweep (with `adaptive.Learner2D`) over `a` and `b` and do that for each value of `c`.

:::

This poor man's version runs a `pipeline.map` for each iteration in the adaptive sweep, creating a new `run_folder` for each iteration.

---

## Setting the stage

Let's set the stage by setting up a simple pipeline with a reduction operation.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int, c: int) -> int:
    return 2 * x + c


@pipefunc(output_name="sum_")
def take_sum(y: list[int], d: int) -> float:
    return sum(y) / d


pipeline = Pipeline([double_it, take_sum])

inputs = {"x": [0, 1, 2, 3], "c": 1, "d": 2}
run_folder = "my_run_folder"
results = pipeline.map(inputs, run_folder=run_folder)
print(results["y"].output.tolist())
assert results["y"].output.tolist() == [1, 3, 5, 7]
assert results["sum_"].output == 8.0
```

This pipeline returns a single number, which is the sum of the inputs.

However, often we want to run a pipeline for a range of inputs, on e.g., a 2D grid on `c` and `d`.

```{code-cell} ipython3
pipeline2d = pipeline.copy()
pipeline2d.add_mapspec_axis("c", axis="j")
pipeline2d.add_mapspec_axis("d", axis="k")
```

Now let's run this on a 2D grid of `c` and `d`:

```{code-cell} ipython3
import numpy as np

inputs = {"x": [0, 1, 2, 3], "c": np.linspace(0, 100, 20), "d": np.linspace(-1, 1, 20)}
run_folder = "my_run_folder"
results = pipeline2d.map(inputs, run_folder=run_folder)
```

We can load the results into an xarray dataset and plot them.

```{code-cell} ipython3
from pipefunc.map import load_xarray_dataset

ds = load_xarray_dataset(run_folder=run_folder)
ds.sum_.astype(float).plot(x="c", y="d")
```

## Using `adaptive` for adaptive sweeps

```{code-cell} ipython3
import adaptive

adaptive.notebook_extension()
```

We redefine the `pipeline` with the single reduction operation.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int, c: int) -> int:
    return 2 * x + c


@pipefunc(output_name="sum_")
def take_sum(y: list[int], d: int) -> float:
    return sum(y) / d


pipeline = Pipeline([double_it, take_sum])
```

### Using `adaptive.Learner1D` for a 1D adaptive sweep

```{code-cell} ipython3
from pipefunc.map.adaptive import to_adaptive_learner

run_folder_template = "adaptive_1d/run_folder_{}"
learner1d = to_adaptive_learner(
    pipeline,
    inputs={"x": [0, 1, 2, 3], "d": 1},
    adaptive_dimensions={"c": (0, 100)},
    adaptive_output="sum_",
    run_folder_template=run_folder_template,
)
```

Then we can drive the learner sequentially because the `pipeline.map` is already parallelized.

```{code-cell} ipython3
adaptive.runner.simple(learner1d, npoints_goal=10)
```

We can now inspect the results of the `adaptive_output` in the learner

```{code-cell} ipython3
learner1d.to_numpy()
```

```{code-cell} ipython3
learner1d.plot()
```

Or inspect all the underlying data

```{code-cell} ipython3
from pathlib import Path

from pipefunc.map import load_xarray_dataset

all_folders = Path(run_folder_template).parent.glob("*")
all_folders = sorted(all_folders)
datasets = [load_xarray_dataset(run_folder=folder) for folder in all_folders]
```

```{code-cell} ipython3
datasets[0]  # just look at the first dataset
```

### Using `adaptive.Learner2D` for a 2D adaptive sweep

```{code-cell} ipython3
run_folder_template = "adaptive_2d/run_folder_{}"
learner2d = to_adaptive_learner(
    pipeline,
    inputs={"x": [0, 1, 2, 3]},
    adaptive_dimensions={"c": (0, 100), "d": (-1, 1)},
    adaptive_output="sum_",
    run_folder_template=run_folder_template,
)
```

Even though the we are doing a 2D sweep, we can still use the `adaptive.Runner` to also run "doubly" parallel, where multiple `pipeline.map` are run in parallel, in addition to the parallelization of the `pipeline.map` itself.

```{code-cell} ipython3
runner = adaptive.Runner(learner2d, npoints_goal=10)
runner.live_info()
```

We can now inspect the results of the `adaptive_output` in the learner

```{code-cell} ipython3
learner2d.plot()
```

```{code-cell} ipython3
learner2d.to_numpy()
```

Or inspect all the underlying data

```{code-cell} ipython3
from pathlib import Path

from pipefunc.map import load_xarray_dataset

all_folders = Path(run_folder_template).parent.glob("*")
all_folders = sorted(all_folders)
datasets = [load_xarray_dataset(run_folder=folder) for folder in all_folders]
```

```{code-cell} ipython3
datasets[0]  # just look at the first dataset
```
