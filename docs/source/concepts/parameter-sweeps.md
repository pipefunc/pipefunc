---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parameter Sweeps

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

The `pipefunc.sweep` module provides a convenient way to contruct parameter sweeps.
It was developed before `pipeline.map` which can perform sweep operations in parallel.
However, by itself {class}`pipefunc.sweep.Sweep` might still be useful for cases where you have a pipeline that has no `mapspec`.

```{code-cell} ipython3
from pipefunc.sweep import Sweep

combos = {
    "a": [0, 1, 2],
    "b": [0, 1, 2],
    "c": [0, 1, 2],
}
# This means a Cartesian product of all the values in the lists
# while zipping ("a", "b").
sweep = Sweep(combos, dims=[("a", "b"), "c"])
sweep.list()[:10]  # show the first 10 combinations
```

The function `set_cache_for_sweep` then enables caching for nodes in the pipeline that are expected to be executed two or more times during the parameter sweep.

```python
from pipefunc.sweep import set_cache_for_sweep

set_cache_for_sweep(output_name, pipeline, sweep, min_executions=2, verbose=True)
```

We can now run the sweep using e.g.,

```python
results = [
    pipeline.run(output_name, kwargs=combo, full_output=True) for combo in sweep.list()
]
```
