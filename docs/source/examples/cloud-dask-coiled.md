---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: pipefunc
  language: python
  name: python3
---

# Cloud run with Dask and Coiled

Let's see how easy it is to run a pipefunc Pipeline in the cloud!

We will be using the code from the [Physics Simulation](physics-simulation.md) example.

```{code-cell} ipython3
from physics_pipeline import pipeline_charge

# Let's add some more axes to the pipeline, to sweep over the x, y, mesh_size, and coarse_mesh_size.
pipeline_sweep = pipeline_charge.copy()

# Add a cross-product of x and y
pipeline_sweep.add_mapspec_axis("x", axis="a")
pipeline_sweep.add_mapspec_axis("y", axis="b")

# And also a cross-product of the zipped mesh_size and coarse_mesh_size
pipeline_sweep.add_mapspec_axis("mesh_size", axis="c")
pipeline_sweep.add_mapspec_axis("coarse_mesh_size", axis="c")

# This results in a final 3D array for the `average_charge`:
pipeline_sweep.visualize_graphviz()
```

```{code-cell} ipython3
import coiled

cluster = coiled.Cluster(n_workers=1)
client = cluster.get_client()
cloud_executor = client.get_executor()
```

```{code-cell} ipython3
import numpy as np

inputs = {
    "V_left": np.linspace(0, 2, 10),
    "V_right": np.linspace(-0.5, 0.5, 10),
    "x": np.linspace(0.1, 0.2, 3),
    "y": np.linspace(0.2, 0.3, 3),
    "mesh_size": [0.01, 0.02],
    "coarse_mesh_size": [0.05, 0.06],
}
results = pipeline_sweep.map(
    inputs,
    run_folder=None,
    executor=cloud_executor,
    show_progress=True,
    scheduling_strategy="eager",
)
results.to_xarray()
```

```{code-cell} ipython3
client.shutdown()
```
