---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: pipefunc
  language: python
  name: python3
---

# Physics Based Example

```{try-notebook}
```

This example demonstrates using the `pipefunc` for a physics-based simulation. The goal is to create a pipeline for geometry creation, meshing, material assignment, and electrostatics calculations, culminating in computing the average charge.

> Note: this example is based on the [`aiida-dynamic-workflows` tutorial](https://github.com/microsoft/aiida-dynamic-workflows/blob/4d452ed3be4192dc5b2c8f40690f82c3afcaa7a8/examples/02-workflows.md).

We start with defining a few `dataclasses` to represent the geometry, mesh, and material properties. We then define functions to create the geometry, mesh it, assign materials, and calculate the electrostatics.

```{code-cell} ipython3
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float


@dataclass(frozen=True)
class Mesh:
    geometry: Geometry
    mesh_size: float


@dataclass(frozen=True)
class Materials:
    geometry: Geometry
    materials: list[str]


@dataclass(frozen=True)
class Electrostatics:
    mesh: Mesh
    materials: Materials
    voltages: list[float]
```

```{code-cell} ipython3
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs


@pipefunc(output_name="geo")
def make_geometry(x: float, y: float) -> Geometry:
    return Geometry(x, y)


@pipefunc(output_name=("mesh", "coarse_mesh"))
def make_mesh(
    geo: Geometry,
    mesh_size: float,
    coarse_mesh_size: float,
) -> tuple[Mesh, Mesh]:
    return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)


@pipefunc(output_name="materials")
def make_materials(geo: Geometry) -> Materials:
    return Materials(geo, ["i", "j", "c"])


@pipefunc(output_name="electrostatics", mapspec="V_left[i], V_right[j] -> electrostatics[i, j]")
def run_electrostatics(
    mesh: Mesh,
    materials: Materials,
    V_left: float,  # noqa: N803
    V_right: float,  # noqa: N803
) -> Electrostatics:
    return Electrostatics(mesh, materials, [V_left, V_right])


@pipefunc(output_name="charge", mapspec="electrostatics[i, j] -> charge[i, j]")
def get_charge(electrostatics: Electrostatics) -> float:
    # obviously not actually the charge; but we should return _some_ number that
    # is "derived" from the electrostatics.
    return sum(electrostatics.voltages)


# No mapspec: function receives the full 2D array of charges!
@pipefunc(output_name="average_charge")
def average_charge(charge: np.ndarray) -> float:
    return np.mean(charge)


pipeline_charge = Pipeline(
    [make_geometry, make_mesh, make_materials, run_electrostatics, get_charge, average_charge],
)
pipeline_charge.visualize(orient="TB")
```

Let's run the map for some inputs:

```{code-cell} ipython3
inputs = {
    "V_left": np.linspace(0, 2, 3),
    "V_right": np.linspace(-0.5, 0.5, 2),
    "x": 0.1,
    "y": 0.2,
    "mesh_size": 0.01,
    "coarse_mesh_size": 0.05,
}

run_folder = "my_run_folder"
results = pipeline_charge.map(inputs, run_folder=run_folder, parallel=False)
assert results["average_charge"].output == 1.0
assert results["average_charge"].output_name == "average_charge"
assert load_outputs("average_charge", run_folder=run_folder) == 1.0
```

This example highlighted how to run a simulation that included a map-reduce operation.
Often we want to sweep this over multiple parameters.
You could add all the `mapspec`s required to map an additional parameter.
Alternatively, you can use the `pipeline.add_mapspec_axis` method to add an axis to parameters of the pipeline.

See the example below, where we extend the `mapspec`s.

```{code-cell} ipython3
# Add a cross-product of x and y
pipeline_charge.add_mapspec_axis("x", axis="a")
pipeline_charge.add_mapspec_axis("y", axis="b")

# And also a cross-product of the zipped mesh_size and coarse_mesh_size
pipeline_charge.add_mapspec_axis("mesh_size", axis="c")
pipeline_charge.add_mapspec_axis("coarse_mesh_size", axis="c")

# Finally, the mapspecs become, which shows a 3D array for the `average_charge`:
pipeline_charge.mapspecs_as_strings
```

```{code-cell} ipython3
pipeline_charge.visualize(orient="TB")
```

Let's run it on a 2x2x2 grid of inputs:

```{code-cell} ipython3
inputs = {
    "V_left": np.linspace(0, 2, 3),
    "V_right": np.linspace(-0.5, 0.5, 2),
    "x": np.linspace(0.1, 0.2, 2),
    "y": np.linspace(0.2, 0.3, 2),
    "mesh_size": [0.01, 0.02],
    "coarse_mesh_size": [0.05, 0.06],
}
results = pipeline_charge.map(inputs, run_folder=run_folder, parallel=False)
output = results["average_charge"].output
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")
```

We can also load all data as `xarray.Dataset`:

```{code-cell} ipython3
from pipefunc.map import load_xarray_dataset

ds = load_xarray_dataset(run_folder=run_folder)
ds
```

Or specify the `output_name` to load only specific outputs:

```{code-cell} ipython3
avg_charge = load_xarray_dataset("average_charge", run_folder=run_folder)
avg_charge
```

Now imagine that the electrostatics object is a very large object that we cannot afford to save and load from disk.
For this purpose there is the `pipfunc.NestedPipeFunc` class that allows to combine multiple functions into a single function. We can then tell it to not return the output of the intermediate functions by specifying which outputs to return.

```{code-cell} ipython3
pipeline_charge2 = pipeline_charge.copy()
nested_func = pipeline_charge2.nest_funcs(
    {"electrostatics", "charge"},
    new_output_name="charge",
    # We can also specify `("charge", "electrostatics")` to get both outputs
)
```

This `nested_func` contains an internal pipeline:

```{code-cell} ipython3
nested_func.pipeline.visualize()
```

When visualizing the pipeline, you can see that the `NestedFunc` is shown as a single node.

```{code-cell} ipython3
pipeline_charge2.visualize(orient="TB")
```
