# pipefunc: function composition magic for Python

> Lightweight function pipeline creation: 📚 Less Bookkeeping, 🎯 More Doing

[![Python](https://img.shields.io/pypi/pyversions/pipefunc)](https://pypi.org/project/pipefunc/)
[![PyPi](https://img.shields.io/pypi/v/pipefunc?color=blue)](https://pypi.org/project/pipefunc/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pytest](https://github.com/pipefunc/pipefunc/actions/workflows/pytest.yml/badge.svg)](https://github.com/pipefunc/pipefunc/actions/workflows/pytest.yml)
[![Conda](https://img.shields.io/badge/install%20with-conda-green.svg)](https://anaconda.org/conda-forge/pipefunc)
[![Coverage](https://img.shields.io/codecov/c/github/pipefunc/pipefunc)](https://codecov.io/gh/pipefunc/pipefunc)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/pipefunc/pipefunc)
[![Documentation](https://readthedocs.org/projects/pipefunc/badge/?version=latest)](https://pipefunc.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/pipefunc.svg)](https://anaconda.org/conda-forge/pipefunc)
[![GitHub](https://img.shields.io/github/stars/pipefunc/pipefunc.svg?style=social)](https://github.com/pipefunc/pipefunc/stargazers)


![](https://user-images.githubusercontent.com/6897215/253785642-cf2a6941-2ea6-41b0-8225-b3e52e94c4de.png)

<!-- toc-start -->
## :books: Table of Contents
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [:thinking: What is this?](#thinking-what-is-this)
- [:rocket: Key Features](#rocket-key-features)
- [:test_tube: How does it work?](#test_tube-how-does-it-work)
- [:notebook: Jupyter Notebook Example](#notebook-jupyter-notebook-example)
- [:computer: Installation](#computer-installation)
- [:hammer_and_wrench: Development](#hammer_and_wrench-development)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- toc-end -->

## :thinking: What is this?

`pipefunc` is a Python library for creating and running function pipelines. By annotating functions and specifying their outputs, it forms a pipeline that automatically organizes the execution order to satisfy dependencies. Just specify the names of the outputs you want to compute, and `pipefunc` will handle the rest by leveraging the parameter names of the annotated functions.

Whether you're working with data processing, scientific computations, machine learning (AI) workflows, or any other scenario involving interdependent functions, `pipefunc` helps you focus on the logic of your code while it handles the intricacies of function dependencies and execution order.

## :rocket: Key Features

1. 🚀 **Function Composition and Pipelining**: Create pipelines by using the `@pipefunc` decorator; execution order is automatically handled.
2. 📊 **Pipeline Visualization**: Generate visual graphs of your pipelines to better understand the flow of data.
3. 👥 **Multiple Outputs**: Handle functions that return multiple results, allowing each result to be used as input to other functions.
4. 🔁 **Map-Reduce Support**: Perform "map" operations to apply functions over data and "reduce" operations to aggregate results, allowing n-dimensional mappings.
5. ➡️ **Pipeline Simplification**: Merge nodes in complex pipelines to run multiple functions in a single step.
6. 🎛️ **Resource Usage Profiling**: Get reports on CPU usage, memory consumption, and execution time to identify bottlenecks and optimize your code.
7. 🔄 **Automatic parallelization**: Automatically runs pipelines in parallel (local or remote) with shared memory and disk caching options.
8. 🔍 **Parameter Sweep Utilities**: Generate parameter combinations for parameter sweeps and optimize the sweeps with result caching.
9. 💡 **Flexible Function Arguments**: Call functions with different argument combinations, letting `pipefunc` determine which other functions to call based on the provided arguments.
10. 🏗️ **Leverages giants**: Builds on top of [NetworkX](https://networkx.org/) for graph algorithms, [NumPy](https://numpy.org/) for multi-dimensional arrays, and optionally [Xarray](https://docs.xarray.dev/) for labeled multi-dimensional arrays, [Zarr](https://zarr.readthedocs.io/) to store results in memory/disk/cloud or any key-value store, and [Adaptive](https://adaptive.readthedocs.io/) for parallel sweeps.
11. 🤓 **Nerd stats**: >500 tests with 100% test coverage, fully typed, only 4 required dependencies, *all* Ruff Rules, *all* public API documented.

## :test_tube: How does it work?

pipefunc provides a Pipeline class that you use to define your function pipeline.
You add functions to the pipeline using the `pipefunc` decorator, which also lets you specify the function's output name.
Once your pipeline is defined, you can execute it for specific output values, simplify it by combining function nodes, visualize it as a directed graph, and profile the resource usage of the pipeline functions.
For more detailed usage instructions and examples, please check the usage example provided in the package.

Here is a simple example usage of pipefunc to illustrate its primary features:

```python
from pipefunc import pipefunc, Pipeline

# Define three functions that will be a part of the pipeline
@pipefunc(output_name="c")
def f_c(a, b):
    return a + b

@pipefunc(output_name="d")
def f_d(b, c):
    return b * c

@pipefunc(output_name="e")
def f_e(c, d, x=1):
    return c * d * x

# Create a pipeline with these functions
pipeline = Pipeline([f_c, f_d, f_e], profile=True)  # `profile=True` enables resource profiling

# Call the pipeline directly for different outputs:
assert pipeline("d", a=2, b=3) == 15
assert pipeline("e", a=2, b=3, x=1) == 75

# Or create a new function for a specific output
h_d = pipeline.func("d")
assert h_d(a=2, b=3) == 15

h_e = pipeline.func("e")
assert h_e(a=2, b=3, x=1) == 75
# Instead of providing the root arguments, you can also provide the intermediate results directly
assert h_e(c=5, d=15, x=1) == 75

# Visualize the pipeline
pipeline.visualize()

# Get all possible argument mappings for each function
all_args = pipeline.all_arg_combinations
print(all_args)

# Show resource reporting (only works if profile=True)
pipeline.print_profiling_stats()
```

This example demonstrates defining a pipeline with `f_c`, `f_d`, `f_e` functions, accessing and executing these functions using the pipeline, visualizing the pipeline graph, getting all possible argument mappings, and reporting on the resource usage.
This basic example should give you an idea of how to use `pipefunc` to construct and manage function pipelines.

The following example demonstrates how to perform a map-reduce operation using `pipefunc`:

```python
from pipefunc import pipefunc, Pipeline
from pipefunc.map import load_outputs
import numpy as np

@pipefunc(output_name="c", mapspec="a[i], b[j] -> c[i, j]")  # the mapspec is used to specify the mapping
def f(a: int, b: int):
    return a + b

@pipefunc(output_name="mean")  # there is no mapspec, so this function takes the full 2D array
def g(c: np.ndarray):
    return np.mean(c)

pipeline = Pipeline([f, g])
inputs = {"a": [1, 2, 3], "b": [4, 5, 6]}
pipeline.map(inputs, run_folder="my_run_folder", parallel=True)
result = load_outputs("mean", run_folder="my_run_folder")
print(result)  # prints 7.0
```

Here the `mapspec` argument is used to specify the mapping between the inputs and outputs of the `f` function, it creates the product of the `a` and `b` input lists and computes the sum of each pair. The `g` function then computes the mean of the resulting 2D array. The `map` method executes the pipeline for the `inputs`, and the `load_outputs` function is used to load the results of the `g` function from the specified run folder.

## :notebook: Jupyter Notebook Example

See the detailed usage example and more in our [example.ipynb](https://github.com/pipefunc/pipefunc/blob/main/example.ipynb).

## :computer: Installation

Install the **latest stable** version from conda (recommended):

```bash
conda install pipefunc
```

or from PyPI:

```bash
pip install "pipefunc[all]"
```

or install **main** with:

```bash
pip install -U https://github.com/pipefunc/pipefunc/archive/main.zip
```

or clone the repository and do a dev install (recommended for dev):

```bash
git clone git@github.com:pipefunc/pipefunc.git
cd pipefunc
pip install -e ".[dev]"
```

## :hammer_and_wrench: Development

We use [`pre-commit`](https://pre-commit.com/) to manage pre-commit hooks, which helps us ensure that our code is always clean and compliant with our coding standards.
To set it up, install pre-commit with pip and then run the install command:

```bash
pip install pre-commit
pre-commit install
```
