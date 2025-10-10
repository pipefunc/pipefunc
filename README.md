# PipeFunc: Structure, Automate, and Simplify Your Computational Workflows 🕸

> **_Stop_** micromanaging execution. Focus on the **science**. Capture your workflow's essence with **function pipelines**, represent **computations as DAGs**, and **automate parallel sweeps**.

[![Python](https://img.shields.io/pypi/pyversions/pipefunc)](https://pypi.org/project/pipefunc/)
[![PyPi](https://img.shields.io/pypi/v/pipefunc?color=blue)](https://pypi.org/project/pipefunc/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytest](https://github.com/pipefunc/pipefunc/actions/workflows/pytest-micromamba.yml/badge.svg)](https://github.com/pipefunc/pipefunc/actions/workflows/pytest-micromamba.yml)
[![Conda](https://img.shields.io/badge/install%20with-conda-green.svg)](https://anaconda.org/conda-forge/pipefunc)
[![Coverage](https://img.shields.io/codecov/c/github/pipefunc/pipefunc)](https://codecov.io/gh/pipefunc/pipefunc)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/pipefunc/pipefunc)
[![Documentation](https://readthedocs.org/projects/pipefunc/badge/?version=latest)](https://pipefunc.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/pipefunc.svg)](https://anaconda.org/conda-forge/pipefunc)
[![GitHub](https://img.shields.io/github/stars/pipefunc/pipefunc.svg?style=social)](https://github.com/pipefunc/pipefunc/stargazers)
[![Discord](https://img.shields.io/discord/1320459922596565103.svg?label=Discord&logo=discord)](https://discord.gg/wUXg2drsNN)

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

[![asciicast](https://asciinema.org/a/q5S3ffIxrAGmoLMOc0hOb3aod.svg)](https://asciinema.org/a/q5S3ffIxrAGmoLMOc0hOb3aod)

**`pipefunc`** is a Python library designed for creating and executing **function pipelines**.
By simply annotating functions and specifying their outputs, it builds a pipeline that **automatically manages the execution order** based on dependencies.
Visualize the pipeline as a directed graph, execute the pipeline for all (or specific) outputs, add multidimensional sweeps, automatically parallelize the pipeline, and get nicely structured data back.

> [!NOTE]
> A _*pipeline*_ is a sequence of interconnected functions, structured as a [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG), where outputs from one or more functions serve as inputs to subsequent ones.
> pipefunc streamlines the creation and management of these pipelines, offering powerful tools to efficiently execute them.

Whether you're working with data processing, scientific computations, machine learning (AI) workflows, or any other scenario involving interdependent functions, `pipefunc` helps you focus on the logic of your code while it handles the intricacies of function dependencies and execution order.

## :rocket: Key Features

1. 🚀 **Function Composition and Pipelining**: Create pipelines by using the `@pipefunc` decorator; execution order is automatically handled.
1. 📊 **Pipeline Visualization**: Generate visual graphs of your pipelines to better understand the flow of data.
1. 👥 **Multiple Outputs**: Handle functions that return multiple results, allowing each result to be used as input to other functions.
1. 🔁 **Map-Reduce Support**: Perform "map" operations to apply functions over data and "reduce" operations to aggregate results, allowing n-dimensional mappings.
1. 👮 **Type Annotations Validation**: Validates the type annotations between functions to ensure type consistency.
1. 🎛️ **Resource Usage Profiling**: Get reports on CPU usage, memory consumption, and execution time to identify bottlenecks and optimize your code.
1. 🔄 **Automatic parallelization**: Automatically runs pipelines in parallel (local or remote) with shared memory and disk caching options.
1. ⚡ **Ultra-Fast Performance**: Minimal overhead of [about 15 µs](https://pipefunc.readthedocs.io/en/latest/faq/#what-is-the-overhead-efficiency-performance-of-pipefunc) per function in the graph, ensuring blazingly fast execution.
1. 🔍 **Parameter Sweep Utilities**: Generate parameter combinations for parameter sweeps and optimize the sweeps with result caching.
1. 💡 **Flexible Function Arguments**: Call functions with different argument combinations, letting `pipefunc` determine which other functions to call based on the provided arguments.
1. 🏗️ **Leverages giants**: Builds on top of [NetworkX](https://networkx.org/) for graph algorithms, [NumPy](https://numpy.org/) for multi-dimensional arrays, and optionally [Xarray](https://docs.xarray.dev/) for labeled multi-dimensional arrays, [Zarr](https://zarr.readthedocs.io/) to store results in memory/disk/cloud or any key-value store, and [Adaptive](https://adaptive.readthedocs.io/) for parallel sweeps.
1. 🤓 **Nerd stats**: >1000 tests with 100% test coverage, fully typed, only 3 required dependencies, _all_ Ruff Rules, _all_ public API documented.

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
assert pipeline("e", a=2, b=3) == 75

# Visualize the pipeline
pipeline.visualize()

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

> [!TIP]
> Have [`uv` installed](https://docs.astral.sh/uv/)?
> Run `uvx --with "pipefunc[docs]" -p 3.13 opennb pipefunc/pipefunc/example.ipynb` to open the example notebook in your browser without the need to setup anything!

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
