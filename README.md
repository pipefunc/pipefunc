# pipefunc: function composition magic for Python

> Lightweight function pipeline creation: 📚 Less Bookkeeping, 🎯 More Doing

[![Python](https://img.shields.io/pypi/pyversions/pipefunc)](https://pypi.org/project/pipefunc/)
[![PyPi](https://img.shields.io/pypi/v/pipefunc?color=blue)](https://pypi.org/project/pipefunc/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pytest](https://github.com/basnijholt/pipefunc/actions/workflows/pytest.yml/badge.svg)](https://github.com/basnijholt/pipefunc/actions/workflows/pytest.yml)
[![Conda](https://img.shields.io/badge/install%20with-conda-green.svg)](https://anaconda.org/conda-forge/pipefunc)
[![Coverage](https://img.shields.io/codecov/c/github/basnijholt/pipefunc)](https://codecov.io/gh/basnijholt/pipefunc)
[![Documentation](https://readthedocs.org/projects/pipefunc/badge/?version=latest)](https://pipefunc.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://img.shields.io/conda/dn/conda-forge/pipefunc.svg)](https://anaconda.org/conda-forge/pipefunc)
[![GitHub](https://img.shields.io/github/stars/basnijholt/pipefunc.svg?style=social)](https://github.com/basnijholt/pipefunc/stargazers)


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

`pipefunc` is a Python library that simplifies the creation and management of complex function pipelines. It allows you to declare the dependencies between functions and automatically organizes the execution order to satisfy these dependencies.

`pipefunc` provides a range of features to streamline your workflow, including pipeline visualization, flexible function arguments, support for multiple outputs, pipeline simplification, resource usage profiling, parallel execution, caching, and parameter sweep utilities.

Whether you're working with data processing, scientific computations, machine learning (AI) workflows, or any other scenario involving interdependent functions, `pipefunc` helps you focus on the logic of your code while it handles the intricacies of function dependencies and execution order.

## :rocket: Key Features

1. 🚀 **Function Composition and Pipelining**: Create pipelines where the output of one function feeds into another, with `pipefunc` handling the execution order.
2. 📊 **Pipeline Visualization**: Generate visual graphs of your pipelines to better understand the flow of data.
3. 💡 **Flexible Function Arguments**: Call functions with different argument combinations, letting `pipefunc` determine which other functions to call based on the provided arguments.
4. 👥 **Multiple Outputs**: Handle functions that return multiple results, allowing each result to be used as input to other functions.
5. 🔁 **Map-Reduce Support**: Utilize "fan-out" to distribute tasks and "fan-in" to aggregate results, streamlining data processing in distributed environments.
6. ➡️ **Pipeline Simplification**: Merge nodes in complex pipelines to improve computational efficiency, trading off visibility into intermediate steps.
7. 🎛️ **Resource Usage Profiling**: Get reports on CPU usage, memory consumption, and execution time to identify bottlenecks and optimize your code.
8. 🔄 **Parallel Execution and Caching**: Run functions in parallel and cache results to avoid redundant computations.
9. 🔍 **Parameter Sweep Utilities**: Generate parameter combinations for parameter sweeps and optimize the sweeps with result caching.
10. 🛠️ **Flexibility and Ease of Use**: Manage complex function dependencies in a clear, intuitive way with this lightweight yet powerful tool.

## :test_tube: How does it work?

pipefunc provides a Pipeline class that you use to define your function pipeline.
You add functions to the pipeline using the `pipefunc` decorator, which also lets you specify a function's output name and dependencies.
Once your pipeline is defined, you can execute it for specific output values, simplify it by combining functions with the same root arguments, visualize it as a directed graph, and profile the resource usage of the pipeline functions.
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
funcs = [f_c, f_d, f_e]
pipeline = Pipeline(funcs, profile=True)

# You can access and call these functions using the func method
h_d = pipeline.func("d")
assert h_d(a=2, b=3) == 15

h_e = pipeline.func("e")
assert h_e(a=2, b=3, x=1) == 75
assert h_e(c=5, d=15, x=1) == 75

# Visualize the pipeline
pipeline.visualize()

# Get all possible argument mappings for each function
all_args = pipeline.all_arg_combinations
print(all_args)

# Show resource reporting (only works if profile=True)
pipeline.resources_report()
```

This example demonstrates defining a pipeline with `f_c`, `f_d`, `f_e` functions, accessing and executing these functions using the pipeline, visualizing the pipeline graph, getting all possible argument mappings, and reporting on the resource usage.
This basic example should give you an idea of how to use pipefunc to construct and manage function pipelines.

## :notebook: Jupyter Notebook Example

See the detailed usage example and more in our [example.ipynb](https://github.com/basnijholt/pipefunc/blob/main/example.ipynb).

## :computer: Installation

Install the **latest stable** version from conda (recommended):

```bash
conda install pipefunc
```

or from PyPI:

```bash
pip install "pipefunc[plotting]"
```

or install **main** with:

```bash
pip install -U https://github.com/basnijholt/pipefunc/archive/main.zip
```

or clone the repository and do a dev install (recommended for dev):

```bash
git clone git@github.com:basnijholt/pipefunc.git
cd pipefunc
pip install -e ".[dev,test,plotting]"
```

## :hammer_and_wrench: Development

We use [`pre-commit`](https://pre-commit.com/) to manage pre-commit hooks, which helps us ensure that our code is always clean and compliant with our coding standards.
To set it up, install pre-commit with pip and then run the install command:

```bash
pip install pre-commit
pre-commit install
```
