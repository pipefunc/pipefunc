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

`pipefunc` is a Python library designed to create and manage complex networks of interdependent functions, often known as function pipelines.

In a function pipeline, each function can have dependencies on the results of other functions. Managing these dependencies, ensuring each function has the inputs it needs, and determining the order of execution can become an annoying bookkeeping task in complex cases.

`pipefunc` simplifies this process by allowing you to declare the dependencies of each function and automatically organizing the execution order to satisfy these dependencies. Additionally, the library provides features for visualizing the function pipeline, simplifying the pipeline graph, caching function results for efficiency, and profiling resource usage for optimization.

For example, imagine you have a set of functions where `function B` needs the output from `function A`, and `function C` needs the outputs from both `function A` and `function B`. `pipefunc` allows you to specify these dependencies when you create the functions and then automatically manages their execution. It also provides tools for visualizing this function network, simplifying it if possible, and understanding the resource usage of each function.

The library is designed to be an efficient and flexible tool for managing complex function dependencies in an intuitive and clear way. Whether you're dealing with data processing tasks, scientific computations, machine learning (AI) workflows, or other scenarios where functions depend on one another, `pipefunc` can help streamline your code and improve your productivity.

## :rocket: Key Features

Some of the key features of `pipefunc` include:

1. 🚀 **Function Composition and Pipelining:** The core functionality of `pipefunc` is to create a pipeline of functions, allowing you to feed the output of one function into another, and execute them in the right order.
1. 📊 **Visualizing Pipelines:** `pipefunc` can generate a visual graph of the function pipeline, making it easier to understand the flow of data.
1. 💡 **Flexible Function Arguments:** `pipefunc` lets you call a function with different combinations of arguments, automatically determining which other functions to call based on the arguments you provide.
1. 👥 **Multiple Outputs:** `pipefunc` supports functions that return multiple results, allowing each result to be used as input to other functions.
1. ➡️ **Reducing Pipelines:** `pipefunc` can simplify a complex pipeline by merging nodes, improving computational efficiency at the cost of losing visibility into some intermediate steps.
1. 🎛️ **Resources Report:** `pipefunc` provides a report on the performance of your pipeline, including CPU usage, memory usage, and execution time, helping you identify bottlenecks and optimize your code.
1. 🔄 **Parallel Execution and Caching:** `pipefunc` supports parallel execution of functions, and caching of results to avoid redundant computation.
1. 🔍 **Parameter Sweeps:** `pipefunc` provides a utility for generating combinations of parameters to use in a parameter sweep, along with the ability to cache results to optimize the sweep.
1. 🛠️ **Flexibility and Ease of Use:** `pipefunc` is a lightweight, flexible, and powerful tool for managing complex function dependencies in a clear and intuitive way, designed to improve your productivity in any scenario where functions depend on one another.


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
all_args = pipeline.all_arg_combinations()
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
conda install "pipefunc[plotting]"
```

or from PyPI:

```bash
pip install pipefunc
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
