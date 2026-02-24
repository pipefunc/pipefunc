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

# Comparing PipeFunc to other libraries

In the landscape of data pipeline and workflow management tools, there are several popular libraries that serve different needs and use cases.
This comparison aims to highlight the unique features of PipeFunc by contrasting it with other well-known tools in the field.
We'll examine how PipeFunc's approach to pipeline construction, execution, and parameter sweeps compares to other libraries, focusing on ease of use, flexibility, and suitability for scientific computing workflows.

We'll compare PipeFunc with the following libraries:

1. [Pydra](https://github.com/nipype/pydra): A dataflow engine designed for scientific workflows, with a focus on neuroimaging.
2. [Airflow](https://github.com/apache/airflow): A platform for programmatically authoring, scheduling, and monitoring workflows.
3. [Dask](https://github.com/dask/dask): A flexible library for parallel computing in Python.
4. [Kedro](https://github.com/kedro-org/kedro): An open-source framework for creating reproducible, maintainable, and modular data science code.
5. [Luigi](https://github.com/spotify/luigi): A Python package that helps you build complex pipelines of batch jobs.

Each comparison will include a brief overview of the library, an example implementation of a simple pipeline, and a discussion of key differences from PipeFunc.
This will help illustrate where PipeFunc fits in the ecosystem and what unique advantages it offers for certain types of workflows, particularly in scientific computing and rapid prototyping scenarios.

Let's start with a basic PipeFunc example that we'll use as a reference point for our comparisons:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

@pipefunc(output_name="c")
def f_c(a: int, b: int) -> int:
    return a + b

@pipefunc(output_name="d")
def f_d(b: int, c: int, x: int = 1) -> int:  # "c" is the output of f_c
    return b * c

@pipefunc(output_name="e")
def f_e(c: int, d: int, x: int = 1) -> int:  # "d" is the output ˝of f_d
    return c * d * x

pipeline = Pipeline([f_c, f_d, f_e])
```

We can call the pipeline for a single set of inputs.

```{code-cell} ipython3
pipeline("e", a=1, b=2)
```

Alternatively, we can call the pipeline for different values of `a` and `b` and store the results in 2D array.

We can do this by adding mapspec axes to the pipeline:

```{code-cell} ipython3
pipeline.add_mapspec_axis("a", axis="i")
pipeline.add_mapspec_axis("b", axis="j")
```

Now we can call the pipeline with inputs for `a` and `b` using the `pipeline.map` method, which automatically runs in parallel with a `~concurrent.futures.ProcessPoolExecutor`.

```{code-cell} ipython3
r = pipeline.map(inputs={"a": [1, 2, 3], "b": [4, 5, 6, 7]}, run_folder="my_run_folder")
```

We can look at any of the intermediate and final results:

```{code-cell} ipython3
r["d"].output
```

```{code-cell} ipython3
r["e"].output
```

Or we load the data as an xarray dataset and plot it:

```{code-cell} ipython3
from pipefunc.map import load_xarray_dataset

ds = load_xarray_dataset(run_folder="my_run_folder")
ds
```

```{code-cell} ipython3
ds.e.astype(float).plot(x="i", y="j")
```

Now, let's see how this same pipeline might be implemented in other libraries and discuss the differences.

## Pydra

Of all the libraries we're comparing, Pydra is the most similar to PipeFunc in terms of its focus on scientific workflows and data processing.
Pydra is a dataflow engine designed for scientific workflows, with a particular focus on neuroimaging.
It's part of the Nipype ecosystem and emphasizes flexibility, scalability, and reproducibility in complex scientific computations.

Here's how you might implement our example pipeline using Pydra:

```{code-cell} ipython3
import nest_asyncio

nest_asyncio.apply()

import pydra


@pydra.mark.task
def f_c(a: int, b: int) -> int:
    return a + b

@pydra.mark.task
def f_d(b: int, c: int, x: int = 1) -> int:
    return b * c

@pydra.mark.task
def f_e(c: int, d: int, x: int = 1) -> int:
    return c * d * x

# Create a workflow
wf = pydra.Workflow(name="example_workflow", input_spec=["a", "b", "x"])
wf.inputs.a = 1
wf.inputs.b = 2
wf.inputs.x = 1

# Add tasks to the workflow
wf.add(f_c(name="task_c", a=wf.lzin.a, b=wf.lzin.b))
wf.add(f_d(name="task_d", b=wf.lzin.b, c=wf.task_c.lzout.out, x=wf.lzin.x))
wf.add(f_e(name="task_e", c=wf.task_c.lzout.out, d=wf.task_d.lzout.out, x=wf.lzin.x))

# Set the workflow output
wf.set_output([("final_output", wf.task_e.lzout.out)])

# Run the workflow
with pydra.Submitter(plugin="cf") as sub:
    sub(wf)

result = wf.result()
print(f"Result: {result.output.final_output}")

# Parameter sweep
wf.split(["a", "b"], a=[1, 2, 3], b=[4, 5, 6])

with pydra.Submitter(plugin="cf") as sub:
    sub(wf)

results = wf.result()
for res in results:
    print(f"Result: {res.output.final_output}")
```

Key differences from PipeFunc:

1. **Task Definition**: Pydra uses the `@mark.task` decorator to define tasks, similar to PipeFunc's `@pipefunc` decorator, but with a focus on type annotations.

2. **Workflow Construction**: Pydra requires explicit workflow construction using the `Workflow` class, whereas PipeFunc infers the workflow structure from function dependencies.

3. **Data Flow**: Pydra uses a system of lazy inputs (`lzin`) and outputs (`lzout`) to define data flow between tasks. PipeFunc's approach is more implicit, based on function arguments and return values.

4. **Parallel Execution**: Pydra's `wf.split` feature provides a way to run parameter sweeps in parallel, similar to PipeFunc's `map` method, but with a different syntax.

5. **Type Annotations**: Both Pydra and PipeFunc support and utilize Python type annotations, allowing to catch type errors early in the development process.

6. **Flexibility**: Pydra is designed to work with Python functions, shell commands, and containers, making it highly flexible for various types of tasks. PipeFunc is primarily focused on Python functions but offers great flexibility within this domain.

7. **Resource Specification**: Both Pydra and PipeFunc allow for specifying computational resources for tasks. PipeFunc's approach is particularly flexible, allowing for dynamic resource allocation based on input parameters.

8. **Audit Trail**: Pydra provides built-in support for generating audit trails, enhancing reproducibility. While this is not a native feature of PipeFunc, its integration with tools like MLflow can provide similar capabilities.

9. **Pipeline Composition**: PipeFunc offers a simple and intuitive way to combine pipelines using the `|` operator, which is not a feature in Pydra.

Pydra excels in scenarios requiring complex scientific workflows, especially those involving neuroimaging tasks.
Its emphasis on type checking and audit trails makes it well-suited for environments where reproducibility and error prevention are critical.

PipeFunc, in comparison, offers a more lightweight and intuitive approach to pipeline construction.
Its simplicity in defining and combining pipelines, along with its straightforward parameter sweep capabilities, makes it particularly suitable for rapid prototyping and iterative development in scientific computing workflows.

Both tools prioritize flexibility and scalability, but PipeFunc's design leans more towards ease of use and quick setup, while Pydra provides a more comprehensive framework for managing complex, long-running scientific workflows.

## Airflow

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows.
While it can be used to create data pipelines, its primary purpose differs from PipeFunc.
Airflow is designed for orchestrating complex workflows, often involving multiple systems, long-running tasks, and scheduled executions.
It's particularly well-suited for ETL processes, machine learning pipelines, and other data-intensive operations that require scheduling and monitoring.

Here's how you might implement a similar pipeline using Airflow:

```{code-cell} ipython3
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Define the functions
def f_c(a, b):
    return a + b

def f_d(b, c, x=1):
    return b * c

def f_e(c, d, x=1):
    return c * d * x

# Define default arguments
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

# Create the DAG
dag = DAG(
    "simple_example",
    default_args=default_args,
    description="A simple DAG example",
    schedule_interval=None,
)

# Define the tasks
task_c = PythonOperator(
    task_id="f_c",
    python_callable=f_c,
    op_kwargs={"a": 1, "b": 2},
    dag=dag,
)

task_d = PythonOperator(
    task_id="f_d",
    python_callable=f_d,
    op_kwargs={"b": 2, "c": task_c.output},
    dag=dag,
)

task_e = PythonOperator(
    task_id="f_e",
    python_callable=f_e,
    op_kwargs={"c": task_c.output, "d": task_d.output},
    dag=dag,
)

# Set task dependencies
task_c >> task_d >> task_e

# For testing: this will run the DAG
if __name__ == "__main__":
    dag.test()
```

There is no easy way to do a parameter sweep in Airflow, but you could create a loop to run the pipeline for different values of `a` and `b`.

Key differences from PipeFunc:

1. **Purpose**: Airflow is designed for orchestrating complex, scheduled workflows, while PipeFunc focuses on creating reusable, composable data pipelines.

2. **Complexity**: Airflow requires more boilerplate code and setup, reflecting its broader scope and capabilities.

3. **Execution Model**: Airflow typically runs tasks as separate processes, which can be distributed across multiple machines. PipeFunc runs functions within a single Python process by default or using `ProcessPoolExecutor`.

4. **Scheduling**: Airflow has built-in scheduling capabilities, allowing DAGs to run on specified intervals or triggers.

5. **Monitoring and UI**: Airflow provides a web interface for monitoring and managing workflows, which is not a feature of PipeFunc.

6. **Data Passing**: In Airflow, data is typically passed between tasks using XComs or external storage. PipeFunc allows direct passing of data between functions.

7. **Parallel Execution**: While Airflow can run tasks in parallel, implementing something like PipeFunc's `map` functionality for parameter sweeps would require additional setup and possibly custom operators.

This Airflow example demonstrates a simple linear pipeline similar to the PipeFunc example. However, it doesn't showcase Airflow's full capabilities for complex workflow management, nor does it replicate PipeFunc's ease of use for quick data pipeline setups and parameter sweeps.

For simple data pipelines or rapid prototyping, PipeFunc offers a more straightforward and lightweight approach. For complex, scheduled workflows involving multiple systems and long-running tasks, Airflow provides a more comprehensive solution.

## Dask

Dask is a flexible library for parallel computing in Python, extending Python libraries like NumPy and Pandas to larger-than-memory datasets and distributed computing.
While powerful for parallelization, it differs from PipeFunc in pipeline management.

Here's a Dask implementation of our pipeline:

```{code-cell} ipython3
import dask
from dask import delayed
from itertools import product

@delayed
def f_c(a, b):
    return a + b

@delayed
def f_d(b, c, x=1):
    return b * c

@delayed
def f_e(c, d, x=1):
    return c * d * x

def pipeline_e(a, b, x=1):
    c = f_c(a, b)
    d = f_d(b, c, x)
    return f_e(c, d, x)

def pipeline_d(a, b, x=1):
    c = f_c(a, b)
    return f_d(b, c, x)

# Single input
result_e = pipeline_e(1, 2).compute()
result_d = pipeline_d(1, 2).compute()

# Multiple inputs
a_values = [1, 2, 3]
b_values = [4, 5, 6, 7]
results_e = [pipeline_e(a, b) for a, b in product(a_values, b_values)]
computed_e = dask.compute(*results_e)
computed_e
```

Dask is much lower-level than PipeFunc, requiring explicit definitions of a pipeline.
PipeFunc can actually use dask as a computational backend (via `adaptive-scheduler`).

Key differences from PipeFunc:

1. **Task creation**: The sweep we do is for just a 3x4 grid, but for larger grids, imagine a 100x100x100 grid, the task creation can become a bottleneck because we create a task for each point in the grid. With pipefunc, the approach is different, here a single numpy array with shape (100, 100, 100) is created and each "task" is simply an index into this array, so creating millions of tasks is very fast.

2. **Pipeline Definition**: Dask requires defining separate functions for different pipeline endpoints. For large pipelines, this means writing a new function for each intermediate result, potentially leading to code duplication and maintenance challenges.

3. **Resource Management**: Unlike PipeFunc, Dask doesn't provide built-in ways to specify different resource requirements (e.g., CPU, memory) for individual pipeline steps. PipeFunc's ability to define resources per function allows for more granular control over execution.

4. **Adaptive Scheduling**: PipeFunc's integration with adaptive scheduling libraries allows for dynamic resource allocation and job splitting, which is not a native feature in Dask.

5. **Intermediate Results**: Accessing intermediate results in Dask often requires redefining the pipeline, whereas PipeFunc allows easy access to any step's output.

6. **Parallelization Model**: Dask focuses on data parallelism and task scheduling, while PipeFunc offers a more structured approach to defining and executing pipelines with varying resource needs.

While Dask excels in handling large datasets and providing flexible parallelization, PipeFunc offers more intuitive pipeline construction, easier access to intermediate results, and fine-grained control over resource allocation for each step in the pipeline.
This makes PipeFunc particularly suitable for complex workflows where different stages have varying computational requirements.

## Kedro

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code.
It provides a standard structure for data pipeline projects, emphasizing best practices for software engineering.

Here's how you might implement our example pipeline using Kedro:

```{code-cell} ipython3
from kedro.pipeline import Pipeline, node

# Define the functions
def f_c(a, b):
    return a + b

def f_d(b, c, x=1):
    return b * c

def f_e(c, d, x=1):
    return c * d * x

# Define the nodes
node_c = node(func=f_c, inputs=["a", "b"], outputs="c")
node_d = node(func=f_d, inputs=["b", "c", "x"], outputs="d")
node_e = node(func=f_e, inputs=["c", "d", "x"], outputs="e")

# Create the pipeline
pipeline = Pipeline([node_c, node_d, node_e])

# To run the pipeline, you would typically use Kedro's run command or a DataCatalog
# This is a simplified example of how you might run it
from kedro.io import DataCatalog, MemoryDataset

data_catalog = DataCatalog(
    {
        "a": MemoryDataset(1),
        "b": MemoryDataset(2),
        "x": MemoryDataset(1),
    }
)

from kedro.runner import SequentialRunner

runner = SequentialRunner()
results = runner.run(pipeline, data_catalog)
```

There is no direct equivalent to PipeFunc's `map` method in Kedro, but you can achieve similar functionality by running a loop over different inputs, which is not very efficient when doing millions of tasks.

Key differences from PipeFunc:

1. **Project Structure**: Kedro enforces standardized structure, beneficial for large projects but potentially overkill for simpler pipelines.

2. **Data Management**: Kedro uses DataCatalog for I/O management, more structured than PipeFunc's direct function calls using standard Python data types. This is advantageous for data science projects but overly verbose for scientific computing.

3. **Pipeline Definition**: Kedro requires explicit node definitions, more verbose than PipeFunc's decorator approach.

4. **Execution**: Both offer sequential and parallel execution, but Kedro lacks built-in support for easy parameter sweeps like PipeFunc's `map` method. Kedro also doesn't support SLURM.

5. **Modularity**: Both encourage modular components, but with different implementations.

6. **Resource Specification**: PipeFunc allows per-function resource specification, not natively supported in Kedro.

7. **Pipeline Composition**: PipeFunc offers simpler pipeline combination (e.g., `pipeline1 | pipeline2`), enhancing reusability and quick assembly of complex workflows.

Kedro excels in structured, large-scale project management with emphasis on reproducibility and best practices.
PipeFunc offers a lightweight, flexible approach with easier prototyping, straightforward parameter sweeps, and simple pipeline composition.
It's particularly strong in managing varying computational requirements and quickly assembling complex workflows.

Both have their strengths: Kedro for comprehensive project management, PipeFunc for flexibility and rapid development.

## Luigi

Luigi is a Python package that helps you build complex pipelines of batch jobs.
It handles dependency resolution, workflow management, visualization, and it provides failure recovery.
Developed by Spotify, Luigi is designed to handle long-running batch processes.

Here's how you might implement our example pipeline using Luigi:

```{code-cell} ipython3
import luigi

class TaskC(luigi.Task):
    a = luigi.IntParameter()
    b = luigi.IntParameter()

    def run(self):
        result = self.a + self.b
        with self.output().open("w") as f:
            f.write(str(result))

    def output(self):
        return luigi.LocalTarget(f"c_{self.a}_{self.b}.txt")

class TaskD(luigi.Task):
    a = luigi.IntParameter()
    b = luigi.IntParameter()
    x = luigi.IntParameter(default=1)

    def requires(self):
        return TaskC(a=self.a, b=self.b)

    def run(self):
        with self.input().open("r") as f:
            c = int(f.read())
        result = self.b * c * self.x
        with self.output().open("w") as f:
            f.write(str(result))

    def output(self):
        return luigi.LocalTarget(f"d_{self.a}_{self.b}_{self.x}.txt")

class TaskE(luigi.Task):
    a = luigi.IntParameter()
    b = luigi.IntParameter()
    x = luigi.IntParameter(default=1)

    def requires(self):
        return {"c": TaskC(a=self.a, b=self.b), "d": TaskD(a=self.a, b=self.b, x=self.x)}

    def run(self):
        with self.input()["c"].open("r") as f:
            c = int(f.read())
        with self.input()["d"].open("r") as f:
            d = int(f.read())
        result = c * d * self.x
        with self.output().open("w") as f:
            f.write(str(result))

    def output(self):
        return luigi.LocalTarget(f"e_{self.a}_{self.b}_{self.x}.txt")

# Run the pipeline
luigi.build([TaskE(a=1, b=2, x=1)], local_scheduler=True)

# For parameter sweep
sweep_tasks = [TaskE(a=a, b=b) for a in [1, 2] for b in [4, 5]]
luigi.build(sweep_tasks, local_scheduler=True)
```

This Luigi example demonstrates a simple linear pipeline similar to the PipeFunc example.
Luigi requires explicit task definitions and dependency management, which can be more verbose than PipeFunc's function-based approach.

Key differences from PipeFunc:

1. **Task Definition**: Luigi requires each step to be defined as a separate `Task` class, which is more verbose than PipeFunc's function-based approach.

2. **Dependency Management**: Dependencies in Luigi are explicitly defined through the `requires` method, while PipeFunc infers dependencies from function inputs and outputs.

3. **Parameter Handling**: Luigi uses `Parameter` classes for input parameters, which provides type checking but adds verbosity compared to PipeFunc's simple function arguments using standard Python type annotations.

4. **Data Passing**: Luigi typically passes data between tasks using files or other persistent storage, while PipeFunc can pass data directly between functions in memory.

5. **Execution Model**: Luigi has a central scheduler that manages task execution, while PipeFunc's execution is more straightforward and can be easily parallelized.

6. **Visualization**: Luigi provides a web interface for visualizing task execution, which PipeFunc doesn't natively offer.

7. **Parameter Sweeps**: While parameter sweeps can be done in Luigi by creating multiple task instances, it's not as streamlined as PipeFunc's `map` method.

8. **Resource Management**: Luigi has some support for resource management, but it's not as fine-grained as PipeFunc's per-function resource specification.

9. **Pipeline Composition**: Luigi doesn't have a simple operator for combining pipelines like PipeFunc's `|` operator.

Luigi excels in managing complex dependencies in large batch processing workflows, especially when intermediate results need to be persisted.
It also provides robust failure recovery mechanisms.

PipeFunc, on the other hand, offers a more lightweight and flexible approach, with easier setup for quick prototyping and straightforward parallel execution for parameter sweeps.
PipeFunc's ability to pass data directly between functions and its simple pipeline composition make it particularly suitable for scientific computing workflows where in-memory data passing and quick pipeline modifications are common.
