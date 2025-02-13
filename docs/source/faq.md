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

# ❓ FAQ: Frequently Asked Questions

```{contents} ToC – Questions
:depth: 3
```

Missing something or is something unclear? Please [open an issue](https://github.com/pipefunc/pipefunc/issues/new)!

## How is this different from Dask, AiiDA, Luigi, Prefect, Kedro, Apache Airflow, Snakemake, etc.?

`pipefunc` fills a unique niche in the Python workflow ecosystem.

### Key Differentiators

What makes `pipefunc` unique:

1. **Simplicity**: Pure Python implementation with minimal dependencies, allowing standard debuggers and profiling tools to work without modification
2. **Flexibility**: Easy to modify pipelines and add parameter sweeps with minimal boilerplate
3. **HPC Integration**: First-class support for traditional HPC clusters
4. **Resource Management**: Fine-grained control over computational resources per function
5. **Development Speed**: Rapid prototyping without infrastructure setup

`pipefunc` is particularly well-suited for scientists and researchers who need to:

- Quickly prototype and iterate on computational workflows
- Run parameter sweeps across multiple dimensions
- Manage varying computational requirements between pipeline steps
- Work with traditional HPC systems
- Maintain readable and maintainable Python code

Let's break down the comparison by categories:

### Low-Level Parallel Computing Tools (e.g., [Dask](https://www.dask.org/))

Dask and `pipefunc` serve different purposes and can be complementary:

- Dask provides low-level control over parallelization, letting you decide exactly what and how to parallelize
- `pipefunc` automatically handles parallelization based on pipeline structure and `mapspec` definitions
- Dask can serve as a computational backend for `pipefunc`
- `pipefunc` provides higher-level abstractions for parameter sweeps without requiring explicit parallel programming

In summary, Dask is a powerful parallel computing library, while pipefunc helps you build and manage scientific workflows with less boilerplate and takes care of parallelization and data saving for you.

### Scientific Workflow Tools (e.g., [AiiDA](https://aiida.readthedocs.io/), [Pydra](https://pydra.readthedocs.io/en/latest/))

Compared to scientific workflow managers, `pipefunc` provides:

- Lighter weight setup with no external dependencies (unlike AiiDA, which requires a daemon, PostgreSQL, and RabbitMQ).
- More intuitive Python-native interface with automatic graph construction from function signatures.
- Simpler debugging as code runs in the same Python process by default.
- Built-in parameter sweeps with automatic parallelization.
- Dynamic resource allocation based on input parameters.

### Job Schedulers/Runners (e.g., [Airflow](https://airflow.apache.org/), [Luigi](https://luigi.readthedocs.io/))

These tools are designed for scheduling and running tasks, often in a distributed environment. They are well-suited for production ETL pipelines and managing dependencies between jobs. Unlike `pipefunc`, they often rely on serialized data or external storage for data exchange between tasks and require custom implementations for parameter sweeps.

**`pipefunc` vs. Job Schedulers:**

- **Focus:** `pipefunc` focuses on creating reusable, composable Python functions within a pipeline. Job schedulers focus on scheduling and executing independent tasks.
- **Complexity:** `pipefunc` is simpler to set up and use for Python-centric workflows. Job schedulers have more features but a steeper learning curve.
- **Flexibility:** `pipefunc` allows for dynamic, data-driven workflows within Python. Job schedulers are more rigid but offer robust scheduling and monitoring.

### Data Pipelines (e.g., [Kedro](https://kedro.org/), [Prefect](https://www.prefect.io/))

These tools provide frameworks for building data pipelines with a focus on data engineering best practices, such as modularity, versioning, and testing.

**`pipefunc` vs. Data Pipelines:**

- **Structure:** `pipefunc` is less opinionated about project structure than Kedro, which enforces a specific layout. Prefect is more flexible but still geared towards defining data flows.
- **Scope:** `pipefunc` is more focused on the computational aspects of pipelines, while Kedro and Prefect offer more features for data management, versioning, and deployment.
- **Flexibility:** `pipefunc` offers more flexibility in how pipelines are defined and executed, while Kedro and Prefect provide more structure and standardization.

### Workflow Definition Languages (e.g., [Snakemake](https://snakemake.readthedocs.io/))

Snakemake uses a domain-specific language (DSL) to define workflows as a set of rules with dependencies. It excels at orchestrating diverse tools and scripts, often in separate environments, through a dedicated workflow definition file (`Snakefile`).
Unlike pipefunc, Snakemake primarily works with serialized data and may require custom implementations for parameter sweeps within the Python code.

**`pipefunc` vs. Snakemake:**

- **Workflow Definition:** `pipefunc` uses Python code with decorators. Snakemake uses a `Snakefile` with a specialized syntax.
- **Focus:** `pipefunc` is designed for Python-centric workflows and automatic parallelization within Python. Snakemake is language-agnostic and handles the execution of diverse tools and steps, potentially in different environments.
- **Flexibility:** `pipefunc` offers more flexibility in defining complex logic within Python functions. Snakemake provides a more rigid, rule-based approach.
- **Learning Curve:** `pipefunc` is generally easier to learn for Python users. Snakemake requires understanding its DSL.

**`pipefunc` within Snakemake:**

`pipefunc` can be integrated into a Snakemake workflow. You could have a Snakemake rule that executes a Python script containing a `pipefunc` pipeline, combining the strengths of both tools.

**In essence:**

`pipefunc` provides a simpler, more Pythonic approach for workflows primarily based on Python functions. It excels at streamlining development, reducing boilerplate, and automatically handling parallelization within the familiar Python ecosystem. While other tools may be better suited for production ETL pipelines, managing complex dependencies, or workflows involving diverse non-Python tools, `pipefunc` is ideal for flexible scientific computing workflows where rapid development and easy parameter exploration are priorities.

## How to use `adaptive` with `pipefunc`?

This section has been moved to [SLURM integration](./concepts/slurm.md).

## How to handle defaults?

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#how-to-handle-defaults).

## How to bind parameters to a fixed value?

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#how-to-bind-parameters-to-a-fixed-value).

## How to rename inputs and outputs?

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#how-to-rename-inputs-and-outputs).

## How to handle multiple outputs?

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#how-to-handle-multiple-outputs).

## How does type checking work in `pipefunc`?

This section has been moved to [Type Checking](./concepts/type-checking.md).

## What is the difference between `pipeline.run` and `pipeline.map`?

This section has been moved to [Parallelism and Execution](./concepts/execution-and-parallelism.md#run-vs-map).

## How to use parameter scopes (namespaces)?

This section has been moved to [Parameter Scopes](./concepts/parameter-scopes.md).

## How to inspect the `Resources` inside a `PipeFunc`?

This section has been moved to [Resource Management](./concepts/resource-management.md#how-to-inspect-the-resources-inside-a-pipefunc).

## How to set the `Resources` dynamically, based on the input arguments?

This section has been moved to [Resource Management](./concepts/resource-management.md#how-to-set-the-resources-dynamically-based-on-the-input-arguments).

## How to use `adaptive` with `pipefunc`?

This section has been moved to [Adaptive integration](./concepts/adaptive-integration.md).

## What is the `ErrorSnapshot` feature in `pipefunc`?

This section has been moved to [Error Handling](./concepts/error-handling.md).

## What is the overhead / efficiency / performance of `pipefunc`?

This section has been moved to [Overhead and Efficiency](./concepts/overhead-and-efficiency.md).

## How to mock functions in a pipeline for testing?

This section has been moved to [Testing](./concepts/testing.md).

## Mixing executors and storage backends for I/O-bound and CPU-bound work

This section has been moved to [Parallelism and Execution](./concepts/execution-and-parallelism.md#mixing-executors-and-storage-backends-for-io-bound-and-cpu-bound-work).

## Get a function handle for a specific pipeline output (`pipeline.func`)

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#get-a-function-handle-for-a-specific-pipeline-output-pipelinefunc).

## `dataclasses` and `pydantic.BaseModel` as `PipeFunc`

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#dataclasses-and-pydanticbasemodel-as-pipefunc).

## What is `VariantPipeline` and how to use it?

This section has been moved to [Variants](./concepts/variants.md).

## How to use post-execution hooks?

This section has been moved to [Parallelism and Execution](./concepts/execution-and-parallelism.md#how-to-use-post-execution-hooks).

## How to collect results as a step in my `Pipeline`?

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#how-to-collect-results-as-a-step-in-my-pipeline).

## `PipeFunc`s with Multiple Outputs of Different Shapes

This section has been moved to [Function Inputs and Outputs](./concepts/function-io.md#pipefuncs-with-multiple-outputs-of-different-shapes).

## Simplifying Pipelines

This section has been moved to [Simplifying Pipelines](./concepts/simplifying-pipelines.md).

## Parameter Sweeps

This section has been moved to [Parameter Sweeps](./concepts/parameter-sweeps.md).
