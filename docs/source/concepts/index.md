# Concepts

```{toctree}
:hidden:

Function Inputs and Outputs <function-io>
Understanding mapspec <mapspec>
Parallelism and Execution <execution-and-parallelism>
Parameter Scopes <parameter-scopes>
Error Handling <error-handling>
Type Checking <type-checking>
Variants <variants>
Caching <caching>
Automatic CLI <cli>
SLURM integration <slurm>
Resource Management <resource-management>
Simplifying Pipelines <simplifying-pipelines>
Adaptive integration <adaptive-integration>
Testing <testing>
Overhead and Efficiency <overhead-and-efficiency.md>
Parameter Sweeps <parameter-sweeps>
```

```{admonition} Getting Started
:class: tip
If you're new to `pipefunc`, we recommend starting with the [Tutorial](../tutorial) to get a hands-on introduction to the library. Then, explore the concepts in this section to deepen your understanding.
```

Welcome to the Concepts section of the `pipefunc` documentation.
Here, we delve into the core ideas and design principles that underpin the library.
Understanding these concepts will help you effectively utilize `pipefunc`'s features to build, manage, and optimize your computational workflows.

Each page in this section covers a specific aspect of `pipefunc`, explained in detail with examples and diagrams.
Whether you're looking to understand the intricacies of data flow with `mapspec`, learn about parallel execution, or explore advanced features like resource management, this section provides the necessary insights.

## Topics Covered

Below are the key concepts discussed in this section. Click on any topic to learn more:

- **[Function Inputs and Outputs](./function-io):** Manage inputs, outputs, defaults, renaming, and multiple returns. Use with `dataclasses` and `pydantic`.
- **[Understanding `mapspec`](mapspec.md):** Define data mappings with `mapspec` for element-wise operations, reductions, and parallelization.
- **[Execution and Parallelism](./execution-and-parallelism):** Control pipeline execution: sequential and parallel, with mixed executors and storage. Includes post-execution hooks.
- **[Parameter Scopes](./parameter-scopes):** Organize pipelines and avoid naming conflicts using parameter scopes.
- **[Error Handling](./error-handling):** Capture detailed error information with `ErrorSnapshot` for debugging.
- **[Type Checking](./type-checking):** How `pipefunc` uses type hints for static and runtime type checking to ensure data integrity.
- **[Variants](./variants):** Use `VariantPipeline` to manage multiple function implementations within a single pipeline.
- **[Caching](./caching):** Cache function results with `memoize` and `Pipeline` cache options. Understand cache types and shared memory caching.
- **[Automatic CLI](./cli):** Automatically generate a CLI for your pipeline, complete with documentation.
- **[SLURM Integration](./slurm):** Submit pipeline.map calls to SLURM clusters with `pipefunc` and [`adaptive-scheduler`](https://adaptive-scheduler.readthedocs.io/en/latest/).
- **[Resource Management](./resource-management):** Specify and dynamically allocate resources (CPU, memory, GPU) for individual functions.
- **[Simplifying Pipelines](./simplifying-pipelines):** Merge nodes with `simplified_pipeline` and `NestedPipeFunc`. Understand the trade-offs.
- **[Adaptive Integration](./adaptive-integration):** Optimize parameter space exploration with `adaptive` library integration.
- **[Testing](./testing):** Best practices for testing, including mocking functions in pipelines.
- **[Overhead and Efficiency](./overhead-and-efficiency):** Measure the performance overhead of `pipefunc`.
- **[Parameter Sweeps](./parameter-sweeps):** Construct parameter sweeps and optimize execution with `pipefunc.sweep`.

## Contributing

We welcome contributions to the `pipefunc` documentation! If you find any issues or have suggestions for improving the concepts explained here, please [open an issue](https://github.com/pipefunc/pipefunc/issues/new) or submit a pull request on our [GitHub repository](https://github.com/pipefunc/pipefunc).
