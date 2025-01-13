# Concepts

```{toctree}
:hidden:

Mapspec Explained <mapspec-explained>
Function Inputs and Outputs <function-io>
Variants <variants>
Error Handling <error-handling>
Resource Management <resource-management>
Parallelism and Execution <parallelism>
Parameter Scopes <parameter-scopes>
Simplifying Pipelines <simplifying-pipelines>
Testing <testing>
Benchmarking <benchmarking>
```

Welcome to the Concepts section of the `pipefunc` documentation.
Here, we delve into the core ideas and design principles that underpin the library.
Understanding these concepts will help you effectively utilize `pipefunc`'s features to build, manage, and optimize your computational workflows.

Each page in this section covers a specific aspect of `pipefunc`, explained in detail with examples and diagrams.
Whether you're looking to understand the intricacies of data flow with `mapspec`, learn about parallel execution, or explore advanced features like resource management, this section provides the necessary insights.

## Topics Covered

Below are the key concepts discussed in this section. Click on any topic to learn more:

- **[mapspec Explained](./mapspec-explained)**: Learn how to use the powerful `mapspec` syntax to define data mappings between functions, enabling element-wise operations, reductions, and parallel computations in your pipelines.
- **[Function Inputs and Outputs](./function-io)**: Discover how `pipefunc` handles function inputs and outputs, including default values, parameter binding, renaming, and managing multiple outputs. Also covers collecting results and using `dataclasses` and `pydantic.BaseModel` with `PipeFunc`.
- **[Variants](./variants)**: Explore the concept of `VariantPipeline` and how to use it to create and manage different implementations of functions within a single pipeline, facilitating experimentation and A/B testing.
- **[Error Handling](./error-handling)**: Understand how `pipefunc` handles errors during pipeline execution and how to use the `ErrorSnapshot` feature to capture detailed error information for debugging.
- **[Resource Management](./resource-management)**: Learn how to specify resource requirements for individual functions, inspect allocated resources, and dynamically allocate resources based on input parameters.
- **[Parallelism and Execution](./parallelism)**: Dive into the different ways to execute pipelines in parallel, including mixing executors and storage backends, and how type checking works in parallel execution scenarios. Also, learn how to use post-execution hooks.
- **[Parameter Scopes](./parameter-scopes)**: Master the use of parameter scopes to organize your pipelines, avoid naming conflicts, and manage complex data flows.
- **[Simplifying Pipelines](./simplifying-pipelines)**: Understand how to use `simplified_pipeline` to merge nodes and create `NestedPipeFunc` objects, along with the associated trade-offs.
- **[Testing](./testing)**: Learn best practices for testing `pipefunc` pipelines, including how to mock functions for controlled testing environments.
- **[Benchmarking](./benchmarking)**: Understand the overhead introduced by the library itself.

## Getting Started

If you're new to `pipefunc`, we recommend starting with the [Tutorial](../tutorial) to get a hands-on introduction to the library. Then, explore the concepts in this section to deepen your understanding.

## Contributing

We welcome contributions to the `pipefunc` documentation! If you find any issues or have suggestions for improving the concepts explained here, please [open an issue](https://github.com/pipefunc/pipefunc/issues/new) or submit a pull request on our [GitHub repository](https://github.com/pipefunc/pipefunc).
