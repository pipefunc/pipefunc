# Examples

```{toctree}
:hidden:

Function Inputs and Outputs <function-io>
Understanding mapspec <mapspec>
Parallelism and Execution <execution-and-parallelism>
Variants <variants>
Type Checking <type-checking>
Error Handling <error-handling>
Resource Management <resource-management>
Parameter Scopes <parameter-scopes>
Simplifying Pipelines <simplifying-pipelines>
Adaptive integration <adaptive-integration>
Testing <testing>
Benchmarking <benchmarking>
Parameter Sweeps <parameter-sweeps>
```

This section provides a collection of practical examples demonstrating various use cases and features of `pipefunc`.

## Basic Usage

- [Basic Pipeline](basic_usage.md): A simple example showing how to create and run a basic pipeline.
  - **Features:** `@pipefunc`, `Pipeline`, sequential execution

## Scientific Computing

- [Physics Simulation](physics_simulation.md): Demonstrates a physics-based simulation using `pipefunc`.
  - **Features:** `@pipefunc`, `Pipeline`, `mapspec`, `dataclasses`, parallel execution, `add_mapspec_axis`, `NestedPipeFunc`
- [Weather Simulation and Analysis](weather_simulation.md): Example of generating and analyzing weather data with `xarray`.
  - **Features:** `@pipefunc`, `Pipeline`, `mapspec`, `xarray`, parallel execution

## Data Processing

- [Sensor Data Processing](sensor_data_processing.md): A pipeline for processing sensor data, including filtering, feature extraction, and anomaly detection.
  - **Features:** `@pipefunc`, `Pipeline`, `scipy.signal`, `seaborn`, `add_mapspec_axis`
- [Image Processing Workflow](image_processing.md): Demonstrates a workflow for image processing, including segmentation and classification.
  - **Features:** `@pipefunc`, `Pipeline`, `scikit-image`, `mapspec`, parallel execution
- [NLP Text Summarization](nlp_text_summarization.md): An example of a natural language processing pipeline for text summarization.
  - **Features:** `@pipefunc`, `Pipeline`, `nltk`, `mapspec`, parallel execution
