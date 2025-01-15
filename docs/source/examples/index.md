# Examples

```{toctree}
:hidden:

Basic Usage <basic-usage>
Physics Simulation <physics-simulation>
Weather Simulation and Analysis <weather-simulation>
Sensor Data Processing <sensor-data-processing>
Image Processing Workflow <image-processing>
NLP Text Summarization <nlp-text-summarization>
```

This section provides a collection of practical examples demonstrating various use cases and features of `pipefunc`.

## Basic Usage

```{admonition} Skip this if you did the [tutorial](../tutorial)!
:class: tip, dropdown

We recommend to look at the [Physics Simulation](physics-simulation.md) example after the [tutorial](../tutorial) to get a hands-on introduction to the library.
```

- [Basic Pipeline](basic-usage.md): A simple example showing how to create and run a basic pipeline.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, sequential execution (`pipeline()`, {meth}`~pipefunc.Pipeline.run`)

## Scientific Computing

- [Physics Simulation](physics-simulation.md): Demonstrates a physics-based simulation using `pipefunc`.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, [`mapspec`](../concepts/mapspec.md) (N-dimensional sweeps, zip, reduction), {meth}`~pipefunc.Pipeline.add_mapspec_axis`, {func}`~pipefunc.map.load_outputs`, {func}`~pipefunc.map.load_xarray_dataset`, {meth}`~pipefunc.Pipeline.nest_funcs`, {class}`~pipefunc.NestedPipeFunc`
- [Weather Simulation and Analysis](weather-simulation.md): Example of generating and analyzing weather data with `xarray`.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, [`mapspec`](../concepts/mapspec.md) (N-dimensional sweeps), {func}`~pipefunc.map.load_xarray_dataset`, parallel execution ({meth}`~pipefunc.Pipeline.map`)

## Data Processing

- [Sensor Data Processing](sensor-data-processing.md): A pipeline for processing sensor data, including filtering, feature extraction, and anomaly detection.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, {meth}`~pipefunc.Pipeline.add_mapspec_axis`, {func}`~pipefunc.map.load_xarray_dataset`, {meth}`~pipefunc.Pipeline.subpipeline`, parallel execution ({meth}`~pipefunc.Pipeline.map`)
- [Image Processing Workflow](image-processing.md): Demonstrates a workflow for image processing, including segmentation and classification.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, [`mapspec`](../concepts/mapspec.md) (element-wise operations), {func}`~pipefunc.map.load_outputs`, parallel execution ({meth}`~pipefunc.Pipeline.map`)
- [NLP Text Summarization](nlp-text-summarization.md): An example of a natural language processing pipeline for text summarization.
  - **Uses:** {func}`~pipefunc.pipefunc`, {class}`~pipefunc.Pipeline`, [`mapspec`](../concepts/mapspec.md) (element-wise operations, reduction), parallel execution ({meth}`~pipefunc.Pipeline.map`)
