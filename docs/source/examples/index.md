# Examples



This section provides a collection of practical examples demonstrating various use cases and features of `pipefunc`.

## Basic Usage

??? tip "Skip this if you did the [tutorial](../tutorial)!"

We recommend to look at the [Physics Simulation](physics-simulation.md) example after the [tutorial](../tutorial) to get a hands-on introduction to the library.


- [Basic Pipeline](basic-usage.md): A simple example showing how to create and run a basic pipeline.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, sequential execution (`pipeline()`, ``pipefunc.Pipeline.run``)

## Scientific Computing

- [Physics Simulation](physics-simulation.md): Demonstrates a physics-based simulation using `pipefunc`.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, [`mapspec`](../concepts/mapspec.md) (N-dimensional sweeps, zip, reduction), ``pipefunc.Pipeline.add_mapspec_axis``, ``pipefunc.map.load_outputs``, ``pipefunc.map.load_xarray_dataset``, ``pipefunc.Pipeline.nest_funcs``, ``pipefunc.NestedPipeFunc``
- [Weather Simulation and Analysis](weather-simulation.md): Example of generating and analyzing weather data with `xarray`.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, [`mapspec`](../concepts/mapspec.md) (N-dimensional sweeps), ``pipefunc.map.load_xarray_dataset``, parallel execution (``pipefunc.Pipeline.map``)

## Data Processing

- [Sensor Data Processing](sensor-data-processing.md): A pipeline for processing sensor data, including filtering, feature extraction, and anomaly detection.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, ``pipefunc.Pipeline.add_mapspec_axis``, ``pipefunc.map.load_xarray_dataset``, ``pipefunc.Pipeline.subpipeline``, parallel execution (``pipefunc.Pipeline.map``)
- [Image Processing Workflow](image-processing.md): Demonstrates a workflow for image processing, including segmentation and classification.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, [`mapspec`](../concepts/mapspec.md) (element-wise operations), ``pipefunc.map.load_outputs``, parallel execution (``pipefunc.Pipeline.map``)
- [NLP Text Summarization](nlp-text-summarization.md): An example of a natural language processing pipeline for text summarization.
  - **Uses:** ``pipefunc.pipefunc``, ``pipefunc.Pipeline``, [`mapspec`](../concepts/mapspec.md) (element-wise operations, reduction), parallel execution (``pipefunc.Pipeline.map``)
