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

- [Basic Pipeline](basic-usage.md): A simple example showing how to create and run a basic pipeline.
  - **Features:** `@pipefunc`, `Pipeline`, sequential execution

## Scientific Computing

- [Physics Simulation](physics-simulation.md): Demonstrates a physics-based simulation using `pipefunc`.
  - **Features:** `@pipefunc`, `Pipeline`, `mapspec`, `dataclasses`, parallel execution, `add_mapspec_axis`, `NestedPipeFunc`
- [Weather Simulation and Analysis](weather-simulation.md): Example of generating and analyzing weather data with `xarray`.
  - **Features:** `@pipefunc`, `Pipeline`, `mapspec`, `xarray`, parallel execution

## Data Processing

- [Sensor Data Processing](sensor-data-processing.md): A pipeline for processing sensor data, including filtering, feature extraction, and anomaly detection.
  - **Features:** `@pipefunc`, `Pipeline`, `scipy.signal`, `seaborn`, `add_mapspec_axis`
- [Image Processing Workflow](image_processing.md): Demonstrates a workflow for image processing, including segmentation and classification.
  - **Features:** `@pipefunc`, `Pipeline`, `scikit-image`, `mapspec`, parallel execution
- [NLP Text Summarization](nlp-text-summarization.md): An example of a natural language processing pipeline for text summarization.
  - **Features:** `@pipefunc`, `Pipeline`, `nltk`, `mapspec`, parallel execution
[Image Processing Workflow](image-processing.md)
