---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: pipefunc
  language: python
  name: python3
---

# Sensor Data Processing Pipeline

```{try-notebook}
```

```{note}
This example uses `scipy` and `seaborn` libraries for data processing and visualization. Make sure to install these libraries before running the code.
```

Let's create a detailed example for the sensor data processing pipeline. This example will simulate the following steps:

1. **Data Collection**: Simulate raw sensor data.
2. **Noise Filtering**: Apply a basic noise filter.
3. **Feature Extraction**: Extract features such as peak values.
4. **Anomaly Detection**: Identify anomalies within the extracted features.
5. **Visualization**: Plot the results.

Here’s how this pipeline can be implemented using `pipefunc`:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from pipefunc import Pipeline, pipefunc


# Step 1: Simulate Sensor Data
@pipefunc(output_name="raw_data")
def collect_data(num_samples: int = 1000, noise_level: float = 0.1):
    time = np.linspace(0, 100, num_samples)
    data = np.sin(time) + noise_level * np.random.randn(num_samples)
    return time, data


# Step 2: Noise Filtering
@pipefunc(output_name="filtered_data")
def filter_noise(raw_data):
    time, data = raw_data
    # Simple moving average filter
    window_size = 5
    filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    adjusted_time = time[: len(filtered_data)]
    return adjusted_time, filtered_data


# Step 3: Feature Extraction
@pipefunc(output_name=("peak_times", "peak_values"))
def extract_features(filtered_data):
    time, data = filtered_data
    # Find peaks in the data
    peaks, _ = find_peaks(data, height=0)
    peak_times = time[peaks]
    peak_values = data[peaks]
    return peak_times, peak_values


# Step 4: Anomaly Detection
@pipefunc(output_name=("anomaly_times", "anomaly_values"))
def detect_anomalies(peak_times, peak_values, threshold: float = 0.8):
    # Simple anomaly detection based on threshold
    anomalies = peak_values > threshold
    anomaly_times = peak_times[anomalies]
    anomaly_values = peak_values[anomalies]
    return anomaly_times, anomaly_values


# Step 5: Visualization
@pipefunc(output_name="visualization")
def visualize(raw_data, filtered_data, peak_times, peak_values, anomaly_times, anomaly_values):
    raw_time, raw_data = raw_data
    filt_time, filt_data = filtered_data

    plt.figure(figsize=(12, 6))
    plt.plot(raw_time, raw_data, label="Raw Data", alpha=0.5)
    plt.plot(filt_time, filt_data, label="Filtered Data")
    plt.scatter(peak_times, peak_values, color="green", label="Peaks")
    plt.scatter(anomaly_times, anomaly_values, color="red", label="Anomalies")
    plt.title("Sensor Data Processing")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(visible=True)
    plt.show()


# Create the pipeline
pipeline_sensor = Pipeline(
    [collect_data, filter_noise, extract_features, detect_anomalies, visualize],
)

pipeline_sensor.visualize(orient="TB")
```

```{code-cell} ipython3
# Run the full pipeline
pipeline_sensor("visualization", num_samples=1000, noise_level=0.1, threshold=0.8)
```

**Explanation:**

- **Data Collection (`collect_data`)**: Simulate time and sine wave data with added Gaussian noise.
- **Noise Filtering (`filter_noise`)**: Use a simple moving average to smooth the data.
- **Feature Extraction (`extract_features`)**: Find peaks in the filtered data using `scipy.signal.find_peaks`.
- **Anomaly Detection (`detect_anomalies`)**: Identify peaks above a certain threshold as anomalies.
- **Visualization (`visualize`)**: Plot raw data, filtered data, detected peaks, and anomalies.

+++

**Do a study for different noise levels and thresholds:**

We can expand the analysis by examining how varying levels of noise and different sample sizes affect the detection of anomalies.

```{code-cell} ipython3
# Create a new pipeline that terminates at the anomaly detection step (so without visualization)
pipeline_sensor2 = pipeline_sensor.subpipeline(output_names={"anomaly_times", "anomaly_values"})

# Also let's add a function to get the number of detected anomalies


@pipefunc(output_name="num_anomalies")
def count_anomalies(anomaly_times):
    return len(anomaly_times)


pipeline_sensor2.add(count_anomalies)

# Add dimensional axes to the input parameters
pipeline_sensor2.add_mapspec_axis("num_samples", axis="i")
pipeline_sensor2.add_mapspec_axis("noise_level", axis="j")

# Run the subpipeline with different configurations
result = pipeline_sensor2.map(
    inputs={"num_samples": [1000, 500, 1000], "noise_level": [0.05, 0.1, 0.2]},
    run_folder="sensor_map_results",
)
```

**Plotting Results for Different Noise Levels and Thresholds:**

To better understand the relationships and impacts of noise and sample size on anomaly detection, visualize the results with a heatmap.

```{code-cell} ipython3
# Load and visualize the resulting xarray dataset
import seaborn as sns

from pipefunc.map import load_xarray_dataset

ds = load_xarray_dataset("num_anomalies", run_folder="sensor_map_results")

# Convert data variables to a numpy array for plotting
num_anomalies_data = ds["num_anomalies"].data.astype(int)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    num_anomalies_data,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=ds["noise_level"].values,
    yticklabels=ds["num_samples"].values,
)

# Add labels
plt.title("Number of Anomalies Heatmap")
plt.xlabel("Noise Level")
plt.ylabel("Number of Samples")

# Show the plot
plt.show()
```
