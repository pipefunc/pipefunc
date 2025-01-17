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

# Weather Simulation and Analysis Pipeline

```{try-notebook}
```

In this example, we'll generate temperature data for multiple cities over several days, compute statistics like mean and variance, and then use `xarray` to load and visualize the results.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_xarray_dataset


# Step 1: Simulate Temperature Data
@pipefunc(output_name="temperature", mapspec="city[i], day[j] -> temperature[i, j]")
def simulate_temperature(city, day):
    np.random.seed(hash(city) % 2**32)  # For reproducibility
    mean_temp = 20 + (hash(city) % 10)  # Base temp varies by city
    temp_variation = 5 * np.sin(day.dayofyear * (2 * np.pi / 365))  # Seasonal variation
    noise = np.random.normal(0, 2)  # Random daily fluctuation
    return float(mean_temp + temp_variation + noise)  # Ensure this is a float


# Step 2: Compute Statistics
@pipefunc(
    output_name=("mean_temp", "variance"),
    mapspec="temperature[i, :] -> mean_temp[i], variance[i]",
    output_picker=dict.__getitem__,
)
def compute_statistics(temperature):
    temp_array = np.array(temperature, dtype=float)  # Ensure it's a numeric array
    mean_temp = np.mean(temp_array)
    var_temp = np.var(temp_array)
    return {"mean_temp": mean_temp, "variance": var_temp}


# Create the pipeline
pipeline_weather = Pipeline([simulate_temperature, compute_statistics])

# Define cities and days
cities = ["New York", "Los Angeles", "Chicago"]
days = pd.date_range("2023-01-01", "2023-01-10")  # 10 days

# Run the pipeline
pipeline_weather.map({"city": cities, "day": days}, run_folder="weather_simulation_results")

# Load and display the xarray dataset
weather_dataset = load_xarray_dataset(run_folder="weather_simulation_results")
display(weather_dataset)

# Plot temperatures for each city
weather_dataset.temperature.astype(float).plot.line(
    x="day",
    hue="city",
    marker="o",
    figsize=(12, 6),
)
```

**Explanation:**

- **Temperature Simulation (`simulate_temperature`)**: Each city has its synthetic daily temperature calculated using a sinusoidal pattern and noise. The `mapspec` `city[i], day[j] -> temperature[i, j]` allows us to handle city-by-day combinations automatically.

- **Statistics Calculation (`compute_statistics`)**: Computes the mean and variance of the daily temperature, mapping over cities.

- **Automatic `xarray.Dataset`**: The `pipeline.map()` call ensures that the data is structured into an N-dimensional format, representing the outputs naturally as an `xarray.Dataset`.

- **Retrieving with `load_xarray_dataset`**: Quickly access the results organized by city and day indices without manually constructing them.

This showcases `pipefunc`'s powerful ability to manage multi-dimensional computations and data structuring, presenting an efficient workflow for simulating and analyzing temperature variations.
