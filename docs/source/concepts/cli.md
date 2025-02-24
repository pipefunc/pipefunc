# Pipeline CLI Interface

The {meth}`~pipefunc.Pipeline.cli()` method provides an out‐of‐the‐box command‐line interface (CLI) for executing your PipeFunc pipelines directly from the terminal.
This feature leverages your pipeline’s input schema—generated automatically via Pydantic—and even extracts detailed parameter descriptions from your docstrings (see the {meth}`~pipefunc.Pipeline.print_documentation` and {meth}`~pipefunc.Pipeline.pydantic_model` methods of {class}`Pipeline`).
With a single call to `pipeline.cli()`, you can run your entire pipeline without writing any additional command-line parsing code.

In other words, you can:

- **Automatically generate a CLI** from your pipeline’s input parameters.
- **Validate and coerce user inputs** using a dynamically created Pydantic model.
- **Extract parameter descriptions** from your function docstrings (if packages like Griffe are installed) so that the CLI help text includes detailed information.
- **Configure mapping options** (e.g., parallel execution, storage method, and cleanup) via dedicated command‐line flags.

---

## Input Modes

The CLI supports two modes:

1. **CLI Mode**
   Each input parameter is provided as an individual command‐line option.
   **Example usage:**

   ```bash
   python cli-example.py cli --x 2 --y 3
   ```

2. **JSON Mode**
   All inputs are supplied via a JSON file. This mode is ideal for complex pipelines or when reusing a set of pre-prepared inputs.
   **Example usage:**

   ```bash
   python cli-example.py json --json-file inputs.json
   ```

In both modes, additional mapping options (prefixed with `--map-`) allow you to control how the pipeline executes, including settings like the run folder, parallel execution, storage backend, and cleanup behavior.

---

## How It Works

When you invoke `pipeline.cli()`, the following steps occur:

1. **Pydantic Model Generation:**
   The CLI inspects your pipeline’s root input parameters and generates a Pydantic model. It automatically uses the default values, type hints, and even the descriptions extracted from your (NumPy, Google, or Sphinx)-style docstrings to create a robust input schema.

2. **Argument Parsing:**
   An `argparse.ArgumentParser` is created with two subcommands:

   - **`cli`**: Accepts individual command-line arguments.
   - **`json`**: Requires a JSON file that contains all inputs.

3. **Mapping Options:**
   Mapping-related options (e.g., `--map-run_folder`, `--map-parallel`, `--map-storage`, and `--map-cleanup`) are added to both subcommands, letting you configure pipeline execution without modifying code.

4. **Input Validation and Execution:**
   The CLI parses and validates the inputs using the generated Pydantic model. Once validated, it executes the pipeline via `Pipeline.map()`, and the results are printed to the terminal using a rich, formatted output.

---

## Example Pipeline with CLI

Below is a complete example that demonstrates how to define a simple pipeline with NumPy-style docstrings and run it via the CLI.
Notice how each function’s docstring provides detailed descriptions for its parameters and return values. These descriptions are automatically extracted and shown in the CLI help text.

```python
import numpy as np
from pipefunc import Pipeline, pipefunc

@pipefunc(output_name="sum")
def add(x: float, y: float) -> float:
    """
    Add two numbers.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        The sum of x and y.
    """
    return x + y

@pipefunc(output_name="product")
def multiply(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Parameters
    ----------
    x : float
        The first factor.
    y : float
        The second factor.

    Returns
    -------
    float
        The product of x and y.
    """
    return x * y

@pipefunc(output_name="result")
def compute_result(sum: float, product: float) -> float:
    """
    Compute the final result.

    Parameters
    ----------
    sum : float
        The result from the add function.
    product : float
        The result from the multiply function.

    Returns
    -------
    float
        The sum of the add and multiply results.
    """
    return sum + product

# Create a pipeline with the three functions
pipeline = Pipeline([add, multiply, compute_result])

if __name__ == "__main__":
    # This will launch the CLI.
    # The CLI automatically extracts default values, type annotations,
    # and parameter descriptions from the NumPy-style docstrings.
    pipeline.cli()
```

### Running the Example

- **CLI Mode:**
  Execute the script and supply individual parameters:

  ```bash
  python cli-example.py cli --x "2" --y "3"
  ```

- **JSON Mode:**
  Create a JSON file (e.g., `inputs.json`) with:

  ```json
  {
    "x": 2,
    "y": 3
  }
  ```

  Then run:

  ```bash
  python cli-example.py json --json-file inputs.json
  ```

In both cases, the CLI uses the detailed parameter descriptions (extracted from the docstrings) to help guide the user.

---

## Dependencies

To use the CLI features, ensure that you have the following packages installed:

- **Pydantic** – For input validation and dynamic model generation.
- **Rich** – For enhanced terminal output and formatted tables.
- **Griffe** – For extracting detailed docstring information.

Install the CLI extras with:

```bash
pip install "pipefunc[cli]"
```

---

By integrating a fully automated CLI into your pipelines, `Pipeline.cli()` makes it straightforward to run and experiment with complex workflows directly from the terminal.
