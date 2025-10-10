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

# Pipeline CLI Interface

The {meth}`~pipefunc.Pipeline.cli()` method provides an out‐of‐the‐box command‐line interface (CLI) for executing your PipeFunc pipelines directly from the terminal.
This feature leverages your pipeline’s input schema—generated automatically via Pydantic—and even extracts detailed parameter descriptions from your docstrings.
With a single call to `pipeline.cli()`, you can run your entire pipeline without writing any additional command-line parsing code, load inputs from a JSON file, or simply view the pipeline documentation.

In other words, you can:

- **Automatically generate a CLI** from your pipeline’s input parameters.
- **Validate and coerce user inputs** using a dynamically created Pydantic model.
- **Extract parameter descriptions** from your function docstrings (if packages like Griffe are installed) so that the CLI help text includes detailed information.
- **Configure mapping options** (e.g., parallel execution, storage method, and cleanup) via dedicated command‐line flags.
- **Print pipeline documentation** directly via the `docs` subcommand.

```{note}
The CLI works with simple input types that can be represented in JSON.
This means that multi-dimensional arrays or any non-JSON-serializable types must be provided in a simplified form (e.g., as nested lists) so that Pydantic can later coerce them to the correct type (such as NumPy arrays) based on your type annotations.
```

---

## Input Modes

The CLI supports three modes:

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

3. **Docs Mode**
   Instead of executing the pipeline, this mode simply prints the pipeline documentation (via `pipeline.print_documentation()`).
   **Example usage:**

   ```bash
   python cli-example.py docs
   ```

In CLI and JSON modes, additional mapping options (prefixed with `--map-`) allow you to control how the pipeline executes, including settings like the run folder, parallel execution, storage backend, and cleanup behavior.

---

## How It Works

When you invoke `pipeline.cli()`, the following steps occur:

1. **Pydantic Model Generation:**
   The CLI inspects your pipeline’s root input parameters and generates a Pydantic model (see {meth}`pipefunc.Pipeline.pydantic_model`).
   It automatically uses the default values, type hints, and even the descriptions extracted from your (NumPy, Google, or Sphinx)-style docstrings (see {meth}`pipefunc.Pipeline.print_documentation`) to create a robust input schema.
   For parameters associated with multi-dimensional arrays (using mapspecs), the CLI expects these values to be represented as nested lists in JSON.
   Pydantic then coerces these lists into the appropriate array type (e.g., NumPy arrays) based on your type annotations.

2. **Argument Parsing:**
   An {class}`argparse.ArgumentParser` is created with three subcommands:
   - **`cli`**: Accepts individual command-line arguments.
   - **`json`**: Requires a JSON file that contains all inputs.
   - **`docs`**: Prints the pipeline documentation.

3. **Mapping Options:**
   Mapping-related options (e.g., `--map-run_folder`, `--map-parallel`, `--map-storage`, and `--map-cleanup`) are added to the `cli` and `json` subcommands, letting you configure pipeline execution without modifying code.

4. **Input Validation and Execution:**
   For the `cli` and `json` subcommands, the CLI parses and validates the inputs using the generated Pydantic model.
   Once validated, it executes the pipeline via {meth}`pipefunc.Pipeline.map()`, and the results are printed to the terminal using a rich, formatted output, and stored to disk in the specified run folder.
   In **docs mode**, instead of executing the pipeline, the CLI simply calls `pipeline.print_documentation()` to display the documentation and then exits.

---

## Example Pipeline with CLI

Below is a complete example that demonstrates how to define a simple pipeline with NumPy-style docstrings and run it via the CLI.
Notice how each function’s docstring provides detailed descriptions for its parameters and return values.
These descriptions are automatically extracted and shown in the CLI help text.

```{code-cell} ipython3
%%writefile cli-example.py

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
# Optional: add a mapspec axis to parameter 'y' for demonstration of multi-dimensional inputs.
pipeline.add_mapspec_axis("y", axis="i")

if __name__ == "__main__":
    # This will launch the CLI.
    # The CLI automatically extracts default values, type annotations,
    # and parameter descriptions from the docstrings.
    # Use 'cli' or 'json' to execute the pipeline, or 'docs' to print documentation.
    pipeline.cli()
```

### Running the Example

- **Help Text:**
  Run the script with the `--help` flag to see the generated help text:

  ```bash
  python cli-example.py --help
  ```

- **CLI Mode:**
  Execute the script and supply individual parameters:

  ```bash
  python cli-example.py cli --x "2" --y "[3, 4, 5]"
  ```

- **JSON Mode:**
  Create a JSON file (e.g., `inputs.json`) with:

  ```json
  {
    "x": 2,
    "y": [3, 4, 5]
  }
  ```

  Then run:

  ```bash
  python cli-example.py json --json-file inputs.json
  ```

- **Docs Mode:**
  To simply view the pipeline documentation without running the pipeline, execute:

  ```bash
  python cli-example.py docs
  ```

In both CLI and JSON modes, the CLI uses the detailed parameter descriptions (extracted from the docstrings) to help guide the user.
For parameters that involve multi-dimensional arrays, ensure that the inputs are represented as nested lists so that they can be properly coerced by Pydantic.

**Example outputs:**

_Click on "*Show code cell output*" to see the CLI output._

```{code-cell} ipython3
:tags: [hide-output]

!python cli-example.py --help
```

```{code-cell} ipython3
:tags: [hide-output]

!python cli-example.py docs
```

```{code-cell} ipython3
:tags: [hide-output]

!python cli-example.py cli -h
```

```{code-cell} ipython3
:tags: [hide-output]

!python cli-example.py cli --x 2 --y "[3, 4, 5]"
```

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
Just remember that the CLI currently supports only simple input types—those that can be represented in JSON and then coerced to the correct type using Pydantic.
Enjoy the simplicity and power of running your pipelines with minimal setup!
