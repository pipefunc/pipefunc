---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Working with Irregular Arrays

```{try-notebook}

```

```{contents} ToC
:depth: 2
```

Irregular (or jagged) arrays are arrays where different elements have different lengths.
`pipefunc` provides built-in support for irregular arrays through special `mapspec` syntax and automatic handling of masked values for outputs whose mapspec uses the `*` suffix.

:::{admonition} Prerequisites
:class: info

This tutorial assumes familiarity with basic `pipefunc` concepts and `mapspec` syntax.
See the [mapspec tutorial](mapspec.md) for an introduction.

:::

## What are Irregular Arrays?

In regular arrays, all elements along a dimension have the same size.
Irregular arrays relax this constraint, allowing elements to have different lengths.

Common examples include:
- Variable-length time series data
- Text processing with sentences of different lengths
- Simulation results with varying numbers of events
- Hierarchical data structures

## The `*` Notation in mapspec

`pipefunc` uses the `*` suffix to denote irregular dimensions in mapspec strings:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
import numpy as np

@pipefunc(output_name="words", mapspec="text[i] -> words[i, j*]")
def split_text(text: str) -> list[str]:
    """Split text into words."""
    return text.split()

@pipefunc(output_name="lengths", mapspec="words[i, j*] -> lengths[i, j*]")
def word_lengths(words: str) -> int:
    """Get the length of each word."""
    # Handle masked values for irregular dimensions
    if np.ma.is_masked(words):
        return np.ma.masked
    return len(words)

pipeline = Pipeline([split_text, word_lengths])

# Different texts have different numbers of words
inputs = {
    "text": ["Hello world", "Python is great", "A"]
}

# Must specify maximum capacity for irregular dimensions
results = pipeline.map(
    inputs=inputs,
    internal_shapes={"words": (3,), "lengths": (3,)},  # Max 3 words
    storage="dict",
    parallel=False,
)

print("Words array shape:", results["words"].output.shape)
print("Words array:\n", results["words"].output)
```

## Understanding Masked Arrays

When pipefunc stores irregular data, it:
1. Allocates arrays with the maximum needed size
2. Fills unused positions with `np.ma.masked` sentinel values
3. Automatically converts those arrays to `numpy.ma.MaskedArray` objects when accessed **if the mapspec declared that output with a `*` irregular dimension**

```{code-cell} ipython3
# The words array is a MaskedArray
words_array = results["words"].output
print("Type:", type(words_array))
print("\nData:")
print(words_array.data)
print("\nMask (True = invalid):")
print(words_array.mask)
```

Because the storage keeps irregular elements in an object array, you will typically see `dtype=object` in these masked arrays. Regular (non-`*`) mapspec dimensions remain ordinary NumPy arrays.

## Working with Downstream Functions

Functions that receive irregular data (declared with a `*` in their mapspec) get MaskedArrays automatically; regular dimensions stay as plain NumPy arrays. Typing with `pipefunc.typing.Array` helps document the expectation but does not change the runtime behaviour.

```{code-cell} ipython3
@pipefunc(output_name="sentence_length", mapspec="words[i, :] -> sentence_length[i]")
def count_words(words: np.ndarray) -> int:
    """Count non-masked words in a sentence."""
    # Check if it's a MaskedArray
    if hasattr(words, "compressed"):
        # compressed() returns only valid (non-masked) values
        return len(words.compressed())
    return len(words)

@pipefunc(output_name="total_chars", mapspec="lengths[i, :] -> total_chars[i]")
def sum_lengths(lengths: np.ndarray) -> int:
    """Sum the lengths of all words."""
    if hasattr(lengths, "compressed"):
        return sum(lengths.compressed())
    return sum(lengths)

# Add to pipeline (using the functions from above)
pipeline2 = Pipeline([split_text, word_lengths, count_words, sum_lengths])

results2 = pipeline2.map(
    inputs=inputs,
    internal_shapes={"words": (3,), "lengths": (3,)},
    storage="dict",
    parallel=False,
)

print("Word counts:", results2["sentence_length"].output)
print("Total characters:", results2["total_chars"].output)
```

## Generating Irregular Output

When your function produces irregular output, return lists of varying lengths:

```{code-cell} ipython3
@pipefunc(output_name="factors", mapspec="n[i] -> factors[i, j*]")
def find_factors(n: int) -> list[int]:
    """Find all factors of n."""
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

@pipefunc(output_name="factor_count", mapspec="factors[i, :] -> factor_count[i]")
def count_factors(factors: np.ndarray) -> int:
    """Count the number of factors."""
    if hasattr(factors, "compressed"):
        return len(factors.compressed())
    return len(factors)

pipeline3 = Pipeline([find_factors, count_factors])

results3 = pipeline3.map(
    inputs={"n": [6, 7, 12]},
    internal_shapes={"factors": (6,)},  # 12 has 6 factors
    storage="dict",
    parallel=False,
)

print("Factors of 6:", results3["factors"].output[0].compressed())
print("Factors of 7:", results3["factors"].output[1].compressed())
print("Factors of 12:", results3["factors"].output[2].compressed())
print("\nFactor counts:", results3["factor_count"].output)
```

## Best Practices

### 1. Always Specify internal_shapes

For irregular dimensions, you must provide the maximum capacity:

```{code-cell} ipython3
factor_lists = [find_factors(n) for n in [6, 7, 12]]
max_size = max(len(factors) for factors in factor_lists)
print(f"Maximum factors needed: {max_size}")

results = pipeline3.map(
    inputs={"n": [6, 7, 12]},
    internal_shapes={"factors": (max_size,)},
    storage="dict",
    parallel=False,
)
```

````{admonition} What if I forget `internal_shapes`?
:class: warning

If `internal_shapes` is omitted (or uses a plain `?` placeholder), `pipefunc` fixes the capacity to the very first shape it encounters. Any later output that needs more space raises an error such as:

```text
ValueError: Irregular output shape (2,) of function '...' (output '...') exceeds the configured internal shape (1,) used in the `mapspec` '...'
```

To avoid that crash, estimate the maximum size up front or pre-compute the results you plan to map over and size `internal_shapes` from their lengths.
````

### 2. Handle Masked Values in Element-wise Operations

When a function operates on individual elements from an irregular dimension (using `j*` notation), it may receive `np.ma.masked` sentinel values:

```{code-cell} ipython3
@pipefunc(output_name="doubled", mapspec="values[i, j*] -> doubled[i, j*]")
def double_value(values: int) -> int:
    """Double a value, handling masked sentinels."""
    # IMPORTANT: Check for masked sentinel first!
    if np.ma.is_masked(values):
        return np.ma.masked
    return values * 2
```

### 3. Handle MaskedArrays in Reductions

When receiving entire rows/slices, check if your input is a MaskedArray:

```{code-cell} ipython3
def process_irregular_data(data: np.ndarray) -> float:
    """Example of proper MaskedArray handling."""
    if hasattr(data, "compressed"):
        # It's a MaskedArray
        valid_values = data.compressed()
        if len(valid_values) == 0:
            return 0.0
        return np.mean(valid_values)
    else:
        # Regular array
        return np.mean(data)

# Test with masked array
test_masked = np.ma.array([1, 2, 3, 4], mask=[False, False, True, True])
print("Mean of [1, 2, masked, masked]:", process_irregular_data(test_masked))
```

### 4. Use compressed() for Simple Operations

The `compressed()` method is efficient for getting valid values:

```{code-cell} ipython3
# Create a masked array
data = np.ma.array([[1, 2, 3], [4, 5, 6]],
                   mask=[[False, False, True], [False, True, True]])

print("Original shape:", data.shape)
print("Compressed (1D):", data.compressed())
print("Sum of valid:", sum(data.compressed()))
print("Mean of valid:", np.mean(data.compressed()))
```

### 5. Preserve Structure When Needed

Note that `compressed()` returns a 1D array. If you need structure:

```{code-cell} ipython3
# If you need to preserve row structure
for i, row in enumerate(data):
    valid_in_row = row.compressed()
    print(f"Row {i} valid values: {valid_in_row}")
```

## Advanced Example: Text Analysis

Here's a complete example analyzing text with irregular arrays:

```{code-cell} ipython3
from pipefunc.typing import Array

@pipefunc(output_name="sentences", mapspec="text[i] -> sentences[i, j*]")
def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

@pipefunc(output_name="words_per_sentence", mapspec="sentences[i, j*] -> words_per_sentence[i, j*]")
def count_words_in_sentence(sentences: str) -> int:
    """Count words in each sentence."""
    # Handle masked values
    if np.ma.is_masked(sentences):
        return np.ma.masked
    return len(sentences.split())

@pipefunc(output_name="avg_words", mapspec="words_per_sentence[i, :] -> avg_words[i]")
def average_words(words_per_sentence: Array[int]) -> float:
    """Calculate average words per sentence."""
    if hasattr(words_per_sentence, "compressed"):
        valid = words_per_sentence.compressed()
        return np.mean(valid) if len(valid) > 0 else 0.0
    return np.mean(words_per_sentence)

# Create pipeline
text_pipeline = Pipeline([split_sentences, count_words_in_sentence, average_words])

# Analyze texts with different numbers of sentences
texts = {
    "text": [
        "Hello world. How are you?",
        "Python is great! It's versatile. Easy to learn. Powerful too!",
        "One sentence only"
    ]
}

results = text_pipeline.map(
    inputs=texts,
    internal_shapes={
        "sentences": (4,),  # Max 4 sentences
        "words_per_sentence": (4,)
    },
    storage="dict",
    parallel=False,
)

for i, text in enumerate(texts["text"]):
    print(f"\nText {i}: '{text[:30]}...'")
    print(f"  Sentences: {results['sentences'].output[i].compressed()}")
    print(f"  Words per sentence: {results['words_per_sentence'].output[i].compressed()}")
    print(f"  Average words: {results['avg_words'].output[i]:.1f}")
```

## Performance Considerations

MaskedArrays have minimal overhead:
- Mask checking is vectorized and fast
- `compressed()` is implemented in C and very efficient
- Storage overhead is one boolean per element for the mask

For large-scale processing, irregular arrays can actually be more memory-efficient than padding with zeros or using lists of lists.

## Troubleshooting

### Common Issues

1. **IndexError when internal_shape is too small**:
   ```python
   # This will fail if any text has >3 words
   internal_shapes={"words": (3,)}
   ```
   Solution: Calculate or estimate the maximum size needed.

2. **Using regular array operations on masked data**:
   ```python
   # Wrong - includes masked values
   total = sum(masked_array)

   # Correct - only sums valid values
   total = sum(masked_array.compressed())
   ```

3. **Forgetting to check for MaskedArray**:
   ```python
   # Better to be defensive
   if hasattr(data, "compressed"):
       values = data.compressed()
   else:
       values = data
   ```

## Summary

- Use `*` in mapspec to denote irregular dimensions (e.g., `j*`)
- Always provide `internal_shapes` for the maximum capacity needed
- Functions automatically receive MaskedArrays for irregular data
- Use `compressed()` to get valid values efficiently
- Check with `hasattr(data, "compressed")` to handle both regular and masked arrays
