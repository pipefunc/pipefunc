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

# Image Processing Workflow

```{try-notebook}
```

```{note}
This example uses `scikit-image` for image processing. If you don't have it installed, you can install it using `pip install scikit-image`.
```

In this example, we'll process a batch of images to:

1. **Load and Preprocess**: Convert each image to grayscale to reduce complexity and prepare it for segmentation.
2. **Image Segmentation**: Detect regions of interest within each individual image using an edge detection technique.
3. **Feature Extraction**: Identify and count the number of detected regions for each processed image.
4. **Classification**: Classify each image as "Complex" or "Simple" based on the extracted features.
5. **Result Aggregation**: Summarize the classification results across all images in the batch.

```{code-cell} ipython3
import numpy as np
from skimage import data, filters, measure
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries

from pipefunc import Pipeline, pipefunc


# Step 1: Image Loading and Preprocessing
@pipefunc(output_name="gray_image", mapspec="image[n] -> gray_image[n]")
def load_and_preprocess_image(image):
    return rgb2gray(image)


# Step 2: Image Segmentation
@pipefunc(output_name="segmented_image", mapspec="gray_image[n] -> segmented_image[n]")
def segment_image(gray_image):
    return filters.sobel(gray_image)


# Step 3: Feature Extraction
@pipefunc(output_name="feature", mapspec="segmented_image[n] -> feature[n]")
def extract_feature(segmented_image):
    boundaries = find_boundaries(segmented_image > 0.1)
    labeled_image = measure.label(boundaries)
    num_regions = np.max(labeled_image)
    return {"num_regions": num_regions}


# Step 4: Object Classification
@pipefunc(output_name="classification", mapspec="feature[n] -> classification[n]")
def classify_object(feature):
    # Classify image as 'Complex' if the number of regions is above a threshold.
    classification = "Complex" if feature["num_regions"] > 5 else "Simple"
    return classification


# Step 5: Result Aggregation
@pipefunc(output_name="summary")
def aggregate_results(classification):
    simple_count = sum(1 for c in classification if c == "Simple")
    complex_count = len(classification) - simple_count
    return {"Simple": simple_count, "Complex": complex_count}


# Create the pipeline
pipeline_img = Pipeline(
    [
        load_and_preprocess_image,
        segment_image,
        extract_feature,
        classify_object,
        aggregate_results,
    ],
)

# Simulate a batch of images (using built-in scikit-image sample images)
images = [
    data.astronaut(),
    data.coffee(),
    data.coffee(),
]  # Repeat the coffee image to simulate multiple images

# Run the pipeline on the images
results_summary = pipeline_img.map({"image": images})
print("Classification Summary:", results_summary["summary"].output)
```

**Explanation:**

- **Image Loading and Preprocessing (`load_and_preprocess_image`)**: Converts each individual image to grayscale, ensuring independent processing via `mapspec`.
- **Image Segmentation (`segment_image`)**: Applies Sobel filtering to detect edges and regions of interest in each grayscale image, taking advantage of parallel processing for the batch.
- **Feature Extraction (`extract_feature`)**: Identifies boundaries and counts distinct regions in each segmented image, returning the count as a feature for classification.
- **Object Classification (`classify_object`)**: Classifies each image as "Complex" or "Simple" based on the detected regions relative to a predefined threshold.
- **Result Aggregation (`aggregate_results`)**: Aggregates classifications to provide a summary of "Simple" and "Complex" images across the batch.

**Key Points:**

- **`mapspec`**: Enables independent and parallel processing of each image by defining input-to-output mappings, removing the need for explicit parallel code.
- **Functional Structure**: Utilizes `pipefunc` to manage dependencies and efficiently execute batch image processing, highlighting the framework's ability to handle complex workflows.
