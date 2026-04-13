# Standard Scaling

Normalize feature columns to zero mean and unit variance.

## Overview

When features in a dataset exist on different scales, distance-based models like `KNearestNeighbors` and `KMeans` give disproportionate weight to the largest values. A credit score ranging from 300 to 850 will dominate a loyalty ratio ranging from 0.0 to 0.56 — not because credit scores matter more, but because the numbers are bigger.

Quiver provides `StandardScaler`, a column-wise z-score scaler that centers each feature at zero and scales it to unit variance using the formula `(value - mean) / std`. For normally distributed data, the resulting values typically fall between -3 and 3.

### When to use standard scaling

`StandardScaler` is the default choice for most ML workflows, and is the scaler used by `Pipeline`. It works well when features have different units or ranges, and is more robust to outliers than min-max scaling. A single extreme value will not compress the rest of the data into a narrow band, because the formula uses mean and standard deviation rather than minimum and maximum.

Use `FeatureScaler` (min-max scaling) instead when the target range matters — for example, mapping pixel intensities to 0...1 or normalizing values for a visualization that expects bounded inputs. See <doc:Feature-Scaling> for details.

### The fit-then-transform workflow

The scaler learns its statistics from training data, then applies them to any dataset. This separation prevents information leaking from the test set into the training process.

```swift
import Quiver

let features: [[Double]] = [
    [619, 15000, 0.08], [502, 78000, 0.04], [699, 0, 0.42],
    [850, 11000, 0.12], [645, 125000, 0.35], [720, 98000, 0.18],
    [410, 45000, 0.06], [780, 0, 0.50], [590, 175000, 0.10],
    [680, 62000, 0.28]
]
let labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

let split = features.stratifiedSplit(labels: labels, testRatio: 0.25, seed: 42)

// Fit on training data only
let scaler = StandardScaler.fit(features: split.trainFeatures)

// Transform both sets using the same statistics
let scaledTrain = scaler.transform(split.trainFeatures)
let scaledTest = scaler.transform(split.testFeatures)
```

Calling `fit` once on training data and `transform` on everything makes the right approach obvious. There is no combined fit-and-transform method, so accidentally re-fitting on test data is not possible.

### Constant columns

If a feature column has the same value for every training sample, its standard deviation is zero. Dividing by zero would produce undefined results. `StandardScaler` handles this by mapping constant columns to zero. No special handling is needed from the caller.

### Pairing the scaler with its model

When a model requires scaled features, the scaler and model must stay paired for correct predictions. `Pipeline` bundles a `StandardScaler` and model into a single value type that scales inputs automatically at prediction time and encodes both as one JSON blob. See <doc:Pipeline> for details.

## Topics

### Scaler
- ``StandardScaler``

### Related
- <doc:Feature-Scaling>
- <doc:Pipeline>
- <doc:Machine-Learning-Primer>
