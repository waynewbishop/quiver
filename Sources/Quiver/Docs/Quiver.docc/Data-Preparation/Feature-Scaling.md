# Feature Scaling

Normalizing feature columns before classification with StandardScaler or FeatureScaler.

## Overview

In real-world datasets, [features](<doc:Machine-Learning-Primer>) often exist on vastly different scales. A customer's account balance might range from 0 to 250,000, while a loyalty ratio ranges from 0.0 to 0.56 — almost six orders of magnitude apart. When features are on different scales, the larger values can dominate the model's calculations, causing it to ignore smaller but equally important features.

Quiver provides two column-wise scalers that solve this problem in different ways. `StandardScaler` applies z-score standardization, centering each column at zero and scaling it to unit variance. `FeatureScaler` applies min-max scaling, mapping each column into a bounded range. Both follow the same fit-then-transform workflow, so the choice between them is a matter of which output shape the downstream model prefers.

### When scaling matters

Not every model requires scaling, but in practice it often makes the difference between a model that works and one that doesn't. Consider a Gaussian Naive Bayes classifier trained on customer data with credit scores (300–850), account balances (0–250,000), and loyalty ratios (0.0–0.56). Without scaling, the balance column overwhelms the probability calculations because its variance is orders of magnitude larger than the other features. The model may predict the same class for every sample, producing zero precision and zero recall.

![Feature Scaler](diagram-feature-scaler)

Scaling all features to a common range gives each one equal influence in the model's calculations. The classifier can then distinguish between classes based on the actual patterns in the data, not on which features happen to have the largest numbers.

### Choosing between StandardScaler and FeatureScaler

`StandardScaler` is the default choice for most machine learning workflows and is the scaler used by `Pipeline`. It works well when features have different units or ranges, and it is more robust to outliers than min-max scaling. A single extreme value will not compress the rest of the data into a narrow band, because the formula uses mean and standard deviation rather than minimum and maximum. For the concept behind z-scores, see <doc:Statistics-Primer>.

`FeatureScaler` applies min-max scaling, which is the right choice when the target range matters. Image pipelines that expect pixel intensities in 0...1, visualizations bounded to a fixed axis, and neural network layers that assume bounded inputs all benefit from a scaler that produces values in a known interval. The trade-off is that an outlier in the training data will compress the rest of the values toward one end of the range.

A useful rule of thumb is to default to `StandardScaler` for distance-based and gradient-based learning, and reach for `FeatureScaler` when a downstream component requires bounded inputs.

### The fit-then-transform workflow

The key rule for scaling is simple: fit on training data, transform everything. The scaler learns its statistics from the training set, then uses those same statistics to scale both the training and test sets. The pattern is identical for both scalers — only the type name changes.

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

This separation matters because the test set is meant to simulate unseen data. If the scaler learned its statistics from both sets combined, it would leak information from the test set into the training process — a subtle bug that makes evaluation results look better than they actually are.

Both scalers cannot change once they are created. Once fitted, the statistics are locked in, which means the same scaler can safely transform training data, test data, and future incoming data with identical behavior. There is no combined fit-and-transform method, so accidentally re-fitting on test data is not possible.

### Custom ranges with FeatureScaler

`FeatureScaler` defaults to a target range of 0 to 1, but some algorithms work better with different bounds. Pass a custom `ClosedRange` to change the target:

```swift
import Quiver

let features: [[Double]] = [[0], [50], [100]]
let scaler = FeatureScaler.fit(features: features, range: -1.0...1.0)
let scaled = scaler.transform(features)
// [[-1.0], [0.0], [1.0]]
```

> Experiment: **The Quiver Notebook** is the right place to confirm that feature scaling preserves shape. Change the `range` from 0...1 to -1...1 and re-run — the output bounds shift to fill the new interval, and the relative spacing between values is unchanged. See <doc:Quiver-Notebook>.

### Constant columns

If a feature column has the same value for every training sample, scaling would otherwise require dividing by zero — by the standard deviation for `StandardScaler`, or by the column's range for `FeatureScaler`. Both scalers handle this case automatically without any special handling from the caller. `StandardScaler` maps constant columns to zero. `FeatureScaler` maps constant columns to the lower bound of the target range.

### Pairing the scaler with its model

When a model requires scaled features, the scaler and model must stay paired for correct predictions. `Pipeline` bundles a `StandardScaler` and a model into a single value type that scales inputs automatically at prediction time and encodes both as one JSON blob. See <doc:Pipeline> for details.

## Topics

### Scalers
- ``StandardScaler``
- ``FeatureScaler``

### Related
- <doc:Pipeline>
- <doc:Machine-Learning-Primer>
- <doc:Statistics-Primer>
