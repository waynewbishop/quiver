# Feature Scaling

Normalize feature columns to a common range before classification.

## Overview

In real-world datasets, [features](<doc:Machine-Learning-Primer>) often exist on vastly different scales. A customer's account balance might range from 0 to 250,000, while a loyalty ratio ranges from 0.0 to 0.56 — almost six orders of magnitude apart. When features are on different scales, the larger values can dominate the model's calculations, causing it to ignore smaller but equally important features.

Quiver provides `FeatureScaler`, a column-wise min-max scaler that normalizes each feature independently so every value falls within a target range. The default range is 0 to 1: the column's minimum maps to 0, its maximum maps to 1, and everything else falls proportionally in between.

### When scaling matters

Not every model requires scaling, but in practice it often makes the difference between a model that works and one that doesn't. Consider a Gaussian Naive Bayes classifier trained on customer data with credit scores (300–850), account balances (0–250,000), and loyalty ratios (0.0–0.56). Without scaling, the balance column overwhelms the probability calculations because its variance is orders of magnitude larger than the other features. The model may predict the same class for every sample, producing zero precision and zero recall.

![Feature Scaler](diagram-feature-scaler)

Scaling all features to the same range gives each one equal influence in the model's calculations. The classifier can then distinguish between classes based on the actual patterns in the data, not on which features happen to have the largest numbers.

### The fit-then-transform workflow

The key rule for scaling is simple: fit on training data, transform everything. The scaler learns the minimum and maximum of each column from the training set, then uses those same statistics to scale both the training and test sets.

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
let scaler = FeatureScaler.fit(features: split.trainFeatures)

// Transform both sets using the same statistics
let scaledTrain = scaler.transform(split.trainFeatures)
let scaledTest = scaler.transform(split.testFeatures)
```

This separation matters because the test set is meant to simulate unseen data. If the scaler learned its statistics from both sets combined, it would leak information from the test set into the training process — a subtle bug that makes evaluation results look better than they actually are.

### Why fit and transform are separate steps

Combining fit and transform into a single call would be convenient, but it would also make it easy to accidentally re-fit on test data. By keeping them separate, the workflow makes the right approach obvious: call `fit` once on training data, then call `transform` on any dataset that needs scaling.

`FeatureScaler` is immutable after creation. Once fitted, its statistics cannot change. This means the same scaler can safely transform training data, test data, and future production data with identical behavior.

### Constant columns

If a feature column has the same value for every training sample (zero range), scaling would require dividing by zero. `FeatureScaler` handles this by mapping constant columns to the lower bound of the target range. No special handling is needed from the caller.

### Custom ranges

The default target range is 0 to 1, but some algorithms work better with different ranges. Pass a custom `ClosedRange` to change the target:

```swift
import Quiver

let features: [[Double]] = [[0], [50], [100]]
let scaler = FeatureScaler.fit(features: features, range: -1.0...1.0)
let scaled = scaler.transform(features)
// [[-1.0], [0.0], [1.0]]
```

## Topics

### Scaler
- ``FeatureScaler``

### Related
- <doc:Machine-Learning-Primer>
