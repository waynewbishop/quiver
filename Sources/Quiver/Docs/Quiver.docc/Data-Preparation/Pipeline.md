# Pipeline

Bundle a scaler and model into a single matched pair.

## Overview

The most common mistake when deploying ML models is saving the model without the scaler that normalized its training data. The model's learned boundaries exist in the scaled coordinate space. If the scaler is lost, new inputs land in a different space and every prediction is silently incorrect.

### Creating a pipeline

Build the scaler and model separately, then combine them:

```swift
import Quiver

let features: [[Double]] = [
    [1, 2], [1.5, 1.8], [1.2, 2.1],
    [5, 8], [6, 9], [5.5, 7.5]
]
let labels = [0, 0, 0, 1, 1, 1]

// Fit the scaler on training data
let scaler = StandardScaler.fit(features: features)

// Train the model on scaled data
let model = KNearestNeighbors.fit(
    features: scaler.transform(features),
    labels: labels, k: 3
)

// Bundle them together. They travel as one from here on.
let pipeline = Pipeline(scaler: scaler, model: model)
```

### Predicting with a pipeline

Pass raw, unscaled features. Pipeline applies the scaler internally:

Without Pipeline, scaling and prediction are two separate calls, and it is easy to forget the first:

```swift
let scaled = scaler.transform([[2, 3], [5, 7]])
let predictions = model.predict(scaled)
// [0, 1]
```

With Pipeline, the scaler is applied automatically:

```swift
let predictions = pipeline.predict([[2, 3], [5, 7]])
// [0, 1]
```

Both produce the same result. The difference is that Pipeline makes the wrong approach impossible. There is no way to accidentally pass unscaled data to the model because the caller never touches the scaler directly.

### Classification and regression

Pipeline works with both classifier and regressor models. The return type adjusts automatically:

```swift
// Classifier pipeline returns [Int] class labels
let classifierPipeline = Pipeline(scaler: scaler, model: knnModel)
let labels = classifierPipeline.predict(newFeatures)  // [0, 1, 1]

// Regressor pipeline returns [Double] predicted values
let regressorPipeline = Pipeline(scaler: scaler, model: regressionModel)
let prices = regressorPipeline.predict(newFeatures)  // [245000, 378000]
```

> Note: All four Quiver models work with `Pipeline`: `KNearestNeighbors`, `GaussianNaiveBayes`, `KMeans`, and `LinearRegression`. Classifier pipelines return integer labels. Regressor pipelines return continuous values.

### Persisting a pipeline

Because `Pipeline` conforms to `Codable`, the entire scaler-model pair encodes and decodes as a single JSON blob:

```swift
import Foundation

// Encode the pipeline, scaler and model together
let data = try JSONEncoder().encode(pipeline)
try data.write(to: pipelineURL)

// On next launch, decode and predict immediately
let saved = try Data(contentsOf: pipelineURL)
let restored = try JSONDecoder().decode(
    Pipeline<KNearestNeighbors>.self, from: saved
)
let predictions = restored.predict(newFeatures)
```

### When to use Pipeline

Use Pipeline when the model requires scaled features, specifically `KNearestNeighbors` and `KMeans`, which measure Euclidean distance and are sensitive to feature magnitudes. The scaler and model must stay paired for correct predictions.

For models that do not require scaling, such as `LinearRegression` and `GaussianNaiveBayes`, Pipeline is optional. Regression coefficients compensate for different magnitudes mathematically, and Naive Bayes evaluates each feature independently. These models can be persisted and used on their own.

> Tip: Even when scaling is not required, Pipeline can still be useful for consistency. A team that always uses Pipeline never has to remember which models need scalers and which do not.

### Verifying round-trip fidelity

Since Pipeline conforms to `Equatable`, verifying that encoding preserved the pipeline exactly is a single expression:

```swift
let original = Pipeline(scaler: scaler, model: model)
let data = try JSONEncoder().encode(original)
let decoded = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)

original == decoded  // true
```

> Experiment: **The Quiver Notebook** is the right place to feel why bundling the scaler and model matters. Fit a `Pipeline` on training data and predict on test data. The scaler stored inside the pipeline applies automatically. Now try the wrong approach: fit a fresh `StandardScaler` on the test data, transform the test features through it, and predict. The two prediction sets will disagree, and the second one is wrong, because the model learned boundaries in the training scaler's coordinate space, not the test scaler's. Watching the predictions diverge is the fastest way to see what data leakage feels like. See <doc:Quiver-Notebook>.

## Topics

### Related types
- ``StandardScaler``
- ``Classifier``
- ``Regressor``
