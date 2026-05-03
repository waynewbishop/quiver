# K-Nearest Neighbors Classification

Classify samples by finding the closest training examples.

## Overview

K-Nearest Neighbors is one of the most intuitive classification algorithms. Given a new sample, it finds the `k` closest points in the training data and predicts the most common label among them. Unlike models that compute parameters during fitting, KNN defers all computation to prediction time — making fit instantaneous but prediction proportionally slower as the training set grows. This makes Nearest Neighbors a "lazy learner," in contrast to `GaussianNaiveBayes` and `LinearRegression` that do their heavy work up front.

### How it works

For each new sample, the algorithm measures the **distance** from that sample to every training point, selects the closest neighbors `(k)` by sorting those distances, and assigns the most common label among them. The algorithm's simplicity is its strength: no assumptions about how the data is distributed, no parameters to optimize, and the decision boundary adapts automatically to the shape of the data. The tradeoff is that prediction requires scanning the entire training set for every query.

![Query point surrounded by labeled neighbors with the closest k highlighted](diagram-nearest-neighbors)

### The distance connection

Nearest Neighbors relies on the same `distance(to:)` operation used throughout Quiver's vector mathematics. This is Euclidean distance — the straight-line distance between two points in n-dimensional space, computed as √Σ(aᵢ − bᵢ)². The same function powers centroid assignment in `KMeans` and similarity operations in <doc:Similarity-Operations>. Understanding this single linear algebra concept, that vectors are points in space and distance measures how far apart they are, unlocks classification, clustering, and similarity search simultaneously.

> Note: Distance builds on vector subtraction. Each (aᵢ − bᵢ) term is one element of the difference vector. For a deeper look at how vector arithmetic works geometrically, see [Vectors](https://waynewbishop.github.io/swift-algorithms/20-vectors.html) in Swift Algorithms & Data Structures.

### Fitting a model

The `fit(features:labels:k:metric:weight:)` static method stores the training data and returns a ready-to-use model. Because Nearest Neighbors is a lazy learner, fitting is instantaneous. No computation happens until prediction:

```swift
import Quiver

// Training data: 4 samples with 2 features each
let features: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8],   // class 0
    [5.0, 8.0], [6.0, 9.0]    // class 1
]
let labels = [0, 0, 1, 1]

let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)
```

### Making predictions

The `predict(_:)` method classifies new samples by finding their nearest neighbors and voting:

```swift
import Quiver

let newSamples: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
let predictions = model.predict(newSamples)
// [0, 1]
```

> Experiment: **The Quiver Notebook** is the right place to see how `k` shapes the decision. Pick a query point near a class boundary and sweep `k` from 1 to 9 — the predicted label flips when `k` is small and stabilizes as `k` grows, exactly as the bias–variance trade-off predicts. See <doc:Quiver-Notebook>.

### Choosing k

The parameter `k` determines how many training vectors the algorithm consults when classifying a new point. After measuring the distance from the new sample to every training vector, the algorithm selects the `k` closest ones and uses their labels to vote on the prediction. A higher `k` means more vectors influence the decision, while a lower `k` means fewer, potentially just one, determine the outcome.

The value of `k` controls the tradeoff between sensitivity and smoothness. A small `k` (e.g., 1 or 3) is sensitive to local patterns, capturing fine-grained boundaries but may [overfit](<doc:Machine-Learning-Primer>) to noisy data points. A large `k` (e.g., 15 or 21) produces smoother decision boundaries that are more robust to noise but may miss local structure. Choosing an odd value avoids ties in binary classification: with two classes and `k=4`, a 2-2 split requires a tiebreaker, while `k=3` guarantees one class always wins. A common starting point is `k = √n` where `n` is the number of training samples, rounded to the nearest odd number.

### Distance metrics

Quiver supports two distance metrics via the `DistanceMetric` enum:

**Euclidean distance** (default) measures straight-line distance between points. It works well when features have similar scales, but can be dominated by high-magnitude features when scales differ. `StandardScaler` is the recommended choice for distance-based classifiers because it centers each feature at zero with unit variance, preventing any single feature from dominating the distance calculation. `FeatureScaler` (min-max scaling) is an alternative when a bounded [0, 1] range is preferred:

```swift
import Quiver

// Learn mean/std from training data
let scaler = StandardScaler.fit(features: trainX)

// Scale training features to zero mean, unit variance
let model = KNearestNeighbors.fit(
    features: scaler.transform(trainX),
    labels: trainY,
    k: 5,
    metric: .euclidean
)

// Scale test data using training statistics (prevents data leakage)
let predictions = model.predict(scaler.transform(testX))
```

**Cosine distance** measures the angle between vectors, ignoring their magnitude. It is preferred for text embeddings and other cases where direction matters more than scale:

```swift
import Quiver

let model = KNearestNeighbors.fit(
    features: embeddings,
    labels: categories,
    k: 5,
    metric: .cosine
)
```

> Note: Cosine similarity measures how closely two vectors point in the same direction, so a high score means similar. **Cosine distance** flips this: `1 − similarity`, so a low score means similar. Nearest Neighbors uses the distance form because the algorithm looks for the smallest values to find the closest neighbors. For more on cosine similarity, see <doc:Similarity-Operations>.

### Vote weighting

By default, each neighbor gets one vote (`VoteWeight/uniform`). With `k: 3`, if the three nearest neighbors have labels [0, 1, 1], label 1 wins 2–1 regardless of how close or far each neighbor is.

With `VoteWeight/distance`, closer neighbors get more influence, so their vote is weighted by `1 / distance`. Consider a new sample where the three nearest neighbors are:

- Label 0 at distance 0.1 → weight 10.0
- Label 1 at distance 0.8 → weight 1.25
- Label 1 at distance 0.9 → weight 1.11

Under uniform voting, label 1 wins 2–1. Under distance weighting, label 0 wins 10.0 to 2.36, because the single very close neighbor outweighs two distant ones. This matters near decision boundaries where the closest point should have the strongest say:

```swift
import Quiver

// Distance-weighted: a single very close neighbor can outweigh
// several distant neighbors from another class
let model = KNearestNeighbors.fit(
    features: features,
    labels: labels,
    k: 5,
    weight: .distance
)
```

### The full pipeline

A typical workflow combines data splitting, optional scaling, model fitting, and evaluation:

```swift
import Quiver

// 10 flowers: petal length, petal width
let features: [[Double]] = [
    [1.4, 0.2], [1.3, 0.2], [1.5, 0.1], [4.5, 1.5],
    [4.7, 1.4], [4.9, 1.5], [5.1, 1.8], [1.6, 0.2],
    [5.9, 2.1], [4.0, 1.3]
]

// 0 = setosa, 1 = versicolor/virginica
let labels = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1]

// Split
let (trainX, testX) = features.trainTestSplit(testRatio: 0.3, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.3, seed: 42)

// Scale, fit, predict, evaluate
let scaler = StandardScaler.fit(features: trainX)
let model = KNearestNeighbors.fit(
    features: scaler.transform(trainX),
    labels: trainY,
    k: 3
)
let predictions = model.predict(scaler.transform(testX))
let cm = predictions.confusionMatrix(actual: testY)
print("Accuracy: \(cm.accuracy)")
```

### Organizing data with Panel

The same pipeline using `Panel` eliminates the need to split features and labels separately. One split keeps all columns aligned automatically:

```swift
import Quiver

let data = Panel([
    ("petalLength", [1.4, 1.3, 1.5, 4.5, 4.7, 4.9, 5.1, 1.6, 5.9, 4.0]),
    ("petalWidth",  [0.2, 0.2, 0.1, 1.5, 1.4, 1.5, 1.8, 0.2, 2.1, 1.3]),
    ("species",     [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
])

// One split — features and labels stay aligned without matching seeds
let (train, test) = data.trainTestSplit(testRatio: 0.3, seed: 42)
let featureColumns = ["petalLength", "petalWidth"]

// Scale, fit, predict, evaluate — same API
let scaler = StandardScaler.fit(features: train.toMatrix(columns: featureColumns))
let model = KNearestNeighbors.fit(
    features: scaler.transform(train.toMatrix(columns: featureColumns)),
    labels: train.labels("species"),
    k: 3
)
let predictions = model.predict(scaler.transform(test.toMatrix(columns: featureColumns)))
let cm = predictions.confusionMatrix(actual: test.labels("species"))
print("Accuracy: \(cm.accuracy)")
```

`Panel` is entirely optional. The classifier accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Panel> for details.

> Tip: When scaling is part of the workflow, `Pipeline` bundles the scaler and model into a single value type. It scales inputs automatically at prediction time and encodes both as one JSON blob. See <doc:Pipeline> for details.

### Structured results with classify

The `predict(_:)` method returns raw class labels as `[Int]` — ideal for evaluation metrics like `accuracy` and `classificationReport`. When exploring results interactively, `classify(_:)` groups the inputs by their predicted label, returning `Classification` objects that pair each label with its assigned points:

```swift
import Quiver

let results = model.classify([[2.0, 2.5], [5.5, 7.0], [6.0, 8.0]])
for group in results {
    print("Class \(group.label): \(group.count) points")
    for point in group {
        print("  \(point)")
    }
}
```

Each `Classification` result conforms to `Sequence` — the same Swift protocol that powers `for-in` loops across the language. Iterating a classification group gives you its data points directly, just like iterating an `Array`.

> Tip: Use `predict(_:)` when feeding results into evaluation methods like `accuracy`, `classificationReport`, or `confusionMatrix`.

### When to use Nearest Neighbors

Nearest Neighbors works best when:
- The dataset is small to medium (hundreds to low thousands of samples)
- The decision boundary is irregular and hard to model parametrically
- There is no strong prior about data distribution
- Interpretability matters, because it is easy to explain "these are the 5 most similar cases"

Nearest Neighbors struggles with large datasets (prediction scans every training point), high-dimensional data (the "curse of dimensionality" makes distances less meaningful), and features on different scales (use `StandardScaler` to mitigate).

### Safe by design

`KNearestNeighbors` follows the same immutable-struct pattern as `GaussianNaiveBayes` and `LinearRegression`. The model is always ready to use after `fit`, training data stays separate from test data, and reproducible splits ensure consistent results. Models and their `Classification` results also conform to Swift's `Equatable` protocol.

## Topics

### Model
- ``KNearestNeighbors``

### Training
- ``KNearestNeighbors/fit(features:labels:k:metric:weight:)``

### Prediction
- ``KNearestNeighbors/predict(_:)``
- ``KNearestNeighbors/classify(_:)``

### Classification result
- ``Classification``
- ``Classifier``

### Configuration
- ``DistanceMetric``
- ``VoteWeight``

### Related
- <doc:Pipeline>
- <doc:Feature-Scaling>
- <doc:Machine-Learning-Primer>
- <doc:Naive-Bayes>
- <doc:Linear-Regression>
