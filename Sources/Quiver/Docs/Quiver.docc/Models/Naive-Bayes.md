# Naive Bayes Classification

Train a Gaussian Naive Bayes classifier.

## Overview

Naive Bayes is one of the simplest and most effective classification algorithms. It applies Bayes' theorem with the **naive** assumption that features are conditionally independent given the class label. Despite this strong assumption, Naive Bayes performs surprisingly well in practice and serves as a reliable baseline for classification tasks.

### How Gaussian classification works

The **Gaussian** in Gaussian Naive Bayes refers to the probability density function (PDF) — the mathematical formula that defines the bell curve of a normal distribution. Given a feature value, a class mean, and a class variance, the PDF answers the question: How likely is this feature value if the sample belongs to this class?

During prediction, the model evaluates the Gaussian PDF for every feature against every class. It then combines these likelihoods with the class prior probabilities (how common each class is in the training data) to determine which class best explains the observed features. The class with the highest combined score wins.

![Naive Bayes Process](diagram-naive-bayes)

### Fitting a model

The `fit(features:labels:)` static method learns class statistics from training data and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable.

> Tip: Classification models predict discrete categories, so labels are `[Int]` — each integer represents a class (e.g., `0` for denied, `1` for approved). To predict continuous values like prices or temperatures, a regression model is needed instead. See <doc:Machine-Learning-Primer> for more on the distinction.

```swift
import Quiver

// Training data: 4 samples with 2 features each
let features: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8],   // class 0
    [5.0, 8.0], [6.0, 9.0]    // class 1
]
let labels = [0, 0, 1, 1]

let model = GaussianNaiveBayes.fit(features: features, labels: labels)
print(model)  // GaussianNaiveBayes: 2 classes, 2 features

// Inspect what the model learned — each class prints cleanly
for stats in model.classes {
    print(stats)
}
// Class 0: prior 50.0%, means [1.25, 1.90], 2 samples
// Class 1: prior 50.0%, means [5.50, 8.50], 2 samples
```

### Making predictions

The `predict(_:)` method classifies new samples by computing the probability of each class and selecting the most likely one:

```swift
import Quiver

let newSamples: [[Double]] = [[2.0, 2.5], [5.5, 7.0]]
let predictions = model.predict(newSamples)
// [0, 1]
```

For deeper inspection, `predictLogProbabilities(_:)` returns the raw log-probabilities for each class, which is useful for understanding how confident the model is in each prediction.

### The full pipeline

A typical workflow combines data splitting, feature scaling, model fitting, and evaluation.

> Tip: For imbalanced datasets where one class is much rarer than another, use `stratifiedSplit(labels:testRatio:seed:)` instead of `trainTestSplit` — it preserves the class ratios in both partitions.

```swift
import Quiver

// 10 customers: credit score, account balance, loyalty ratio
let features: [[Double]] = [
    [619, 15000, 0.08], [502, 78000, 0.04], [699, 0, 0.42],
    [850, 11000, 0.12], [645, 125000, 0.35], [720, 98000, 0.18],
    [410, 45000, 0.06], [780, 0, 0.50], [590, 175000, 0.10],
    [680, 62000, 0.28]
]

// 1 = churned, 0 = retained
let labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

// Split features and labels separately — seeds must match
let (trainX, testX) = features.trainTestSplit(testRatio: 0.25, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.25, seed: 42)

// Scale, fit, predict, evaluate
let scaler = FeatureScaler.fit(features: trainX)
let model = GaussianNaiveBayes.fit(features: scaler.transform(trainX), labels: trainY)
let predictions = model.predict(scaler.transform(testX))
let cm = predictions.confusionMatrix(actual: testY)
print("Accuracy: \(cm.accuracy)")
```

### Organizing data with Panel

The same pipeline using `Panel` eliminates the need to split features and labels separately. One split keeps all columns aligned automatically:

```swift
import Quiver

// Same 10 customers, organized by named columns
let data = Panel([
    ("creditScore", [619.0, 502.0, 699.0, 850.0, 645.0,
                     720.0, 410.0, 780.0, 590.0, 680.0]),
    ("balance", [15000.0, 78000.0, 0.0, 11000.0, 125000.0,
                 98000.0, 45000.0, 0.0, 175000.0, 62000.0]),
    ("loyalty", [0.08, 0.04, 0.42, 0.12, 0.35,
                 0.18, 0.06, 0.50, 0.10, 0.28]),
    ("churned", [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
])

// One split — features and labels stay aligned without matching seeds
let (train, test) = data.trainTestSplit(testRatio: 0.25, seed: 42)
let featureColumns = ["creditScore", "balance", "loyalty"]

// Scale, fit, predict, evaluate — same API
let scaler = FeatureScaler.fit(features: train.toMatrix(columns: featureColumns))
let model = GaussianNaiveBayes.fit(
    features: scaler.transform(train.toMatrix(columns: featureColumns)),
    labels: train.labels("churned")
)
let predictions = model.predict(scaler.transform(test.toMatrix(columns: featureColumns)))
let cm = predictions.confusionMatrix(actual: test.labels("churned"))
print("Accuracy: \(cm.accuracy)")
```

`Panel` is entirely optional. The classifier accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Panel> for details.

### Structured results with classify

The `predict(_:)` method returns raw class labels as `[Int]` for evaluation pipelines. When exploring results interactively, `classify(_:)` groups the inputs by their predicted label:

```swift
import Quiver

let results = model.classify([[700.0, 20000.0, 0.3], [500.0, 90000.0, 0.05]])
for group in results {
    print("Class \(group.label): \(group.count) customers")
    for point in group {
        print("  \(point)")
    }
}
```

Each `Classification` result conforms to `Sequence` — the same Swift protocol that powers `for-in` loops across the language. Iterating a classification group gives you its data points directly, just like iterating an `Array`.

> Tip: Use `predict(_:)` when feeding results into evaluation methods like `accuracy()`, `classificationReport()`, or `confusionMatrix()`.

### Safe by design

`GaussianNaiveBayes` is a Swift struct, which means it cannot be accidentally changed after creation. This design prevents three common mistakes:

**The model is always ready to use.** Calling `fit(features:labels:)` returns a fully trained model in one step. There is no way to create an empty model and forget to train it before making predictions.

**Training data stays separate from test data.** Both `GaussianNaiveBayes` and `FeatureScaler` are immutable once created. Fitting the scaler on training data and applying it to both sets ensures that test data never influences the scaling — a subtle but common source of [data leakage](<doc:Machine-Learning-Primer>) in ML pipelines.

**Reproducible splits.** Each call to `trainTestSplit(testRatio:seed:)` uses its own seed. There is no shared random state that other code can interfere with, so the same seed always produces the same split.

**Direct comparison.** Models and their `Classification` results conform to Swift's `Equatable` protocol. Verifying that two training runs produce the same model is a single expression — no need to compare properties one at a time.

### Numerical stability

Naive Bayes multiplies together one probability for every feature in every class. With many features, these probabilities become extremely small numbers that can round to zero, causing the model to stop distinguishing between classes. Quiver handles this internally by working with logarithms, which keeps the arithmetic accurate regardless of how many features the data contains.

> Tip: The variance calculation uses population variance (dividing by n), which is the standard approach for Gaussian Naive Bayes classifiers. With small training sets (2-4 samples per class), this slightly underestimates the true spread, but the effect is negligible for typical dataset sizes.

## Topics

### Model
- ``GaussianNaiveBayes``
- ``GaussianNaiveBayes/ClassStats``

### Training
- ``GaussianNaiveBayes/fit(features:labels:)``

### Prediction
- ``GaussianNaiveBayes/predict(_:)``
- ``GaussianNaiveBayes/classify(_:)``

### Classification result
- ``Classification``
- ``Classifier``

### Related
- <doc:Machine-Learning-Primer>
- ``GaussianNaiveBayes/predictLogProbabilities(_:)``

### Data Splitting
- ``Swift/Array/trainTestSplit(testRatio:seed:)``
- ``Swift/Array/stratifiedSplit(labels:testRatio:seed:)``

