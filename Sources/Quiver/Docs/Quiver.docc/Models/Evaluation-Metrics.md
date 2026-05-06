# Evaluation Metrics

Measure classifier performance with accuracy, precision, recall, and F1 score.

## Overview

A [classification](<doc:Machine-Learning-Primer>) model is only as useful as its evaluation. Accuracy — the fraction of correct predictions — is the most intuitive metric, but it can be deeply misleading on imbalanced datasets. If 95% of samples belong to one class, a model that always predicts that class achieves 95% accuracy while providing no useful discrimination. Precision, recall, and F1 score give a more complete picture by examining how the model handles the positive class specifically.

Quiver provides these metrics as extensions on `[Int]`, where the calling array represents predicted labels and the `actual:` parameter provides the ground truth.

### How it works

Every binary classification outcome falls into one of four categories. A **true positive** is a correct positive prediction — the model said "yes" and the answer was yes. A **false positive** is an incorrect positive prediction — the model said "yes" but the answer was no. A **true negative** is a correct negative prediction, and a **false negative** is a missed positive. All evaluation metrics are ratios of these four counts: accuracy uses all four, precision focuses on positive predictions, recall focuses on actual positives, and the F1 score balances precision and recall into a single number.

### The confusion matrix

Every binary classification metric derives from four counts — true positives, false positives, true negatives, and false negatives. The `ConfusionMatrix` struct captures all four and computes the derived metrics as properties:

```swift
import Quiver

// 8 predictions compared against ground truth
let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

// Build a confusion matrix from predictions and actual labels
let cm = predictions.confusionMatrix(actual: actual)

cm.truePositives   // 3 — correct positive predictions
cm.falsePositives  // 1 — incorrectly predicted positive
cm.trueNegatives   // 3 — correct negative predictions
cm.falseNegatives  // 1 — missed positive
cm.accuracy        // 0.75
cm.precision       // Optional(0.75)
cm.recall          // Optional(0.75)
cm.f1Score         // Optional(0.75)
```

> Experiment: **The Quiver Notebook** is the right place to see why we report precision and recall rather than accuracy alone. Flip a few correct predictions to wrong ones and watch the metrics move in different directions — false positives hurt precision, false negatives hurt recall, and the F1 score lands between them. See <doc:Quiver-Notebook>.

### Type safety over silent failures

In some ML libraries, computing precision when the model predicts no positives silently returns 0.0. This hides a critical problem — the model is not making any positive predictions at all. A precision of 0.0 could mean "every positive prediction was wrong" or "no positive predictions were made," and the only way to tell the difference is manual inspection.

Quiver returns `nil` instead, surfacing the problem at the type level:

```swift
import Quiver

// A model that predicts all negative — a common failure mode
let predictions = [0, 0, 0, 0, 0]
let actual      = [1, 0, 1, 0, 0]

// nil signals that the metric is undefined, not zero
let p = predictions.precision(actual: actual)  // nil — no positives predicted
let r = predictions.recall(actual: actual)     // Optional(0.0) — caught 0 of 2

// Swift forces explicit handling of the undefined case
if let precision = p {
    print("Precision: \(precision)")
} else {
    print("Precision is undefined — model predicted no positives")
}
```

This design eliminates an entire class of silent bugs. When precision is `nil`, the code cannot proceed as if everything is fine — the `Optional` type requires explicit handling.

### Labeled parameters prevent argument-swap bugs

Quiver's metrics use the calling array as predictions and a labeled `actual:` parameter for ground truth. This makes argument ordering unambiguous:

```swift
import Quiver

// The predictions are always self, the ground truth is always actual:
let f1 = predictions.f1Score(actual: actual)
```

In positional APIs, swapping the two arguments silently produces wrong results with no compile-time or runtime warning. Swift's labeled parameters make this mistake structurally impossible.

### Choosing the right metric

The right metric depends on the cost of errors in the specific domain. In recall-first scenarios — malware detection, medical screening, customer churn — missing a positive case is expensive, so we optimize for catching every true positive even at the cost of some false alarms. In precision-first scenarios — spam filtering, content moderation, fraud alerts — false positives are expensive because flagging a legitimate email or freezing a valid transaction has real consequences for the user. When neither error type clearly dominates, the F1 score provides a single balanced metric. It is the harmonic mean of precision and recall, which penalizes extreme imbalances more heavily than an arithmetic mean would.

### Classification report

When evaluating a model, computing each metric individually and formatting the output is repetitive. The `classificationReport(actual:)` method computes per-class precision, recall, F1, and support in one call, along with overall accuracy and macro/weighted averages:

```swift
import Quiver

let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

print(predictions.classificationReport(actual: actual))
//               precision    recall  f1-score   support
//
//            0       0.75      0.75      0.75         4
//            1       0.75      0.75      0.75         4
//
//     accuracy                           0.75         8
//    macro avg       0.75      0.75      0.75         8
// weighted avg       0.75      0.75      0.75         8
```

Each class gets its own row with precision, recall, F1, and support (sample count). The macro average is the unweighted mean across classes — it treats every class equally regardless of size. The weighted average accounts for class imbalance by weighting each class by its support. Undefined metrics display as 0.00 in the report. The individual `precision`, `recall`, and `f1Score` methods still return `nil` for programmatic access.

### The full pipeline

A typical workflow fits a model, predicts on held-out data, and evaluates the results:

```swift
import Quiver

// 10 samples: credit score, account balance
let features: [[Double]] = [
    [720.0, 15000.0], [650.0, 78000.0], [580.0, 42000.0],
    [710.0, 8000.0], [690.0, 55000.0], [620.0, 91000.0],
    [750.0, 12000.0], [600.0, 63000.0], [680.0, 37000.0],
    [640.0, 84000.0]
]
let labels = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]

// Split into training and test sets
let (trainX, testX) = features.trainTestSplit(testRatio: 0.3, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.3, seed: 42)

// Scale, fit, predict
let scaler = StandardScaler.fit(features: trainX)
let model = GaussianNaiveBayes.fit(
    features: scaler.transform(trainX),
    labels: trainY
)
let predictions = model.predict(scaler.transform(testX))

// Evaluate on data the model never saw during training
print(predictions.classificationReport(actual: testY))
```

## Topics

### Confusion matrix
- ``ConfusionMatrix``

### Metric functions
- ``Swift/Array/confusionMatrix(actual:positiveLabel:)``
- ``Swift/Array/accuracy(actual:positiveLabel:)``
- ``Swift/Array/precision(actual:positiveLabel:)``
- ``Swift/Array/recall(actual:positiveLabel:)``
- ``Swift/Array/f1Score(actual:positiveLabel:)``
- ``Swift/Array/classificationReport(actual:)``

### Related
- <doc:Machine-Learning-Primer>
- <doc:Naive-Bayes>
- <doc:Nearest-Neighbors-Classification>
- <doc:Linear-Regression>
