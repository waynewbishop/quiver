# Evaluation Metrics

Measure classifier performance with accuracy, precision, recall, and F1 score.

## Overview

A classification model is only as useful as its evaluation. We rely on accuracy to understand performance, but a single number often hides the true behavior of our model on imbalanced data. These metrics provide a complete picture of how well the model discriminates between classes.

**Accuracy** measures the fraction of correct predictions. This metric can be misleading when one class is much more common than the others. We might reach ninety-five percent accuracy by always predicting the majority class, but such a model provides no useful information. **Precision**, **recall** and **F1** score help us see how the model handles specific classes to give us a honest assessment of performance.

Quiver provides these metrics as extensions on arrays of integers. The calling array represents the predicted labels while the actual parameter provides the ground truth.

### How it works

We categorize every binary classification outcome into one of four buckets. A true positive is a correct prediction of yes. A false positive is an incorrect prediction of yes. A true negative is a correct prediction of no. A false negative is a missed positive. Every evaluation metric is a ratio of these four counts. Accuracy uses all four, while precision focuses on positive predictions and recall focuses on actual positives. The F1 score balances precision and recall into a single number that reflects the overall performance of the model.

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

The right metric depends on the cost of errors in the specific domain. In recall-first scenarios — malware detection, medical screening, customer churn — missing a positive case is expensive, so we optimize for catching every true positive even at the cost of some false alarms. In precision-first scenarios — spam filtering, content moderation, fraud alerts — false positives are expensive because flagging a legitimate email or freezing a valid transaction has real consequences for the user. When neither error type clearly dominates, the F1 score provides a single balanced metric. The F1 score is the harmonic mean of precision and recall, which penalizes extreme imbalances more heavily than an arithmetic mean would. The difference is stark: a model with 100% precision but 1% recall scores only 0.02 under the harmonic mean, not the 0.505 an arithmetic mean would suggest. The F1 score refuses to reward a model that excels at one metric while collapsing on the other.

The precision-recall trade-off is not fixed — we control it by moving the decision threshold. A probabilistic classifier outputs a confidence for each class, and a prediction becomes positive only when that confidence crosses a threshold. Raising the threshold means we predict positive only when very confident, which lifts precision while lowering recall. Lowering it casts a wider net, raising recall at the expense of precision. In Quiver this choice happens upstream at prediction time, because the metrics here take hard `[Int]` labels rather than probabilities. `GaussianNaiveBayes.predictProbabilities` returns per-class confidences, and that is where we would apply a threshold to convert probabilities into the labels these metrics evaluate.

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
- <doc:Model-Interpretation-Primer>
- <doc:Naive-Bayes>
- <doc:Nearest-Neighbors-Classification>
- <doc:Linear-Regression>
