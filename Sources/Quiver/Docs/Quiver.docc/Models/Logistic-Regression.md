# Logistic Regression

Train a binary classifier that predicts class probabilities.

## Overview

Logistic regression answers a yes-or-no question with a probability. Where <doc:Linear-Regression> predicts a continuous value and <doc:Gradient-Descent> reaches that value iteratively, logistic regression predicts the *probability* that a sample belongs to class 1 — then thresholds that probability at 0.5 to decide the class. It is the workhorse binary classifier: spam or not, churn or retain, pass or fail.

The name is a known source of confusion. Despite the word *regression*, logistic regression is a **classification** model — it predicts a discrete label, not a continuous quantity. The "regression" refers to what happens underneath: the model fits a straight line in *log-odds* space, then squashes that line through the sigmoid function to turn it into a probability between 0 and 1.

### How it works

The model computes the same linear score `Xθ` that a linear model would — a weighted sum of the features — and passes it through the **sigmoid** function σ, which maps any real number into the open interval (0, 1). A large positive score becomes a probability near 1, a large negative score becomes a probability near 0, and a score of exactly 0 becomes 0.5, the decision boundary. The `sigmoid` function is covered on its own in <doc:Activation-Functions>.

Fitting follows the same path as <doc:Gradient-Descent> — start from θ = 0 and step opposite the gradient of the loss at each iteration — but the loss is **cross-entropy** (log loss) rather than squared error. The two gradients share a shape: design-matrix transpose times the residual, ∇L = (1/n)Xᵀ(σ(Xθ) − y). Only the prediction the residual is built from differs — σ(Xθ) here, the raw Xθ for least squares — which is why cross-entropy is the natural loss for a sigmoid hypothesis. The <doc:Optimization-Primer> covers why one descent algorithm serves several models at once.

### Fitting a model

The `fit(features:labels:)` static method learns the coefficient vector from training data and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable.

> Note: Classification models predict discrete categories, so labels are `[Int]`, and logistic regression accepts only the binary values `0` and `1`. For more than two classes, a multinomial model is needed; that is a separate model and out of scope here. See <doc:Machine-Learning-Primer> for the classification-versus-regression distinction.

Logistic regression is trained iteratively, so it is sensitive to feature scale. The default `learningRate` of `0.01` assumes features with unit variance — standardize first with `StandardScaler`, or the default rate fails to descend cleanly. The default rate is also deliberately conservative; on a small, well-scaled dataset a faster rate reaches the minimum in fewer iterations.

```swift
import Quiver

// 14 students: hours studied, labeled pass (1) or fail (0).
// The classes overlap near the boundary, so the fit has a finite minimum.
let hours: [[Double]] = [
    [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0],
    [8.0], [3.5], [4.5], [5.5], [2.5], [6.5], [4.0]
]
let labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

// Standardize, then fit at a rate suited to this small set.
let scaler = StandardScaler.fit(features: hours)
let model = try LogisticRegression.fit(
    features: scaler.transform(hours), labels: labels, learningRate: 0.5)

print(model)
// LogisticRegression: 1 feature, converged in 48 iterations (loss: 0.6284)
```

The printed summary reports whether the run converged or hit the iteration cap, alongside the final loss — the same one-line observability that <doc:Gradient-Descent> provides.

### Making predictions

The `predict(_:)` method classifies new samples by computing each probability and thresholding at 0.5. Query points must pass through the same fitted scaler used in training, so they live on the scale the model learned:

```swift
import Quiver

let query = scaler.transform([[6.5], [2.5]])
let predictions = model.predict(query)
// [1, 0]
```

When the caller needs the probability rather than the bare label — for threshold tuning, ranking, or cost-sensitive decisions — `predictProbabilities(_:)` returns the probability of class 1 for each sample, one value per row:

```swift
import Quiver

let probs = model.predictProbabilities(query)
// [0.693, 0.314]
```

> Note: Unlike the per-class distribution returned by `GaussianNaiveBayes.predictProbabilities(_:)`, a binary logistic model has a single degree of freedom. The returned value is P(class = 1), and P(class = 0) is its complement — one probability per sample, not a row that sums to 1.0 across classes.

Having the probability is what makes a custom threshold possible. The default `predict` accepts any sample at or above 0.5; a stricter cutoff flags only high-confidence positives, trading recall for precision. Applying a 0.8 cutoff to the same query rejects the first sample, whose probability of 0.693 cleared the default boundary but not the stricter one:

```swift
import Quiver

var flagged: [Int] = []
for probability in probs {
    flagged.append(probability >= 0.8 ? 1 : 0)
}
// flagged is [0, 0]
```

For inspecting the model in log-odds space — plotting the decision boundary, comparing per-sample margins, or thresholding somewhere other than 0.5 — `decisionFunction(_:)` returns the raw score `Xθ` *before* the sigmoid. Zero is the boundary, positive favors class 1, negative favors class 0:

```swift
import Quiver

let scores = model.decisionFunction(query)
// [0.813, -0.781] — above 0 predicts class 1, below 0 predicts class 0
```

The label, the probability, and the log-odds are three views of one quantity: `predict` is the sign of `decisionFunction`, and `predictProbabilities` is its sigmoid. The scores `0.813` and `−0.781` pass through the sigmoid to the probabilities `0.693` and `0.314` seen above, and their signs give the labels `1` and `0`. On a single-feature model, `predict` and `decisionFunction` also accept a bare value; `predictProbabilities` takes the array form, so wrap a lone query in a row to read its probability.

> Experiment: **The Quiver Notebook** is the right place to watch the sigmoid turn a line into a probability. Fit a one-feature model, sweep the input across its range, and print `decisionFunction` next to `predictProbabilities` — watching the unbounded log-odds compress into (0, 1) makes the squashing concrete. See <doc:Quiver-Notebook>.

### Standardize features

Logistic regression's default learning rate is calibrated for features with unit variance. On raw-scale features the loss-surface curvature scales with the squared feature magnitude, and the default rate stops descending cleanly. The fix is to fit a `StandardScaler` on the training features and apply it to both the training data and every prediction input — the same transform on both sides, so training and prediction never drift apart. See <doc:Feature-Scaling>.

The single most common mistake is scaling the training data but forgetting to scale the query points at prediction time. A raw query value then reads as many standard deviations from the mean, and the prediction is silently wrong. `Pipeline` removes the hazard by bundling the scaler and the model into one value whose `predict(_:)` scales raw inputs internally:

```swift
import Quiver

// Pipeline fits the scaler and the model together, then scales
// query points automatically — the forgotten-scaling mistake
// becomes impossible.
let pipeline = try Pipeline.fit(
    features: hours, labels: labels, learningRate: 0.5)

let predictions = pipeline.predict([[6.5], [2.5]])
// [1, 0]
```

See <doc:Pipeline> for the bundled training-and-prediction pattern.

### The full pipeline

A typical workflow combines data splitting, feature scaling, fitting, and evaluation.

```swift
import Quiver

// 12 applicants: credit score, debt ratio — approved (1) or denied (0).
let features: [[Double]] = [
    [619, 0.32], [502, 0.51], [740, 0.18], [688, 0.40],
    [605, 0.45], [710, 0.22], [560, 0.55], [780, 0.15],
    [640, 0.38], [533, 0.60], [700, 0.25], [615, 0.48]
]
let labels = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

// Split features and labels separately — seeds must match.
let (trainX, testX) = features.trainTestSplit(testRatio: 0.25, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.25, seed: 42)

// Scale, fit, predict, evaluate.
let scaler = StandardScaler.fit(features: trainX)
let model = try LogisticRegression.fit(
    features: scaler.transform(trainX), labels: trainY,
    learningRate: 0.3, maxIterations: 5000)
let predictions = model.predict(scaler.transform(testX))
let cm = predictions.confusionMatrix(actual: testY)
print("Accuracy: \(cm.accuracy)")
```

> Tip: For imbalanced datasets where one class is much rarer than the other, use `stratifiedSplit(labels:testRatio:seed:)` instead of `trainTestSplit` — it preserves the class ratios in both partitions.

### Grouping results by predicted class

The `predict(_:)` method returns raw class labels as `[Int]` for evaluation pipelines. When exploring results interactively, `classify(_:)` groups the inputs by their predicted label:

```swift
import Quiver

let results = model.classify(scaler.transform([[700, 0.20], [540, 0.58]]))
for group in results {
    print("Class \(group.label): \(group.count) applicants")
    for point in group {
        print("  \(point)")
    }
}
```

Each `Classification` result conforms to `Sequence` — the same Swift protocol that powers `for-in` loops across the language. Iterating a classification group gives us its data points directly, just like iterating an `Array`.

> Tip: Use `predict(_:)` when feeding results into evaluation methods like `accuracy`, `classificationReport`, or `confusionMatrix`.

### Watching the descent

Like <doc:Gradient-Descent>, a fitted model carries the full loss trajectory and an outcome flag, so convergence is observable rather than assumed. The `lossHistory` array begins with the cross-entropy at θ = 0 — which is exactly log 2, since every sample starts at probability 0.5 — and ends at `finalLoss`. The `outcome` distinguishes a converged run from one that exhausted the iteration cap.

The coefficients follow the intercept-first layout: when the model fits a bias term, `coefficients[0]` is the intercept and the remaining elements are the feature weights in input order. Reading a weight back out means indexing past the intercept.

> Important: A `.maxIterationsReached` outcome is necessary but not sufficient for trustworthiness. On *linearly separable* data — data a single boundary splits with no errors — the maximum-likelihood fit has no finite minimum. The coefficients grow without bound and the loss keeps shrinking, so the run reaches the cap by design. The predictions are still correct, but the coefficient magnitudes are arbitrary. Compare `lossHistory.first` to `lossHistory.last` to confirm meaningful descent, and treat a clean `.converged` on overlapping data as the trustworthy case.

### When to use logistic regression

Logistic regression is the right reach when the target is a binary yes-or-no label and a single boundary separates the classes reasonably well. It predicts a calibrated probability rather than a bare label, so it suits ranking, threshold tuning, and cost-sensitive decisions where the confidence of a prediction matters as much as its class. Because the decision boundary is linear in the features, the model excels when the classes are close to linearly separable and underperforms when the true boundary curves — a case better served by a non-parametric classifier such as <doc:Nearest-Neighbors-Classification>.

Two constraints define its scope. The labels must be binary; more than two classes calls for a multinomial model, which is out of scope here. And because the fit is iterative, the features must be standardized first, or the default learning rate fails to descend. Within those constraints it is the workhorse binary classifier, and the probability it returns is its defining advantage over a model that emits only a label.

### Safe by design

The `LogisticRegression` model is a Swift struct, which means it cannot be accidentally changed after creation. This design prevents several common mistakes.

The model is always ready to use, because calling `fit(features:labels:)` returns a fully trained model in one step. There is no way to create an empty model and forget to train it before making predictions. Training data also stays separate from test data: both `LogisticRegression` and `StandardScaler` are immutable once created, so fitting the scaler on training data and applying it to both sets ensures that test data never influences the scaling — a subtle but common source of [data leakage](<doc:Machine-Learning-Primer>) in ML pipelines.

Divergence is surfaced rather than hidden. When an over-large learning rate makes the loss climb, `fit` throws rather than returning a model full of meaningless coefficients, so the caller must acknowledge the failure, exactly as the closed-form <doc:Linear-Regression> throws on a singular system. And because models and their `Classification` results conform to Swift's `Equatable` protocol, verifying that two training runs produce the same model is a single expression.

## Topics

### Model
- ``LogisticRegression``
- ``LogisticRegression/Outcome``

### Training
- ``LogisticRegression/fit(features:labels:learningRate:maxIterations:tolerance:intercept:)-([[Double]],_,_,_,_,_)``
- ``LogisticRegression/fit(features:labels:learningRate:maxIterations:tolerance:intercept:)-([Double],_,_,_,_,_)``

### Prediction
- ``LogisticRegression/predict(_:)``
- ``LogisticRegression/predictProbabilities(_:)``
- ``LogisticRegression/decisionFunction(_:)->[Double]``
- ``LogisticRegression/decisionFunction(_:)->Double``
- ``LogisticRegression/classify(_:)``

### Diagnostics
- ``LogisticRegression/coefficients``
- ``LogisticRegression/featureCount``
- ``LogisticRegression/hasIntercept``
- ``LogisticRegression/learningRate``
- ``LogisticRegression/finalLoss``
- ``LogisticRegression/lossHistory``
- ``LogisticRegression/iterations``
- ``LogisticRegression/outcome``

### Classification result
- ``Classification``
- ``Classifier``

### Related
- <doc:Activation-Functions>
- <doc:Gradient-Descent>
- <doc:Machine-Learning-Primer>
- <doc:Optimization-Primer>
- <doc:Feature-Scaling>
- <doc:Pipeline>
- <doc:Nearest-Neighbors-Classification>

### Data Splitting
- ``Swift/Array/trainTestSplit(testRatio:seed:)``
- ``Swift/Array/stratifiedSplit(labels:testRatio:seed:)``
