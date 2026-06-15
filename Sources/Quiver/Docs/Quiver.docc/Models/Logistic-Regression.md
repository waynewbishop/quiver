# Logistic Regression

Train a binary classifier that predicts class probabilities.

## Overview

Logistic regression answers a yes-or-no question with a probability. While <doc:Linear-Regression> predicts continuous values and <doc:Gradient-Descent> fits iteratively, logistic regression predicts the probability a sample belongs to class 1, then thresholds that probability at 0.5 to decide the class. It is our go-to binary classifier for spam, churn, or pass/fail scenarios.

Despite the name, logistic regression is a **classification** model. The "regression" refers to fitting a straight line in log-odds space, which we then squash through the sigmoid function to produce a probability between 0 and 1.

### How it works

We compute the linear score `Xθ`, a weighted sum of features, and pass it through the **sigmoid** function σ, which maps any real number into the open interval (0, 1). A large positive score yields a probability near 1, a large negative score near 0, and a score of exactly 0 yields 0.5, our decision boundary. The `sigmoid` function is covered on its own in <doc:Activation-Functions>.

Fitting follows the same gradient descent loop as our other models: start from θ = 0 and step opposite the gradient of our objective. Here, our objective is **cross-entropy** (log loss). The gradient $\nabla L = (1/n)X^T(\sigma(X\theta) - y)$ shares the same shape as least squares, but uses the sigmoid-transformed residual. Cross-entropy is naturally convex for this model, ensuring our descent leads us to a single global minimum. The <doc:Optimization-Primer> covers why one descent algorithm serves several models at once.

### Fitting a model

The `fit(features:labels:)` method learns the coefficient vector from training data and returns a ready-to-use model. The returned struct is immutable and immediately usable.

> Note: We predict discrete categories, so labels are `[Int]`, and we only support binary values `0` and `1`. Multinomial classification is out of scope here. See <doc:Machine-Learning-Primer> for the distinction between classification and regression.

Because we train iteratively, we are sensitive to feature scale. The default `learningRate` of `0.01` assumes features with unit variance, so standardize first with `StandardScaler`, or the default rate may struggle. On smaller, well-scaled datasets, a faster rate often reaches the minimum in fewer iterations.

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

The printed summary reports whether the run converged or hit the iteration cap, alongside the final loss: the same one-line observability that <doc:Gradient-Descent> provides.

### Making predictions

The `predict(_:)` method classifies new samples by computing each probability and thresholding at 0.5. Query points must pass through the same fitted scaler used in training, ensuring they live on the scale the model learned:

```swift
import Quiver

let query = scaler.transform([[6.5], [2.5]])
let predictions = model.predict(query)
// [1, 0]
```

When we need probabilities for threshold tuning, ranking, or cost-sensitive decisions, `predictProbabilities(_:)` returns the probability of class 1 for each sample:

```swift
import Quiver

let probs = model.predictProbabilities(query)
// [0.693, 0.314]
```

> Note: Unlike `GaussianNaiveBayes.predictProbabilities(_:)`, which returns a per-class distribution, a binary logistic model has a single degree of freedom. We return P(class = 1); P(class = 0) is its complement.

Having the probability is what makes a custom threshold possible. The default `predict` accepts any sample at or above 0.5; a stricter cutoff flags only high-confidence positives, trading recall for precision. Applying a 0.8 cutoff to the same query rejects the first sample, whose probability of 0.693 cleared the default boundary but not the stricter one:

```swift
import Quiver

var flagged: [Int] = []
for probability in probs {
    flagged.append(probability >= 0.8 ? 1 : 0)
}
// flagged is [0, 0]
```

To inspect the model in log-odds space (for plotting boundaries, margin analysis, or custom thresholding), `decisionFunction(_:)` returns the raw score `Xθ` before the sigmoid. Zero is the boundary, positive favors class 1, and negative favors class 0:

```swift
import Quiver

let scores = model.decisionFunction(query)
// [0.813, -0.781] — above 0 predicts class 1, below 0 predicts class 0
```

The label, probability, and log-odds are three views of the same quantity: `predict` is the sign of `decisionFunction`, and `predictProbabilities` is its sigmoid. On a single-feature model, `predict` and `decisionFunction` also accept a bare value; `predictProbabilities` takes the array form, so wrap a lone query in a row to read its probability.

> Experiment: **The Quiver Notebook** is the right place to watch the sigmoid turn a line into a probability. Fit a one-feature model, sweep the input across its range, and print `decisionFunction` next to `predictProbabilities`: watching the unbounded log-odds compress into (0, 1) makes the squashing concrete. See <doc:Quiver-Notebook>.

### Standardize features

Because the fit is iterative, query points must pass through the same scaler the model trained on, or raw values read as the wrong number of standard deviations and predictions become silently wrong. `Pipeline` removes this hazard by bundling the scaler and model into one value; its `predict(_:)` scales raw inputs internally, making the forgotten-scaling mistake impossible:

```swift
import Quiver

let pipeline = try Pipeline.fit(
    features: hours, labels: labels, learningRate: 0.5)

let predictions = pipeline.predict([[6.5], [2.5]])
// [1, 0]
```

See <doc:Feature-Scaling> for why standardization matters and <doc:Pipeline> for the bundled training-and-prediction pattern.

### An end-to-end run

Split, scale, fit, predict, and evaluate in one pass. The split uses a shared seed so features and labels partition the same way; see <doc:Train-Test-Split> for the mechanics.

```swift
import Quiver

// 12 applicants: credit score, debt ratio — approved (1) or denied (0).
let features: [[Double]] = [
    [619, 0.32], [502, 0.51], [740, 0.18], [688, 0.40],
    [605, 0.45], [710, 0.22], [560, 0.55], [780, 0.15],
    [640, 0.38], [533, 0.60], [700, 0.25], [615, 0.48]
]
let labels = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

let (trainX, testX) = features.trainTestSplit(testRatio: 0.25, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.25, seed: 42)

let scaler = StandardScaler.fit(features: trainX)
let model = try LogisticRegression.fit(
    features: scaler.transform(trainX), labels: trainY,
    learningRate: 0.3, maxIterations: 5000)
let predictions = model.predict(scaler.transform(testX))
let cm = predictions.confusionMatrix(actual: testY)
print("Accuracy: \(cm.accuracy)")
```

> Tip: When one class is much rarer than the other, use `stratifiedSplit(labels:testRatio:seed:)` to preserve the class ratios in both partitions.

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

Each `Classification` result conforms to `Sequence`, so iterating a group yields its data points directly.

> Tip: Use `predict(_:)` when feeding results into evaluation methods like `accuracy`, `classificationReport`, or `confusionMatrix`.

### Watching the descent

Like <doc:Gradient-Descent>, a fitted model carries the full loss trajectory and an outcome flag, making convergence observable rather than assumed. The `lossHistory` array begins with the cross-entropy at θ = 0 (exactly log 2, as every sample starts at probability 0.5) and ends at `finalLoss`. The `outcome` flag distinguishes a converged run from one that hit the iteration cap.

The trajectory shape is our diagnostic. A healthy run falls steeply at first and then flattens near the minimum; a trajectory that rises signals a learning rate too large for the data, while one that barely moves signals a rate too small. Reading the curve tells us more than the final number alone.

The coefficients follow the intercept-first layout: when we fit a bias term, `coefficients[0]` is the intercept and the remaining elements are feature weights in input order.

> Important: A `.maxIterationsReached` outcome is not automatically a failure. On linearly separable data the fit has no finite minimum, so the loss keeps shrinking and the run hits the cap by design—predictions remain correct, but coefficient magnitudes are arbitrary. Compare `lossHistory.first` to `lossHistory.last` to confirm meaningful descent.

### When to use logistic regression

Logistic regression is our tool of choice when we have a binary target and a boundary that separates classes reasonably well. It predicts a calibrated probability rather than a bare label, making it perfect for ranking, threshold tuning, and cost-sensitive decisions where confidence matters. Because the decision boundary is linear, the model excels when classes are nearly linearly separable; when the true boundary curves, a non-parametric classifier like <doc:Nearest-Neighbors-Classification> is a better choice.

Two constraints define its scope. Labels must be binary; more than two classes require a multinomial model, which is out of scope here. Because fitting is iterative, we must standardize features first, or the default learning rate fails to descend. Within these constraints, it remains our workhorse binary classifier, with its calibrated probability being its defining advantage.

### Safe by design

`LogisticRegression` is an immutable struct created only through `fit`. An untrained model cannot be misused, and a fitted one cannot drift. When an over-large learning rate makes the loss climb, `fit` throws rather than returning meaningless coefficients, ensuring we acknowledge the failure. The model and its `Classification` results conform to `Equatable`, making it trivial to confirm that two runs produce identical results.

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
