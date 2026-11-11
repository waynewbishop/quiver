# Naive Bayes

Using Naive Bayes to build fast classifiers.

## Overview

Naive Bayes provides a reliable baseline for classification by applying probability to our features. It works effectively even under the assumption that features are independent, yielding a resilient classifier that is simple to train and easy to explain.

We apply **Bayes theorem** assuming features are independent of one another. Despite this assumption, the model performs well in practice, making it one of our most effective classification tools.

### How Gaussian classification works

The "Gaussian" in Gaussian Naive Bayes refers to the probability density function known as the PDF. This is the mathematical formula that defines the bell curve of a normal distribution. Given a feature value and a class mean and variance the PDF answers how likely this feature value is if the sample belongs to that class. See <doc:Working-With-Distributions> for the public Distributions.normal API the model uses internally.

During prediction the model evaluates the Gaussian PDF for every feature against every class. We then combine these likelihoods with the class prior probabilities to determine which class best explains the observed features. The prior probabilities tell us how common each class is in the training data. The class with the highest combined score wins. Class priors are simply a frequency table over the labels. See <doc:Frequency-Tables> for details.

![Three panels showing Gaussian naive Bayes: training points with per-class means and variances, the learned bell curve for each class, and a new sample classified by likelihood times prior](diagram-naive-bayes)


### Fitting a model

The `fit(features:labels:)` static method learns class statistics from training data and returns a ready-to-use model. There is no separate unfitted state: the returned struct is immediately usable.

> Note: Classification models predict discrete categories, so labels are `[Int]`, where each integer represents a class (e.g., `0` for denied, `1` for approved). To predict continuous values like prices or temperatures, a regression model is needed instead. See <doc:Machine-Learning-Primer> for more on the distinction.

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

When the caller needs **calibrated probabilities** rather than the raw log-space values (for cost-sensitive decisions, threshold tuning, or downstream probabilistic reasoning), `predictProbabilities(_:)` applies the softmax transform and returns probabilities that sum to `1.0` across classes for each row:

```swift
import Quiver

// Calibrated probabilities for two new samples
let probs = model.predictProbabilities([[2.0, 2.5], [5.5, 7.0]])
// probs[0] sums to 1.0 across classes, ditto probs[1]
// Each value at probs[i][c] is P(class = classes[c] | features[i])
```

> Experiment: **The Quiver Notebook** is the right place to watch a single feature steer a prediction. Hold every other feature fixed, sweep one through its range, and watch the class probability move. Naive Bayes treats each feature as independent evidence, and the snippet makes that visible. See <doc:Quiver-Notebook>.

The output ordering matches the model's `classes` array. The argmax of each row is identical to what `predict(_:)` would return: `predictProbabilities` adds calibrated confidence on top of the same decision.

### The full pipeline

A typical workflow combines data splitting, feature scaling, model fitting, and evaluation.

> Tip: For imbalanced datasets where one class is much rarer than another, use `stratifiedSplit(labels:testRatio:seed:)` instead of `trainTestSplit`; it preserves the class ratios in both partitions.

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
let scaler = StandardScaler.fit(features: trainX)
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
let scaler = StandardScaler.fit(features: train.toMatrix(columns: featureColumns))
let model = GaussianNaiveBayes.fit(
    features: scaler.transform(train.toMatrix(columns: featureColumns)),
    labels: train.labels("churned")
)
let predictions = model.predict(scaler.transform(test.toMatrix(columns: featureColumns)))
let cm = predictions.confusionMatrix(actual: test.labels("churned"))
print("Accuracy: \(cm.accuracy)")
```

The `Panel` type is entirely optional. The classifier accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Working-With-Panels> for the type itself and <doc:Panel-Workflows> for the train-test-predict pattern with named columns, including the typed snapshot a panel returns from `summary()`.

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

Each `Classification` result conforms to `Sequence`, the same Swift protocol that powers `for-in` loops across the language. Iterating a classification group gives us its data points directly, just like iterating an `Array`.

> Tip: Use `predict(_:)` when feeding results into evaluation methods like `accuracy`, `classificationReport`, or `confusionMatrix`.

### Safe by design

The `GaussianNaiveBayes` model is a Swift struct, which means it cannot be accidentally changed after creation. This design prevents three common mistakes:

**The model is always ready to use.** Calling `fit(features:labels:)` returns a fully trained model in one step. There is no way to create an empty model and forget to train it before making predictions.

**Training data stays separate from test data.** Both `GaussianNaiveBayes` and `StandardScaler` are immutable once created. Fitting the scaler on training data and applying it to both sets ensures that test data never influences the scaling, a subtle but common source of [data leakage](<doc:Machine-Learning-Primer>) in ML pipelines.

**Reproducible splits.** Each call to `trainTestSplit(testRatio:seed:)` uses its own seed. There is no shared random state that other code can interfere with, so the same seed always produces the same split.

**Direct comparison.** Models and their `Classification` results conform to Swift's `Equatable` protocol. Verifying that two training runs produce the same model is a single expression — no need to compare properties one at a time.

### Numerical stability

Naive Bayes multiplies together one probability for every feature in every class. With many features, these probabilities become extremely small numbers that can round to zero, causing the model to stop distinguishing between classes. Quiver handles this internally by working with logarithms, which keeps the arithmetic accurate regardless of how many features the data contains.

A second instability comes from the variance itself. The Gaussian density divides by the variance, so a feature that is constant within a class produces zero variance. That zero variance drives the density toward infinity through division by zero. Quiver guards against this by applying a small variance floor of `1e-9` during fitting: any variance below the floor is raised to it. This floor is the Gaussian event model's form of smoothing. Where count-based formulations add one to a tally to avoid a zero probability, the Gaussian model instead keeps the variance strictly positive, so the density stays finite and the class remains comparable.

> Note: `GaussianNaiveBayes` implements the Gaussian event model, which treats each feature as a continuous value drawn from a normal distribution. For features that are simply present or absent, `BernoulliNaiveBayes` takes the count-based form instead, described in the next section.

> Note: The variance calculation uses population variance (dividing by n), which is the standard approach for Gaussian Naive Bayes classifiers. With small training sets (2-4 samples per class), this slightly underestimates the true spread, but the effect is negligible for typical dataset sizes.

### Classifying yes or no features

Many useful features are not measurements at all. Whether a word ends in "-tion," whether an email contains a link, and whether a user enabled notifications are each a yes or no question, recorded as `1.0` or `0.0`. **Bernoulli Naive Bayes** models each of these features as a single event that is either present or absent. Instead of a mean and variance, it learns one number per feature per class: the probability that the feature is present.

Absence counts as evidence too. When a feature is usually present in a class, a sample without it scores lower for that class. The example below predicts the grammatical gender of French nouns from two word endings:

```swift
import Quiver

// [ends in "-tion", ends in "-age"]; 0 = masculine, 1 = feminine
let endings: [[Double]] = [
    [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],   // nation, station, plage
    [0.0, 1.0], [0.0, 1.0]                // fromage, voyage
]
let genders = [1, 1, 1, 0, 0]

let model = BernoulliNaiveBayes.fit(features: endings, labels: genders)
print(model)  // BernoulliNaiveBayes: 2 classes, 2 features, alpha: 1
```

The model accepts any number of classes. A three-gender language such as German adds a third label for neuter, and the same `fit(features:labels:alpha:)` call handles it without any other change.

### Reading feature probabilities

Each class stores its learned probabilities in `featureProbabilities`, one entry per feature, in the same order as the feature columns. Three feminine nouns appear in training, and two of them end in "-tion." The model does not report that as `2 / 3`. It applies **Laplace smoothing**, which adds one imaginary sample with the feature present and one with it absent to every class:

```
P(feature = 1 | class) = (presentCount + α) / (classCount + 2α)
```

With the default `alpha` of `1.0`, the feminine class gives `(2 + 1) / (3 + 2) = 0.6` for "-tion." The masculine class never saw the ending, so it gives `(0 + 1) / (2 + 2) = 0.25` rather than zero:

```swift
for stats in model.classes {
    print(stats)
}
// Class 0: prior 40.0%, feature probabilities [0.25, 0.75], 2 samples
// Class 1: prior 60.0%, feature probabilities [0.60, 0.40], 3 samples
```

Prediction multiplies the prior by one probability per feature, using p for a present feature and 1 − p for an absent one. For "information," which ends in "-tion" but not "-age," the feminine score is 0.6 · 0.6 · 0.6 = 0.216 and the masculine score is 0.4 · 0.25 · 0.25 = 0.025. Normalizing gives 0.216 / 0.241 ≈ 0.896:

```swift
// information, garage
let newWords: [[Double]] = [[1.0, 0.0], [0.0, 1.0]]

model.predict(newWords)  // [1, 0]
model.predictProbabilities(newWords)  // [[≈ 0.104, ≈ 0.896], [≈ 0.701, ≈ 0.299]]
```

### Smoothing unseen features

The difference between the two models shows most clearly on a feature that one class never saw. No masculine noun in training ends in "-tion," so its value is `0.0` in every masculine sample. The Gaussian model sees zero variance and raises it to the `1e-9` floor. A value of `1.0` then sits far outside that narrow bell curve, and the masculine probability for "information" underflows to exactly zero:

```swift
let gaussian = GaussianNaiveBayes.fit(features: endings, labels: genders)
gaussian.predictProbabilities([[1.0, 0.0]])  // [[0.0, 1.0]]

model.predictProbabilities([[1.0, 0.0]])  // [[≈ 0.104, ≈ 0.896]]
```

Both models choose feminine, but only the Bernoulli model keeps an honest measure of doubt. That matters whenever the probability drives a decision, such as flagging a word for review only when the model is unsure.

The `alpha` parameter sets the smoothing strength. Smaller values trust the observed counts more closely. With `alpha: 0.5`, the masculine class gives "-tion" a probability of (0 + 0.5) / (2 + 1) ≈ 0.167, and the feminine probability for "information" rises to ≈ 0.955:

```swift
let lightSmoothing = BernoulliNaiveBayes.fit(features: endings, labels: genders, alpha: 0.5)
lightSmoothing.predictProbabilities([[1.0, 0.0]])  // [[≈ 0.045, ≈ 0.955]]
```

### Choosing between the two models

The deciding question is whether a feature is measured or checked. Prices, temperatures, and credit scores are measured, so they belong to `GaussianNaiveBayes`. Word endings, keywords, and on or off settings are checked, so they belong to `BernoulliNaiveBayes`. Binary features need no scaling. Passing them through `StandardScaler` would turn each `0.0` and `1.0` into other values, and the model rejects any feature value other than exactly `0.0` or `1.0`.

The Bernoulli model breaks down in two ways. First, it records only whether a feature is present, so a word that appears three times in an email counts the same as a word that appears once. When frequency carries the signal, a yes or no feature discards it. Second, it treats every feature as independent evidence. Two features that almost always fire together, such as endings "-ion" and "-tion," count the same evidence twice and push the probabilities toward overconfidence. Choosing features that each ask a distinct question keeps the reported confidence meaningful.

## Topics

### Model
- ``GaussianNaiveBayes``
- ``GaussianNaiveBayes/ClassStats``
- ``BernoulliNaiveBayes``
- ``BernoulliNaiveBayes/ClassStats``

### Training
- ``GaussianNaiveBayes/fit(features:labels:)``
- ``BernoulliNaiveBayes/fit(features:labels:alpha:)``

### Prediction
- ``GaussianNaiveBayes/predict(_:)``
- ``GaussianNaiveBayes/classify(_:)``
- ``BernoulliNaiveBayes/predict(_:)``
- ``BernoulliNaiveBayes/classify(_:)``

### Classification result
- ``Classification``
- ``Classifier``

### Bayesian updating
- ``Bayes``
- ``BayesPrior``
- ``BayesLikelihood``
- ``BayesPosterior``

### Related
- <doc:Machine-Learning-Primer>
- <doc:Working-With-Distributions>
- <doc:Feature-Scaling>
- <doc:Working-With-Pipelines>
- <doc:Notebook-Datasets>
- ``GaussianNaiveBayes/predictLogProbabilities(_:)``
- ``GaussianNaiveBayes/predictProbabilities(_:)``
- ``BernoulliNaiveBayes/predictLogProbabilities(_:)``
- ``BernoulliNaiveBayes/predictProbabilities(_:)``

### Data Splitting
- ``Swift/Array/trainTestSplit(testRatio:seed:)``
- ``Swift/Array/stratifiedSplit(labels:testRatio:seed:)``

