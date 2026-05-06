# Machine Learning Primer

Core vocabulary and concepts for understanding machine learning workflows in Quiver.

## Overview

Machine learning is the practice of training a program to recognize patterns in data so it can make predictions on new, unseen examples. This primer defines the vocabulary that appears throughout Quiver's classification documentation.

### Features and labels

Every supervised learning problem starts with a dataset where each row is one example and each column is one measurement. The columns the model uses to make predictions are called **features**. The column the model is trying to predict is called the **label** (also known as the target). Consider a dataset for predicting loan approval:

```swift
import Quiver

let data = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0]),
    ("balance",     [15000.0, 78000.0, 42000.0, 8000.0]),
    ("approved",    [1.0, 0.0, 0.0, 1.0])
])
```

Here, `creditScore` and `balance` are features, the information the model receives as input. `approved` is the label, the outcome we want the model to learn to predict. The model never sees the label at prediction time; it must infer the answer from the features alone.

> Note: A good mental model is features = question, label = answer. We train the model on many question-answer pairs, then ask it new questions and check whether it gives the right answers.

### Training and test data

If we evaluate a model on the same data it learned from, we get a misleadingly optimistic score, like grading a student on questions they already saw. To get an honest measure of how well the model generalizes, we split the data into two partitions:

- **Training set** — the examples the model learns from (typically 80% of the data)
- **Test set** — the examples held back for evaluation (typically 20%)

```swift
import Quiver

let features = [[720.0, 15000.0], [650.0, 78000.0], [580.0, 42000.0],
                [710.0, 8000.0], [690.0, 55000.0], [620.0, 91000.0],
                [750.0, 12000.0], [600.0, 63000.0], [680.0, 37000.0],
                [640.0, 84000.0]]

let (train, test) = features.trainTestSplit(testRatio: 0.2, seed: 42)
// train: 8 rows for learning
// test:  2 rows for evaluation
```

The `seed` parameter ensures the same split every time, making experiments reproducible. When using a `Panel`, the split is atomic, so all columns are partitioned by the same rows, so features and labels stay aligned automatically.

### Stratified splitting

When classes are imbalanced, say 95% approved and 5% denied, a random split might leave the test set with no denied examples at all. A **stratified split** preserves class proportions in both partitions:

```swift
import Quiver

let labels = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]

let (trainX, testX, trainY, testY) = features.stratifiedSplit(
    labels: labels, testRatio: 0.2, seed: 42
)
// Both partitions reflect the original class balance
```

### Data leakage

**Data leakage** occurs when information from the test set influences the training process. The most common form is fitting a preprocessor (like a scaler) on the entire dataset before splitting. If the scaler learns the minimum and maximum from all rows, including the test rows, then the training process has indirectly "seen" the test data, and evaluation results will be overly optimistic.

The fix is simple: fit on training data only, then transform both sets using the same learned statistics:

```swift
import Quiver

// Correct: fit on training data, transform both
let scaler = FeatureScaler.fit(features: trainFeatures)
let scaledTrain = scaler.transform(trainFeatures)
let scaledTest = scaler.transform(testFeatures)
```

This pattern — fit once on training data, apply everywhere — prevents leakage and gives us an honest evaluation. `Pipeline` enforces this automatically by bundling the scaler and model together, so the caller passes raw features and Pipeline handles scaling internally. See <doc:Pipeline> for details.

### Feature engineering and scaling

Raw data rarely arrives in a form that works well for models. **Feature engineering** is the process of transforming raw inputs into features that better represent the underlying patterns. This might involve combining columns (ratio of balance to income), extracting components (day of week from a timestamp), or encoding categories as numbers.

**Feature scaling** addresses a specific problem: when features have very different magnitudes, larger values can dominate the model's calculations. A credit score ranging from 300–850 and an account balance ranging from 0–250,000 are nearly six orders of magnitude apart. Scaling brings all features to a comparable range so each one contributes proportionally:

```swift
import Quiver

// Min-max scaling: transforms each column to 0–1 range
let scaler = FeatureScaler.fit(features: trainFeatures)
let scaled = scaler.transform(trainFeatures)
```

Quiver's `FeatureScaler` uses min-max normalization by default, scaling each column independently based on its observed range in the training data. For details on custom ranges and constant-column handling, see <doc:Feature-Scaling>.

### Overfitting and underfitting

A model can fail in two opposite ways:

**Overfitting** means the model has memorized the training data, including its noise and quirks, rather than learning the underlying pattern. It performs well on training data but poorly on new examples. Signs of overfitting include near-perfect training accuracy paired with significantly lower test accuracy.

**Underfitting** means the model is too simple to capture the pattern in the data. It performs poorly on both training and test data. This can happen when the model lacks the capacity to represent the relationship, or when important features are missing.

The goal is a model that generalizes, one that learns the true pattern well enough to make accurate predictions on data it has never seen. Splitting data into training and test sets (and checking both scores) is the primary tool for detecting these problems.

### Classification and regression

Supervised learning problems fall into two categories:

**Classification** predicts a discrete category: spam or not spam, approved or denied, which digit (0–9) an image contains. The label is a class identifier, and the model's output is a predicted class:

```swift
import Quiver

let features: [[Double]] = [[720, 15000], [650, 78000], [580, 42000], [710, 8000]]
let labels = [1, 0, 0, 1]

// Train on labeled examples
let model = GaussianNaiveBayes.fit(features: features, labels: labels)

// Predict a class label for a new sample
let predictions = model.predict([[690, 30000]])
print(predictions)  // [1]
```

**Regression** predicts a continuous value: tomorrow's temperature, a house's sale price, how long a user session will last. The label is a number, and the model's output is a number:

```swift
import Quiver

let sqft   = [1200.0, 1500.0, 1800.0, 2100.0]
let prices = [250000.0, 320000.0, 380000.0, 440000.0]

// Train on historical sales data
let model = try LinearRegression.fit(features: sqft, targets: prices)

// Predict a continuous value for a new listing
let prediction = model.predict([2000.0])
print(prediction)  // [~421000.0]
```

The distinction matters because evaluation metrics differ. Classification uses accuracy, precision, and recall. Regression uses measures like mean squared error and R².

### Fit and predict

Every Quiver model follows the same two-step pattern: **fit**, then **predict**. Fitting is the learning phase where the model examines the training data and builds whatever internal representation it needs. For `LinearRegression`, fitting computes the optimal coefficients. For `GaussianNaiveBayes`, it calculates the mean and variance of each feature per class. For `KNearestNeighbors`, fitting simply stores the training data (all the real work happens later). The result of `fit` is always a ready-to-use model.

Predicting is the application phase. We hand the fitted model new samples it has never seen, and it returns answers. Classification models return class labels; regression models return continuous values:

```swift
import Quiver

let features: [[Double]] = [[1, 2], [1.5, 1.8], [5, 8], [6, 9]]
let labels = [0, 0, 1, 1]

// Fit learns from training data
let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)
print(model)
// KNearestNeighbors: k=3, euclidean, 4 training points, 2 features

// Predict applies to new, unseen samples
let predictions = model.predict([[2, 2.5], [5.5, 7]])
print(predictions)  // [0, 1]
```

The model uses what it learned during fitting but never modifies itself. Calling `predict` twice on the same input always gives the same result.

Some classifiers can also return **calibrated probabilities** — a probability distribution across classes that sums to `1.0` for each sample, useful for cost-sensitive decisions and threshold tuning. `GaussianNaiveBayes.predictProbabilities(_:)` exposes this for the Naive Bayes classifier; see <doc:Naive-Bayes>.

### Evaluating models

Accuracy, the fraction of correct predictions, is the most intuitive metric, but it can be misleading. If 95% of loan applications are approved, a model that always predicts "approved" achieves 95% accuracy while providing zero useful information.

Better metrics examine the types of errors a model makes. **Precision** measures how many of the model's positive predictions were actually correct, so high precision means few false alarms. **Recall** measures how many of the actually positive examples the model caught, so high recall means few missed cases. The **F1 score** is the harmonic mean of precision and recall, balancing both concerns into a single number.

Which metric matters most depends on the cost of each error type. Missing a fraudulent transaction (low recall) is worse than flagging a legitimate one (low precision):

```swift
import Quiver

let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

// Per-class precision, recall, F1, and support in one call
let report = predictions.classificationReport(actual: actual)
print(report)
```

For a full treatment of these metrics and the `ConfusionMatrix` type, see <doc:Evaluation-Metrics>.

### Choosing an algorithm

**Gaussian Naive Bayes** trains quickly and works well with small datasets, but assumes features are independent of each other. When that assumption roughly holds, it is hard to beat as a starting point. See <doc:Naive-Bayes>.

**K-Nearest Neighbors** makes no assumptions about data distribution and classifies new points by finding the most similar training examples. The tradeoff is performance: every prediction scans the entire training set, and feature scaling is critical because `distance(to:)` is sensitive to magnitude differences. See <doc:Nearest-Neighbors-Classification>.

**Linear Regression** predicts continuous values rather than categories. Its coefficients are directly interpretable ("each additional bedroom adds $X to the price"), but it assumes a linear relationship between features and target. See <doc:Linear-Regression>.

**K-Means** is unsupervised and discovers natural groupings in data that has no labels. Useful for segmentation and anomaly detection, but we must choose the number of clusters in advance. See <doc:KMeans-Clustering>.

> Tip: Start simple: Naive Bayes for classification, Linear Regression for continuous targets, Nearest Neighbors when the decision boundary is nonlinear, K-Means for unlabeled data. The evaluation techniques in the previous section tell us whether our choice is working.

> Tip: For a course adopting Quiver, the <doc:Quiver-Notebook-For-Classrooms> page covers the fork-and-distribute model, adding custom assignments to the `examples/` folder, and pinning a specific Quiver release for the duration of a semester.

