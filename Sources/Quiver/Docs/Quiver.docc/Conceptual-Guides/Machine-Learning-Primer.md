# Machine Learning Primer

Core vocabulary and concepts for understanding machine learning workflows in Quiver.

## Overview

Machine learning trains programs to recognize patterns in data, allowing them to make predictions on unseen examples. This primer defines the vocabulary we use throughout Quiver's classification documentation.

### Features and labels

Every supervised learning problem starts with a dataset where each row is an example and each column is a measurement. Columns the model uses to make predictions are **features**; the column we try to predict is the **label** (or target). Consider a dataset for predicting loan approval:

```swift
import Quiver

let data = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0]),
    ("balance",     [15000.0, 78000.0, 42000.0, 8000.0]),
    ("approved",    [1.0, 0.0, 0.0, 1.0])
])
```

`creditScore` and `balance` are features—input information—and `approved` is the label—the target outcome. The model infers the answer from the features alone; the label is hidden at prediction time.

### Training and test data

Evaluating a model on the data it learned from produces unreliable results, much like grading students on questions they have already seen. To measure generalization, we split data into two partitions:

- **Training set**: Examples the model learns from (typically 80%).
- **Test set**: Examples held back for evaluation (typically 20%).

```swift
import Quiver

let features = [[720.0, 15000.0], [650.0, 78000.0], [580.0, 42000.0],
                [710.0, 8000.0], [690.0, 55000.0], [620.0, 91000.0],
                [750.0, 12000.0], [600.0, 63000.0], [680.0, 37000.0],
                [640.0, 84000.0]]

let (train, test) = features.trainTestSplit(testRatio: 0.2, seed: 42)
```

The `seed` ensures reproducible splits. When using a Panel, splitting is atomic: all columns are partitioned by the same rows, keeping features and labels aligned automatically.

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

**Data leakage** occurs when test set information influences the training process. The most common form is fitting a preprocessor—like a scaler—on the entire dataset before splitting. If the scaler learns the minimum and maximum from all rows, including the test rows, the training process indirectly "sees" the test data, making evaluation results overly optimistic.

Fit on training data only, then transform both sets using the same statistics:

```swift
import Quiver

// Correct: fit on training data, transform both
let scaler = StandardScaler.fit(features: trainFeatures)
let scaledTrain = scaler.transform(trainFeatures)
let scaledTest = scaler.transform(scaledTestFeatures)
```

This pattern—fit once on training data, apply everywhere—prevents leakage and ensures honest evaluation. The Pipeline type enforces this automatically by bundling the scaler and model, so raw inputs are scaled internally and the forgotten-scaling mistake becomes impossible.

### Feature engineering and scaling

Raw data rarely arrives ready for models. **Feature engineering** transforms raw inputs into features that better represent underlying patterns: combining columns (e.g., balance-to-income ratio), extracting components (e.g., day of week from a timestamp), or encoding categories as numbers.

Interaction terms are one crucial move. A linear model is **additive**: it combines features by adding weighted contributions (`cost = a·length + b·width`). If the target depends on a product (e.g., tiling cost depends on area = length × width), no straight line through the original columns can capture it. Computing the product `area = lengths.multiply(widths)` and handing it to the model as a new feature captures this interaction linearly.

**Feature scaling** makes inputs comparable. A credit score (300–850) and an account balance (0–250,000) are six orders of magnitude apart; scaling ensures one does not dominate the calculation.

```swift
import Quiver

// Standardization: transforms each column to zero mean and unit variance
let scaler = StandardScaler.fit(features: trainFeatures)
let scaled = scaler.transform(trainFeatures)
```

`StandardScaler` centers each column at zero with unit variance, a robust default. When features require a bounded range, `FeatureScaler` maps each column to a 0–1 interval instead.

### Overfitting and underfitting

A model fails in two ways:

- **Overfitting**: The model memorizes training noise and quirks instead of the underlying pattern. It performs well on training data but poorly on unseen data.
- **Underfitting**: The model is too simple to capture the pattern. It performs poorly on both training and test data, often due to missing features or a lack of representational capacity.

These represent the **bias-variance tradeoff**—underfitting is high bias (model is too rigid), and overfitting is high variance (model follows noise). The goal is generalization. Splitting data and comparing training and test scores is the primary tool for detecting these failures.

### Fitting remedies

The gap between training and test scores dictates the remedy:

- **Overfitting**: Simplify the model, add regularization, or reduce feature reach.
- **Underfitting**: Increase capacity by adding features, creating interaction terms, or reducing regularization.

### Classification and regression

Supervised learning problems split into two categories:

**Classification** predicts discrete categories (spam vs. not, approved vs. denied). The output is a predicted class label:

```swift
import Quiver

// Train on labeled examples
let model = GaussianNaiveBayes.fit(features: features, labels: labels)

// Predict a class label for a new sample
let predictions = model.predict([[690, 30000]])
```

**Regression** predicts continuous values (temperature, price, session length). The output is a number:

```swift
import Quiver

// Train on historical sales data
let model = try LinearRegression.fit(features: sqft, targets: prices)

// Predict a continuous value
let prediction = model.predict(2000.0)
```

### Fit and predict

Quiver models follow a two-step pattern: **fit**, then **predict**. Fitting is the learning phase where the model builds its internal representation. For `LinearRegression`, fitting computes optimal coefficients; for `KNearestNeighbors`, it stores the training data. The result is always a ready-to-use model.

Predicting is the application phase. The fitted model takes unseen samples and returns answers:

```swift
import Quiver

// Fit learns from training data
let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)

// Predict applies to new, unseen samples
let predictions = model.predict([[2, 2.5], [5.5, 7]])
```

Predict is a stateless application; identical inputs always produce identical results.

### Evaluating models

Accuracy is intuitive but can be misleading—a model that always predicts the majority class may achieve high accuracy while providing zero useful information.

Better metrics examine the types of errors: **Precision** measures the correctness of positive predictions; **Recall** measures how many actual positive examples the model captured. The **F1 score** balances both into a single number.

Which metric matters depends on the error cost. Missing fraud (low recall) is often worse than a false alarm (low precision):

```swift
import Quiver

// Per-class precision, recall, F1, and support in one call
let report = predictions.classificationReport(actual: actual)
```

### Choosing an algorithm

Start simple:

- **Naive Bayes** for classification, especially when features are roughly independent.
- **Linear Regression** for continuous targets.
- **Nearest Neighbors** when decision boundaries are non-linear (remember to scale features, as `distance(to:)` is sensitive to magnitude).
- **KMeans** to discover natural groupings in unlabeled data.

The evaluation techniques above determine if the choice is effective.

