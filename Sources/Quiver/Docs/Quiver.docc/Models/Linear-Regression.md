# Linear Regression

Train an ordinary least squares regression model.

## Overview

Linear regression finds the best-fit line — or hyperplane in higher dimensions — through training data by minimizing the sum of squared residuals. It predicts continuous values like prices, temperatures, scores, or any numerical quantity, and is the workhorse model when the relationship between features and a target is roughly linear.

> Important: Linear regression is **supervised** — every training row is paired with a known target value, and the model learns the relationship between the features and that target. Unlike clustering models like `KMeans` that discover structure on their own, linear regression needs labelled data to find anything at all.

![Scatter plot of training points with the fitted regression line passing through them](diagram-linear-regression)

### How it works

Linear regression models the relationship between features and a target as a linear equation: `ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`. The goal is to find the coefficients θ that minimize the total squared error between predicted and actual values. Quiver solves this using the **normal equation** `θ = (XᵀX)⁻¹Xᵀy`, which gives an exact closed-form solution in a single pass — no iteration, no learning rate, no convergence check. The route uses the matrix operations already shipped in Quiver: transposition, multiplication, and inversion.

The two-point case shows the closed form on numbers we can check by hand. The line `y = 1 + 2x` passes exactly through `(1, 3)` and `(2, 5)`; the normal equation recovers the intercept and slope directly:

```swift
let x = [1.0, 2.0]
let y = [3.0, 5.0]

let model = try LinearRegression.fit(features: x, targets: y)
model.intercept     // 1.0
model.coefficients  // [2.0]
```

Adding more points overdetermines the system. The normal equation then returns the line that minimizes the sum of squared vertical distances rather than passing through every point.

### Fitting a model

The `fit(features:targets:intercept:)` static method computes the optimal coefficients and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable. Single-feature regression takes a flat `[Double]`:

```swift
import Quiver

// Square footage → price
let sqft = [1200.0, 1800.0, 2400.0, 1600.0, 2000.0, 2800.0]
let price = [180000.0, 260000.0, 350000.0, 230000.0, 290000.0, 420000.0]

let model = try LinearRegression.fit(features: sqft, targets: price)
print(model)
// LinearRegression: 1 feature, intercept: -7469.39, slope: 150.41
```

Multi-feature regression takes `[[Double]]` where each row is one sample:

```swift
import Quiver

// Each row: [square footage, bedrooms]
let features: [[Double]] = [
    [1200, 2], [1800, 3], [2400, 4],
    [1600, 3], [2000, 3], [2800, 5]
]
let price = [180000.0, 260000.0, 350000.0, 230000.0, 290000.0, 420000.0]

let model = try LinearRegression.fit(features: features, targets: price)
print(model.coefficients)  // [137.84, 7162.16] — dollars per sqft, dollars per bedroom
```

### Making predictions

The `predict(_:)` method computes target values for new samples using the fitted coefficients. Each row of the input is one sample with the same features used in training:

```swift
import Quiver

let newHomes: [[Double]] = [[1800, 3], [3500, 5]]
let prices = model.predict(newHomes)
// prices in the trained model's units (dollars)
```

For single-feature models, a convenience overload accepts a flat `[Double]` directly — useful with `Array.linspace` to generate a smooth trend line across the feature range for charting.

### Evaluating the fit

Regression metrics tell us how close the model's predictions land to the actual values. R² (coefficient of determination) measures the fraction of variance explained, where 1.0 is perfect and 0.0 means the model is no better than predicting the mean. Mean squared error and its square root express the average prediction error — RMSE in the same units as the target:

```swift
import Quiver

let predictions = model.predict(features)
let r2 = predictions.rSquared(actual: price)
let rmse = predictions.rootMeanSquaredError(actual: price)
```

R² answers how well the line fits the points we trained on. A separate question is whether the slope itself is large enough — given how noisy the data is — to be confident the underlying relationship is real, rather than a pattern that happened to land in this sample. The fitted coefficients are estimates from a sample, and the same sample-versus-population thinking from the <doc:Inferential-Statistics-Primer> governs how much we should trust them.

That is what `summary` answers. The `summary(features:targets:level:)` method returns a `RegressionSummary` value carrying standard errors, p-values, confidence intervals, and adjusted R² for every coefficient. See <doc:Regression-Summary> for the full inferential vocabulary and how to read each field.

### Polynomial regression

**Polynomial regression** extends the straight-line form `y = θ₀ + θ₁x` to a curve: `y = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ`. The fit is still ordinary least squares — the columns of the design matrix are `[x, x², ..., xⁿ]` instead of independent features — so `LinearRegression.fit` solves it directly when we hand-build that matrix. The convenience path is `[Double].polyfit(x:y:degree:)`, which builds the design matrix for us and returns a `Polynomial` we can evaluate, differentiate, and compose.

Reach for `LinearRegression.fit` when standard errors and confidence intervals matter or when several features share the model; reach for `polyfit` when the input is a single variable and the curve itself is the return value. See <doc:Polynomials> for the polynomial path, the equivalence in code, and the conditioning limits that put a practical ceiling on degree.

### When the normal equation fails

The normal equation requires inverting `XᵀX`. If the features are linearly dependent — for example, including both temperature in Celsius and Fahrenheit — the matrix is [singular](<doc:Determinants-Primer>) and cannot be inverted. In this case `fit` throws `MatrixError.singular`, and the fix is to remove the redundant features before fitting. The determinant tells us in advance whether the fit will succeed:

```swift
import Quiver

let healthy: [[Double]] = [[1, 3], [2, 5]]
healthy.determinant     // -1.0 — independent columns, fit will succeed

let redundant: [[Double]] = [[1, 2], [1, 2]]
redundant.determinant   //  0.0 — duplicate rows, fit will throw
```

The same throw also stops `summary` from returning a corrupted variance-covariance matrix; the caller learns immediately that inference is not available rather than reading a struct of meaningless standard errors.

### The full pipeline

A typical workflow combines holding out test data, fitting on the training portion, and evaluating on the held-out portion. The R² computed on the training rows always flatters the fit; the honest measure is performance on rows the model has never seen:

```swift
import Quiver

let features: [[Double]] = [
    [1200, 2], [1800, 3], [2400, 4], [1600, 3], [2000, 3],
    [2800, 5], [1400, 2], [2200, 4], [1000, 2], [3000, 5]
]
let price = [180000.0, 260000.0, 350000.0, 230000.0, 290000.0,
             420000.0, 195000.0, 320000.0, 160000.0, 450000.0]

let (trainX, testX) = features.trainTestSplit(testRatio: 0.2, seed: 42)
let (trainY, testY) = price.trainTestSplit(testRatio: 0.2, seed: 42)

let model = try LinearRegression.fit(features: trainX, targets: trainY)
let heldOutR2 = model.predict(testX).rSquared(actual: testY)
```

The seeded split makes the partition reproducible — two runs with the same seed produce the same train and test rows.

### Organizing data with Panel

The same pipeline using `Panel` keeps column names attached to the data throughout and partitions every column by the same rows in a single call:

```swift
import Quiver

let data = Panel([
    ("sqft", [1200.0, 1800, 2400, 1600, 2000, 2800, 1400, 2200, 1000, 3000]),
    ("bedrooms", [2.0, 3, 4, 3, 3, 5, 2, 4, 2, 5]),
    ("price", [180000.0, 260000, 350000, 230000, 290000, 420000, 195000, 320000, 160000, 450000])
])

let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
let model = try LinearRegression.fit(
    features: train.toMatrix(columns: ["sqft", "bedrooms"]),
    targets: train["price"]
)
let heldOutR2 = model.predict(test.toMatrix(columns: ["sqft", "bedrooms"])).rSquared(actual: test["price"])
```

The `Panel` type is entirely optional. The regression model accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Panel> for the type itself and <doc:Panel-Workflows> for the train-test-predict pattern with named columns.

### When to use linear regression

Linear regression works best when the relationship between features and target is roughly linear, the features are not heavily collinear (no temperature-in-Celsius-and-Fahrenheit), and the residuals are roughly normal and constant in spread across the range of fitted values. The closed-form normal equation is exact and one pass — no learning rate to tune, no convergence to watch.

Linear regression struggles with strongly non-linear relationships (try polynomial regression or a transformation of the inputs first), with very high feature counts where matrix inversion becomes expensive, and with the kind of categorical or sparse data that violates the linearity assumption outright. The cost of the closed form is O(*n*·*f*² + *f*³), where *n* is the number of samples and *f* is the feature count. The *f*³ term is the inversion: the normal equation inverts the *f*×*f* matrix XᵀX, and Gaussian elimination over an *f*×*f* matrix touches each of its *f*² entries across *f* elimination passes, giving *f*³. That term is fixed by the feature count alone — adding samples never reduces it. `GradientDescent` avoids inversion entirely: each iteration is a single matrix-vector product, so its cost is O(*k*·*n*·*f*) for *k* iterations, linear in the feature count rather than cubic. As *f* grows, the cubic *f*³ term eventually dominates the closed form while the iterative cost stays linear in *f*, so reach for `GradientDescent` once the feature count is high enough that inversion outweighs the per-iteration overhead. When inference matters — standard errors, p-values, confidence intervals — pair `LinearRegression.fit` with `summary` and read <doc:Regression-Summary> for the interpretive vocabulary.

### Safe by design

The `LinearRegression` model follows the same immutable-struct pattern as `GaussianNaiveBayes`, `KMeans`, and `KNearestNeighbors`. The model is always ready to use after `fit`, the training data stays separate from the result, and seeded splits ensure reproducible runs.

`LinearRegression` conforms to Swift's `Equatable` protocol. When two runs use the same data, the closed-form solver returns identical coefficients:

```swift
import Quiver

let run1 = try LinearRegression.fit(features: features, targets: price)
let run2 = try LinearRegression.fit(features: features, targets: price)
run1 == run2  // true
```

This is useful for unit tests, debugging, and verifying that a pipeline produces stable output.

> Experiment: **The Quiver Notebook** is the right place to see outlier leverage. Take the workflow above, push one entry of `price` far above the rest, refit, and compare R² and the coefficients — the line bends to chase the outlier and the metric drops. The bent line and the lower R² are the signal that one point is doing disproportionate work, leverage made visible. See <doc:Quiver-Notebook>.

## Topics

### Model
- ``LinearRegression``

### Training
- ``LinearRegression/fit(features:targets:intercept:)-8lsme``
- ``LinearRegression/fit(features:targets:intercept:)-20bry``

### Evaluation
- ``Swift/Array/rSquared(actual:)``
- ``Swift/Array/meanSquaredError(actual:)``
- ``Swift/Array/rootMeanSquaredError(actual:)``

### Inference
- ``LinearRegression/summary(features:targets:level:)``

### Related
- <doc:Regression-Summary>
- <doc:Gradient-Descent>
- <doc:Polynomials>
- <doc:Pipeline>
- <doc:Machine-Learning-Primer>
