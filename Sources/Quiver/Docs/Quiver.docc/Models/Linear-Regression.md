# Linear Regression

Train an ordinary least squares regression model.

## Overview

Linear regression finds the best-fit line (or hyperplane in higher dimensions) through training data by minimizing the sum of squared residuals. We call this **ordinary least squares** (OLS) because the model's objective is to minimize the sum of the squared differences (the "least squares") between the observed target values and the values predicted by the linear model. The "ordinary" distinguishes this from more complex variations that apply weight penalties or handle non-normal error distributions. The objective function we minimize is:
```
min ‖Xθ − y‖²
```

Linear regression is the workhorse choice when the relationship between features and a target is roughly linear, predicting continuous values like prices, temperatures, or scores.

> Important: Linear regression is **supervised**: every training row is paired with a known target value, and the model learns the relationship between the features and that target. Unlike clustering models like ``KMeans`` that discover structure on their own, linear regression needs labelled data to find anything at all.

![Scatter plot of training points with the fitted regression line passing through them](diagram-linear-regression)

### How it works

We model the relationship between features and a target as a linear equation: `ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`. To find the coefficients θ that minimize our squared error, we use the **normal equation** `θ = (XᵀX)⁻¹Xᵀy`. This provides an exact closed-form solution in one pass: we need no iteration, no learning rate, and no convergence check. This approach relies entirely on the matrix operations shipped in Quiver: transposition, multiplication, and inversion.

The two-point case demonstrates this closed form on numbers we can verify by hand. The line `y = 1 + 2x` passes exactly through `(1, 3)` and `(2, 5)`; the normal equation recovers the intercept and slope directly:

```swift
let x = [1.0, 2.0]
let y = [3.0, 5.0]

let model = try LinearRegression.fit(features: x, targets: y)
model.coefficients      // [1.0, 2.0] — intercept first, then slope
model.coefficients[0]   // 1.0 — the intercept
model.coefficients[1]   // 2.0 — the slope
```

When we add more points and overdetermine the system, the normal equation returns the line that minimizes the sum of squared vertical distances rather than passing through every point.

### Fitting a model

The `fit(features:targets:intercept:)` static method computes the optimal coefficients and returns a ready-to-use model. There is no separate unfitted state; the returned struct is immediately usable. Single-feature regression takes a flat `[Double]`:

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
print(model.coefficients)  // [-6621.62, 137.84, 7162.16] — intercept, dollars per sqft, dollars per bedroom
```

### Making predictions

The `predict(_:)` method computes target values for new samples using the fitted coefficients. Each row of the input is one sample with the same features used in training:

```swift
import Quiver

let newHomes: [[Double]] = [[1800, 3], [3500, 5]]
let prices = model.predict(newHomes)
// prices in the trained model's units (dollars)
```

For single-feature models, a convenience overload accepts a flat `[Double]` directly, useful with `Array.linspace` to generate a smooth trend line across the feature range for charting.

### Evaluating the fit

Regression metrics tell us how well the model predicts. R² (coefficient of determination) explains the fraction of target variance; 1.0 is perfect, while 0.0 means the model performs no better than predicting the mean. Mean squared error and RMSE express prediction error, with RMSE appearing in the same units as the target:

```swift
import Quiver

let predictions = model.predict(features)
let r2 = predictions.rSquared(actual: price)
let rmse = predictions.rootMeanSquaredError(actual: price)
```

R² shows how well our line fits the training data. For deeper insight into whether our coefficients are statistically sound given our sample's noise, we use `summary(features:targets:level:)`. This method returns a ``RegressionSummary`` value carrying standard errors, p-values, confidence intervals, and adjusted R² for every coefficient. See Regression Summary for the full inferential vocabulary.

### Polynomial regression

**Polynomial regression** extends the straight-line form `y = θ₀ + θ₁x` to a curve: `y = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ`. The fit is still ordinary least squares (the columns of the design matrix are `[x, x², ..., xⁿ]` instead of independent features), so `LinearRegression.fit` solves it directly when we hand-build that matrix. The convenience path is `[Double].polyfit(x:y:degree:)`, which builds the design matrix for us and returns a ``Polynomial`` we can evaluate, differentiate, and compose.

Reach for `LinearRegression.fit` when standard errors and confidence intervals matter or when several features share the model; reach for `polyfit` when the input is a single variable and the curve itself is the return value. See <doc:Polynomials> for the polynomial path, the equivalence in code, and the conditioning limits that put a practical ceiling on degree.

### When the normal equation fails

The normal equation requires inverting `XᵀX`. If the features are linearly dependent (for example, including both temperature in Celsius and Fahrenheit), the matrix is [singular](<doc:Determinants-Primer>) and cannot be inverted. In this case `fit` throws `MatrixError.singular`, and we must remove redundant features before fitting. The determinant tells us in advance whether the fit will succeed:

```swift
import Quiver

let healthy: [[Double]] = [[1, 3], [2, 5]]
healthy.determinant     // -1.0 — independent columns, fit will succeed

let redundant: [[Double]] = [[1, 2], [1, 2]]
redundant.determinant   //  0.0 — duplicate rows, fit will throw
```

This prevents `summary` from returning a corrupted variance-covariance matrix, ensuring we acknowledge that inference is not available rather than reading meaningless errors.

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

The seeded split makes the partition reproducible: two runs with the same seed produce the same train and test rows.

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

Linear regression is our tool of choice when relationships are roughly linear, features are not heavily collinear, and errors are roughly normal and constant across the target range. The normal equation is exact and one-pass: no learning rate tuning or convergence checks required.

We should reach for other models when:
*   Relationships are strongly non-linear (try polynomial regression or input transformations first).
*   Feature counts are very high, making matrix inversion expensive.
*   Data is categorical or sparse, violating the linearity assumption.

For these cases, `GradientDescent` scales better, as it avoids matrix inversion and uses a per-iteration cost linear in the number of features. For inferential questions (standard errors, p-values, confidence intervals), we pair `LinearRegression.fit` with `summary`. For reading coefficients, understanding slope units, and recognizing when collinearity makes weights untrustworthy, see the Model Interpretation Primer.

### Safe by design

`LinearRegression` is an immutable struct created only through `fit`. An untrained model cannot be misused, and a fitted one cannot drift. Seeded splits ensure reproducible runs, and the model conforms to `Equatable`, making it trivial to verify that two runs produce identical results:

```swift
import Quiver

let run1 = try LinearRegression.fit(features: features, targets: price)
let run2 = try LinearRegression.fit(features: features, targets: price)
run1 == run2  // true
```

This is useful for unit tests, debugging, and verifying that a pipeline produces stable output.

> Experiment: **The Quiver Notebook** is the right place to see outlier leverage. Take the workflow above, push one entry of `price` far above the rest, refit, and compare R² and the coefficients—the line bends to chase the outlier and the metric drops. The bent line and the lower R² are the signal that one point is doing disproportionate work, leverage made visible. See [Quiver Notebook](<doc:Quiver-Notebook>).

## Topics

### Model
- ``LinearRegression``

### Training
- ``LinearRegression/fit(features:targets:intercept:)-8lsme``
- ``LinearRegression/fit(features:targets:intercept:)-20bry``

### Reading the fit
- ``Coefficients``
- ``Coefficients/equation()``

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
