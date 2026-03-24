# Linear Regression

Train an ordinary least squares regression model.

## Overview

Linear regression finds the best-fit line (or hyperplane) through training data by minimizing the sum of squared residuals. Unlike classification models that predict discrete categories, regression models predict continuous values like prices, temperatures, scores, or any numerical quantity.

### How it works

Linear regression models the relationship between features and a target as a linear equation: ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ. The goal is to find the coefficients θ that minimize the total squared error between predicted and actual values.

Quiver solves this using the **normal equation** θ = (X'X)⁻¹X'y, which gives an exact closed-form solution. This approach uses the matrix operations already available in Quiver (transposition, multiplication, and inversion) rather than iterative gradient descent. The result is a precise answer computed in a single pass.

### Fitting a model

The `fit(features:targets:intercept:)` static method computes the optimal coefficients and returns a ready-to-use model. There is no separate unfitted state, so the returned struct is immediately usable.

> Tip: Regression models predict continuous `Double` values, so targets are `[Double]`. To predict discrete categories like "approved" or "denied", use a classification model instead. See <doc:Machine-Learning-Primer> for more on the distinction.

```swift
import Quiver

// Training data: square footage → price
let sqft   = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

let model = try LinearRegression.fit(features: sqft, targets: prices)

// Inspect the coefficients
print("Intercept: \(model.coefficients[0])")   // bias term
print("Slope: \(model.coefficients[1])")        // price per sqft
```

For single-feature regression, `fit` accepts a flat `[Double]` array directly — no need to wrap each value in `[[Double]]`. Multi-feature regression uses the standard `fit(features: [[Double]], targets:)` form shown below.

### Making predictions

The `predict(_:)` method computes target values for new samples using the fitted coefficients:

```swift
import Quiver

// Each row is one sample with the same features used in training
let newHomes: [[Double]] = [[1800], [3500]]
let prices = model.predict(newHomes)
// prices ≈ [236000, 423000]
```

For single-feature models, a convenience overload accepts a flat `[Double]` instead of wrapping each value in an array. Combined with `linspace`, this generates a smooth trend line across the feature range:

```swift
import Quiver

// Generate a trend line from x=500 to x=3500
let trendX = Array.linspace(start: 500.0, end: 3500.0, count: 50)
let trendY = model.predict(trendX)

// trendX and trendY are parallel arrays ready for charting
```

> Important: The single-feature `predict(_:)` overload requires a model trained on exactly one feature. For multi-feature models, use the standard `predict([[Double]])` form.

### Multiple features

Linear regression naturally extends to multiple features. Each feature gets its own coefficient weight:

```swift
import Quiver

// Each row: [square footage, bedrooms, age in years]
let features: [[Double]] = [
    [1200, 2, 20], [1800, 3, 10], [2400, 4, 5],
    [1600, 3, 15], [2000, 3, 8], [2800, 5, 2]
]
let targets = [180000.0, 260000.0, 350000.0,
               230000.0, 290000.0, 420000.0]

// Fit produces one coefficient per feature plus an intercept
let model = try LinearRegression.fit(features: features, targets: targets)
print("Features: \(model.featureCount)")  // 3
```

### Evaluating the fit

Regression metrics tell us how well the model's predictions match the actual values. R² (coefficient of determination) measures the fraction of variance explained, where 1.0 is perfect and 0.0 means the model is no better than predicting the mean:

```swift
import Quiver

// Predict on the training data to check how well the model fits
let predictions = model.predict(features)

// R² measures fraction of variance explained (1.0 = perfect)
let r2   = predictions.rSquared(actual: targets)

// MSE and RMSE measure average prediction error
let mse  = predictions.meanSquaredError(actual: targets)
let rmse = predictions.rootMeanSquaredError(actual: targets)

print("R²: \(r2)")      // closer to 1.0 is better
print("RMSE: \(rmse)")  // in the same units as the target
```

### The full pipeline

A typical workflow combines data splitting, model fitting, and evaluation:

```swift
import Quiver

// 10 houses: [square footage, bedrooms]
let features: [[Double]] = [
    [1200, 2], [1800, 3], [2400, 4], [1600, 3], [2000, 3],
    [2800, 5], [1400, 2], [2200, 4], [1000, 2], [3000, 5]
]
let targets = [180000.0, 260000.0, 350000.0, 230000.0, 290000.0,
               420000.0, 195000.0, 320000.0, 160000.0, 450000.0]

// Hold out 20% for evaluation — seed ensures reproducible splits
let (trainX, testX) = features.trainTestSplit(testRatio: 0.2, seed: 42)
let (trainY, testY) = targets.trainTestSplit(testRatio: 0.2, seed: 42)

// Train on 80%, predict on the held-out 20%
let model = try LinearRegression.fit(features: trainX, targets: trainY)
let predictions = model.predict(testX)

// Evaluate on data the model never saw during training
let r2 = predictions.rSquared(actual: testY)
let rmse = predictions.rootMeanSquaredError(actual: testY)
print("R²: \(r2), RMSE: \(rmse)")
```

### Organizing data with Panel

The same pipeline using `Panel` eliminates the need to split features and targets separately. One split keeps all columns aligned automatically:

```swift
import Quiver

// Named columns keep features and targets together in one structure
let data = Panel([
    ("sqft", [1200.0, 1800.0, 2400.0, 1600.0, 2000.0,
              2800.0, 1400.0, 2200.0, 1000.0, 3000.0]),
    ("bedrooms", [2.0, 3.0, 4.0, 3.0, 3.0,
                  5.0, 2.0, 4.0, 2.0, 5.0]),
    ("price", [180000.0, 260000.0, 350000.0, 230000.0, 290000.0,
               420000.0, 195000.0, 320000.0, 160000.0, 450000.0])
])

// One split partitions all columns by the same rows automatically
let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
let featureColumns = ["sqft", "bedrooms"]

// Extract feature matrix and target vector by column name
let model = try LinearRegression.fit(
    features: train.toMatrix(columns: featureColumns),
    targets: train["price"]
)

// Predict and evaluate on the held-out partition
let predictions = model.predict(test.toMatrix(columns: featureColumns))
let r2 = predictions.rSquared(actual: test["price"])
print("R²: \(r2)")
```

`Panel` is entirely optional. The regression model accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Panel> for details.

### When the normal equation fails

The normal equation requires inverting the matrix X'X. If the features are linearly dependent (for example, including both temperature in Celsius and Fahrenheit), the matrix is singular and cannot be inverted. In this case, `fit` throws `MatrixError.singular`. The fix is to remove redundant features before fitting.

To check beforehand, inspect the determinant of your feature matrix. A non-zero value means the columns are independent and the model can be fitted:

```swift
import Quiver

// Independent columns — determinant is non-zero
let healthy: [[Double]] = [[1.0, 3.0], [2.0, 5.0]]
healthy.determinant  // -1.0 → safe to fit

// Redundant columns — determinant is zero
let redundant: [[Double]] = [[1.0, 2.0], [1.0, 2.0]]
redundant.determinant  // 0.0 → fit will throw MatrixError.singular
```

> Tip: A singular matrix means the determinant is zero, because the features collapse into a lower-dimensional space and the equation has no unique solution. For a deeper look at what determinants measure and why singularity matters, see <doc:Determinants-Primer>.

### Safe by design

`LinearRegression` follows the same immutable-struct pattern as `GaussianNaiveBayes`. The model is always ready to use after `fit`, training data stays separate from test data, and reproducible splits ensure consistent results. Models support direct comparison with Swift's `Equatable` protocol', so verifying two training runs produce the same coefficients is a single expression.

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

### Related
- <doc:Machine-Learning-Primer>
- <doc:Naive-Bayes>
