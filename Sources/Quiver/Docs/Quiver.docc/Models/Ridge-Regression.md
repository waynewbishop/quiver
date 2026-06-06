# Ridge Regression

Shrinking regression coefficients with a penalty that curbs overfitting.

## Overview

Ridge is a regularized alternative to `LinearRegression` — it uses the same linear model, then adds a penalty on the size of the coefficients to curb [overfitting](<doc:Machine-Learning-Primer>). The penalty is the squared [magnitude](<doc:Vector-Operations>) of the coefficients, and its strength is set by a single parameter, `lambda`. As `lambda` grows, the penalty bites harder, shrinking the coefficients toward zero and making the model less sensitive to the noise and collinearity in the training data.

The two models are interchangeable in code, because both conform to the `Regressor` protocol — the same `fit` and `predict` — so Ridge drops into any pipeline written for `LinearRegression` without structural changes. They sit on a spectrum in concept too: at a `lambda` of zero the penalty vanishes and ridge reproduces ordinary least squares exactly, while raising `lambda` trades a little accuracy on the training data for steadier predictions on data the model has never seen.

> Note: This page documents the `Ridge` model. For the concept behind it — why a penalty curbs overfitting, what shrinkage does to the coefficients, and how to choose `lambda` — see the <doc:Regularization-Primer>.

> Important: Ridge assumes the features share a scale. The penalty compares coefficient magnitudes across features, so a feature measured in thousands and a feature measured in fractions cannot be penalized fairly until both are standardized. Standardize with `StandardScaler` before fitting. The intercept is never penalized — only the slopes are asked to shrink.

### How it works

Ordinary least squares minimizes the squared error alone. Ridge minimizes the squared error plus a penalty proportional to the squared size of the weights:

```
minimize   (1/n)‖Xθ − y‖²  +  λ‖θ‖²
```

The dial `lambda` sets how much the penalty counts. At `lambda` of zero the penalty vanishes and the result is ordinary least squares — exactly. The <doc:Regularization-Primer> walks through what the penalty does to the coefficients and why; this page is the model that applies it. Ridge is fit by gradient descent on the penalized objective, sharing the optimizer behind <doc:Gradient-Descent>, so the same convergence and divergence behavior applies here.

### Fitting a model

The `fit(features:targets:lambda:learningRate:maxIterations:tolerance:intercept:)` static method runs the penalized descent and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable.

```swift
import Quiver

// Houses described by square footage and bedroom count, with sale prices.
let features: [[Double]] = [
    [1400, 3], [1600, 3], [1700, 2], [1875, 3], [1100, 2],
    [1550, 2], [2350, 4], [2450, 4], [1425, 3], [1700, 3]
]
let prices = [245000.0, 312000, 279000, 308000, 199000,
              219000, 405000, 324000, 319000, 255000]

// Standardize first — the penalty compares coefficient magnitudes.
let scaler = StandardScaler.fit(features: features)
let scaled = scaler.transform(features)

let model = try Ridge.fit(features: scaled, targets: prices, lambda: 0.1)
print(model)
// Ridge: 2 features, λ=0.1, converged in 386 iterations (loss: 1146834644.9318)
```

The `lambda` parameter is the only one specific to ridge. The remaining parameters — `learningRate`, `maxIterations`, `tolerance`, `intercept` — match `GradientDescent` and carry the same defaults, calibrated for standardized features.

### Recovering ordinary least squares

The clearest way to see that ridge is least squares plus one term is to set `lambda` to zero. The penalty drops out, and the fit approaches the same coefficients the closed-form normal equation produces. Continuing with the `scaled` features from above:

```swift
let ridge = try Ridge.fit(features: scaled, targets: prices, lambda: 0.0)
let ols = try LinearRegression.fit(features: scaled, targets: prices)

ridge.coefficients  // [286487, 20994, 29008] — gradient descent, default settings
ols.coefficients    // [286500, 20484, 29518] — the closed-form answer it is approaching
```

At `lambda` of zero the two are minimizing the identical objective, so they share a minimum; the small gap is the gradient-descent solver stopping at its default tolerance rather than running to the exact closed form. Tightening `tolerance` or raising `maxIterations` closes it. Ridge departs from this fit in earnest only as `lambda` rises and the penalty begins to pull the weights in.

### Making predictions

`predict(_:)` returns continuous values for one or more samples. When the model has an intercept, the bias term is handled internally — callers pass raw feature rows, not augmented rows with a leading constant. New houses must be scaled with the same `scaler` the model trained on:

```swift
let newHouses: [[Double]] = [[1500, 3], [2200, 4]]
let predictions = model.predict(scaler.transform(newHouses))
```

A model trained on standardized inputs expects standardized inputs at prediction time, so the transform uses the scaler fit on the training data — never a fresh one fit on the new rows.

### Evaluating the fit

Ridge predictions are a plain `[Double]`, so the regression metrics that score any model score this one — there is no ridge-specific evaluation path. Scoring the model on a held-out set of houses it never trained on:

```swift
let testHouses: [[Double]] = [[1500, 3], [2000, 3], [2600, 4]]
let testPrices = [262000.0, 295000, 430000]

let predicted = model.predict(scaler.transform(testHouses))
predicted.rSquared(actual: testPrices)             // proportion of variance explained
predicted.meanSquaredError(actual: testPrices)     // average squared error
predicted.rootMeanSquaredError(actual: testPrices) // same units as price
```

The held-out R² is where the penalty earns its keep. A ridge model usually scores a little worse than ordinary least squares on the training data and a little better on data held back — the whole point of the trade.

### Organizing data with Panel

Using `Panel` keeps column names attached to the data and partitions every column by the same rows in a single call. The scaler still fits on the training matrix alone, so no held-out statistics leak into the fit:

```swift
import Quiver

let data = Panel([
    ("sqft", [1200.0, 1800, 2400, 1600, 2000, 2800, 1400, 2200, 1000, 3000]),
    ("bedrooms", [2.0, 3, 4, 3, 3, 5, 2, 4, 2, 5]),
    ("price", [180000.0, 260000, 350000, 230000, 290000, 420000, 195000, 320000, 160000, 450000])
])

let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
let scaler = StandardScaler.fit(features: train.toMatrix(columns: ["sqft", "bedrooms"]))

let model = try Ridge.fit(
    features: scaler.transform(train.toMatrix(columns: ["sqft", "bedrooms"])),
    targets: train["price"],
    lambda: 0.5
)
let heldOutR2 = model.predict(scaler.transform(test.toMatrix(columns: ["sqft", "bedrooms"]))).rSquared(actual: test["price"])
```

The `Panel` type is entirely optional — `Ridge` accepts arrays directly. See <doc:Panel> for the type itself and <doc:Panel-Workflows> for the train-test-predict pattern with named columns.

### Choosing lambda

`lambda` is a dial, and a dial needs a gauge. The operational rule is short: score a small grid of candidate values on data the model did not train on, and keep the one whose held-out error is lowest. Training error cannot make this choice — it falls as `lambda` falls, always nominating zero, the unpenalized fit. The <doc:Regularization-Primer> covers the reasoning, the bias-variance trade behind it, and how cross-validation supplies the held-out score.

The grid itself is a multiplicative ladder, not an even spacing. Because `lambda` acts on the loss across orders of magnitude, the useful candidates spread out geometrically — start at zero, then climb in roughly doubling steps until the penalty is clearly too strong. A ladder like `0, 0.01, 0.02, 0.04, 0.08, ...` up into the tens covers four orders of magnitude in a dozen fits:

```swift
import Quiver

// A doubling ladder from no penalty up into the tens.
let grid = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]

var best: (lambda: Double, error: Double)? = nil
for lambda in grid {
    let model = try Ridge.fit(features: scaled, targets: prices, lambda: lambda)
    let error = model.predict(scaler.transform(testHouses)).meanSquaredError(actual: testPrices)
    if best == nil || error < best!.error {
        best = (lambda, error)   // keep the lambda with the lowest held-out error
    }
}
```

The exact ratio between rungs is not sacred — doubling is convenient, tripling is just as valid. What matters is that the rungs multiply rather than add, so a single short ladder reaches both the gentle penalties near zero and the heavy ones that begin to underfit.

> Warning: Quiver's `lambda` is the bare coefficient on `λ‖θ‖²` added to the `1/n` mean squared error — it is not the `λ/2m` convention some textbooks write. A `lambda` value does not port across that difference: the same number means a different penalty strength under each, so a `lambda` chosen against one formula must be re-tuned, not copied, for the other.

Each candidate in that grid fits independently, so the sweep parallelizes cleanly: `Ridge` is `Sendable` and `fit` is a synchronous value-returning call, which lets each `lambda` be fit in its own task and the held-out scores gathered as they finish. See <doc:Concurrency-Primer> for the task-based pattern.

### Taming unstable coefficients

The clearest case for ridge is collinear features — two columns carrying almost the same information. Here floor area is recorded twice, in square feet and square meters, so the two columns move together almost perfectly. Ordinary least squares has no reason to prefer one over the other, so it hands one a large positive weight and the other a large negative one that nearly cancels it:

```swift
import Quiver

// Floor area recorded two ways: the columns move together almost perfectly.
let areaFeatures: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]
let areaPrices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]
let areaScaled = StandardScaler.fit(features: areaFeatures).transform(areaFeatures)

let ols = try LinearRegression.fit(features: areaScaled, targets: areaPrices)
ols.coefficients  // [344750, 608719, -512315] — intercept, then two wild opposing weights
```

Those weights are not findings about floor area; they are the model balancing two near-identical columns on a knife's edge, and a different sample would swing them just as violently the other way. Ridge penalizes large weights, so it keeps the pair small and balanced. As `lambda` rises, the two weights shrink and converge toward each other:

```swift
for lambda in [1.0, 10.0] {
    let model = try Ridge.fit(features: areaScaled, targets: areaPrices, lambda: lambda)
    print(model.coefficients)
}
// lambda = 1  → [344480, 32137, 32132]   small and balanced
// lambda = 10 → [344320,  8034,  8033]   smaller still, the slopes nearly equal
```

The intercept stays near the same value throughout because it is never penalized; only the slopes give ground. The wild `608719 / -512315` pair becomes a steady one a new sample will not overturn. This is what ridge buys on collinear data: not a better fit to the training rows, but coefficients that mean something.

### When to use ridge

Reach for ridge when ordinary least squares overfits — most often when features are collinear or nearly so, when the feature count is high relative to the number of samples, or when a model fit on a small sample needs to generalize. Collinearity is the textbook case: when two features carry almost the same information, least squares can assign them large opposing weights, and ridge keeps those weights small and balanced instead.

The decision does not have to be a guess. The `conditionNumber` of the standardized feature matrix measures how close it is to singular, which is exactly the instability ridge corrects. A small value means the features are well separated and ordinary least squares is on solid ground; a large value means they overlap and the unpenalized fit is unstable. Comparing two datasets makes the gap plain — the same sqft-and-bedrooms houses against the floor-area columns recorded twice:

```swift
import Quiver

let clean: [[Double]] = [
    [1400, 3], [1600, 3], [1700, 2], [1875, 3], [1100, 2],
    [1550, 2], [2350, 4], [2450, 4], [1425, 3], [1700, 3]
]
let collinear: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]

let cleanScaled = StandardScaler.fit(features: clean).transform(clean)
let collinearScaled = StandardScaler.fit(features: collinear).transform(collinear)

cleanScaled.transposed().multiplyMatrix(cleanScaled).conditionNumber          // 8 — well separated
collinearScaled.transposed().multiplyMatrix(collinearScaled).conditionNumber  // 402610 — near singular
```

A condition number in the low tens is fine; once it climbs into the thousands, ordinary least squares is unstable and ridge earns its keep. For ordinary, well-conditioned problems where no penalty is needed, `LinearRegression` is the simpler choice — closed-form, exact, one pass. Ridge adds value precisely when the unpenalized fit is unstable.

### Safe by design

`Ridge` follows the same immutable-struct pattern as the other regression models. The model is ready to use the moment `fit` returns, the training data stays separate from the result, and the penalized fit carries the same guarantees as the optimizer it shares.

Divergence throws rather than returning corrupted coefficients. A large `lambda` makes the fit harder for the optimizer to settle, and if the learning rate is too big for it the search overshoots — so `fit` throws `GradientDescentError` rather than returning a model with meaningless weights. A large penalty may simply call for a smaller learning rate, and the error names that problem rather than hiding it.

`Ridge` conforms to `Equatable`. Two runs on the same data with the same `lambda` produce identical coefficients, which is useful for tests and for confirming a pipeline produces stable output:

```swift
import Quiver

let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
let scaled = StandardScaler.fit(features: features).transform(features)

let run1 = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)
let run2 = try Ridge.fit(features: scaled, targets: targets, lambda: 0.1)
run1 == run2  // true
```

> Experiment: **The Quiver Notebook** is the right place to watch a penalty work. Take a small standardized dataset, fit `Ridge` across a range of `lambda` from `0.0` upward, and print the coefficients beside the held-out R² at each step. Watching every weight shrink toward zero as `lambda` grows — while the held-out R² climbs to a peak and then falls — is what makes the accuracy-for-stability trade concrete. The peak is the `lambda` worth keeping. See <doc:Quiver-Notebook>.

## Topics

### Model
- ``Ridge``

### Diagnostics
- ``Ridge/lambda``
- ``Ridge/lossHistory``
- ``Ridge/iterations``
- ``Ridge/finalLoss``
- ``Ridge/outcome``
- ``Ridge/Outcome``
- ``Ridge/learningRate``

### Evaluation
- ``Swift/Array/rSquared(actual:)``
- ``Swift/Array/meanSquaredError(actual:)``
- ``Swift/Array/rootMeanSquaredError(actual:)``

### Errors
- ``GradientDescentError``

### Related
- <doc:Regularization-Primer>
- <doc:Linear-Regression>
- <doc:Gradient-Descent>
- <doc:Feature-Scaling>
- <doc:Train-Test-Split>
