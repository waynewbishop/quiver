# Residual Model

Wrapping a fitted regressor to read the part of a signal it could not explain.

## Overview

A **residual** is what a prediction leaves behind: the observed value minus the predicted one, computed for each sample. Where a regressor answers "what value do these features predict?", the residual answers "how far did the real value land from that prediction?" — the information a single number throws away. ``ResidualModel`` wraps an already-fitted ``Regressor`` and surfaces that gap through three reads: the expected values, the residuals across a batch, and the residual for one sample.

The technique is the standard move behind residual analysis: model out the part of an outcome a set of inputs explains, then study what is left. A sale price inflated by a renovation reads the same as one inflated by location until a baseline predicts the price the square footage and bedroom count alone should produce. The gap between that prediction and the real price isolates the rest. The math name for partialling out an explained part this way is Frisch–Waugh–Lovell, and the residual is its output.

> Important: Read residuals out of sample. Fit the baseline on one block of data, then residualize a *later* block the baseline never trained on. Residuals computed on the same rows the baseline was fit on are optimistically small — the fit was tuned to those rows, so it explains them better than it will ever explain fresh data. The reading understates what the model actually misses.

### How it works

There is one operation underneath all three reads: subtract the prediction from the observation. For a feature matrix `X` and observed targets `y`, the baseline produces predictions `ŷ = predict(X)`, and the residual is `y − ŷ`, element by element. The wrapper forwards prediction to the model it holds and does the subtraction.

The matrix form of the same idea is the residual-maker `M = I − X(XᵀX)⁻¹Xᵀ`, which maps observations straight to residuals as `My`. That identity is worth knowing because it names what residualizing *is*, but it is conceptual only. We never form `M`. We compute `ŷ` with `predict` and subtract, which is cheaper and numerically steadier than building and applying an n×n matrix.

There is a separation worth making explicit here, because it governs how the type behaves on a memory-constrained device. Once `init(model:)` returns, the wrapper holds exactly one thing: the fitted baseline's coefficients. It is set at construction and never grows. The features and targets a residual is computed from are not part of that state — they arrive as arguments to `residuals(features:targets:)`, pass through the subtraction, and are gone when the call returns. Nothing accumulates inside the wrapper from one call to the next. A watch streaming a reading per second calls the same fixed-size value thousands of times without it ever holding more than the baseline it was built with.

### Wrapping a fitted model

The wrapper holds a model it is handed rather than training one of its own, the same posture ``Pipeline`` takes when it composes a stage. We fit the baseline ourselves, then pass it to `init(model:)`. That keeps the wrapper free of any training signature — any model that conforms to `Regressor` drops in, and the residual math is identical no matter how the baseline was fit. Here a linear model learns sale price from square footage on an earlier block of houses, and the wrapper holds the fitted result:

```swift
import Quiver

// Square footage (thousands) → sale price (thousands), an earlier block of houses.
let baseSqft = [1.2, 1.6, 2.0, 2.4]
let basePrice = [220.0, 260.0, 300.0, 340.0]

let baseline = try LinearRegression.fit(features: baseSqft, targets: basePrice)

// The wrapper holds only the fitted baseline now — no residual is computed yet.
// Residuals appear later, when features and observations are passed to a read.
let residualModel = ResidualModel(model: baseline)

print(residualModel)
// ResidualModel: wrapping LinearRegression
```

The block sits exactly on the line `price = 100 + 100 · sqft`, so the fit recovers an intercept of `100` and a slope of `100`. The wrapper now carries that baseline and is ready to report the gap on any later sample.

### Reading the residuals

`residuals(features:targets:)` takes both the features *and* the observed targets, because there is no `observed − predicted` without the observed value. This is the one deviation from a scaler's unary `transform(_:)` — a scaler needs only the inputs, a residual needs the answer too. Continuing with the `residualModel` above, we score a later block of houses the baseline never saw:

```swift
// A later block. Predictions follow the line; observations include the gaps.
let laterSqft: [[Double]] = [[1.4], [1.8], [2.2]]
let laterPrice = [255.0, 280.0, 360.0]

residualModel.expected(laterSqft)                                  // [240.0, 280.0, 320.0]
residualModel.residuals(features: laterSqft, targets: laterPrice) // [15.0, 0.0, 40.0]
```

Each residual is checkable by hand: the line predicts `240`, `280`, and `320`, so the gaps are `255 − 240 = 15`, `280 − 280 = 0`, and `360 − 320 = 40`. The middle house sold for exactly what its size predicts. The third sold `40` above — a gap worth flagging, the part of the price the square footage does not account for.

For a single house, `residual(features:observed:)` skips the array wrapping and returns one number:

```swift
residualModel.residual(features: [2.2], observed: 360.0)  // 40.0
```

### Reading coefficients through the wrapper

The wrapper computes no coefficients of its own. The values come straight from the model it holds — `residualModel.coefficients` reads `baseline.coefficients` and hands the array back unchanged, so the two are the same numbers from the same fit:

```swift
residualModel.coefficients  // [100.0, 100.0] — intercept, then slope, forwarded from the wrapped baseline
baseline.coefficients       // [100.0, 100.0] — the identical array, read off the model directly
```

Those forwarded coefficients *are* the model — there is nothing else to its prediction. Rendering them turns the fit into readable math the reader can check by hand:

```swift
residualModel.coefficients.asExpression(form: .inline)  // "⟨100, 100⟩"
```

Read `⟨100, 100⟩` back into a line and the baseline is `price = 100 + 100 · sqft` — the same rule the fit recovered from the earlier block. See <doc:Rendering-Math-Primer> for the rendering family this draws on.

This forwarding is decided at compile time, not at runtime. ``LinearRegression``, ``Ridge``, and ``GradientDescent`` all conform to ``Coefficients``, so the property is available on a `ResidualModel` wrapping any of them. A model with no coefficient vector has no `.coefficients` to forward, and reading it is a compile error rather than a value that surprises us at runtime. The capability follows the wrapped type, and the compiler enforces the match.

### Using it out of sample

The held-out posture is the whole point, so it deserves its own example. We fit the baseline on one period and read residuals on a strictly later one:

```swift
import Quiver

// Baseline period: houses already sold, used to fit.
let trainSqft = [1.2, 1.6, 2.0, 2.4]
let trainPrice = [220.0, 260.0, 300.0, 340.0]
let baseline = try LinearRegression.fit(features: trainSqft, targets: trainPrice)

// Later period: fresh sales the baseline never trained on.
let model = ResidualModel(model: baseline)
let recentSqft: [[Double]] = [[1.4], [1.8], [2.2]]
let recentPrice = [255.0, 280.0, 360.0]
let drift = model.residuals(features: recentSqft, targets: recentPrice)  // [15.0, 0.0, 40.0]
```

Residualizing `recentPrice` against a baseline fit on `trainPrice` gives an honest reading of what the size-to-price relationship misses on new sales. Had we residualized `trainPrice` instead, the fit would explain those rows better than it explains anything new, and the residuals would read smaller than the true gap. The <doc:Train-Test-Split> pattern supplies the earlier-and-later partition.

### When to use a residual model

Reach for `ResidualModel` when a raw signal mixes an effect we can model with one we want to study. Predicting the part we understand and subtracting it leaves the remainder isolated — the over-and-under-priced houses after size is accounted for, or the workload-explained portion of a sensor reading removed so the rest stands alone. The downstream effort classifier in <doc:Building-An-Effort-Model> is built on exactly this step.

The quality of the residual is only as good as the baseline. A held-out R² from <doc:Evaluation-Metrics> is the first check: a baseline that explains little of the target leaves residuals that mix model error with the signal worth studying. On collinear or ill-conditioned feature matrices — two columns carrying nearly the same information — ordinary least squares hands the pair large opposing weights that swing with the sample, and residuals computed from those unstable predictions inherit the instability. A stabilized regressor such as ``Ridge`` keeps the weights small and steady, so prefer it as the baseline when the features overlap. The `conditionNumber` of the standardized feature matrix is the gauge: a value in the low tens is fine, while one in the thousands signals the overlap that makes ``Ridge`` the safer baseline. The <doc:Model-Interpretation-Primer> covers reading that number before any fit.

### Safe by design

`ResidualModel` is an immutable value type. Once `init(model:)` returns, the wrapper holds its baseline and can be reused across many residual computations without copying or guarding shared state.

The wrapper conforms to `Equatable`, and equality is coefficient-equality inherited from the wrapped model. Two closed-form fits on the same data are bit-identical, so the wrappers around them compare equal. Two iteratively fit baselines match only when they converged to the same numbers:

```swift
import Quiver

let sqft = [1.2, 1.6, 2.0, 2.4]
let price = [220.0, 260.0, 300.0, 340.0]

let a = ResidualModel(model: try LinearRegression.fit(features: sqft, targets: price))
let b = ResidualModel(model: try LinearRegression.fit(features: sqft, targets: price))
a == b  // true — same closed-form coefficients
```

`ResidualModel` is also `Codable` and `Sendable` whenever its wrapped model is, so a wrapped baseline saves to disk and crosses task boundaries with the same guarantees the model carries on its own. See <doc:Model-Persistence> for the persistence path.

> Experiment: **The Quiver Notebook** is the right place to watch in-sample residuals flatter a fit. Fit a baseline on a block of houses, read the residuals on that same block, then read them again on a held-out block — the in-sample residuals cluster near zero while the held-out ones spread wider. The gap between the two is the optimism the held-out read removes. See <doc:Quiver-Notebook>.

## Topics

### Model
- ``ResidualModel``

### Reading residuals
- ``ResidualModel/expected(_:)``
- ``ResidualModel/residuals(features:targets:)``
- ``ResidualModel/residual(features:observed:)``

### Coefficients
- ``Coefficients``
- ``ResidualModel/coefficients``

### Related
- <doc:Ridge-Regression>
- <doc:Model-Interpretation-Primer>
- <doc:Evaluation-Metrics>
- <doc:Train-Test-Split>
- <doc:Building-An-Effort-Model>
