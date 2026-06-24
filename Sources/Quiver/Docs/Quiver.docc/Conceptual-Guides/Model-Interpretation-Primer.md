# Model Interpretation Primer

Interpret Quiver models by analyzing their coefficients and geometric structures.

## Overview

A trained model gives us much more than a simple prediction. It reveals the internal logic it used to reach a result. We can learn to read these values to ensure our models make sound decisions.

### Coefficients as sensitivity sliders

A linear model predicts by multiplying each feature by its fitted weight and summing them:

`ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ`

Each coefficient `θᵢ` represents the slope of the target along one feature axis: the amount the prediction moves when that one feature increases by a unit while others hold still. This last part assumes the others *can* hold still, which only holds true when features are nearly independent. A large weight means the prediction reacts sharply to that input; a near-zero weight means the prediction barely moves, suggesting the feature is either irrelevant or redundant.

```swift
import Quiver

let sqft   = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

let model = try LinearRegression.fit(features: sqft, targets: prices)
// LinearRegression: 1 feature, intercept: 38000.00, slope: 110.00

model.coefficients   // [38000.0, 110.0] — intercept first, then one slope per feature
model.equation()     // "y = 38000.00 + 110.00x" — the same fit, read as algebra
```

This layout is worth holding onto. When `hasIntercept` is `true`, the first element is the bias term and the remaining elements are the feature weights in input order, so `coefficients[0]` is the intercept, not the first feature's slope. When `hasIntercept` is `false`, the array shifts down, and `coefficients[0]` becomes the first feature's slope. ``GradientDescent`` and ``Ridge`` use the same layout, making them interchangeable when we read their output. The ``Coefficients/equation()`` method renders any of them as the intercept-first formula above, the same numbers in the form `ŷ = θ₀ + θ₁x₁ + …`.

A near-zero weight does not necessarily mean the feature is unimportant. On raw units, a tiny weight can sit on a feature measured in large numbers. Under collinearity, a feature's weight can collapse toward zero simply because a correlated twin absorbed its share. Read coefficient magnitudes only after scaling, and only once collinearity has been ruled out. Comparing weights provides a ranking, not a measurement of absolute importance; the next sections show how to make that ranking honest.

### What scaling does to the units

A coefficient's unit depends entirely on whether we scaled the feature before fitting.

*   **Unscaled coefficients** are expressed in the feature's raw units. A weight of `110.0` on square footage means each additional square foot adds 110 to the predicted price: real units in, real units out.
*   **Scaled coefficients** (after ``StandardScaler``) report the change in target per **one standard deviation** of that feature, because the transform converts each feature to a z-score. This makes weights comparable in scale: a high-magnitude heart-rate signal and a low-magnitude cadence signal, which start on completely different scales, become directly comparable once both are standardized.

Comparing standardized weights gives a useful first pass at how strongly each feature drives the prediction, and standardizing is our sensible default whenever that comparison is the goal. Treat the comparison as a starting point rather than a verdict; standardized magnitudes still ignore feature correlation, and they discard the effect's sign, so two correlated inputs can split a single underlying influence.

> Note: ``StandardScaler`` centers each feature at zero with unit variance; ``FeatureScaler`` maps to a bounded [0, 1] range instead. Both expose the same `fit` / `transform` workflow.

### Diagnosing the collinearity tug-of-war

When two features carry nearly identical information (like a runner's pace and running power), the model cannot separate their contributions. This breaks our clean interpretation from the previous section: features no longer "hold still" individually, so no single coefficient describes a realizable change. How this ambiguity manifests depends on the solver and how identical the columns are.

The mechanics and cure belong to the <doc:Regularization-Primer>; here we focus on the diagnosis: how to spot the failure signature directly off the console.

### Measuring instability before the fit

Before evaluating how the solver behaves, check the matrix. The closed-form solver inverts `XᵀX`, so its conditioning determines stability. The `conditionNumber` of `XᵀX` measures proximity to singularity: the exact instability that wrecks coefficient interpretation:

```swift
import Quiver

// Two floor-area columns: square feet and square metres of the same homes.
// They track each other almost perfectly.
let collinear: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]

let scaled = StandardScaler.fit(features: collinear).transform(collinear)
scaled.transposed().multiplyMatrix(scaled).conditionNumber   // 402610 — near-singular
```

A condition number in the low tens is healthy. Once it climbs into the thousands, ordinary least squares is on unstable ground, and its coefficients can no longer be trusted. Computing this on `XᵀX` rather than on `X` is crucial: forming `XᵀX` squares the condition number, which is why the matrix the solver actually inverts degrades so much faster than the data alone suggests. The <doc:Determinants-Primer> explores this diagnostic in detail.

### Reading the failure in the coefficients

For near-identical columns, ``LinearRegression`` still returns an answer—a pathological one. The closed-form solver pushes one weight up and its mirror hard the other way, producing a large, opposing pair that predicts accurately while meaning nothing individually:

```swift
let prices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]

let ols = try LinearRegression.fit(features: scaled, targets: prices)
ols.coefficients   // a lopsided pair — roughly 608719 against -512315
```

These two numbers are not findings about floor area. They are the solver chasing quirks across two columns that should have been one. A different sample would swing them just as violently. The general signature is not "one positive, one negative," but two large weights that nearly cancel and which a different sample would throw just as hard.

By contrast, ``GradientDescent`` meets near-collinear data and converges quietly. It avoids matrix inversion, so it reports `.converged` with sound predictions. Collinearity creates a flat valley of near-equal solutions, and the walk settles somewhere along it. Predictions are reliable because every point in that valley predicts about equally well, but individual weights are not because the valley has no single bottom. Quiet success is not the same as a trustworthy coefficient. See <doc:Optimization-Primer>.

For perfectly collinear columns, where one is an exact multiple of the other, `XᵀX` becomes singular and there is no unique answer. ``LinearRegression`` throws `MatrixError.singular` rather than returning corrupted numbers:

```swift
// Pace recorded twice — minutes per km and minutes per mile
// try LinearRegression.fit(features:targets:) throws MatrixError.singular
```

This throw is honest: with no unique solution, the model provides none. Quiver's `determinant` reports `0.0` for any matrix singular to within its tolerance, and the solver throws on that same condition. A `0.0` here is a reliable advance warning that the fit will throw. See <doc:Determinants-Primer>.

### Resolving the signature

When an audit reveals this signature, resolve it via one of two pathways:

1.  **Feature selection**: Prune one of the mirrored columns. If two sensor streams tell the same physical story, the model needs only one to map the space cleanly.
2.  **Ridge regression**: Keep both columns but penalize large weights, collapsing the opposing pair into small, stable, balanced values without dropping either feature.

```swift
import Quiver

let ridge = try Ridge.fit(features: scaled, targets: prices, lambda: 1.0)
ridge.coefficients   // two small, balanced weights instead of the lopsided pair
```

The penalty's mechanism, how it shrinks weights and how to choose its strength, is the subject of the <doc:Regularization-Primer>. See <doc:Ridge-Regression> for the model itself.

### Verifying non-parametric structure

``KMeans`` and ``KNearestNeighbors`` compute no equation and expose **no coefficients**: there is no weight vector to print. They operate entirely on spatial proximity, so we validate them with geometry.

``KMeans`` reports `inertia`: the total squared distance from every point to its assigned centroid (within-cluster sum of squares). Lower inertia means tighter clusters. Inertia is not comparable across different feature sets, so we use it to choose `k` within one feature set, not to evaluate feature importance.

`elbowMethod` sweeps a range of `k` to find where extra clusters stop paying off:

```swift
import Quiver

let data: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
    [5.0, 5.0], [5.5, 4.8], [4.8, 5.2],
    [9.0, 8.0], [8.5, 8.5], [9.2, 7.8]
]

let results = KMeans.elbowMethod(data: data, kRange: 1...5, seed: 7)
for r in results {
    print("k=\(r.k): inertia=\(r.inertia)")
}
// k=1: inertia=145.63
// k=2: inertia=37.26
// k=3: inertia=1.03    ← elbow
// k=4: inertia=0.55    (diminishing returns)
// k=5: inertia=0.25
```

The sharp drop from `k=2` to `k=3` and the subsequent flattening mark the natural cluster count. `bestFit(data:k:attempts:)` guards against unlucky random starts by keeping the lowest-inertia run.

### An angular view of cohesion

For a single group of vectors, `clusterCohesion()` returns the average pairwise cosine similarity (0.0 to 1.0); higher means members point in nearly the same direction:

```swift
let groupA = [[0.8, 0.3, 0.9], [0.7, 0.4, 0.8], [0.9, 0.2, 0.9]]
groupA.clusterCohesion()   // ~0.98 — a tight, coherent group
```

> Important: `clusterCohesion()` measures angular agreement, while ``KMeans`` groups by Euclidean distance. They answer related but different questions—two vectors can point the same way yet sit far apart. Cohesion is most meaningful when features are scaled and direction is what matters. Quiver does not expose a silhouette metric; use `inertia` for Euclidean tightness and `elbowMethod` to compare whole clusterings.

### Evaluating a classifier by prediction

``KNearestNeighbors`` is a lazy learner: `fit` only stores data, and all work happens at `predict`. With no weights to inspect, we judge it by how well it classifies held-out data. Feed `predict(_:)` output into our evaluation metrics:

```swift
import Quiver

let predictions = model.predict(testFeatures)
let matrix = predictions.confusionMatrix(actual: testLabels)
// accuracy and classificationReport accept the same actual: labels
```

A scaled feature set that produces clean, well-separated classes shows high accuracy and a confusion matrix concentrated on the diagonal. Features on mismatched scales blur neighborhoods, so standardize first with ``StandardScaler``. See <doc:Nearest-Neighbors-Classification>.

### From reading a model to trusting it

Reading coefficients and geometry is the first move; trusting them is the second. The <doc:Regularization-Primer> takes the collinearity signature diagnosed here and supplies the cure: the penalty that turns a flat valley of equally-good answers into a single defensible one. The <doc:Optimization-Primer> explains why ``GradientDescent`` converges quietly onto an arbitrary point in that valley and what its loss history reveals about the walk. For the broader arc, <doc:Machine-Learning-Primer> is our map, <doc:Linear-Regression> provides our foundation, and <doc:Feature-Scaling> ensures our readings remain honest. Reading coefficients is one half of inspecting a fit; reading what the fit leaves behind is the other. <doc:Residual-Model> wraps a fitted regressor to surface the observed value minus the predicted one.

> Experiment: **The Quiver Notebook** is the right place to watch a coefficient lose its meaning. Take the two near-identical floor-area columns, standardize them, and fit ``LinearRegression``—then print `conditionNumber` and the coefficients side by side. Now swap in ``Ridge`` and sweep `lambda` upward until the opposing weights collapse into a small, balanced pair. Seeing the same data go from pathological to stable is the fastest way to feel why a coefficient is only as trustworthy as the matrix behind it. See <doc:Quiver-Notebook>.
