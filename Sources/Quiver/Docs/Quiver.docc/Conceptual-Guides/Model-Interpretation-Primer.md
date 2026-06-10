# Model Interpretation Primer

Reading, auditing, and troubleshooting the coefficients and geometry that Quiver's models return.

## Overview

A trained model does not return arbitrary numbers — it returns geometric adjustments. Parametric models (``LinearRegression``, ``GradientDescent``, ``Ridge``, ``LogisticRegression``) expose those adjustments as **coefficients**, one weight per feature. Non-parametric models (``KNearestNeighbors``, ``KMeans``) hold no weights at all; they store positions in space and are judged by **distance and cohesion** instead.

This primer covers three reading skills: interpreting parametric coefficients, diagnosing the collinearity failure mode, and verifying non-parametric structure geometrically.

### Coefficients as sensitivity sliders

A linear model predicts by multiplying each feature by its fitted weight and summing them:

`ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ`

Each coefficient `θᵢ` is the slope of the target along one feature axis — how much the prediction moves when that one feature increases by a unit and the others hold still. That last phrase carries an assumption: the others *can* hold still, which is only true when the features are close to independent. Reading a coefficient is reading a sensitivity: a large weight means the prediction reacts sharply to that input, a near-zero weight means the prediction barely moves with this feature as the model has parameterized it.

```swift
import Quiver

let sqft   = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

let model = try LinearRegression.fit(features: sqft, targets: prices)
// LinearRegression: 1 feature, intercept: 38000.00, slope: 110.00

model.coefficients   // [38000.0, 110.0] — intercept first, then one slope per feature
```

The layout is worth holding onto, because it is easy to misread. When `hasIntercept` is `true`, the **first** element is the bias term and the remaining elements are the feature weights in input order — so `coefficients[0]` is the intercept, not the first feature's slope. ``GradientDescent`` and ``Ridge`` use the same layout, so the three regressors are interchangeable when we read their output.

A near-zero weight does not, on its own, mean the feature is unimportant. On raw units a tiny weight can sit on a feature measured in large numbers, and under collinearity a feature's weight can collapse toward zero simply because a correlated twin absorbed its share. Read coefficient magnitudes only after scaling, and only once collinearity has been ruled out — the next section is what rules it out.

### What scaling does to the units

The *unit* of a coefficient depends entirely on whether the feature was scaled before fitting.

 * **Unscaled coefficients** are expressed in the feature's raw units. A weight of `110.0` on square footage means each additional square foot adds 110 to the predicted price — real units in, real units out. A weight of `12.0` on a runner's pace means each 1 m/s of speed adds a fixed 12 points to the predicted effort score.
 * **Scaled coefficients** (after ``StandardScaler``) report the change in target per **one standard deviation** of that feature, because the transform converts each feature to a Z-score. That single change makes the weights comparable in scale: a high-magnitude heart-rate signal and a low-magnitude cadence signal, raw on completely different scales, become comparable once both are standardized.

Comparing standardized weights gives a useful first pass at how strongly each feature drives the prediction, and standardizing before fitting is the sensible default whenever that comparison is the goal. See <doc:Feature-Scaling>. Treat the comparison as a starting point rather than a verdict, though: standardized magnitudes still ignore the correlation between features that the next section is about, and they discard the sign of the effect, so two correlated inputs can split a single underlying influence between them. They rank coefficients, which is not the same as measuring importance.

> Note: ``StandardScaler`` centers each feature at zero with unit variance; ``FeatureScaler`` maps to a bounded [0, 1] range instead. Both expose the same `fit` / `transform` pair.

### Diagnosing the collinearity tug-of-war

When two features carry nearly the same information — a runner's pace and their running power, say, which rise and fall together — the model cannot separate their individual contributions. This is the failure of the clean reading from the previous section: the features can no longer hold still one at a time, so no single coefficient describes a realizable change. How that ambiguity surfaces depends on which solver we use, and on how identical the columns are.

The mechanism and its cure belong to the <doc:Regularization-Primer>; what follows is the *diagnosis* — how to read the failure signature directly off the console and recognize it for what it is.

### Measuring it before the fit

Before reading any coefficients, check whether the feature matrix is well-conditioned. The `conditionNumber` of `XᵀX` measures how close the matrix is to singular — the exact instability that wrecks coefficient interpretation:

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

A condition number in the low tens is healthy. Once it climbs into the thousands, ordinary least squares is on unstable ground and the coefficients it returns can no longer be read as feature importances. The <doc:Determinants-Primer> introduces this diagnostic in full.

### Reading the failure in the coefficients

For *near*-identical columns, ``LinearRegression`` still returns an answer — but a pathological one. The closed-form solver pushes one weight up and pushes the mirrored weight hard the other way, producing a large, opposing pair that predicts correctly while meaning nothing individually:

```swift
let prices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]

let ols = try LinearRegression.fit(features: scaled, targets: prices)
ols.coefficients   // a lopsided pair — roughly 608719 against -512315
```

Those two numbers are not findings about floor area. They are the solver chasing quirks across two columns that should have been one, and a different sample would swing them just as violently the other way. The signs here follow from the two columns rising together; the general signature is not "one positive, one negative" but "two large weights that nearly cancel, and that a different sample would throw just as hard."

For *perfectly* collinear columns — one an exact multiple of the other — `XᵀX` becomes singular and there is no unique answer to return. ``LinearRegression`` **throws `MatrixError.singular`** rather than hand back corrupted numbers:

```swift
// Pace recorded twice — minutes per km and minutes per mile — one an exact
// multiple of the other, so the columns are perfectly linearly dependent.
// try LinearRegression.fit(features:targets:) throws MatrixError.singular
```

The throw is honest: with no unique solution, the model gives none. Quiver's `determinant` reports `0.0` for any matrix singular to within its tolerance, and the solver throws on that same condition — so a `0.0` here is a reliable advance warning that the fit will throw. See <doc:Determinants-Primer>.

> Note: ``GradientDescent`` meets the same near-collinear data and converges quietly — it never inverts a matrix, so it reports `.converged` with sound predictions. Collinearity creates a flat valley of near-equal solutions, and the walk settles somewhere along it. The predictions are reliable because every point in that valley predicts about equally well, but the individual weights are not, because the valley has no single bottom. Quiet success is not the same as a trustworthy coefficient. See <doc:Optimization-Primer>.

### Resolving the signature

When an audit reveals this signature, resolve it along one of two pathways. The first is **feature selection**: prune one of the mirrored columns, since if two sensor streams tell the same physical story the model needs only one to map the space cleanly. The second is **ridge regression**: keep both columns but penalize large weights, collapsing the opposing pair into small, stable, balanced values without dropping either feature.

```swift
import Quiver

let ridge = try Ridge.fit(features: scaled, targets: prices, lambda: 1.0)
ridge.coefficients   // two small, balanced weights instead of the lopsided pair
```

The penalty's mechanism — why it shrinks and balances the weights, and how to choose its strength — is the subject of the <doc:Regularization-Primer>. See <doc:Ridge-Regression> for the model itself.

### Verifying non-parametric structure

``KNearestNeighbors`` and ``KMeans`` compute no equation and expose **no coefficients** — there is no weight vector to print. They operate entirely on spatial proximity, so we validate them with geometry rather than a weight matrix.

``KMeans`` reports `inertia` — the total squared distance from every point to its assigned centroid, the quantity often called within-cluster sum of squares. Lower inertia means tighter clusters, for a fixed set of features and a fixed `k`. Inertia is not comparable across different feature sets — adding a feature changes the space the distances live in — so we use it to choose `k` within one feature set, not to decide whether a feature belongs.

`elbowMethod` sweeps a range of `k` and returns the inertia at each, so we can find where extra clusters stop paying off:

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

The sharp drop from `k=2` to `k=3` and the flattening after it mark the natural cluster count. `bestFit(data:k:attempts:)` guards against unlucky random starts by keeping the lowest-inertia run.

### An angular view of cohesion

For a single group of vectors, `clusterCohesion()` returns the average pairwise cosine similarity, scaled 0.0 to 1.0 — higher means the members point in more nearly the same direction:

```swift
let groupA = [[0.8, 0.3, 0.9], [0.7, 0.4, 0.8], [0.9, 0.2, 0.9]]
groupA.clusterCohesion()   // ~0.98 — a tight, coherent group
```

> Important: `clusterCohesion()` measures *angular* agreement using cosine similarity (0.0 to 1.0), while ``KMeans`` groups by *Euclidean distance*. They answer related but different questions — two vectors can point the same way yet sit far apart — so cohesion is most meaningful when the features are scaled and direction is what we care about. It is not a silhouette score, and it does not measure separation *between* clusters; Quiver does not currently expose a silhouette metric. Use `inertia` to check Euclidean cluster tightness, `elbowMethod` to compare whole clusterings, and `clusterCohesion()` to inspect the angular agreement within a single group.

### Evaluating a classifier by prediction

``KNearestNeighbors`` is a lazy learner — `fit` only stores the data, and all work happens at `predict`. Because there are no weights to inspect, we judge it by how well it classifies held-out data. Feed `predict(_:)` output into the evaluation metrics:

```swift
import Quiver

let predictions = model.predict(testFeatures)
let matrix = predictions.confusionMatrix(actual: testLabels)
// `accuracy` and `classificationReport` accept the same actual: labels
```

A scaled feature set that produces clean, well-separated classes shows high accuracy and a confusion matrix concentrated on the diagonal. Features on mismatched scales blur the neighborhoods, so reach for ``StandardScaler`` first. See <doc:Nearest-Neighbors-Classification>.

### From reading a model to trusting it

Reading coefficients and geometry is the first move; trusting what we read is the second, and the two sections that follow most directly are about exactly that gap. The <doc:Regularization-Primer> takes the collinearity signature diagnosed here and supplies the cure — the penalty that turns a flat valley of equally-good answers into a single defensible one. The <doc:Optimization-Primer> explains why ``GradientDescent`` converges quietly onto an arbitrary point in that valley, and what its loss history reveals about the walk. For the broader arc — how fitting, evaluating, and generalizing fit together — the <doc:Machine-Learning-Primer> is the map, and <doc:Feature-Scaling> is the transform that makes most of these readings honest in the first place.

> Experiment: **The Quiver Notebook** is the right place to watch a coefficient lose its meaning. Take the two near-identical floor-area columns, standardize them, and fit ``LinearRegression`` — then print `conditionNumber` and the coefficients side by side. Append a third column that is the first one plus a trickle of noise and refit, watching the condition number climb and the lopsided pair grow. Now swap in ``Ridge`` and sweep `lambda` upward until the opposing weights collapse into a small, balanced pair. Seeing the same data go from honest to pathological and back is the fastest way to feel why a coefficient is only as trustworthy as the matrix behind it. See <doc:Quiver-Notebook>.

## Further reading

The intercept-first layout and the readings built on it connect to several other pages. These continue the thread from here:

- [Linear Regression](https://waynewbishop.github.io/quiver/documentation/quiver/linear-regression) — where the coefficients and intercept come from, and how `fit` produces them.
- [Feature Scaling](https://waynewbishop.github.io/quiver/documentation/quiver/feature-scaling) — the transform that decides whether a coefficient reads in raw units or standard deviations.
- [Regularization Primer](https://waynewbishop.github.io/quiver/documentation/quiver/regularization-primer) — the cure for the collinearity tug-of-war diagnosed above.
- [Ridge Regression](https://waynewbishop.github.io/quiver/documentation/quiver/ridge-regression) — the model that collapses the opposing pair into small, balanced weights.
- [Optimization Primer](https://waynewbishop.github.io/quiver/documentation/quiver/optimization-primer) — why ``GradientDescent`` converges quietly onto one point in a flat collinear valley.
- [Determinants Primer](https://waynewbishop.github.io/quiver/documentation/quiver/determinants-primer) — the condition-number and singularity diagnostics that warn a fit will fail.
- [Machine Learning Primer](https://waynewbishop.github.io/quiver/documentation/quiver/machine-learning-primer) — the broader map of how fitting, evaluating, and generalizing fit together.
