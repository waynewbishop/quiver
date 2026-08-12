# Principal Component Analysis

Projecting correlated features onto the directions of largest variance to compress and chart data.

## Overview

**Principal component analysis** finds the directions along which a dataset actually varies, orders them by how much variance each one carries, and keeps only the few that matter. A workout log with three correlated columns becomes two columns that preserve 97% of the variation — few enough to draw in a scatter plot, store compactly, or hand to a distance-based model.

The method builds on two primitives that ship alongside it. `covarianceMatrix(ddof:)` measures how features move together, and `eigenDecomposed()` splits that matrix into directions and magnitudes. `PCA` packages the full pipeline — center, decompose, project — into the same fit-then-use shape as the other models.

### From covariance to components

Covariance is where the structure lives. Six workouts, described by duration in minutes and active energy in kilocalories:

```swift
import Quiver

// [duration, activeEnergy] — minutes and kilocalories per workout
let sessions = [
    [35.0, 385.0],
    [41.0, 295.0],
    [67.0, 790.0],
    [49.0, 510.0],
    [66.0, 715.0],
    [36.0, 495.0]
]

let covariance = sessions.covarianceMatrix()
// Optional([[208.4, 2438.0],
//           [2438.0, 35936.67]])
```

The diagonal holds each feature's variance — `208.4` for duration, `35936.67` for energy — and the off-diagonal `2438.0` says the two move together: longer workouts tend to burn more. The matrix is p × p for p feature columns, and it is symmetric by construction.

Eigendecomposition turns that summary into directions — the <doc:Eigenvalues-Primer> builds the idea from geometry. Each eigenvector is a direction through the data, and its eigenvalue is the variance found along it:

```swift
if let covariance = sessions.covarianceMatrix() {
    let eigen = try covariance.eigenDecomposed()
    eigen.eigenvalues      // [36102.26, 42.81] — descending
    eigen.eigenvectors[0]  // [0.068, 0.998]
}
```

The two eigenvalues split the same total: their sum, `36145.07`, is the trace of the covariance matrix. But the top direction, `[0.068, 0.998]`, points almost entirely at energy. That is a finding about units, not workouts. Kilocalories run in the hundreds and minutes in the tens, so energy's variance swamps duration's, and the decomposition faithfully reports the swamping. Measure duration in seconds instead of minutes and its variance multiplies by `3600`. The top eigenvector then swings to point at duration instead.

> Note: `eigenDecomposed()` works on symmetric matrices only, where every eigenvalue is guaranteed real. A covariance matrix is symmetric by construction; a hand-built matrix must be, or the call throws ``MatrixError/notSymmetric``.

### Fitting a model

Standardizing first puts every feature on the same scale, so the components report structure instead of units. The `fit(features:componentCount:)` static method then centers the data, forms the sample covariance matrix with the same `n − 1` divisor `covarianceMatrix(ddof:)` defaults to, and eigendecomposes it in one call. There is no separate unfitted state; the returned struct is immediately usable:

```swift
import Quiver

// [duration, activeEnergy, averageHeartRate] per workout —
// minutes, kilocalories, and beats per minute
let workouts = [
    [35.0, 385.0, 135.0],
    [41.0, 295.0, 132.0],
    [67.0, 790.0, 137.0],
    [49.0, 510.0, 136.0],
    [66.0, 715.0, 146.0],
    [36.0, 495.0, 130.0]
]

let scaler = StandardScaler.fit(features: workouts)
let scaled = scaler.transform(workouts)

let model = PCA.fit(features: scaled, componentCount: 2)
print(model)
// PCA: 2 components, 3 features (97.2% variance explained)
```

The fit centers the data internally, so the caller never subtracts means by hand. What it cannot do is equalize units — that is the scaler's job, the same division of labor <doc:Feature-Scaling> establishes for the regression models. The fitted `components` store one direction per row: `components[0]` is the direction of largest variance, and each row is a unit vector.

> Important: PCA finds directions of variance wherever the variance comes from. Fit on raw features with mixed units and the top component reports the units. Fit on standardized features and it reports the structure. Standardize first whenever the columns are measured in different scales.

### Reading the explained variance

Fitting with every component retained shows the full spectrum, and the spectrum is how we choose how many to keep:

```swift
let full = PCA.fit(features: scaled, componentCount: 3)
full.explainedVariances       // [3.059, 0.44, 0.101]
full.explainedVarianceRatios  // [0.85, 0.122, 0.028]
```

The first component carries 85% of the variance — a shared workout-size direction that duration, energy, and heart rate all follow. The second carries 12.2%, separating high-burn sessions from long easy ones. The third holds 2.8%: measurement noise more than structure. Because the ratios sum to `1.0` across the full set of components, they act as a strict variance budget. Keeping two components preserves `0.85 + 0.122`, the `97.2%` the model description reported, and that cumulative read is the working rule: keep components until the running total crosses the coverage the application needs.

### Projecting data

`transform(_:)` projects samples onto the retained components. Rows stay samples; columns become component scores:

```swift
let projected = model.transform(scaled)
projected[0]  // [-1.24, 0.51]
projected[4]  // [2.47, 0.75]
```

Six workouts that lived in three correlated columns now live in two independent ones. The first score places each workout on the overall-size axis — the fifth workout, 66 minutes and 715 kilocalories, sits far right at `2.47` while the first sits left of center at `-1.24`. The second score separates intensity from duration at a given size. Two columns is exactly what a scatter plot can draw, which makes the projection the natural bridge from a high-dimensional feature table to a chart a person can read.

### Measuring the cost of compression

`inverseTransform(_:)` maps component scores back to feature space. What returns is the original data as the retained components saw it — the dropped component's contribution is gone:

```swift
let restored = model.inverseTransform(projected)
restored[0]  // [-0.84, -1.01, -0.27]
scaled[0]    // [-1.06, -0.85, -0.2]
```

The restored row is close to the original but not equal to it, and the gap is exactly the discarded variance. For features standardized with `StandardScaler`, the average squared reconstruction error across all cells equals the variance ratio of the dropped components — `0.028` here, the third ratio read earlier. An error in squared units can match a unitless ratio only because standardization fixes the data's total variance: the share of variance dropped and the average error per cell become the same number. Compression and reconstruction error are the same quantity seen from two sides, which is what makes the ratios trustworthy as a budget: the variance a component claims to carry is precisely what disappears when it is dropped.

### When to use principal components

Reach for PCA when features are many and correlated. Collinear columns that destabilize a regression collapse into a smaller set of independent ones. For nearest-neighbor classification, `Pipeline.fit(features:labels:componentCount:k:metric:weight:)` bundles the whole chain — scaler, reducer, and classifier — into one value that scales and projects every query with the fitted training parameters automatically. Embedding vectors compress before a similarity search over them, trading a controlled slice of variance for less storage and faster distance computations — see <doc:Semantic-Search> for the search pipeline itself. Embedding dimensions already share one scale, so the fit runs on them directly, with no scaler step. And any high-dimensional dataset becomes chartable by keeping its top two components, the same unsupervised spirit as <doc:KMeans-Clustering>: both find structure in unlabeled data.

One boundary applies to all three uses: the projection is linear. Structure that bends — points along a curve, clusters around a ring — survives only as well as a flat axis can represent it, and the variance a curved pattern carries can spread across many components instead of concentrating in a few.

The method has a failure mode to check before trusting individual components. When two eigenvalues nearly tie, the variance along the pair is real but its split into two specific directions is not. The scatter across that pair of axes is nearly circular, and no axis through a circle is better than any other. Any rotation of the tied pair explains the data equally well, and a handful of new samples can swap or blend them. The ratios stay honest. The component *directions* are what to distrust. Reading the gaps in `explainedVariances` first is the guard: here `3.059` stands seven times clear of `0.44`, so the leading direction is stable. A second check matters when features are nearly collinear: forming a covariance matrix squares the condition number of the data, concentrating rounding error in the smallest eigenvalues. The `conditionNumber` diagnostic from the <doc:Determinants-Primer> reads that risk on the covariance matrix directly.

### Safe by design

`PCA` is an immutable value type. `fit` is the only public way to construct one, so a model in hand is always a fitted model — no partially initialized state can exist. The decomposition is deterministic: the same features and component count produce the same model, bit for bit, and `Equatable` conformance makes that checkable in tests.

Shape problems are treated as programmer errors rather than recoverable ones. A ragged feature matrix, a non-finite value, or a component count above the feature count stops `fit` at the call site instead of throwing an error the caller could absorb. The throwing path belongs to `eigenDecomposed()`, whose symmetry check depends on the data itself.

The model is `Codable` for persistence, and its decoder validates. Decoding rejects mismatched shapes — a components matrix that disagrees with the stored counts, a means vector of the wrong length, any non-finite value — by throwing `DecodingError.dataCorrupted` rather than constructing a model that would fail later. A model decoded from disk carries the same guarantees as one just fitted. See <doc:Model-Persistence> for the encode-once, decode-on-launch pattern.

> Experiment: **The Quiver Notebook** is the right place to watch the variance budget balance. Fit `PCA` on the standardized workouts at `componentCount` of 1, 2, and 3, and print the ratio sum beside the mean squared reconstruction error at each step. The two numbers always sum to `1.0` — every point of variance kept is a point of error avoided. See <doc:Quiver-Notebook>.

## Topics

### Model

- ``PCA``

### Fitting and projecting

- ``PCA/fit(features:componentCount:)``
- ``PCA/transform(_:)``
- ``PCA/inverseTransform(_:)``

### Reading the fit

- ``PCA/components``
- ``PCA/means``
- ``PCA/explainedVariances``
- ``PCA/explainedVarianceRatios``
- ``PCA/componentCount``
- ``PCA/featureCount``

### Supporting primitives

- ``Swift/Array/covarianceMatrix(ddof:)``
- ``Swift/Array/eigenDecomposed()``
- ``EigenDecomposition``
- ``StandardScaler``
- ``Transformer``

### Errors

- ``MatrixError``

### Related

- <doc:Feature-Scaling>
- <doc:Determinants-Primer>
