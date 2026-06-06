# Quiver — A Swift package for statistics, linear algebra, and machine learning.

Complete reference for the Quiver Swift package. Upload this file to a Claude Project or conversation to get accurate assistance with Quiver code.

**Repository:** https://github.com/waynewbishop/quiver
**Cookbook:** https://github.com/waynewbishop/quiver-cookbook — 42 interactive recipes for learning vector math, statistics, and ML models in Swift
**Module:** `import Quiver`
**Platforms:** macOS 12+, iOS 15+, tvOS 15+, watchOS 8+, visionOS 1+
**Swift:** 5.9+
**Dependencies:** None (pure Swift)
**Compiled size:** ~1.9 MB (release) — 33 files, 7,784 lines. Uses 2.5% of watchOS's 75 MB app bundle limit

---

## What Quiver Is

Quiver fills the gap between Swift's standard library and trained-model inference. Swift gives you `Array` with basic operations. Between basic arrays and a finished model sits a wide space of real-time numerical computing — statistics, vector math, matrix operations, similarity search, clustering, regression — that the standard library does not address. Quiver fills that gap with 350+ APIs, zero dependencies, and a footprint small enough for watchOS.

**Built for Swift developers.** Quiver serves developers who need numerical computing in the language they already use — on iOS, watchOS, visionOS, server-side Linux, and in Xcode Playgrounds. No context switching, no model files, no Accelerate wrappers.

Quiver provides the computational building blocks for working with data directly: `mean()`, `percentile()`, `cosineOfAngle(with:)`, `trainTestSplit()`, `KMeans.fit()`, `LinearRegression.fit()`. It offers ad-hoc statistical queries, pairwise semantic comparison, and transparent ML model training on raw arrays.

**Validated against industry-standard implementations.** Quiver has 312 unit tests plus a separate cross-validation suite (44 checks + 29 tests, all passing). Quiver produces identical results for identical inputs.

---

## How Quiver Works

Quiver extends Swift's `Array` type with numerical computing methods. There are no custom container types — everything operates on `[Double]`, `[[Double]]`, `[Int]`, `[Bool]`, and `[String]`. This means every Quiver array inherits Swift's full standard library (`map`, `filter`, `sorted`, `reduce`, `enumerated`, `zip`, etc.) for free.

Internal logic lives in `_Vector` types that are not part of the public API. The public surface is clean Array extensions constrained by `Numeric`, `FloatingPoint`, `Comparable`, or `BinaryFloatingPoint`.

ML models are immutable value types created via static `fit()` methods. There is no unfitted state — you cannot call `predict()` on a model that hasn't been trained.

All models and result types conform to `Codable`, enabling JSON serialization for persistence, network transfer, and cross-device deployment. The Swift compiler auto-synthesizes encoding/decoding since all stored properties are basic Swift types (`[Double]`, `Int`, `Bool`, etc.). Train once, encode to JSON, decode on any platform — identical predictions guaranteed.

All model and result types conform to `CustomStringConvertible`, producing clean summaries when printed:

```swift
print(model)    // KMeans: 3 clusters, 9 points, converged in 4 iterations (inertia: 1.08)
print(cluster)  // Cluster: center [1.23, 1.97], 3 points
print(cm)       // TP: 3  FP: 1  TN: 3  FN: 1  (accuracy: 75.0%)
print(knn)      // KNearestNeighbors: k=3, euclidean, 6 training points, 2 features
print(nb)       // GaussianNaiveBayes: 2 classes, 2 features
print(lr)       // LinearRegression: 1 feature, intercept: 38000.00, slope: 110.00
print(scaler)   // FeatureScaler: 2 features, range 0.0...1.0
print(group)    // Class 0: 3 points
```

Individual properties remain accessible for detailed inspection — `model.labels`, `cluster.centroid`, `cm.truePositives`, etc.

All models and result types conform to `Equatable`, enabling direct comparison with `==` in tests and assertions:

```swift
let run1 = KMeans.fit(data: points, k: 3, seed: 42)
let run2 = KMeans.fit(data: points, k: 3, seed: 42)
run1 == run2  // true
```

Models: `KMeans`, `KNearestNeighbors`, `GaussianNaiveBayes`, `LinearRegression`, `GradientDescent`, `Ridge`. Data: `Panel`. Result types: `ConfusionMatrix`, `Classification`, `Cluster`, `FeatureScaler`, `ClassStats`. Supporting types: `DistanceMetric`, `VoteWeight`, `Fraction`, `MatrixError`, `GradientDescentError`.

---

## Vector Arithmetic

Element-wise operations use **named methods**, not operators:

```swift
let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0, 6.0]

a.add(b)        // [5.0, 7.0, 9.0]
a.subtract(b)   // [-3.0, -3.0, -3.0]
a.multiply(b)   // [4.0, 10.0, 18.0]
a.divide(b)     // [0.25, 0.4, 0.5]
```

**Scalar broadcast operators** are supported:

```swift
2.0 + a    // [3.0, 4.0, 5.0]
a - 1.0    // [0.0, 1.0, 2.0]
3.0 * a    // [3.0, 6.0, 9.0]
a / 2.0    // [0.5, 1.0, 1.5]
```

**Important:** `a + b` where both are arrays will NOT compile. Use `.add()`. Scalar broadcast (`a + 2.0`) works fine.

## Matrix Arithmetic

Same pattern — named methods for matrix-matrix, operators for scalar:

```swift
let A = [[1.0, 2.0], [3.0, 4.0]]
let B = [[5.0, 6.0], [7.0, 8.0]]

A.add(B)        // element-wise add
A.subtract(B)   // element-wise subtract
A.multiply(B)   // Hadamard product (element-wise)
A.divide(B)     // element-wise divide

A + 10.0        // scalar broadcast (works)
2.0 * A         // scalar broadcast (works)
```

## Dot Product and Projections

All on `[FloatingPoint]`, all return non-optional values:

```swift
let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0, 6.0]

a.dot(b)                        // 32.0
a.scalarProjection(onto: b)     // projection length
a.vectorProjection(onto: b)     // vector component in direction of b
a.orthogonalComponent(to: b)    // perpendicular remainder
```

## Magnitude and Distance

```swift
let v = [3.0, 4.0]
v.magnitude          // 5.0 (non-optional)
v.normalized         // [0.6, 0.8] (non-optional)
v.distance(to: w)    // Euclidean distance (non-optional)
```

## Angular Operations

```swift
a.cosineOfAngle(with: b)    // cosine similarity [-1, 1] (non-optional)
a.angle(with: b)             // angle in radians (non-optional)
a.angleInDegrees(with: b)    // angle in degrees (non-optional)
```

## Matrix Transformations

```swift
let M = [[1.0, 2.0], [3.0, 4.0]]
let v = [1.0, 0.0]

M.transform(v)              // matrix-vector multiply (non-optional)
v.transformedBy(M)          // same result, vector-first syntax (non-optional)
M.transpose()                // swap rows and columns (non-optional)
M.transposed()               // same as transpose()
M.multiplyMatrix(other)      // matrix-matrix multiply (non-optional)
M.column(at: 0)              // extract column vector (non-optional)

M.determinant                // determinant value (non-optional)
M.conditionNumber            // condition number, 1-norm (non-optional)
M.logDeterminant             // LogDeterminant struct (non-optional)
  // .sign (-1, 0, or 1)
  // .logAbsValue
  // .value (reconstructed)

try M.inverted()             // matrix inverse (throws MatrixError)
```

`MatrixError` cases: `.notSquare`, `.singular`

## Shape and Size

```swift
let M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
M.shape          // (rows: 2, columns: 3) (non-optional)
M.size           // 6 (non-optional)
M.flattened()    // [1, 2, 3, 4, 5, 6] (non-optional)

let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
flat.reshaped(rows: 2, columns: 3)   // back to 2x3 (non-optional)
```

## Linear Systems

```swift
// 2x +  y = 5
//  x + 3y = 10
let A = [[2.0, 1.0],
         [1.0, 3.0]]
let b = [5.0, 10.0]

A.solve(b)   // [1.0, 3.0]?
```

`solve(_:)` returns `x` such that `A · x = b`, computed by inverting `A` and applying the inverse to `b`. Returns `nil` if `A` is not square, has inconsistent row lengths, the row count disagrees with `b.count`, or `A` is singular. Near-singular matrices (condition number above roughly `10¹⁰`) return a non-nil value that may be numerically unreliable — check `A.conditionNumber` before trusting the result.

## Array Generation

All called on `[Double]` (not `[[Double]]`), even for 2D:

```swift
// 1D
[Double].zeros(5)
[Double].ones(5)
[Double].full(5, value: 3.14)

// 2D — note: called on [Double], returns [[Double]]
[Double].zeros(2, 3)
[Double].ones(2, 3)
[Double].full(2, 3, value: 7.0)

// Special matrices
[Double].identity(3)           // 3x3 identity
[Double].diag([1.0, 2.0, 3.0]) // diagonal matrix

// Sequences
[Double].linspace(start: 0.0, end: 1.0, count: 5)   // [0.0, 0.25, 0.5, 0.75, 1.0]
[Double].arange(0.0, 10.0, step: 2.5)                // [0.0, 2.5, 5.0, 7.5]
```

## Random Generation

```swift
// Uniform
[Double].random(5)                      // 1D, [0, 1]
[Double].random(2, 3)                   // 2D, [0, 1]
[Double].random(5, in: 10.0...20.0)     // custom range
[Double].random(2, 3, in: -1.0...1.0)   // 2D custom range
[Int].random(5, in: 0..<100)            // Int, half-open range

// Normal distribution
[Double].randomNormal(5, mean: 0.0, standardDeviation: 1.0)
[Double].randomNormal(2, 3, mean: 5.0, standardDeviation: 2.0)

// Exponential distribution (1.2.0) — long right tail, rate = 1/mean
[Double].randomExponential(1_000, rate: 0.5)        // 1D, mean = 1/rate = 2.0
[Double].randomExponential(3, 4, rate: 1.0)         // 2D

// Binomial distribution (1.2.0) — count of successes in n Bernoulli trials
[Double].randomBinomial(1_000, n: 20, p: 0.5)       // 1D
[Double].randomBinomial(3, 4, n: 10, p: 0.3)        // 2D
```

### Reproducible randomness (1.2.0)

`SeededRandomNumberGenerator(seed: UInt64)` is a public struct conforming to `RandomNumberGenerator`. Every Quiver random method has a `using:` overload that accepts an `inout` generator, mirroring the standard library's `Array.shuffled(using:)` pattern:

```swift
var rng = SeededRandomNumberGenerator(seed: 42)
let normal = [Double].randomNormal(1_000, mean: 0, standardDeviation: 1, using: &rng)
let expon  = [Double].randomExponential(1_000, rate: 0.5, using: &rng)
let bins   = [Double].randomBinomial(1_000, n: 20, p: 0.5, using: &rng)
let shuf   = [1, 2, 3, 4, 5].shuffled(using: &rng)
```

The generator is passed by `inout` because each call advances its internal state — passing by value would produce the same numbers on every call. Two runs with the same seed produce identical sequences across the whole Quiver random surface (uniform, normal, exponential, binomial) and the standard library random methods that accept `using:`.

## Statistical Operations

### Return optionals (must unwrap):

```swift
let data = [3.0, 1.0, 4.0, 1.0, 5.0]

if let avg = data.mean() { print(avg) }           // Double?
if let mid = data.median() { print(mid) }         // Double?
if let v = data.variance() { print(v) }           // Double? (sample, default ddof: 1)
if let s = data.standardDeviation() { print(s) }  // Double? (sample, default ddof: 1)
if let e = data.standardError() { print(e) }      // Double? (sample, default ddof: 1)
if let q = data.quartiles() { print(q.iqr) }      // Quartiles?
if let m = data.mode().first { print(m) }         // [Element] — empty for empty input
if let lo = data.argMin() { print(lo) }            // Int?
if let hi = data.argMax() { print(hi) }            // Int?
if let p = data.percentile(90) { print(p) }        // Double?
if let sk = data.skewness() { print(sk) }          // Double? — asymmetry (1.3.0)
if let ku = data.kurtosis() { print(ku) }          // Double? — tail weight (1.3.0)
if let r = data.skewnessReport() { print(r) }      // SkewnessReport? (1.3.0)
```

**Shape diagnostics (1.3.0):** `skewness(bias:)` and `kurtosis(bias:)` return `Element?` (nil for too-small input). `skewnessReport()` pairs an outlier-sensitive measure with an outlier-resistant one and flags when they disagree — a signal that extreme values are warping the distribution.

**Critical rule (1.2.0):** `variance(ddof:)`, `standardDeviation(ddof:)`, and `standardError(ddof:)` all default to `ddof: 1` (sample statistics, dividing by `n - 1`). This matches the formula in introductory statistics textbooks. Pass `ddof: 0` explicitly when the array represents an entire population rather than a sample. The 1.2.0 rename from `std()` to `standardDeviation()` shipped with this default flip in the same commit.

`mode()` is defined on `Array where Element: Hashable` and returns `[Element]` (not optional). When multiple values tie for highest frequency, all are returned — bimodal distributions surface honestly. Empty input returns an empty array.

### Return non-optional:

```swift
data.sum()                // Double
data.product()            // Double
data.cumulativeSum()      // [Double]
data.cumulativeProduct()  // [Double]
data.percentileRank(of: 3.0)  // Double
data.percentileRanks()        // [Double]
data.histogram(bins: 6)       // [(midpoint: Double, count: Int)]
data.outlierMask(threshold: 2.0)  // [Bool]
```

### Multi-vector (return optionals):

```swift
let vectors: [[Double]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
if let mv = vectors.meanVector() { print(mv) }   // [Double]?
if let avg = vectors.averaged() { print(avg) }   // [Double]?
```

### Resampling and confidence intervals:

```swift
let scores = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]

// Resampled distribution of any statistic (sampling with replacement)
let medians = scores.resampled(iterations: 1000, seed: 42) { resample in
    resample.median() ?? 0.0
}

// Percentile-based confidence interval from a resampled distribution
if let ci = medians.percentileCI(level: 0.95) {
    print(ci.lower, ci.upper)   // 2.5th and 97.5th percentiles
}
```

`resampled` returns the distribution of a statistic across `iterations` resamples (the technique known in statistics as the bootstrap). Pair it with `percentileCI(level:)` to turn that distribution into a confidence interval. Both use Quiver's seeded generator — same seed, same result.

## Probability Distributions

`Distributions` is a stateless namespace for probability density, cumulative density, log-density, and quantile functions. Every function returns `Double?` and produces `nil` for out-of-domain input (`standardDeviation <= 0`, `p` outside `(0, 1)`, non-finite results, negative `df` for `t` or `chiSquared`) — matching Quiver's existing pattern for `mean`, `variance`, etc.

```swift
// Probability density at the mean of a standard normal
Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)        // ≈ 0.3989

// Log-density (numerically stable for tail values or repeated multiplication)
Distributions.normal.logPDF(x: 4, mean: 0, standardDeviation: 1)     // ≈ -8.919

// Cumulative probability P(X <= x)
Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)     // ≈ 0.975

// Quantile (inverse CDF) — the 95% critical value
Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1)  // ≈ 1.96
```

The quantile uses the Beasley-Springer-Moro rational approximation — roughly 7-decimal accuracy in the body of the distribution, 4-decimal in the tails. Values closer than `1e-15` to `0` or `1` return `nil`.

### Student's t-distribution

```swift
// Probability that a t-distributed value with df = 10 falls at or below 2.228
Distributions.t.cdf(x: 2.228, df: 10)       // ≈ 0.975

// Critical value at p = 0.975 for a 95% confidence interval, df = 10
Distributions.t.quantile(p: 0.975, df: 10)  // ≈ 2.228

// Same probability under the 95th-percentile cutoff
Distributions.t.quantile(p: 0.95, df: 10)   // ≈ 1.812
```

The t-distribution is centered at zero, symmetric, and has heavier tails than the standard normal. As `df` grows, it converges to the normal — at `df = 1000` the two are visually indistinguishable. Used for small-sample inference (`n < 30`) where we estimate both the mean and the standard deviation from the same sample. The classic recipe for a one-sample t-statistic uses `df = n - 1`.

Implementation: regularized incomplete beta function with bisection for the quantile. Verified against the reference test grid in `Tests/QuiverTests/DistributionsTests.swift`.

### Chi-squared distribution

```swift
// Cumulative probability for a chi-squared value at df = 5
Distributions.chiSquared.cdf(x: 11.07, df: 5)    // ≈ 0.95

// Same at df = 10
Distributions.chiSquared.cdf(x: 18.307, df: 10)  // ≈ 0.95
```

The chi-squared distribution has support on `[0, ∞)` (negative `x` returns `0.0`, not `nil`). Used for variance inference and goodness-of-fit testing — the test statistic for "do observed category counts match expected counts" follows a chi-squared distribution under the null hypothesis. Implementation: regularized lower incomplete gamma function. Closed-form anchor at `df = 2`: `cdf(x, df: 2) = 1 - exp(-x / 2)`.

`GaussianNaiveBayes` uses `Distributions.normal.logPDF` internally; the public exposure lets other callers reuse the same well-tested implementation.

## Inferential Output Types

Quiver methods that return multiple related quantities use typed value types instead of dictionaries or anonymous tuples. Every type below conforms to `Codable`, `Sendable`, `Equatable`, and `CustomStringConvertible`. The conformances buy round-trip persistence to JSON, cross-task safety, equality checks for testing, and readable `print()` output.

### `Quartiles`

Returned by `Array.quartiles()` on `Array where Element: FloatingPoint`. Generic over `T`. Fields: `min`, `q1`, `median`, `q3`, `max`, `iqr`. The five-number summary plus the interquartile range as one value. `print(q)` emits all six labeled values in a single formatted block (`min:`, `q1:`, etc.); read fields directly when only one is needed.

### `ColumnSummary`

Returned by `Array.summary()` on `Array where Element == Double`. Fields: `count` (`Int`), `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, `iqr` (all `Double`). Computed once at construction time. Format helpers `markdownTable()` and `csvRows()` render the summary for inclusion in PRs, reports, or pipelines.

### `PanelSummary`

Returned by `Panel.summary()`. Fields: `columnNames` (`[String]`), `columns` (`[String: ColumnSummary]`). Per-column statistics across an entire panel in a single value. Iteration order matches `columnNames`.

### `RegressionSummary`

Returned by `LinearRegression.summary(features:targets:confidenceLevel:)`. Throws `MatrixError.singular` if `(XᵀX)` is non-invertible. Fields:

- `coefficients: [Double]` — fitted parameter vector (intercept first if the model was fit with intercept)
- `standardErrors: [Double]` — standard error for each coefficient
- `tStatistics: [Double]` — coefficient divided by its standard error
- `pValues: [Double]` — two-tailed p-value under the null `β = 0`, computed against `Distributions.t.cdf` with the residual degrees of freedom
- `confidenceIntervals: [ConfidenceInterval]` — `(lower, upper)` band per coefficient at the requested `confidenceLevel` (default `0.95`)
- `rSquared: Double` — coefficient of determination
- `adjustedRSquared: Double` — penalized for parameter count, useful for comparing models of different dimensionality
- `n: Int` — sample size
- `degreesOfFreedom: Int` — `n − p` where `p` is the number of fitted parameters
- `residualStandardError: Double` — square root of the residual mean square
- `confidenceLevel: Double` — the level used to build `confidenceIntervals`

`ConfidenceInterval` is a small `(lower: Double, upper: Double)` value type, also conforming to the standard four protocols.

### `ClassificationReport` and `ClassMetrics`

Returned by `predictions.classificationReport(actual:)` on `[Int]`. Two-tier shape:

- `ClassMetrics`: `label: Int`, `precision: Double?`, `recall: Double?`, `f1Score: Double?`, `support: Int`. Optionals are `nil` when the denominator would be zero (e.g., `precision` is `nil` when the class has no predicted positives).
- `ClassificationReport`: `perClass: [Int: ClassMetrics]`, `classOrder: [Int]`, `accuracy: Double`, `macroAverage: ClassMetrics`, `weightedAverage: ClassMetrics`, `totalSupport: Int`. `classOrder` preserves the deterministic class ordering for iteration.

## Similarity and Clustering

```swift
let docs: [[Double]] = [...]
let query = [1.0, 0.0, 1.0]

docs.cosineSimilarities(to: query)      // [Double] (non-optional)
docs.findDuplicates(threshold: 0.9)     // [(index1, index2, similarity)]
docs.clusterCohesion()                  // Double
```

## Ranking and Sorting

```swift
let scores = [0.9, 0.1, 0.7, 0.3, 0.5]

scores.topIndices(k: 3)                                         // [(index, score)]
scores.topIndices(k: 2, labels: ["A", "B", "C", "D", "E"])    // [(label, score)]
scores.sortedIndices()                                          // [Int] (argsort)
```

## Embedding Dictionary Search

```swift
let king  = [0.9, 0.2, 0.8, 0.7]
let queen = [0.3, 0.9, 0.8, 0.7]
let man   = [0.8, 0.1, 0.2, 0.6]
let woman = [0.2, 0.8, 0.2, 0.6]

let embeddings: [String: [Double]] = [
    "king":  king,
    "queen": queen,
    "man":   man,
    "woman": woman
]

// Rank dictionary entries by cosine similarity to a query vector.
// Returns [(rank: Int, word: String, score: Double)] sorted by score descending.
embeddings.nearest(to: queryVector, k: 5)

// Analogy lookup — king - man + woman should land near queen
let target = king.subtract(man).add(woman)
embeddings.nearest(to: target, k: 1)  // [(1, "queen", 1.0)]
```

Entries whose vector dimension does not match the query are silently skipped. Zero-magnitude vectors score 0.0 (perpendicular by convention). Default k = 5.

## Broadcasting

### Scalar (named methods):

```swift
data.broadcast(adding: 10.0)
data.broadcast(subtracting: 1.0)
data.broadcast(multiplyingBy: 2.0)
data.broadcast(dividingBy: 3.0)
data.broadcast(with: 2.0, operation: { $0 + $1 })
```

### Matrix (row/column vector):

```swift
matrix.broadcast(addingToEachRow: rowVec)
matrix.broadcast(addingToEachColumn: colVec)
matrix.broadcast(multiplyingEachRowBy: rowVec)
matrix.broadcast(multiplyingEachColumnBy: colVec)
matrix.broadcast(withRowVector: rowVec, operation: +)
matrix.broadcast(withColumnVector: colVec, operation: *)
```

## Boolean and Comparison

### Comparison (returns `[Bool]`):

```swift
data.isGreaterThan(3.0)
data.isLessThan(3.0)
data.isGreaterThanOrEqual(3.0)
data.isLessThanOrEqual(3.0)
data.isEqual(to: otherArray)    // takes an array, not a scalar
```

### Boolean operations:

```swift
mask1.and(mask2)       // element-wise AND
mask1.or(mask2)        // element-wise OR
mask1.not              // element-wise NOT (computed property)
mask1.trueIndices      // [Int] — indices where true
```

### Boolean indexing:

```swift
data.masked(by: mask)                          // select where true
data.choose(where: mask, otherwise: otherArray) // conditional (takes array, not scalar)
```

## Element-wise Math

Available on `[Double]` and `[Float]`:

```swift
vals.power(2.0)    // raise to power
vals.sqrt()        // square root
vals.square()      // x^2
vals.log()         // natural log
vals.log10()       // base-10 log
vals.exp()         // e^x
vals.sin()         // sine
vals.cos()         // cosine
vals.tan()         // tangent
vals.floor()       // round down
vals.ceil()        // round up
vals.round()       // round to nearest
```

### Activation Functions:

```swift
logits.softMax()    // probability distribution summing to 1.0
logits.sigmoid()    // element-wise 1/(1+e^-x), each in (0,1)
```

## Time Series

```swift
series.rollingMean(window: 3)
series.diff(lag: 1)
series.percentChange(lag: 1)

// Rate of change — discrete derivative (difference ÷ spacing); output is one shorter
series.derivative(sampleRate: 1.0)        // [Double]

// Total from a rate — discrete integral via the trapezoid rule (1.3.0)
series.trapezoidalIntegral(dt: 1.0)       // Double? (nil if fewer than 2 samples)
series.cumulativeTrapezoidal(dt: 1.0)     // [Double] — running total at each step
```

`derivative(sampleRate:)` and `trapezoidalIntegral(dt:)` are inverse operations: the derivative turns a quantity into its rate (speed → acceleration), the integral turns a rate back into a total (speed → distance, power → energy). `dt`/`sampleRate` is the spacing between samples.

## Data Visualization Helpers

```swift
// Scaling
revenues.scaled(to: 10.0...50.0)    // min-max to range
revenues.standardized()               // z-score (mean=0, std=1)
revenues.asPercentages()              // share of total

// Aggregation
sales.groupBy(regions, using: .sum)            // [String: Double]
sales.groupedData(by: regions, using: .mean)   // [(category, value)]
data.downsample(factor: 6, using: .mean)       // reduce resolution
// AggregationMethod: .sum, .mean, .count, .min, .max

// Multi-series
series.stackedCumulative()     // for stacked area charts
series.stackedPercentage()     // for 100% stacked bars
vectors.correlationMatrix()    // Pearson correlation matrix
vectors.heatmapData(labels:)   // [(x, y, value)] for RectangleMark
```

## Text Operations

```swift
"Hello World! This is a test.".tokenize()   // ["hello", "world", "this", "is", "a", "test"]
"Hello, World!".tokenize(removingPunctuation: false)  // ["hello,", "world!"]

let embeddings: [String: [Double]] = ["hello": [0.1, 0.2], ...]
tokens.embed(using: embeddings)   // [[Double]] — vectors for known words
```

## Fraction Type

```swift
let frac = Fraction(numerator: 3, denominator: 4)
frac.value          // 0.75
frac.description    // "3/4"

Fraction(0.333)              // auto-detect: 333/1000 or similar
0.75.asFraction()            // Fraction
[0.5, 0.25].asFractions()   // [Fraction]
[[0.5, 0.25]].asFractions()  // [[Fraction]]
```

## Polynomial Type

`Polynomial` represents a single-variable polynomial `a₀ + a₁x + a₂x² + ... + aₙxⁿ` as ordered coefficients in a `[Double]`. Element `i` is the coefficient of `xⁱ` — constant term first, ascending powers after. Value type, `Codable`, `Equatable`, `Sendable`.

```swift
// 2x² + 3x + 1
let p = Polynomial([1, 3, 2])

p.coefficients      // [1.0, 3.0, 2.0]
p.degree            // 2
p.description       // "2x² + 3x + 1"

// Evaluation uses Horner's method (numerically stable)
p(2)                // 15.0 — single point
p([0, 1, 2, 3])     // [1.0, 6.0, 15.0, 28.0] — vectorized for plotting

// Calculus and canonicalization
p.derivative()      // 4x + 3
p.trimmed()         // strips trailing zero coefficients

// Math rendering (1.3.0) — legible expression for teaching and console output
p.asExpression()    // "2x² + 3x + 1"

// Arithmetic
let q = Polynomial([4, -3, -2])
p + q               // term-by-term addition
p * q               // polynomial multiplication (convolution)
3.0 * p             // scalar broadcast
```

### Polynomial fitting

```swift
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [6.0, 15.0, 28.0, 45.0, 66.0]   // 2x² + 3x + 1

if let p = [Double].polyfit(x: x, y: y, degree: 2) {
    p.coefficients   // ≈ [1.0, 3.0, 2.0]
    p(6)             // ≈ 91.0
}
```

`polyfit` builds a Vandermonde-style design matrix and defers to `LinearRegression` to solve the normal equation. For `degree: 1` the result matches `LinearRegression.fit(features: x, targets: y)` exactly. For `degree: 0` the polynomial collapses to the mean of `y`. Returns `nil` on mismatched lengths, fewer points than `degree + 1`, negative degree, or an ill-conditioned system.

## Info and Debugging

```swift
[1.0, 2.0, 3.0].info()             // pretty-printed stats
[[1.0, 2.0], [3.0, 4.0]].info()    // matrix info with shape

// Math rendering (1.3.0) — arrays, matrices, and polynomials as legible math
[1.0, 2.0, 3.0].asExpression()           // column or row vector notation
[[1.0, 2.0], [3.0, 4.0]].asExpression()  // bracketed matrix
```

`asExpression()` is presentation only — every value is still computed in `Double`; it changes how a result reads in the console, not what it equals.

## Sampling and Splitting

```swift
// Basic split — use matching seeds for paired arrays
let (trainX, testX) = features.trainTestSplit(testRatio: 0.2, seed: 42)
let (trainY, testY) = labels.trainTestSplit(testRatio: 0.2, seed: 42)

// Stratified split — preserves class proportions
let (trainF, testF, trainL, testL) = features.stratifiedSplit(
    labels: labels, testRatio: 0.2, seed: 42
)

// Single random sample (1.3.0) — reproducible subset, with or without replacement
let draw = data.sample(3, replace: false, seed: 42)   // [Element]

// K-fold cross-validation (1.3.0) — leak-free index pairs, caller drives the loop
let folds = features.kFoldIndices(k: 5, seed: 42)      // [(train: [Int], validation: [Int])]
for fold in folds {
    // fit on fold.train indices, score on fold.validation; every index validated once
}
```

`kFoldIndices(k:seed:)` returns index sets, not sliced data — so a scaler can fit on the training indices alone and validation rows never leak into the fit. Bounds: `2 ≤ k ≤ count`. Every position is validated exactly once across the `k` folds; fold sizes differ by at most one. After cross-validation picks the best configuration, retrain that choice on the full dataset before deploying.

## Feature Scaling

```swift
let scaler = FeatureScaler.fit(features: trainX, range: 0.0...1.0)
let scaledTrain = scaler.transform(trainX)
let scaledTest = scaler.transform(testX)

scaler.minimums      // [Double]
scaler.maximums      // [Double]
scaler.range         // ClosedRange<Double>
scaler.featureCount  // Int
```

## Classification Metrics

```swift
let cm = predictions.confusionMatrix(actual: truth)
cm.truePositives     // Int
cm.falsePositives    // Int
cm.trueNegatives     // Int
cm.falseNegatives    // Int
cm.accuracy          // Double (non-optional)
cm.precision         // Double? (nil if TP+FP == 0)
cm.recall            // Double? (nil if TP+FN == 0)
cm.f1Score           // Double? (nil if precision or recall is nil)

// Standalone
predictions.accuracy(actual: truth)    // Double
predictions.precision(actual: truth)   // Double?
predictions.recall(actual: truth)      // Double?
predictions.f1Score(actual: truth)     // Double?
```

## Regression Metrics

```swift
predicted.rSquared(actual: actual)              // Double (non-optional)
predicted.meanSquaredError(actual: actual)       // Double (non-optional)
predicted.rootMeanSquaredError(actual: actual)   // Double (non-optional)
```

## Linear Regression

```swift
let model = try LinearRegression.fit(features: trainX, targets: trainY)
// throws MatrixError.singular if features are linearly dependent

model.coefficients    // [Double] — [intercept, weight1, weight2, ...]
model.featureCount    // Int
model.hasIntercept    // Bool

let predictions = model.predict(testX)          // [[Double]] → [Double]
let singleFeature = model.predict(xValues)      // [Double] → [Double] (featureCount == 1)
```

## Gradient Descent

```swift
// Standardize features first — defaults assume unit variance.
let scaled = StandardScaler.fit(features: trainX).transform(trainX)

let gd = try GradientDescent.fit(
    features: scaled, targets: trainY,
    learningRate: 0.01, maxIterations: 1000, tolerance: 1.0e-6
)
// throws GradientDescentError.divergedNonFinite or .divergedIncreasing on divergence

gd.coefficients     // [Double] — same layout as LinearRegression: [intercept, weight1, ...]
gd.featureCount     // Int
gd.hasIntercept     // Bool
gd.learningRate     // Double — echoes the hyperparameter used
gd.iterations       // Int — count when the loop stopped
gd.finalLoss        // Double — loss at the returned coefficients
gd.lossHistory      // [Double] — loss at every iteration, including iteration 0
gd.outcome          // .converged | .maxIterationsReached

print(gd)           // GradientDescent: 2 features, converged in 142 iterations (loss: 0.0034)

let predictions = gd.predict(testX)             // [[Double]] → [Double]
let singleFeature = gd.predict(xValues)         // [Double] → [Double] (featureCount == 1)
```

Same `Regressor` protocol as `LinearRegression`. Same `coefficients` layout (intercept at index 0 when `hasIntercept` is true). The iterative route exists for the cases where no closed form is available, or where a penalty is added to the objective — `Ridge` (below) is the first model built on this optimizer, and the same descent loop will fit the iterative models that follow (logistic regression, SVM).

`Outcome.maxIterationsReached` is necessary but not sufficient for trustworthiness — confirm meaningful descent by comparing `lossHistory.first` to `lossHistory.last` before relying on the coefficients.

## Ridge Regression (1.3.0)

```swift
// Standardize first — the penalty compares coefficient magnitudes across features.
let scaled = StandardScaler.fit(features: trainX).transform(trainX)

let ridge = try Ridge.fit(
    features: scaled, targets: trainY,
    lambda: 0.1, learningRate: 0.01, maxIterations: 1000, tolerance: 1.0e-6
)
// throws GradientDescentError on divergence — shares Gradient Descent's failure modes

ridge.coefficients  // [Double] — [intercept, weight1, ...]; the intercept is never penalized
ridge.lambda        // Double — the penalty strength used
ridge.lossHistory   // [Double] — same observable trajectory as GradientDescent
ridge.outcome       // .converged | .maxIterationsReached

let predictions = ridge.predict(testX)
```

L2-regularized regression: minimizes `(1/n)‖Xθ − y‖² + λ‖θ‖²`, the squared-error objective plus a penalty on coefficient size that curbs overfitting and steadies the unstable fits collinear features produce. At `lambda` of zero the penalty vanishes and the fit reproduces ordinary least squares; as `lambda` grows the slopes shrink toward zero. The intercept is never penalized. Conforms to `Regressor`, so it substitutes for `LinearRegression` in any pipeline, and is fit by the same descent optimizer behind `GradientDescent`. Note `lambda` scales a bare penalty against a `1/n` error term, so its values are not interchangeable with conventions that fold in a `1/2m` or `λ/2m` factor. When the need for regularization is unclear, a large `conditionNumber` on the feature matrix is the collinearity the penalty is built to absorb.

## K-Means Clustering

```swift
let km = KMeans.fit(data: features, k: 3, seed: 42)
km.centroids     // [[Double]]
km.labels        // [Int]
km.inertia       // Double
km.iterations    // Int
km.featureCount  // Int
km.predict(newData)  // [Int]

// Best of multiple runs
let best = KMeans.bestFit(data: features, k: 3, attempts: 10)

// Elbow method
let inertias = KMeans.elbowMethod(data: features, kRange: Array(1...8), seed: 42)
```

## K-Nearest Neighbors

```swift
let knn = KNearestNeighbors.fit(
    features: trainX, labels: trainY,
    k: 3, metric: .euclidean, weight: .uniform
)
// DistanceMetric: .euclidean, .cosine
// VoteWeight: .uniform, .distance

knn.predict(testX)   // [Int] — raw labels for evaluation pipelines
knn.classify(testX)  // [Classification] — grouped by predicted label
knn.k                // Int
knn.metric           // DistanceMetric
knn.featureCount     // Int

// classify() groups inputs by predicted label (Classifier protocol)
// Each Classification conforms to Sequence — iterate with for-in
for group in knn.classify(testX) {
    print(group)  // Class 0: 3 points
    for point in group { print(point) }
}
```

## Gaussian Naive Bayes

```swift
let gnb = GaussianNaiveBayes.fit(features: trainX, labels: trainY)
print(gnb)  // GaussianNaiveBayes: 2 classes, 2 features

gnb.predict(testX)                    // [Int] — raw labels for evaluation pipelines
gnb.classify(testX)                   // [Classification] — grouped by predicted label
gnb.predictLogProbabilities(testX)    // [[Double]] — unnormalized log-probabilities
gnb.predictProbabilities(testX)       // [[Double]] — calibrated probabilities, each row sums to 1.0
gnb.featureCount                      // Int
gnb.classes                           // [ClassStats]

// ClassStats prints cleanly
for stats in gnb.classes {
    print(stats)  // Class 0: prior 50.0%, means [83.75, 32.50], 4 samples
}
```

## Panel (Columnar Data)

```swift
// Create
let panel = Panel([("age", [25.0, 30.0, 35.0]), ("salary", [50000.0, 60000.0, 70000.0])])
print(panel)  // Panel: 2 columns, 3 rows
let panel2 = Panel(["age": [25.0, 30.0], "salary": [50000.0, 60000.0]])
let panel3 = Panel(matrix: [[25.0, 50000.0], [30.0, 60000.0]], columns: ["age", "salary"])

// Access
panel["age"]              // [Double]
panel.labels("age")       // [Int]
panel.columnNames         // [String]
panel.rowCount            // Int
panel.shape               // (rows: Int, columns: Int) — same as matrix .shape
panel.toMatrix()          // [[Double]]

// Filter and split
panel.filtered(where: [true, false, true])
let (train, test) = panel.trainTestSplit(testRatio: 0.2, seed: 42)

// Display and summary statistics
panel.head()       // Tabular output showing first 10 rows
panel.head(n: 3)   // First 3 rows in tabular format
panel.summary()    // Per-column statistics (count, mean, std, min, max)
```

---

## Common Patterns

### Full ML pipeline

```swift
import Quiver

// Load data
let panel = Panel([("f1", features1), ("f2", features2), ("label", labels)])
let features = panel.toMatrix(columns: ["f1", "f2"])
let labels = panel.labels("label")

// Split
let (trainF, testF, trainL, testL) = features.stratifiedSplit(
    labels: labels, testRatio: 0.2, seed: 42
)

// Scale
let scaler = FeatureScaler.fit(features: trainF)
let scaledTrain = scaler.transform(trainF)
let scaledTest = scaler.transform(testF)

// Train
let model = GaussianNaiveBayes.fit(features: scaledTrain, labels: trainL)

// Evaluate
let predictions = model.predict(scaledTest)
let cm = predictions.confusionMatrix(actual: testL)
print(cm.accuracy)
```

### Semantic search

```swift
import Quiver

let query = "running shoes".tokenize()
let embeddings: [String: [Double]] = [...]   // pre-trained word vectors
if let queryVector = query.embed(using: embeddings).meanVector() {
    let results = documents.cosineSimilarities(to: queryVector)
    let top3 = results.topIndices(k: 3, labels: docNames)
    print(top3)
}
```

### Data visualization pipeline (Quiver + Swift Charts)

```swift
import Quiver

let raw = [100.0, 200.0, 300.0, 400.0, 500.0]
let scaled = raw.scaled(to: 0.0...1.0)       // for chart sizing
let zScores = raw.standardized()               // for comparison overlay
let shares = raw.asPercentages()               // for pie/donut charts
let bins = raw.histogram(bins: 5)              // for bar charts
```

---

## Performance Characteristics

92% of Quiver's API surface is O(n) or better. The remaining 8% are operations where higher complexity is mathematically inherent. All public APIs with complexity greater than O(n) include `/// - Complexity:` documentation in the source.

### O(n³) — Matrix algebra
`determinant`, `inverted()`, `logDeterminant`, `conditionNumber` — perform well for matrices up to a few hundred rows.

### O(n²) — Pairwise operations
`findDuplicates(threshold:)`, `clusterCohesion()` — perform well for collections up to low thousands. `multiplyMatrix(_:)` is O(n·m·p). `correlationMatrix()`, `heatmapData(labels:)` are O(n²·m).

### O(n·k·d) — ML model training and prediction
`KMeans.fit`, `bestFit`, `elbowMethod`, `clusters(from:)`, `predict` — scale with samples × clusters × features × iterations. `KNearestNeighbors.predict` — scales with queries × training samples × features. `LinearRegression.fit` — O(n·f² + f³) where f is feature count.

### O(n log n) — Sorting-based
`median()`, `percentile(_:)`, `quartiles()`, `percentileRanks()`, `topIndices(k:)`, `sortedIndices()` — sort internally. When computing multiple percentiles, `quartiles()` sorts once.

### Benchmarks (release build, Apple Silicon M-series, March 2026)

Quiver's performance is optimized for educational datasets (hundreds to low thousands of samples), on-device inference (real-time sensor data), and server-side queries (similarity search on embedded vectors).

| Operation | Data Size | Time | Memory Δ |
|---|---|---|---|
| Naive Bayes fit + predict | 10K train, 1K query, 20 features | 1ms | +0.2 MB |
| KMeans fit | 1K samples, k=5, 10 features | 1ms | +0.0 MB |
| Linear Regression fit | 5K samples, 10 features | 2ms | +2.5 MB |
| KNN Euclidean predict | 1K train, 100 query, 10 features | 3ms | +0.0 MB |
| KNN Cosine predict | 1K train, 100 query, 10 features | 4ms | +0.1 MB |
| Matrix Multiply | 100×100 | 30ms | +0.0 MB |
| Transpose | 500×500 | 32ms | +3.9 MB |
| Determinant | 150×150 | 97ms | +0.3 MB |
| findDuplicates | 500 vectors, 20 dims | <1ms | +0.0 MB |
| clusterCohesion | 500 vectors, 20 dims | <1ms | +0.0 MB |

**What these numbers mean for app developers:** Training a Naive Bayes classifier on 10,000 samples takes 1ms — fast enough for real-time use on watchOS. KNN prediction on 1,000 training points completes in 3ms — well within a 60fps frame budget. Matrix operations are comfortable up to a few hundred rows.

**Where performance scales and where it doesn't:** ML models, statistics, similarity operations, and boolean masking all scale linearly and handle tens of thousands of elements comfortably. Matrix multiplication, inversion, and determinant are cubic — they perform well up to a few hundred rows but grow rapidly beyond that. Pairwise operations (findDuplicates, clusterCohesion) are quadratic — use them on subsets, not entire datasets.

Run benchmarks locally: `swift test -c release --filter QuiverStressTests`

---

## Quick Reference: What Returns Optional?

| Returns Optional | Returns Non-Optional |
|---|---|
| `mean()` → `Double?` | `sum()` → `Double` |
| `median()` → `Double?` | `product()` → `Double` |
| `variance(ddof:)` → `Double?` | `magnitude` → `Double` |
| `standardDeviation(ddof:)` → `Double?` | `normalized` → `[Double]` |
| `quartiles()` → tuple? | `dot(_:)` → `Double` |
| `argMin()` → `Int?` | `cosineOfAngle(with:)` → `Double` |
| `argMax()` → `Int?` | `determinant` → `Double` |
| `percentile(_:)` → `Double?` | `softMax()` → `[Double]` |
| `meanVector()` → `[Double]?` | `sigmoid()` → `[Double]` |
| `averaged()` → `[Double]?` | `scaled(to:)` → `[Double]` |
| `cm.precision` → `Double?` | `standardized()` → `[Double]` |
| `cm.recall` → `Double?` | `cm.accuracy` → `Double` |
| `cm.f1Score` → `Double?` | `rSquared(actual:)` → `Double` |

---

## Documentation Guide

The sections below summarize Quiver's full documentation. Each topic explains **when** and **why** to use the APIs, not just how.

### Linear Algebra Primer

Quiver treats Swift arrays as mathematical vectors. A `[Double]` gains `magnitude`, `normalized`, `dot()`, `cosineOfAngle(with:)`, and `distance(to:)` through constrained extensions. Key concepts:

- **Arrays are vectors.** `[3.0, 4.0]` is a point in 2D vector space. `magnitude` = √(3² + 4²) = 5. `normalized` divides each element by magnitude → `[0.6, 0.8]` (unit vector preserving direction).
- **Vector space.** Every element in the array is a dimension. Two vectors close together = similar items. This is how ML works — flowers, documents, customers become comparable once represented as arrays.
- **Dot product.** Sum of element-wise products. Positive = same direction, zero = perpendicular, negative = opposite. Foundation for cosine similarity.
- **Cosine similarity.** Dot product divided by both magnitudes. Cancels length, measures only angle. Range: -1 (opposite) to 1 (identical). Powers recommendation engines, search, duplicate detection.
- **Matrices** are rectangular grids that transform vectors. `transformedBy()` applies the rule. Matrices can rotate, scale, reflect, shear, and compose transformations.
- **Distance** connects linear algebra to ML. `magnitude` = distance from origin. `distance(to:)` = distance between any two points. Every Quiver ML model uses distance internally.

### Determinants Primer

The determinant measures how a matrix scales space. For a 2×2 matrix `[[a,b],[c,d]]`, determinant = ad − bc.

- **Geometric meaning:** A determinant of 2 means the matrix doubles area. A determinant of 0 means the matrix collapses space into a lower dimension (singular — not invertible).
- **Invertibility:** Only matrices with non-zero determinant can be inverted. `try matrix.inverted()` throws `MatrixError.singular` if det = 0.
- **Condition number:** `matrix.conditionNumber` measures numerical stability. Values near 1 = well-conditioned. Values > 1000 = results may be unreliable. Always check before trusting an inverse.
- **Log determinant:** `matrix.logDeterminant` returns a `LogDeterminant` struct with `.sign`, `.logAbsValue`, and `.value`. Prevents overflow for large matrices where the raw determinant would exceed `Double.greatestFiniteMagnitude`.
- **How Quiver uses determinants:** `LinearRegression.fit()` solves the normal equation θ = (X'X)⁻¹X'y, which requires inverting X'X. If the feature vectors are linearly dependent, the determinant of X'X is zero and `fit()` throws `MatrixError.singular`. The determinant tells us whether the features contain enough independent information to solve the problem.
- **Diagnostic chain:** Check determinant → check condition number → attempt inversion → verify with `matrix.multiplyMatrix(inverse)` ≈ identity.

### Machine Learning Primer

Quiver's ML models follow a consistent pattern: `fit()` → `predict()` → evaluate.

- **Classification vs regression.** Classification predicts discrete categories (`[Int]` labels). Regression predicts continuous values (`[Double]` targets).
- **Features and labels.** Features are the input measurements (`[[Double]]` matrix, rows = samples, columns = measurements). Labels are what we predict (`[Int]` for classification, `[Double]` for regression).
- **Train/test split.** Never evaluate on training data. Use `trainTestSplit(testRatio:seed:)` or `stratifiedSplit(labels:testRatio:seed:)` to hold out evaluation data.
- **Feature scaling.** Use `FeatureScaler.fit(features:)` on training data only. Transform both train and test sets with the same scaler. Prevents features with large ranges from dominating. Distance-based models (`KNearestNeighbors`, `KMeans`) and iterative optimizers (`GradientDescent`) require scaling — the scaler and model must be persisted together. `LinearRegression` and `GaussianNaiveBayes` do not require scaling.
- **Models available:** GaussianNaiveBayes, KNearestNeighbors, KMeans, LinearRegression, GradientDescent, Ridge. All use static `fit()` methods — no unfitted state exists.
- **Model persistence.** All models conform to `Codable`. Train once, encode to JSON with `JSONEncoder`, decode on any platform with `JSONDecoder` — identical predictions guaranteed by `Equatable`. When scaling is used, persist both the scaler and model together. See the Model-Persistence documentation page for platform-specific guidance (iOS, watchOS, Vapor, SwiftData).
- **Naive Bayes variance.** The variance calculation uses population variance (dividing by n), which is the standard approach for Gaussian Naive Bayes classifiers. With small training sets (2-4 samples per class), this slightly underestimates the true spread, but the effect is negligible for typical dataset sizes.
- **Evaluation (after training):** `confusionMatrix(actual:)` for classification (accuracy, precision, recall, F1). `rSquared(actual:)`, `meanSquaredError(actual:)` for regression. `classificationReport(actual:)` for a formatted summary.
- **Loss (during training):** `GradientDescent.lossHistory` exposes per-iteration MSE loss as a `[Double]`, compatible with `rollingMean()` and Swift Charts. The first entry is the loss at θ = 0, the last is `finalLoss`. Other models use closed-form solutions (LinearRegression) or single-pass statistics (NaiveBayes), so they carry no iterative loss. KMeans `inertia` is the closest closed-form analog — sum of squared distances to centroids.

### Vector Operations

Vectors have magnitude (length), direction (normalized), and relationships to other vectors (dot product, angle, distance).

- **Magnitude:** `v.magnitude` — Pythagorean theorem extended to any dimension. `[3.0, 4.0].magnitude` = 5.0.
- **Normalization:** `v.normalized` — unit vector (length 1) preserving direction. Zero vector returns zero vector.
- **Dot product:** `v1.dot(v2)` — sum of element-wise products. Zero means perpendicular.
- **Angle:** `v1.angle(with: v2)` (radians), `v1.angleInDegrees(with: v2)` (degrees), `v1.cosineOfAngle(with: v2)` (raw cosine value).
- **Distance:** `v1.distance(to: v2)` — Euclidean distance. Used internally by KNN and KMeans.
- **Arithmetic:** `.add()`, `.subtract()`, `.multiply()` (Hadamard), `.divide()`. Named methods, not operators.
- **Matrix-vector:** `vector.transformedBy(matrix)` or `matrix.transform(vector)` — two syntaxes, same result.
- **Averaging:** `vectors.averaged()` and `vectors.meanVector()` — both return optionals. Key for building document vectors from word embeddings.

### Vector Projections

Decompose any vector into parallel and perpendicular components relative to a reference direction.

- **Scalar projection:** `v.scalarProjection(onto: ref)` — how far v reaches along ref (a number).
- **Vector projection:** `v.vectorProjection(onto: ref)` — the component of v parallel to ref (a vector).
- **Orthogonal component:** `v.orthogonalComponent(to: ref)` — the perpendicular remainder.
- **Reconstruction:** `parallel.add(perpendicular)` always equals the original vector.
- **Use cases:** Force decomposition on ramps, ball reflection off surfaces (`v − 2 × proj(v onto normal)`), course correction (groundspeed vs crosswind).
- **Connection to regression:** The normal equation projects the target vector onto the feature column space. The prediction is the parallel component; the residual error is the orthogonal component.

### Boolean Masking

Filter and select array elements using comparisons, logical conditions, and boolean masks.

- **Comparisons return `[Bool]`:** `isGreaterThan()`, `isLessThan()`, `isGreaterThanOrEqual()`, `isLessThanOrEqual()`, `isEqual(to:)` (takes array, not scalar).
- **Combine masks:** `.and()`, `.or()`, `.not` (computed property).
- **Apply masks:** `data.masked(by: mask)` extracts matching elements. `mask.trueIndices` returns positions.
- **Conditional selection:** `data.choose(where: mask, otherwise: otherArray)` — picks from first array where true, second where false. Both parameters are arrays.
- **Panel integration:** `panel.filtered(where: mask)` applies the same mask to all columns simultaneously.

### Statistical Operations

Three questions about any dataset: where is the center, how spread out, and which values are unusual.

- **Aggregation:** `sum()`, `product()` (non-optional). `argMin()`, `argMax()` (optional — return indices).
- **Central tendency:** `mean()`, `median()` — both optional. When they diverge, data is skewed.
- **Dispersion:** `variance(ddof:)`, `standardDeviation(ddof:)`, `standardError(ddof:)` — all optional. Default `ddof: 1` (sample); pass `ddof: 0` for an entire population.
- **Cumulative:** `cumulativeSum()`, `cumulativeProduct()` — non-optional. Running totals.
- **Outlier detection:** `outlierMask(threshold:)` — z-score method, returns `[Bool]`. Default std of 1.0 when all values identical.
- **Vector averaging:** `meanVector()` — element-wise mean across multiple vectors. Returns optional.
- **Info:** `.info()` — quick summary (count, mean, std, min, max; adds shape/size for matrices).

### Matrix Operations

Work with 2D arrays using element-wise arithmetic and linear algebra.

- **Element-wise:** `.add()`, `.subtract()`, `.multiply()` (Hadamard), `.divide()` — same as vectors but for matrices.
- **Scalar broadcast:** `matrix * 2.0`, `matrix + 10.0` — operators work for scalar-matrix.
- **Transpose:** `.transpose()` or `.transposed()` — flip rows and columns.
- **Matrix multiply:** `.multiplyMatrix(other)` — true matrix multiplication (dot products of rows × columns). Inner dimensions must match.
- **Column access:** `.column(at: index)` — extract a column as `[Double]`.
- **Determinant:** `.determinant` — non-optional. Zero means singular.
- **Inverse:** `try .inverted()` — throws `MatrixError.singular` or `.notSquare`.
- **Fractions:** `.asFractions()` on inverted matrices reveals rational structure behind decimal results.

### Shape and Size

Inspect matrix dimensions and convert between 1D and 2D without altering data.

- `.shape` returns `(rows: Int, columns: Int)` named tuple — only on `[[Numeric]]`.
- `.size` returns total element count (rows × columns). Differs from `.count` which returns row count only.
- Supports tuple destructuring: `let (stores, days) = sales.shape`.
- Compile-time safety: calling `.shape` on `[Double]` or `[String]` is a compile-time error.
- `flat.reshaped(rows:columns:)` — 1D → 2D, fills row-major. Total elements must equal rows × columns.
- `matrix.flattened()` — 2D → 1D, concatenates rows.
- `matrix.reshaped(rows:columns:)` — 2D → different 2D, flattens internally then reshapes.
- Round-trip: `matrix.flattened().reshaped(rows:columns:)` restores original.

### Array Generation

Static methods on `[Double]` (or `[Int]`) to create arrays with specific patterns.

- **1D:** `[Double].zeros(5)`, `.ones(5)`, `.full(5, value: 3.14)`.
- **2D:** `[Double].zeros(2, 3)`, `.ones(2, 3)`, `.full(2, 3, value: 7.0)` — called on `[Double]`, returns `[[Double]]`.
- **Special:** `[Double].identity(3)` (3×3 identity), `[Double].diag([1,2,3])` (diagonal matrix).
- **Sequences:** `[Double].linspace(start: 0, end: 1, count: 5)` — includes both endpoints. `[Double].arange(0, 10, step: 2.5)` — excludes end.

### Random Number Generation

Generate random arrays for testing, simulation, and initialization.

- **Uniform:** `[Double].random(5)` (0–1), `[Double].random(5, in: -1.0...1.0)`, `[Double].random(2, 3)` (2D).
- **Normal:** `[Double].randomNormal(5, mean: 0.0, standardDeviation: 1.0)`, `[Double].randomNormal(2, 3, mean: 5.0, standardDeviation: 2.0)` — uses Box-Muller transform.
- **Integer:** `[Int].random(5, in: 0..<100)` — half-open range.
- Works with both `Float` and `Double`.

### Broadcasting Operations

Apply operations between arrays and scalars or between arrays of different dimensions.

- **Scalar methods:** `.broadcast(adding:)`, `.broadcast(subtracting:)`, `.broadcast(multiplyingBy:)`, `.broadcast(dividingBy:)`.
- **Scalar operators:** `array + 2.0`, `2.0 * array`, `matrix * 0.5` — commutative.
- **Matrix-vector:** `.broadcast(addingToEachRow:)`, `.broadcast(addingToEachColumn:)`, `.broadcast(multiplyingEachRowBy:)`, `.broadcast(multiplyingEachColumnBy:)`.
- **Custom:** `.broadcast(with: value, operation: { $0 + $1 })`, `.broadcast(withRowVector:operation:)`.
- **When to use:** Broadcasting for scalar math on arrays (reads like math notation). `map` for custom/non-numeric transformations.

### Matrix Transformations

Matrices transform vectors through multiplication. Each row produces one element of the result via dot product.

- **Basic usage:** `vector.transformedBy(matrix)` or `matrix.transform(vector)`.
- **Basis vectors:** Column 1 = where i-hat [1,0] lands. Column 2 = where j-hat [0,1] lands. The result is a linear combination of columns weighted by the vector.
- **Identity:** `[Double].identity(2)` — leaves vectors unchanged. Starting point for building transformations.
- **Diagonal:** `[Double].diag([2.0, 3.0])` — scales each axis independently.
- **Dimension rule:** Matrix columns must match vector length.
- **Rotation:** `[[cos θ, -sin θ], [sin θ, cos θ]]`. 90° = `[[0,-1],[1,0]]`. Preserves magnitude.
- **Scaling:** Diagonal matrix. Uniform = same factor all axes. Non-uniform = different per axis.
- **Reflection:** Negate one axis. `reflectX = [[1,0],[0,-1]]`, `reflectY = [[-1,0],[0,1]]`. Preserves distances.
- **Shear:** Off-diagonal element. Horizontal shear `[[1,k],[0,1]]` shifts x proportionally to y.

### Composing Transformations

Combine multiple transformations via matrix multiplication.

- `A.multiplyMatrix(B)` means "first apply B, then apply A" (right-to-left order).
- **Two approaches:** Compose first then apply once (efficient for many vectors), or apply sequentially (simpler for single vectors).
- **Order matters:** `rotate.multiplyMatrix(scale) ≠ scale.multiplyMatrix(rotate)`.
- **Rotate around a point:** Translate to origin → rotate → translate back.
- **Performance:** Precompute composed matrix when applying to many vectors. Matrix multiplication is O(n³); matrix-vector is O(n²).
- **Inverse transformations:** Rotation inverse is the negative angle. Scaling inverse is the reciprocal. `try matrix.inverted()` for general inverse.

### Similarity Operations

Measure how related two vectors are using cosine similarity and distance metrics.

- **Dot product:** `v1.dot(v2)` — foundation for cosine similarity. Mixes alignment with magnitude.
- **Cosine similarity:** `v1.cosineOfAngle(with: v2)` — normalized dot product. Range -1 to 1. Direction only, magnitude ignored.
- **Magnitude vs distance:** Magnitude = "how far from origin." Distance = "how far between two points." Both use Pythagorean theorem.
- **Normalization:** `v.normalized` creates unit vector. For normalized vectors, dot product equals cosine similarity.
- **Angle:** `angle(with:)` applies acos to cosine value → radians. `angleInDegrees(with:)` → degrees.
- **Batch:** `database.cosineSimilarities(to: query)` — compare one vector against many.
- **Duplicate detection:** `documents.findDuplicates(threshold: 0.95)` — returns `[(index1, index2, similarity)]` sorted by similarity.
- **Cluster validation:** `cluster.clusterCohesion()` — average pairwise similarity within a group (0 to 1).

### Semantic Search

Find information by meaning, not keywords. Full pipeline: tokenize → embed → average → compare.

1. **Tokenize:** `"Running Shoes".tokenize()` → `["running", "shoes"]`. Lowercases, splits on whitespace, and removes punctuation by default. Pass `removingPunctuation: false` to keep punctuation.
2. **Embed:** `tokens.embed(using: embeddings)` → `[[Double]]`. Looks up each token in a `[String: [Double]]` dictionary. Unknown words silently skipped.
3. **Average:** `vectors.meanVector()` → `[Double]?`. Combines word vectors into one document vector.
4. **Compare:** `docVectors.cosineSimilarities(to: queryVector)` → `[Double]`. Rank by similarity.
5. **Top results:** `scores.topIndices(k: 3, labels: docNames)` → `[(label, score)]`.
- **Vector arithmetic captures relationships:** `king - man + woman ≈ queen` in vector space.
- **Pre-compute:** Store document vectors; only build query vector at search time.

### Panel

Organizes named columns of numeric data into a lightweight container focused on numeric ML workflows.

- **Create:** `Panel([("age", [25.0, 30.0]), ("income", [50000.0, 60000.0])])` or `Panel(matrix:columns:)`.
- **Access:** `panel["age"]` → `[Double]`. `panel.labels("age")` → `[Int]`. `panel.toMatrix(columns:)` → `[[Double]]`.
- **Properties:** `.columnNames`, `.rowCount`, `.shape` → `(rows: Int, columns: Int)` — same format as matrix `.shape`.
- **Filter:** `panel.filtered(where: boolMask)` — applies mask to all columns simultaneously.
- **Split:** `panel.trainTestSplit(testRatio:seed:)` — splits all columns atomically by the same rows.
- **Head:** `panel.head()` — tabular display of first 10 rows. `panel.head(n: 3)` for custom count.
- **Summary:** `panel.summary()` — per-column summary statistics.
- **Design:** Value type, fixed schema, all `Double` columns. Models accept `[[Double]]` and `[Int]` directly — Panel organizes data, models consume raw arrays.

### Train-Test Split

Split data into training and testing subsets for honest evaluation.

- **Basic:** `data.trainTestSplit(testRatio: 0.2, seed: 42)` → `(train, test)` named tuple. Works on any array type.
- **Paired arrays:** Use same seed for features and labels to keep rows aligned.
- **Reproducible:** Same seed + same data = same split every time.
- **Stratified:** `features.stratifiedSplit(labels:testRatio:seed:)` preserves class proportions. Returns 4-tuple: `(trainFeatures, testFeatures, trainLabels, testLabels)`.
- **Choosing ratio:** 0.2 is standard. 0.1 for small datasets. 0.3 for large datasets.
- **Class balance diagnostics:** `labels.classDistribution()` → `[Int: Int]` mapping each label to its count. `labels.imbalanceRatio()` → `Double?` ratio of largest to smallest class (1.0 = balanced, 4.0 = 4x imbalance, nil for empty/single-class). Developers set their own threshold to decide when to oversample.
- **Oversample:** `features.oversample(labels: labels)` → `(features: [[Double]], labels: [Int])`. Auto-detects smaller classes, generates synthetic points by interpolating between existing samples. Handles multi-class. Call before splitting.

### Feature Scaling

Normalize features to a common range so no single feature dominates.

- **Fit:** `FeatureScaler.fit(features: trainX)` or `FeatureScaler.fit(features: trainX, range: -1.0...1.0)`.
- **Transform:** `scaler.transform(data)` — applies the learned min/max statistics.
- **Rule:** Fit on training data only. Transform both train and test with the same scaler. Prevents data leakage.
- **Constant columns:** Mapped to lower bound of target range (no division by zero).
- **Properties:** `.minimums`, `.maximums`, `.range`, `.featureCount`.
- **Immutable:** Once fitted, statistics cannot change. Safe to reuse on production data.

### Data Visualization

Prepare data for Swift Charts and other visualization frameworks.

- **Scaling:** `scaled(to: range)` (min-max), `standardized()` (z-score), `asPercentages()` (share of total).
- **Stacking:** `series.stackedCumulative()` (absolute stacked area), `series.stackedPercentage()` (100% stacked bars).
- **Heatmap:** `vectors.correlationMatrix()` → `[[Double]]`. `vectors.heatmapData(labels:)` → `[(x, y, value)]` for `RectangleMark`.
- **Downsampling:** `data.downsample(factor:using:)` with `AggregationMethod`: `.mean`, `.max`, `.min`, `.sum`, `.count`.
- **Grouping:** `data.groupBy(categories, using: .sum)` → `[String: Double]`. `data.groupedData(by:using:)` → `[(category, value)]`.
- **Boolean filtering:** Split data into series using `.masked(by:)` for normal vs outlier chart layers.
- **ML visualization:** Confusion matrix → heatmap, KMeans elbow → line chart, regression → scatter + fitted line overlay, softMax → bar chart.

### Gaussian Naive Bayes

Simplest effective classifier. Assumes features are conditionally independent given the class label.

- **Fit:** `GaussianNaiveBayes.fit(features: trainX, labels: trainY)` — learns per-class means, variances, and priors.
- **Predict:** `model.predict(testX)` → `[Int]` — raw labels for evaluation pipelines.
- **Classify:** `model.classify(testX)` → `[Classification]` — groups inputs by predicted label, same pattern as KNN.
- **Log probabilities:** `model.predictLogProbabilities(testX)` → `[[Double]]` — one row per sample, one column per class. Unnormalized.
- **Probabilities:** `model.predictProbabilities(testX)` → `[[Double]]` — softmax of the log-probabilities, each row sums to 1.0. Use this for cost-sensitive decisions or threshold tuning rather than just the argmax label from `predict`.
- **Inspect:** `model.classes` → `[ClassStats]` with `.label`, `.prior`, `.means`, `.variances`, `.count`. Each `ClassStats` conforms to `CustomStringConvertible`: `print(stats)` shows `Class 0: prior 50.0%, means [...], N samples`.
- **Internal:** Uses `Distributions.normal.logPDF` for per-feature log-densities — the same public function callers can reach directly.
- **When to use:** Baseline classifier. Fast training. Works well with small datasets. Good first model before trying more complex approaches.

### K-Nearest Neighbors

Classifies by finding the k closest training examples and voting on their labels.

- **Fit:** `KNearestNeighbors.fit(features:labels:k:metric:weight:)`.
- **Metrics:** `.euclidean` (default), `.cosine` (for high-dimensional text/embeddings).
- **Weights:** `.uniform` (majority vote), `.distance` (closer neighbors have more influence).
- **Predict:** `model.predict(testX)` → `[Int]` — raw labels for evaluation pipelines.
- **Classify:** `model.classify(testX)` → `[Classification]` — groups inputs by predicted label. Each `Classification` conforms to `Sequence` with `.label`, `.points`, `.count`.
- **Properties:** `.k`, `.metric`, `.featureCount`.
- **When to use:** Non-linear decision boundaries. When similar items should be classified similarly. Feature scaling recommended.

### K-Means Clustering

Unsupervised grouping of data into k clusters.

- **Fit:** `KMeans.fit(data:k:seed:)` — assigns each point to nearest centroid, iterates until stable.
- **Properties:** `.centroids`, `.labels`, `.inertia` (total within-cluster distance), `.iterations`, `.featureCount`.
- **Predict:** `model.predict(newData)` → `[Int]` — assigns new points to existing clusters.
- **Best fit:** `KMeans.bestFit(data:k:attempts:)` — runs multiple times, returns lowest inertia.
- **Elbow method:** `KMeans.elbowMethod(data:kRange:seed:)` → `[Double]` inertias. Plot against k to find the "elbow."
- **Clusters:** `model.clusters(from: data)` → `[Cluster]`. Each `Cluster` has `.centroid` (`[Double]`), `.points` (`[[Double]]`), `.count` (`Int`), and conforms to `Sequence`.

### Linear Regression

Fits a line (or hyperplane) to continuous data using the normal equation.

- **Fit:** `try LinearRegression.fit(features: trainX, targets: trainY)` — throws `MatrixError.singular` if features are linearly dependent.
- **Predict:** `model.predict(testX)` → `[Double]` for `[[Double]]` input. `model.predict(xValues)` → `[Double]` for single-feature `[Double]` input.
- **Properties:** `.coefficients` (includes intercept as first element), `.featureCount`, `.hasIntercept`.
- **Evaluation:** `predicted.rSquared(actual:)`, `predicted.meanSquaredError(actual:)`, `predicted.rootMeanSquaredError(actual:)`.

### Evaluation Metrics

Measure model performance after prediction.

- **Classification:** `predictions.confusionMatrix(actual: truth)` → `ConfusionMatrix` with `.truePositives`, `.falsePositives`, `.trueNegatives`, `.falseNegatives`, `.accuracy` (non-optional), `.precision` (optional), `.recall` (optional), `.f1Score` (optional). Conforms to `Equatable` and `CustomStringConvertible`.
- **Classification report:** `predictions.classificationReport(actual: truth)` → per-class precision, recall, F1, and support with accuracy, macro avg, and weighted avg.
- **Standalone:** `predictions.accuracy(actual:)`, `.precision(actual:)`, `.recall(actual:)`, `.f1Score(actual:)`.
- **Regression:** `predicted.rSquared(actual:)` (1.0 = perfect), `.meanSquaredError(actual:)`, `.rootMeanSquaredError(actual:)`.

### Activation Functions

Convert raw scores (logits) into probabilities.

- **SoftMax:** `logits.softMax()` → `[Double]` summing to 1.0. For multi-class classification. Uses numerically stable variant (subtracts max before exp). Handles large values like `[1000, 1001, 1002]` without overflow.
- **Sigmoid:** `logits.sigmoid()` → `[Double]` each in (0,1). For binary classification. Element-wise, independent. Property: σ(x) + σ(−x) = 1.0.
- **When to use:** SoftMax for "which one?" (exactly one label). Sigmoid for "yes or no?" (per-element binary).
- **With Quiver models:** Built-in models handle probability conversion internally. These functions are most useful for external model output or custom scoring systems.

### Installation

Add via SPM: `.package(url: "https://github.com/waynewbishop/quiver", from: "1.0.0")`. Or use Xcode → File → Add Package Dependencies. Zero external dependencies. Verify with `[1.0, 2.0, 3.0].dot([4.0, 5.0, 6.0])` → 32.0.

### Usage (Xcode Playground Macro)

The `#Playground` macro (Xcode 26+) enables interactive exploration of Quiver APIs. Unlike `.playground` files, `#Playground` compiles as part of the project and can import SPM dependencies. Use `import Playgrounds` and `import Quiver`, wrap code in `#Playground { ... }`. Named blocks like `#Playground("Dot Product") { ... }` organize multiple experiments in one file. The Canvas shows each variable's value inline — no build-and-run cycle needed.

---

## Companion Book

Quiver is the companion framework for *Swift Algorithms & Data Structures* (5th Edition) by Wayne Bishop. Chapters 20–23 cover Quiver in depth. The book is available at https://waynewbishop.github.io/swift-algorithms/
