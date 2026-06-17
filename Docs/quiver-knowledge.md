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

Models: `KMeans`, `KNearestNeighbors`, `GaussianNaiveBayes`, `LinearRegression`, `GradientDescent`, `Ridge`, `LogisticRegression`. Data: `Panel`. Result types: `ConfusionMatrix`, `Classification`, `Cluster`, `FeatureScaler`, `ClassStats`. Supporting types: `DistanceMetric`, `VoteWeight`, `Fraction`, `MatrixError`, `GradientDescentError`.

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

## Embedding Sources (1.4.0)

```swift
// Conform any text-to-vector source to Embedder — Quiver ships the contract, not the embedder.
struct TableEmbedder: Embedder {
    let table: [String: [Double]]
    func embed(_ text: String) -> [Double]? {
        text.tokenize().embed(using: table).meanVector()   // nil when no words are recognized
    }
}

let docs = ["a long slow rise", "knead the dough well", "proof the yeast"]
let embedded = docs.embedded(using: TableEmbedder(table: embeddings))
// [(text: String, vector: [Double])] — strings that embed to nil are dropped, text stays paired with its vector

if let query = TableEmbedder(table: embeddings).embed("how long should bread rise") {
    let hits = embedded.mostSimilar(to: query, k: 3)
    // [(rank: Int, text: String, score: Double)] — highest score first
}
```

`Embedder` is a one-method protocol — `embed(_ text: String) -> [Double]?` — that turns text into a fixed-dimension vector, or `nil` for empty text or text with no recognized tokens. Quiver defines the contract and ships no embedder: swapping a small word-vector table for a richer on-device sentence model changes the one line that creates the embedder, and every line that tokenizes, ranks, and reports stays as written. `[String].embedded(using:)` batches the call into `(text, vector)` pairs (dropping `nil` results, so a skipped string never shifts another's label), and `.mostSimilar(to:k:)` ranks those pairs against a query as `(rank, text, score)`. The produced vectors are plain `[Double]`, so they flow straight into `cosineSimilarities`/`topIndices`. This is the pluggable generalization of the dictionary-based Embedding Dictionary Search above (the retrieval step in a retrieval-augmented generation pipeline); `embed(_:)` here is distinct from the `[String].embed(using:)` token-table lookup it may call internally.

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

```swift
// Class balance — inspect and correct imbalance before training a classifier
let counts = labels.classDistribution()        // [Int: Int] — samples per label
let ratio = labels.imbalanceRatio()             // Double? — largest class ÷ smallest (nil if empty)
let (balancedX, balancedY) = features.oversample(labels: labels)  // synthesizes minority rows to parity
```

`classDistribution()` counts samples per label; `imbalanceRatio()` returns the largest class divided by the smallest (`nil` on empty input), where `1.0` is perfectly balanced and higher values signal skew. `oversample(labels:)` brings every smaller class up to the majority count by generating synthetic points interpolated between existing class members (not plain duplicates), returning the rebalanced features and labels together. Oversample before splitting, and never on the test set alone.

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
let oneValue = model.predict(2000.0)            // Double → Double (single sample, featureCount == 1)
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
let oneValue = gd.predict(2000.0)               // Double → Double (single sample, featureCount == 1)
```

Same `Regressor` protocol as `LinearRegression`. Same `coefficients` layout (intercept at index 0 when `hasIntercept` is true). The iterative route exists for the cases where no closed form is available, or where a penalty is added to the objective. This one descent loop is shared across all three iterative models: `GradientDescent` (squared-error loss), `Ridge` (squared-error plus an L2 penalty), and `LogisticRegression` (cross-entropy loss with a sigmoid hypothesis). The same step rule, convergence test, and divergence guard serve all three — only the gradient and loss handed to the loop change.

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
let oneValue = ridge.predict(2000.0)            // Double → Double (single sample, featureCount == 1)
```

L2-regularized regression: minimizes `(1/n)‖Xθ − y‖² + λ‖θ‖²`, the squared-error objective plus a penalty on coefficient size that curbs overfitting and steadies the unstable fits collinear features produce. At `lambda` of zero the penalty vanishes and the fit reproduces ordinary least squares; as `lambda` grows the slopes shrink toward zero. The intercept is never penalized. Conforms to `Regressor`, so it substitutes for `LinearRegression` in any pipeline, and is fit by the same descent optimizer behind `GradientDescent`. Note `lambda` scales a bare penalty against a `1/n` error term, so its values are not interchangeable with conventions that fold in a `1/2m` or `λ/2m` factor. When the need for regularization is unclear, a large `conditionNumber` on the feature matrix is the collinearity the penalty is built to absorb.

## Logistic Regression (1.4.0)

```swift
// Standardize first — defaults assume unit variance. Binary labels (0/1) only.
let scaled = StandardScaler.fit(features: trainX).transform(trainX)

let lr = try LogisticRegression.fit(
    features: scaled, labels: trainY,
    learningRate: 0.5, maxIterations: 1000, tolerance: 1.0e-6
)
// throws GradientDescentError on divergence — shares Gradient Descent's failure modes

lr.coefficients         // [Double] — [intercept, weight1, ...]
lr.featureCount         // Int
lr.hasIntercept         // Bool
lr.finalLoss            // Double — final cross-entropy (log loss)
lr.lossHistory          // [Double] — loss at every iteration, starting at log 2 (θ = 0)
lr.outcome              // .converged | .maxIterationsReached

print(lr)               // LogisticRegression: 2 features, converged in 48 iterations (loss: 0.6284)

let labels = lr.predict(scaledTest)              // [[Double]] → [Int] (threshold at 0.5)
let one = lr.predict(2.5)                        // Double → Int (single sample, featureCount == 1)
let probs = lr.predictProbabilities(scaledTest)  // [[Double]] → [Double] — P(class = 1) per sample
let scores = lr.decisionFunction(scaledTest)     // [[Double]] → [Double] — raw log-odds Xθ
let groups = lr.classify(scaledTest)             // [Classification], grouped by predicted label
```

Binary classifier trained by gradient descent on cross-entropy loss: the linear score `Xθ` is squashed through `sigmoid` into a probability, and the gradient `(1/n)Xᵀ(σ(Xθ) − y)` shares its shape with the squared-error gradient — the same descent loop behind `GradientDescent`, applied to a different loss. Conforms to `Classifier`, so `predict(_:)` returns `[Int]` and `classify(_:)` is provided for free. `predictProbabilities(_:)` returns a single probability per sample — P(class = 1) — not a per-class row that sums to 1.0 (that is the `GaussianNaiveBayes` shape). `decisionFunction(_:)` exposes the raw log-odds before the sigmoid for threshold tuning and margin inspection. Labels must be `0` or `1`; multinomial is out of scope. On linearly separable data the maximum-likelihood fit has no finite minimum, so the run reaches `maxIterationsReached` by design while still predicting correctly — confirm a `.converged` outcome on overlapping data before relying on the coefficient magnitudes.

## Residual Model (1.4.0)

```swift
// Fit a regressor first, then wrap it — ResidualModel holds no state of its own.
let baseline = try LinearRegression.fit(features: trainX, targets: trainY)
let residualModel = ResidualModel(model: baseline)

let gaps = residualModel.residuals(features: laterX, targets: laterY)  // [Double] — observed − predicted, per sample
let gap  = residualModel.residual(features: oneRow, observed: 162.0)   // Double — scalar form
let yhat = residualModel.expected(laterX)                              // [Double] — pass-through to the wrapped model's predict
residualModel.model                                                    // the wrapped regressor
```

`ResidualModel<Model: Regressor>` wraps any fitted regressor and reports what the model could not explain: the residual `observed − predicted`. It composes like `Pipeline` — the caller fits the regressor, then wraps it — and forwards `expected(_:)` straight to the wrapped model's `predict`. Computing a residual needs both the features and the observed target (there is no `observed − predicted` without the observed). Fit the baseline on one period and residualize a *later* sample: residualizing the training data understates the true error.

The `Coefficients` protocol travels with this type. It exposes a uniform `coefficients: [Double]` (intercept first), and `LinearRegression`, `Ridge`, and `GradientDescent` conform; a `ResidualModel` forwards its wrapped model's coefficients when that model conforms. `LogisticRegression` deliberately does not conform — it is a classifier, and a residual is undefined for a probability or a 0/1 label. Distance- and tree-based models do not conform either, which is why the capability is its own protocol rather than a requirement on every regressor.

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
knn.predict(8.0)     // Int — single sample, featureCount == 1
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
gnb.predict(8.0)                      // Int — single sample, featureCount == 1
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

Quiver is optimized for hundreds to low thousands of samples: linear operations (ML models, statistics, similarity, boolean masking) handle tens of thousands comfortably, cubic matrix operations stay fast to a few hundred rows, and quadratic pairwise operations are best on subsets. Run benchmarks locally with `swift test -c release --filter QuiverStressTests`.

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
- **Models available:** GaussianNaiveBayes, KNearestNeighbors, KMeans, LinearRegression, GradientDescent, Ridge, LogisticRegression. All use static `fit()` methods — no unfitted state exists.
- **Scalar prediction.** Single-feature models accept a scalar in place of a one-row matrix: `model.predict(3500.0)` returns one value — `Double` for regressors, `Int` for classifiers. The convenience is protocol-provided on every `Regressor` and `Classifier`; it applies only when the model was trained on a single feature.
- **Model persistence.** All models conform to `Codable`. Train once, encode to JSON with `JSONEncoder`, decode on any platform with `JSONDecoder` — identical predictions guaranteed by `Equatable`. When scaling is used, persist both the scaler and model together. See the Model-Persistence documentation page for platform-specific guidance (iOS, watchOS, Vapor, SwiftData).
- **Naive Bayes variance.** The variance calculation uses population variance (dividing by n), which is the standard approach for Gaussian Naive Bayes classifiers. With small training sets (2-4 samples per class), this slightly underestimates the true spread, but the effect is negligible for typical dataset sizes.
- **Evaluation (after training):** `confusionMatrix(actual:)` for classification (accuracy, precision, recall, F1). `rSquared(actual:)`, `meanSquaredError(actual:)` for regression. `classificationReport(actual:)` for a formatted summary.
- **Loss (during training):** `GradientDescent.lossHistory` exposes per-iteration MSE loss as a `[Double]`, compatible with `rollingMean()` and Swift Charts. The first entry is the loss at θ = 0, the last is `finalLoss`. Other models use closed-form solutions (LinearRegression) or single-pass statistics (NaiveBayes), so they carry no iterative loss. KMeans `inertia` is the closest closed-form analog — sum of squared distances to centroids.

### Vector Operations

See the Vector Arithmetic section above for signatures. Key idea: vectors carry magnitude (length), direction (normalized), and relationships to other vectors (dot product zero = perpendicular; distance drives KNN and KMeans). Arithmetic uses named methods (`.add()`, `.subtract()`, `.multiply()`), not operators.

### Vector Projections

Decompose any vector into parallel and perpendicular components relative to a reference direction.

- **Scalar projection:** `v.scalarProjection(onto: ref)` — how far v reaches along ref (a number).
- **Vector projection:** `v.vectorProjection(onto: ref)` — the component of v parallel to ref (a vector).
- **Orthogonal component:** `v.orthogonalComponent(to: ref)` — the perpendicular remainder.
- **Reconstruction:** `parallel.add(perpendicular)` always equals the original vector.
- **Use cases:** Force decomposition on ramps, ball reflection off surfaces (`v − 2 × proj(v onto normal)`), course correction (groundspeed vs crosswind).
- **Connection to regression:** The normal equation projects the target vector onto the feature column space. The prediction is the parallel component; the residual error is the orthogonal component.

### Boolean Masking

See the Boolean Masking section above for signatures. Key idea: comparisons return `[Bool]`, combine with `.and()`/`.or()`/`.not`, apply with `data.masked(by:)` or `data.choose(where:otherwise:)`. Panel integration: `panel.filtered(where: mask)` applies the same mask to all columns simultaneously.

### Statistical Operations

See the Statistics section above for signatures. Key idea: when `mean()` and `median()` diverge, the data is skewed. Dispersion methods (`variance`, `standardDeviation`, `standardError`) default to `ddof: 1` (sample); pass `ddof: 0` for a full population.

### Matrix Operations

See the Matrix Operations section above for signatures. Key idea: `.asFractions()` on an inverted matrix reveals the rational structure behind decimal results — useful for teaching and verification.

### Shape and Size

See the Shape and Size section above for signatures. Two gotchas worth keeping: `.size` (total elements, rows × columns) differs from `.count` (row count only), and calling `.shape` on a `[Double]` or `[String]` is a compile-time error — it exists only on `[[Numeric]]`.

### Broadcasting Operations

See the Broadcasting section above for signatures. When to use: broadcasting reads like math notation for scalar/row/column math on arrays; reach for `map` when the transformation is custom or non-numeric.

### Matrix Transformations

See the Matrix Transformations section above for signatures. Key idea: a matrix's columns are where the basis vectors land — column 1 is where `[1,0]` goes, column 2 where `[0,1]` goes — and the result is a linear combination of those columns. Rotation `[[cos θ, -sin θ], [sin θ, cos θ]]` preserves magnitude; a diagonal matrix scales each axis.

### Composing Transformations

`A.multiplyMatrix(B)` means "first apply B, then apply A" (right-to-left). Order matters: `rotate.multiplyMatrix(scale) ≠ scale.multiplyMatrix(rotate)`. Compose once then apply to many vectors for efficiency.
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

See the Panel (Columnar Data) section above for signatures. Design note: `Panel` is a value type with a fixed schema and all-`Double` columns. It organizes data, but models consume raw `[[Double]]`/`[Int]` directly — Panel never sits between the data and the model.

### Train-Test Split

See the Sampling and Splitting section above for signatures (`trainTestSplit`, `stratifiedSplit`, `classDistribution`, `imbalanceRatio`, `oversample`). Choosing a ratio: 0.2 is standard, 0.1 for small datasets, 0.3 for large. Always use the same seed across paired feature/label arrays to keep rows aligned.

### Feature Scaling

See the Feature Scaling section above for signatures. Two rules worth keeping: fit on training data only, then transform both train and test with the same scaler (fitting on the full set leaks the test distribution into the fit); and constant columns map to the lower bound of the target range, so there is no division by zero.

### Data Visualization

See the Data Visualization section above for signatures. ML-specific mappings: confusion matrix → heatmap, KMeans elbow → line chart, regression → scatter plus fitted-line overlay, softMax → bar chart.

### Gaussian Naive Bayes

See the Gaussian Naive Bayes section above for signatures. When to use: a fast baseline classifier that works well on small datasets — a good first model before trying more complex approaches. Prefer `predictProbabilities` over the argmax `predict` label when decisions are cost-sensitive or need threshold tuning.

### K-Nearest Neighbors

See the K-Nearest Neighbors section above for signatures. When to use: non-linear decision boundaries, or when similar items should be classified similarly. Feature scaling is recommended, and `.cosine` suits high-dimensional text/embeddings.

### K-Means Clustering

See the K-Means Clustering section above for signatures. When to use: unsupervised grouping. Use `elbowMethod(data:kRange:seed:)` to find the elbow that suggests a good `k`, and `bestFit(data:k:attempts:)` to escape unlucky initializations by keeping the lowest-inertia run.

### Linear Regression

See the Linear Regression section above for signatures. When to use: roughly linear relationships, exact one-pass closed-form fit. `.coefficients` carries the intercept as its first element (the `Coefficients` protocol; see Residual Model). For inferential questions (standard errors, p-values), pair `fit` with `summary`.

### Logistic Regression

See the Logistic Regression section above for signatures. When to use: binary classification (labels 0/1). It is trained iteratively on cross-entropy, so standardize features first and confirm `.outcome == .converged` before trusting coefficient magnitudes — on perfectly separable data it reaches the iteration cap by design. Use `predictProbabilities` for P(class = 1) and `decisionFunction` for raw log-odds. See the Logistic-Regression documentation page.

### Residual Model

See the Residual Model section above for signatures. When to use: to surface what a fitted regressor could not explain (`observed − predicted`). Wrap any fitted `Regressor`, then residualize a *later* sample — residualizing the training data understates the true error. The `Coefficients` protocol is documented there too: `LinearRegression`, `Ridge`, and `GradientDescent` conform, exposing a uniform `coefficients: [Double]`; `LogisticRegression` does not, because a residual is undefined for a classifier. See the Residual-Model documentation page.

### Embedding Sources

See the Embedding Sources section above for signatures. When to use: to plug any text-to-vector source into Quiver's similarity surface (the retrieval step in a retrieval-augmented generation pipeline). Conform a type to `Embedder` (one method, `embed(_:) -> [Double]?`), batch with `[String].embedded(using:)`, and rank with `.mostSimilar(to:k:)`. This is the pluggable generalization of the dictionary-based Embedding Dictionary Search; the contract stays fixed when the source changes. See the Embedding-Sources documentation page.

### Evaluation Metrics

See the Classification Metrics and Regression Metrics sections above for signatures (`confusionMatrix`, `classificationReport`, standalone `accuracy`/`precision`/`recall`/`f1Score`, `rSquared`, `meanSquaredError`, `rootMeanSquaredError`).

### Activation Functions

See the Activation section above for signatures. When to use: SoftMax for "which one?" (exactly one label, outputs sum to 1.0); Sigmoid for "yes or no?" (per-element, independent). Quiver's built-in models convert probabilities internally — these are for external model output or custom scoring.

### Installation

Add via SPM: `.package(url: "https://github.com/waynewbishop/quiver", from: "1.0.0")`. Or use Xcode → File → Add Package Dependencies. Zero external dependencies. Verify with `[1.0, 2.0, 3.0].dot([4.0, 5.0, 6.0])` → 32.0.

### Usage (Xcode Playground Macro)

The `#Playground` macro (Xcode 26+) enables interactive exploration of Quiver APIs. Unlike `.playground` files, `#Playground` compiles as part of the project and can import SPM dependencies. Use `import Playgrounds` and `import Quiver`, wrap code in `#Playground { ... }`. Named blocks like `#Playground("Dot Product") { ... }` organize multiple experiments in one file. The Canvas shows each variable's value inline — no build-and-run cycle needed.

---

## Companion Book

Quiver is the companion framework for *Swift Algorithms & Data Structures* (5th Edition) by Wayne Bishop. Chapters 20–23 cover Quiver in depth. The book is available at https://waynewbishop.github.io/swift-algorithms/
