# Statistical Operations

Computing centers, spreads, and outliers from arrays of numerical data.

## Overview

Statistical functions are the foundation of data analysis in Quiver. They answer three questions about any dataset: where is the center, how spread out are the values, and which values are unusual. These measures feed directly into visualization, scaling, and machine learning workflows throughout the framework.

### Quick data overview

The `info` method provides a quick statistical summary of any array or matrix. For vectors, it reports count, type, mean, standard deviation, min, and max. For matrices, it adds shape and size:

```swift
import Quiver

let matrix: [[Double]] = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]

print(matrix.info())
// Matrix Information:
// Shape: (2, 3)
// Size: 6
// Type: Double.Type
// Mean: 3.5
// Std: 1.707825127659933
// Min: 1.0
// Max: 6.0
//
// First 2 rows:
// [0]: [1.0, 2.0, 3.0]
// [1]: [4.0, 5.0, 6.0]
```

These properties are also available individually on any matrix:

```swift
matrix.shape        // (rows: 2, columns: 3)
matrix.size         // 6
matrix.transposed() // [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

> Tip: For a deeper look at shape, size, and dimension operations, see <doc:Shape-And-Size>.

### Aggregation functions

Quiver provides functions to calculate basic aggregations on arrays:

```swift
let data = [4.0, 7.0, 2.0, 9.0, 3.0]

// Calculate the sum of all elements
let total = data.sum()  // 25.0

// Find the indices of minimum and maximum values
if let minIndex = data.argMin(), let maxIndex = data.argMax() {
    print(minIndex, maxIndex)  // 2, 3
}
```

> Tip: Use `argMin` and `argMax` to find not just the extreme values but also where they occur in the data. For the values themselves, use Swift's built-in `min` and `max` methods.

### Central tendency

`mean` and `median` both describe the center of a distribution, but they respond differently to extreme values. For the conceptual treatment of when to reach for each, see <doc:Statistics-Primer>.

```swift
import Quiver

let responseTimes = [12.0, 15.0, 14.0, 13.0, 16.0, 11.0, 450.0]

if let mean = responseTimes.mean(), let median = responseTimes.median() {
    print(mean)    // 75.86 (pulled up by 450)
    print(median)  // 14.0  (unaffected)
}
```

### Dispersion and variation

`variance` and `std` measure how far values spread from the mean. The `ddof` parameter (Delta Degrees of Freedom) selects between population statistics (`ddof: 0`, the default) and sample statistics (`ddof: 1`). For the conceptual background on variance and standard deviation, see <doc:Statistics-Primer>.

```swift
import Quiver

let data = [4.0, 7.0, 2.0, 9.0, 3.0]

// Population statistics (default, ddof: 0)
if let variance = data.variance(), let std = data.std() {
    print(variance)  // 6.8
    print(std)       // 2.61
}

// Sample statistics (ddof: 1) for data representing a subset
if let sampleVar = data.variance(ddof: 1), let sampleStd = data.std(ddof: 1) {
    print(sampleVar)  // 8.5
    print(sampleStd)  // 2.92
}
```

### Cumulative operations

Cumulative functions replace each element with the running total or running product up to that position. These are useful for tracking growth over time, computing running balances, and building empirical distribution functions.

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let cumSum = data.cumulativeSum()      // [1.0, 3.0, 6.0, 10.0, 15.0]
let cumProd = data.cumulativeProduct() // [1.0, 2.0, 6.0, 24.0, 120.0]
```

### Anomaly detection

`outlierMask(threshold:)` returns a boolean mask flagging values that exceed the given z-score threshold. The threshold is in units of standard deviations. For the concept behind z-scores and guidance on choosing a threshold, see <doc:Statistics-Primer>.

```swift
import Quiver

let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]

let mask = data.outlierMask(threshold: 2.0)
// [false, false, false, false, false, true, false]

// Extract the flagged values using boolean masking
let outliers = data.masked(by: mask)  // [35.0]

// Pre-calculate statistics when processing multiple arrays against the same baseline
guard let mean = data.mean(), let std = data.std() else {
    fatalError("Unable to calculate statistics for empty array")
}
let mask2 = data.outlierMask(threshold: 3.0, mean: mean, std: std)
```

> Important: When all values are identical (zero standard deviation), `outlierMask` defaults the standard deviation to `1.0` and no values are flagged as outliers.

### Resampling and inference

The `resampled(iterations:seed:statistic:)` method estimates the variability of any statistic by drawing many resamples from the original data with replacement and recomputing the statistic on each resample. Pair it with `percentileCI(level:)` on the resulting distribution to read off a percentile confidence interval. The two methods are designed to compose — resample a statistic, then take the percentile interval of the result. For the conceptual treatment of resampling and what a percentile interval means, see <doc:Statistics-Primer>.

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

// Resampled distribution of the sample mean
let resampledMeans = sessionSeconds.resampled(iterations: 1000, seed: 42) { resample in
    resample.mean() ?? 0.0
}

// 95% percentile confidence interval for the mean
if let ci = resampledMeans.percentileCI(level: 0.95) {
    print(ci.lower, ci.upper)   // ≈ 246, 254 — a plausible range for the population mean
}
```

The closure can return any statistic — median, quartile, ratio, difference of group means — and the same percentile-interval pattern applies. The `seed` parameter pins the randomness so the same input always produces the same resampled distribution.

### Vector averaging

Calculate the mean vector by averaging corresponding elements across multiple vectors. This operation computes the element-wise mean — essential for creating document vectors from word embeddings, computing centroids for clustering, and averaging feature vectors in machine learning:

```swift
import Quiver

// Average word embeddings to create a document vector
let wordEmbeddings = [
    [0.2, 0.5, -0.3, 0.8],   // "swift"
    [0.1, 0.6, 0.2, -0.4]    // "algorithms"
]
if let documentVector = wordEmbeddings.meanVector() {
    print(documentVector)  // [0.15, 0.55, -0.05, 0.2]
}
```

> Tip: The `meanVector` method is a key step in building semantic search systems — it combines multiple word vectors into a single document vector for similarity comparison. See <doc:Semantic-Search> for a complete walkthrough.

## Topics

### Basic aggregations
- ``Swift/Array/sum()``
- ``Swift/Array/product()``
- ``Swift/Array/argMin()``
- ``Swift/Array/argMax()``

### Central tendency
- ``Swift/Array/mean()``
- ``Swift/Array/median()``

### Dispersion measures
- ``Swift/Array/variance(ddof:)``
- ``Swift/Array/std(ddof:)``

### Cumulative statistics
- ``Swift/Array/cumulativeSum()``
- ``Swift/Array/cumulativeProduct()``

### Resampling and inference
- ``Swift/Array/resampled(iterations:seed:statistic:)``
- ``Swift/Array/percentileCI(level:)``

### Vector operations
- ``Swift/Array/meanVector()->[Double]?``
- ``Swift/Array/meanVector()->[Float]?``

### Related articles
- <doc:Data-Visualization>
- <doc:Train-Test-Split>
- <doc:Vector-Operations>
- <doc:Fourier-Transform>
