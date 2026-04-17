# Statistical Operations

Calculate common statistical measures from arrays of numerical data.

## Overview

Statistical functions are the foundation of data analysis in Quiver. They answer three questions about any dataset: where is the center, how spread out are the values, and which values are unusual. These measures feed directly into visualization, scaling, and machine learning workflows throughout the framework.

### Quick data overview

The `info()` method provides a quick statistical summary of any array or matrix. For vectors, it reports count, type, mean, standard deviation, min, and max. For matrices, it adds shape and size:

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

> Tip: Use `argMin()` and `argMax()` to find not just the extreme values but also where they occur in the data. For the values themselves, use Swift's built-in `min()` and `max()` methods.

### Central tendency

Mean and median both describe the center of a distribution, but they respond differently to extreme values. The mean shifts toward outliers; the median ignores them. When the two diverge significantly, it signals a skewed distribution — and `outlierMask()` can help identify the values responsible.

```swift
import Quiver

let responseTimes = [12.0, 15.0, 14.0, 13.0, 16.0, 11.0, 450.0]

if let mean = responseTimes.mean(), let median = responseTimes.median() {
    print(mean)    // 75.86 (pulled up by 450)
    print(median)  // 14.0  (unaffected)
}

// The gap between mean and median signals an outlier
let outliers = responseTimes.outlierMask(threshold: 2.0)
// [false, false, false, false, false, false, true]
```

### Dispersion and variation

Variance and standard deviation measure how far values spread from the mean. A low standard deviation means values cluster tightly; a high one means they are scattered. These two measures are the inputs to z-score standardization — dividing by the standard deviation converts any dataset to a common scale where values represent distance from the mean in standard-deviation units.

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

> Important: The `ddof` parameter (Delta Degrees of Freedom) controls the denominator. Use `ddof: 0` for population statistics when the data is the complete set. Use `ddof: 1` for sample statistics when the data is a subset of all possible observations.

### Cumulative operations

Cumulative functions replace each element with the running total or running product up to that position. These are useful for tracking growth over time, computing running balances, and building empirical distribution functions.

```swift
let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let cumSum = data.cumulativeSum()      // [1.0, 3.0, 6.0, 10.0, 15.0]
let cumProd = data.cumulativeProduct() // [1.0, 2.0, 6.0, 24.0, 120.0]
```

### Anomaly detection

Find values that deviate significantly from the norm using the z-score method. A z-score measures how many standard deviations a value is from the mean — the threshold parameter sets the cutoff:

```swift
import Quiver

let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]

// Find outliers (values more than 2 standard deviations from mean)
let mask = data.outlierMask(threshold: 2.0)
// [false, false, false, false, false, true, false]

// Extract outlier values using boolean masking
let outliers = data.masked(by: mask)  // [35.0]

// Pre-calculate statistics when processing multiple arrays
guard let mean = data.mean(), let std = data.std() else {
    fatalError("Unable to calculate statistics for empty array")
}
let mask2 = data.outlierMask(threshold: 3.0, mean: mean, std: std)
```

> Important: When all values are identical (zero standard deviation), `outlierMask` defaults the standard deviation to `1.0` and no values are flagged as outliers.

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

> Tip: The `meanVector()` method is a key step in building semantic search systems — it combines multiple word vectors into a single document vector for similarity comparison. See <doc:Semantic-Search> for a complete walkthrough.

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

### Vector operations
- ``Swift/Array/meanVector()->[Double]?``
- ``Swift/Array/meanVector()->[Float]?``

### Related articles
- <doc:Data-Visualization>
- <doc:Train-Test-Split>
- <doc:Vector-Operations>
- <doc:Fourier-Transform>
