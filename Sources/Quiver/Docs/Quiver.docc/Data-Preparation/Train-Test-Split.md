# Train-Test Split

Splitting arrays into training and testing subsets with reproducible, class-balanced partitions.

## Overview

Training a model on the same data used to evaluate it produces misleadingly optimistic results. The standard practice is to split data into two partitions — a [training set](<doc:Machine-Learning-Primer>) the model learns from and a test set held back for evaluation. Quiver provides `trainTestSplit(testRatio:seed:)` as an extension on `Array`, making this fundamental operation available on any array type.

> Tip: For an introduction to matrices as structured datasets — where rows represent samples and columns represent features — see [Matrices](https://waynewbishop.github.io/swift-algorithms/21-matrices.html) in Swift Algorithms & Data Structures.

### Basic usage

Split an array into training and testing subsets by specifying the fraction of elements reserved for testing:

```swift
import Quiver

let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)

// train — 8 elements for learning
// test  — 2 elements for evaluation
```

The method returns a named tuple of `(train: [Element], test: [Element])`. This follows the same named tuple pattern as `.shape`, which returns `(rows: Int, columns: Int)` — the labels are built into the return type, so they are available automatically at the call site.

### Reproducible splits with seeds

The `seed` parameter ensures that the same array with the same seed always produces the same split. This is essential for reproducible experiments — running the same code twice should yield the same training and test sets:

```swift
import Quiver

let scores = [85.0, 92.0, 78.0, 95.0, 88.0, 73.0, 91.0, 84.0]

// Both calls produce identical results
let first = scores.trainTestSplit(testRatio: 0.25, seed: 7)
let second = scores.trainTestSplit(testRatio: 0.25, seed: 7)

// first.train == second.train  ✓
// first.test  == second.test   ✓
```

Changing the seed produces a different shuffle order, which means different elements land in training versus testing. This allows us to experiment with multiple random splits to verify that results are not sensitive to a particular partition.

### Splitting paired arrays

In a typical machine learning workflow, we have two parallel arrays — features and labels — that must stay aligned after splitting. The seed parameter guarantees identical shuffling across both calls, so each feature row remains paired with its correct label:

```swift
import Quiver

// Feature matrix — each row is a house [sq_ft, bedrooms]
let features: [[Double]] = [
    [1400, 3], [1600, 3], [1700, 2], [1875, 3], [1100, 2],
    [1550, 2], [2350, 4], [2450, 4], [1425, 3], [1700, 3]
]

// Labels — price for each house
let prices: [Double] = [245000, 312000, 279000, 308000, 199000,
                         219000, 405000, 324000, 319000, 255000]

// Same seed keeps features and labels aligned
let (trainFeatures, testFeatures) = features.trainTestSplit(testRatio: 0.2, seed: 42)
let (trainPrices, testPrices) = prices.trainTestSplit(testRatio: 0.2, seed: 42)
```

The same seed produces the same index permutation regardless of element type. A `[[Double]]` feature matrix and a `[Double]` label array shuffled with the same seed will place index 0 in the same partition, index 1 in the same partition, and so on.

### Splitting features and labels together

In other numerical computing ecosystems, train-test splitting is typically a single function that accepts multiple arrays and returns all partitions at once. This produces four or more return values that must be unpacked in the correct positional order — and getting that order wrong is a common source of silent bugs.

Quiver takes a different approach. Each call splits one array and returns a named tuple with exactly two values: `.train` and `.test`. This design has three advantages:

**No positional ordering bugs**. With a single multi-array function, swapping two variables in the unpack silently corrupts the data. With named tuples, each destructure is self-contained — `trainFeatures` and `testFeatures` cannot be confused with `trainPrices` and `testPrices`.

**Works with any number of arrays**. If we have features, labels, and sample weights, we add a third call with the same seed. A single-function approach would need a new parameter for every additional array.

### Choosing a test ratio

A `testRatio` of `0.2` (80% training, 20% testing) is the most common choice and a good default for most datasets. For smaller datasets where every training sample matters, `0.1` reserves more data for learning. For very large datasets, even `0.3` or higher can work since there are enough training samples either way.

The ratio must be between 0 and 1, exclusive — a ratio of 0 or 1 would produce an empty partition, which is never useful.

### Stratified splitting

Random splitting works well for balanced datasets, but it can produce skewed partitions when classes are imbalanced. If only 5% of samples belong to the positive class, a random split might concentrate most of them in one partition, leaving the other with too few to learn from or evaluate against.

`stratifiedSplit(labels:testRatio:seed:)` solves this by splitting each class independently, so both partitions reflect the original class ratios:

```swift
import Quiver

let features: [[Double]] = [
    [619, 15000], [502, 78000], [699, 0],
    [850, 11000], [645, 125000], [720, 98000],
    [410, 45000], [780, 0], [590, 175000], [680, 62000]
]
let labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

let split = features.stratifiedSplit(labels: labels, testRatio: 0.2, seed: 42)
// split.trainFeatures, split.testFeatures
// split.trainLabels, split.testLabels
// Both partitions preserve the 50/50 class ratio
```

Unlike `trainTestSplit`, which takes one array at a time, `stratifiedSplit` takes both features and labels together. This is necessary because the method needs to see the class labels to determine how to divide each group. The return value is a named 4-tuple, so each partition is unambiguous at the call site.

### Works with any element type

Because splitting is pure index shuffling and slicing, `trainTestSplit` has no type constraint on the array's elements. It works on `[Double]`, `[String]`, `[[Double]]`, custom structs, or any other Swift type:

```swift
import Quiver

let labels = ["cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog"]
let (trainLabels, testLabels) = labels.trainTestSplit(testRatio: 0.25, seed: 42)
```

### Detecting class imbalance

Before training, `imbalanceRatio` measures how skewed the class distribution is. A ratio of 1.0 means all classes have the same number of samples. Higher values indicate greater imbalance — a ratio of 4.0 means the largest class has four times as many samples as the smallest:

```swift
import Quiver

let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
labels.classDistribution()  // [0: 8, 1: 2]
labels.imbalanceRatio()     // 4.0
```

Developers set their own threshold based on the domain. A common guideline: ratios above 3.0 suggest oversampling before training:

```swift
if let ratio = labels.imbalanceRatio(), ratio > 3.0 {
    let (balanced, balancedLabels) = features.oversample(labels: labels)
    // train on balanced data
}
```

### Oversampling imbalanced data

Stratified splitting preserves class ratios, but if the smaller class has very few samples, even a proportional split leaves the model with too little data to learn its pattern. `oversample(labels:)` addresses this by generating synthetic samples before splitting:

```swift
import Quiver

let features: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [1.2, 2.1],
    [7.0, 8.0], [7.5, 8.5]
]
let labels = [0, 0, 0, 0, 1, 1]

// Balance first, then split
let (balanced, balancedLabels) = features.oversample(labels: labels)
let split = balanced.stratifiedSplit(
    labels: balancedLabels, testRatio: 0.2, seed: 42
)
```

The method auto-detects which classes are smaller and generates new samples by interpolating between existing points in vector space. For multi-class data, every class below the largest count is oversampled independently. Call `oversample` before `stratifiedSplit` so that both the training and test sets contain enough examples from every class.

## Topics

### Splitting
- ``Swift/Array/trainTestSplit(testRatio:seed:)``
- ``Swift/Array/stratifiedSplit(labels:testRatio:seed:)``
- ``Swift/Array/oversample(labels:)``

### Related
- <doc:Pipeline>
- <doc:Feature-Scaling>
- <doc:Machine-Learning-Primer>

