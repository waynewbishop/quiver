# Panel

A Quiver type that organizes named columns of numeric data into a single container.

## Overview

Panel takes a matrix of rows and pivots it into named columns, where each column represents a [feature or label](<doc:Machine-Learning-Primer>). Each column is a `[Double]` — effectively a named vector. The data is the same, just organized by column instead of by row. Without Panel, features offer no indication of what each column represents, and splitting or filtering requires careful coordination across parallel arrays to keep rows aligned.

With Panel, each column gets a name and all rows stay together as a unit. It serves as a lightweight container for labeled column data, scoped to Quiver's numeric focus.

> Important: `Panel` does not replace Quiver's array and matrix operations — it organizes them. Each column is a standard `[Double]` that supports Quiver vector operations like `.mean()`, `.std()`, and boolean masking.

### Creating a panel

Build a panel from an ordered list of named columns. All columns must have the same number of elements:

```swift
import Quiver

let data = Panel([
    ("age", [25.0, 30.0, 35.0, 28.0]),
    ("income", [50000.0, 60000.0, 75000.0, 55000.0]),
    ("score", [0.8, 0.6, 0.9, 0.7])
])
```

### Creating a panel from a matrix

`Panel` can also be created from an existing matrix by providing column names. This is useful when we already have data in `[[Double]]` form — from a computation, a file import, or another Quiver operation — and want to add column labels:

```swift
import Quiver

// Existing matrix from a previous calculation
let results = [[8.5, 2.1], [7.2, 2.4], [9.1, 1.9]]

// Wrap it in a Panel to give columns meaningful names
let athletes = Panel(matrix: results, columns: ["speed", "jumpHeight"])
if let avgSpeed = athletes["speed"].mean() {
    print(avgSpeed)  // 8.27
}
if let stdJump = athletes["jumpHeight"].std() {
    print(stdJump)  // 0.21
}
```

### Column access

Access any column by name with subscript syntax. The returned array is a standard `[Double]` vector, so all Quiver vector operations work immediately:

```swift
import Quiver

let data = Panel([
    ("age", [25.0, 30.0, 35.0, 28.0]),
    ("income", [50000.0, 60000.0, 75000.0, 55000.0])
])

// Access by name — returns a [Double] vector
if let avgIncome = data["income"].mean() {
    print(avgIncome)  // 60000.0
}
data["income"].standardized()  // z-scores
```

For classification workflows, use `.labels()` to extract a column as `[Int]` for classifiers:

```swift
let trainLabels = data.labels("age")  // [25, 30, 35, 28]
```

### Extracting matrices

Convert selected columns into a matrix for classifiers, feature scaling, or linear algebra. Columns appear in the order specified:

```swift
import Quiver

let data = Panel([
    ("a", [1.0, 4.0]),
    ("b", [2.0, 5.0]),
    ("c", [3.0, 6.0])
])

let all = data.toMatrix()                    // [[1, 2, 3], [4, 5, 6]]
let selected = data.toMatrix(columns: ["c", "a"])  // [[3, 1], [6, 4]]
```

Once extracted as a matrix, columns are accessible by index using Quiver's `.column(at:)` method. This is useful for linear algebra calculations where column names are no longer needed:

```swift
let matrix = data.toMatrix()
let secondColumn = matrix.column(at: 1)  // [2.0, 5.0]
secondColumn.magnitude                   // 5.385...
```

### Converting to vectors and matrices

Panel stores everything as `[Double]` columns, but ML models need different shapes. Three methods handle every conversion:

```swift
import Quiver

let data = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0]),
    ("balance",     [15000.0, 78000.0, 42000.0, 8000.0]),
    ("approved",    [1.0, 0.0, 0.0, 1.0])
])

// Vector — single column as [Double] for regression targets or statistics
let balances = data["balance"]              // [15000.0, 78000.0, 42000.0, 8000.0]
if let avgBalance = balances.mean() {
    print(avgBalance)  // 35750.0
}

// Integer labels — single column as [Int] for classification labels
let labels = data.labels("approved")        // [1, 0, 0, 1]

// Matrix — multiple columns as [[Double]] for feature input
let features = data.toMatrix(columns: ["creditScore", "balance"])
// [[720.0, 15000.0], [650.0, 78000.0], [580.0, 42000.0], [710.0, 8000.0]]
```

The subscript returns a `[Double]` vector for continuous values like regression targets. The `labels()` method converts to `[Int]` for classifiers that expect integer class identifiers. The `toMatrix()` method assembles selected columns into the `[[Double]]` format that every Quiver model accepts.

Quiver's models — `LinearRegression`, `GaussianNaiveBayes`, `KNearestNeighbors`, and `KMeans` — all accept `[[Double]]` and `[Double]` or `[Int]` directly. None of them accept Panel. This is a deliberate design choice: models stay simple and decoupled from how data is organized. Panel handles the naming and alignment; the extraction step above converts to the shapes models expect.

### Filtering with boolean masks

Combine `Panel` with Quiver's boolean comparison operations to filter rows across all columns simultaneously:

```swift
import Quiver

let data = Panel([
    ("value", [10.0, 20.0, 30.0, 40.0]),
    ("label", [0.0, 1.0, 0.0, 1.0])
])

let mask = data["value"].isGreaterThan(15.0)
let filtered = data.filtered(where: mask)
// filtered["value"] == [20.0, 30.0, 40.0]
// filtered["label"] == [1.0, 0.0, 1.0]
```

### Splitting for machine learning

Split a `Panel` into training and testing subsets with a single call. All columns are split atomically — the same rows go to training and testing across every column:

```swift
import Quiver

// 5 samples with two feature columns and a binary label
let data = Panel([
    ("feature1", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("feature2", [10.0, 20.0, 30.0, 40.0, 50.0]),
    ("label", [0.0, 1.0, 0.0, 1.0, 0.0])
])

// Split 80/20 — all columns partitioned by the same rows
let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)

// Extract features as a matrix and labels as integers
let trainFeatures = train.toMatrix(columns: ["feature1", "feature2"])
let trainLabels = train.labels("label")
```

This eliminates the need to match seeds across parallel array splits, which is error-prone and a common source of row misalignment bugs.

### Inspecting data

Panel provides three levels of detail for inspecting data:

```swift
import Quiver

let data = Panel([
    ("age", [25.0, 30.0, 35.0, 28.0, 42.0]),
    ("income", [50000.0, 62000.0, 75000.0, 58000.0, 95000.0]),
    ("score", [88.0, 92.0, 85.0, 91.0, 78.0])
])

print(data)        // Panel: 3 columns, 5 rows
print(data.shape)  // (rows: 5, columns: 3)

print(data.head())
//        age    income  score
// 0     25.0   50000.0  88.0
// 1     30.0   62000.0  92.0
// 2     35.0   75000.0  85.0
// 3     28.0   58000.0  91.0
// 4     42.0   95000.0  78.0

print(data.summary())
// Prints count, mean, std, min, and max for each column
```

`print()` gives a quick structural overview. `.shape` returns the dimensions as a `(rows: Int, columns: Int)` named tuple — the same format as matrix `.shape`, so the API is consistent across both types. `head()` shows the actual row data in tabular format — column headers with right-aligned values and a row index. `summary()` provides per-column summary statistics. Together they provide a complete sanity check on the data before feeding it into a model. By default, `head()` displays up to 10 rows. Pass a count to limit the output:

```swift
print(data.head(n: 3))
//        age    income  score
// 0     25.0   50000.0  88.0
// 1     30.0   62000.0  92.0
// 2     35.0   75000.0  85.0
```

> Tip: Use `head()` in a Playground to visually verify your data after loading or filtering. Catching a column of unexpected zeros early saves debugging time later.

### Classification pipeline

`Panel` integrates directly with Quiver's classification workflow. A typical pipeline scales features, fits a classifier on training data, and evaluates predictions — all while keeping columns aligned:

```swift
import Quiver

let data = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0, 690.0]),
    ("balance", [15000.0, 78000.0, 42000.0, 8000.0, 55000.0]),
    ("approved", [1.0, 0.0, 0.0, 1.0, 1.0])
])

// Split preserves row alignment across all columns
let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
let featureColumns = ["creditScore", "balance"]

// Scale features using training data only (prevents data leakage)
let scaler = FeatureScaler.fit(features: train.toMatrix(columns: featureColumns))
let trainScaled = scaler.transform(train.toMatrix(columns: featureColumns))
let testScaled = scaler.transform(test.toMatrix(columns: featureColumns))

// Fit and predict
let model = GaussianNaiveBayes.fit(
    features: trainScaled,
    labels: train.labels("approved")
)
let predictions = model.predict(testScaled)
```

> Tip: `Panel` is a convenience, not a requirement. Every Quiver classifier accepts standard `[[Double]]` matrices and `[Int]` label arrays directly. `Panel` simply keeps columns named and rows aligned — use it when that organization helps, skip it when raw arrays are simpler.

### Design scope

`Panel` is intentionally focused on numeric columnar data for ML workflows. It is a value type with a fixed schema — columns are defined at creation and all values are `Double`. This focused design keeps `Panel` lightweight and predictable, optimized for the split-scale-train-evaluate cycle that classification workflows require.

Panels support direct comparison with `==`. Two panels are equal when they have the same column names in the same order and the same data in every column. This is useful for verifying that a filtering or splitting operation produced the expected result.

