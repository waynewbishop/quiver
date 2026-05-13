# Panel

A Quiver type that organizes named columns of numeric data into a single container.

## Overview

Panel takes a matrix of rows and pivots it into named columns, where each column represents a [feature or label](<doc:Machine-Learning-Primer>). Each column is a `[Double]` — effectively a named vector. The data is the same, just organized by column instead of by row. Without Panel, features offer no indication of what each column represents, and splitting or filtering requires careful coordination across parallel arrays to keep rows aligned.

With Panel, each column gets a name and all rows stay together as a unit. It serves as a lightweight container for labeled column data, scoped to Quiver's numeric focus.

> Important: `Panel` does not replace Quiver's array and matrix operations — it organizes them. Each column is a standard `[Double]` that supports Quiver vector operations like `.mean()`, `.standardDeviation()`, and boolean masking.

### Creating a panel

Build a panel from an ordered list of named columns. All columns must have the same number of elements:

```swift
import Quiver

let employees = Panel([
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
if let jumpSpread = athletes["jumpHeight"].standardDeviation() {
    print(jumpSpread)  // 0.25
}
```

### Wrapping a single array as a panel

Sometimes the data already lives in a plain `[Double]` and we want the Panel surface — typed summaries, head printing, charting — without writing the literal constructor. `toPanel()` on `Array where Element == Double` wraps the array in a single-column Panel. By default the column is named `"values"`; pass a string to give it a semantic name:

```swift
import Quiver

let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]

// One-shot inspection — the default name is fine
print(scores.toPanel().summary())

// Named — preferred when the column appears in later prints, charts, or summaries
let panel = scores.toPanel("scores")
print(panel.head())
```

This is the bridge that connects every Quiver vector method to the Panel surface. Any `[Double]` — a sensor stream, a column extracted from another panel, the output of an aggregation — becomes addressable by name and inspectable with `head`, `summary`, and the charting helpers.

### Column access

Access any column by name with subscript syntax. The returned array is a standard `[Double]` vector, so all Quiver vector operations work immediately:

```swift
import Quiver

let employees = Panel([
    ("age", [25.0, 30.0, 35.0, 28.0]),
    ("income", [50000.0, 60000.0, 75000.0, 55000.0])
])

// Access by name — returns a [Double] vector
if let avgIncome = employees["income"].mean() {
    print(avgIncome)  // 60000.0
}
employees["income"].standardized()  // z-scores
```

For classification workflows, use `.labels()` to extract a column as `[Int]` for classifiers:

```swift
let ages = employees.labels("age")  // [25, 30, 35, 28]
```

### Extracting matrices

Convert selected columns into a matrix for classifiers, feature scaling, or linear algebra. Columns appear in the order specified:

```swift
import Quiver

let cells = Panel([
    ("a", [1.0, 4.0]),
    ("b", [2.0, 5.0]),
    ("c", [3.0, 6.0])
])

let all = cells.toMatrix()                          // [[1, 2, 3], [4, 5, 6]]
let selected = cells.toMatrix(columns: ["c", "a"])  // [[3, 1], [6, 4]]
```

Once extracted as a matrix, columns are accessible by index using Quiver's `.column(at:)` method. This is useful for linear algebra calculations where column names are no longer needed:

```swift
let matrix = cells.toMatrix()
let secondColumn = matrix.column(at: 1)  // [2.0, 5.0]
secondColumn.magnitude                   // 5.385...
```

### Converting to vectors and matrices

Panel stores everything as `[Double]` columns, but ML models need different shapes. Three methods handle every conversion:

```swift
import Quiver

let applications = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0]),
    ("balance",     [15000.0, 78000.0, 42000.0, 8000.0]),
    ("approved",    [1.0, 0.0, 0.0, 1.0])
])

// Vector — single column as [Double] for regression targets or statistics
let balances = applications["balance"]              // [15000.0, 78000.0, 42000.0, 8000.0]
if let avgBalance = balances.mean() {
    print(avgBalance)  // 35750.0
}

// Integer labels — single column as [Int] for classification labels
let labels = applications.labels("approved")        // [1, 0, 0, 1]

// Matrix — multiple columns as [[Double]] for feature input
let features = applications.toMatrix(columns: ["creditScore", "balance"])
// [[720.0, 15000.0], [650.0, 78000.0], [580.0, 42000.0], [710.0, 8000.0]]
```

The subscript returns a `[Double]` vector for continuous values like regression targets. The `labels` method converts to `[Int]` for classifiers that expect integer class identifiers. The `toMatrix` method assembles selected columns into the `[[Double]]` format that every Quiver model accepts.

Quiver's models — `LinearRegression`, `GaussianNaiveBayes`, `KNearestNeighbors`, and `KMeans` — all accept `[[Double]]` and `[Double]` or `[Int]` directly. None of them accept Panel. This is a deliberate design choice: models stay simple and decoupled from how data is organized. Panel handles the naming and alignment; the extraction step above converts to the shapes models expect.

### Filtering with boolean masks

Combine `Panel` with Quiver's boolean comparison operations to filter rows across all columns simultaneously:

```swift
import Quiver

let samples = Panel([
    ("value", [10.0, 20.0, 30.0, 40.0]),
    ("label", [0.0, 1.0, 0.0, 1.0])
])

let mask = samples["value"].isGreaterThan(15.0)
let filtered = samples.filtered(where: mask)
// filtered["value"] == [20.0, 30.0, 40.0]
// filtered["label"] == [1.0, 0.0, 1.0]
```

### Splitting for machine learning

Split a `Panel` into training and testing subsets with a single call. All columns are split atomically — the same rows go to training and testing across every column:

```swift
import Quiver

// 5 samples with two feature columns and a binary label
let dataset = Panel([
    ("feature1", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("feature2", [10.0, 20.0, 30.0, 40.0, 50.0]),
    ("label", [0.0, 1.0, 0.0, 1.0, 0.0])
])

// Split 80/20 — all columns partitioned by the same rows
let (train, test) = dataset.trainTestSplit(testRatio: 0.2, seed: 42)

// Extract features as a matrix and labels as integers
let trainFeatures = train.toMatrix(columns: ["feature1", "feature2"])
let trainLabels = train.labels("label")
```

This eliminates the need to match seeds across parallel array splits, which is error-prone and a common source of row misalignment bugs.

### Inspecting data

Panel provides three quick inspections for any new panel:

```swift
import Quiver

let employees = Panel([
    ("age", [25.0, 30.0, 35.0, 28.0, 42.0]),
    ("income", [50000.0, 62000.0, 75000.0, 58000.0, 95000.0]),
    ("score", [88.0, 92.0, 85.0, 91.0, 78.0])
])

print(employees)        // Panel: 3 columns, 5 rows
print(employees.shape)  // (rows: 5, columns: 3)

print(employees.head())
//        age    income  score
// 0     25.0   50000.0  88.0
// 1     30.0   62000.0  92.0
// 2     35.0   75000.0  85.0
// 3     28.0   58000.0  91.0
// 4     42.0   95000.0  78.0
```

`print` shows the structure. `shape` returns dimensions as a named tuple, matching the matrix API. `head` displays row data in tabular format and by default shows up to 10 rows — pass a count to limit the output:

```swift
print(employees.head(n: 3))
//        age    income  score
// 0     25.0   50000.0  88.0
// 1     30.0   62000.0  92.0
// 2     35.0   75000.0  85.0
```

The fourth inspection — `summary()` — returns a typed snapshot of per-column statistics and gets its own section below.

> Experiment: **The Quiver Notebook** is the right place to catch data-quality issues before they propagate. Call `head` after every load or filter step and watch the tabular output update as the panel changes — a column of unexpected zeros is the kind of issue that quietly breaks downstream models if it slips past the eye. See <doc:Quiver-Notebook>.

### Typed column summaries

`summary()` returns a `PanelSummary` — a typed snapshot keyed by column name. The previous String-returning version printed the same table, but every downstream caller had to parse it back into numbers. The typed return removes that round-trip. Each column's statistics live in a `ColumnSummary` value addressable by field, so the same call serves a human reader (via `print`) and a downstream calculation (via property access):

```swift
import Quiver

let quiz = Panel([
    ("score", [60.0, 70.0, 80.0, 90.0, 100.0])
])

let summary = quiz.summary()

// Reads like a report
print(summary)
// column  count  mean      std   min    max
// -------------------------------------------
// score       5  80.0  15.8114  60.0  100.0

// Reads like data
if let score = summary.columns["score"] {
    score.count    // 5
    score.mean     // 80.0
    score.std      // 15.8114... — sample standard deviation, ddof = 1
    score.min      // 60.0
    score.q1       // 70.0
    score.median   // 80.0
    score.q3       // 90.0
    score.max      // 100.0
    score.iqr      // 20.0
}
```

The nine fields are the same five-number summary used elsewhere in Quiver — `count`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, `iqr`. `std` is the sample standard deviation; the formula divides by `n - 1`, matching the default of `[Double].standardDeviation()`.

`PanelSummary` and `ColumnSummary` are both `Codable`, `Sendable`, and `Equatable`. Crossing a task boundary, persisting a snapshot to disk, or comparing two snapshots from different runs is a direct conformance call — `JSONEncoder().encode(summary)` for the first, `==` for the third. The `CustomStringConvertible` conformance is what makes `print(summary)` reproduce the formatted table.

For deliverables, `ColumnSummary` and `PanelSummary` both expose `markdownTable()` and `csvRows()` formatters. The Markdown variant pastes cleanly into a PR comment or a stakeholder report. The CSV variant moves a snapshot into a spreadsheet or another tool without intermediate parsing:

```swift
if let score = summary.columns["score"] {
    print(score.markdownTable())
    // | Statistic | Value |
    // | --- | --- |
    // | count | 5 |
    // | mean | 80.0 |
    // | std | 15.8114 |
    // ...
}

print(summary.csvRows())
// column,count,mean,std,min,max
// score,5,80.0,15.811388300841896,60.0,100.0
```

### Classification pipeline

`Panel` integrates directly with Quiver's [classification](<doc:Machine-Learning-Primer>) workflow. A typical pipeline scales features, fits a classifier on training data, and evaluates predictions — all while keeping columns aligned. `Pipeline.fit` bundles the scaler and classifier into a single value, so the scaler trained on the training set is the exact scaler applied at predict time:

```swift
import Quiver

let loans = Panel([
    ("creditScore", [720.0, 650.0, 580.0, 710.0, 690.0]),
    ("balance", [15000.0, 78000.0, 42000.0, 8000.0, 55000.0]),
    ("approved", [1.0, 0.0, 0.0, 1.0, 1.0])
])

// Split preserves row alignment across all columns
let (train, test) = loans.trainTestSplit(testRatio: 0.2, seed: 42)
let featureColumns = ["creditScore", "balance"]

// One call fits the scaler and the classifier together as a Pipeline
let pipeline = Pipeline.fit(
    features: train.toMatrix(columns: featureColumns),
    labels: train.labels("approved")
)

// Predict against the test set — the pipeline applies its own scaler first
let predictions = pipeline.predict(test.toMatrix(columns: featureColumns))
```

`Pipeline.fit` takes it from there: it fits a `StandardScaler` on the raw features, applies it, trains the `GaussianNaiveBayes` model on the scaled data, and returns the two as one bundled value. The `predict` call applies the stored scaler before running the model, which is what keeps every prediction in the same coordinate system the model was trained on. For the full Pipeline surface, see <doc:Pipeline>.

> Tip: `Panel` is a convenience, not a requirement. Every Quiver classifier accepts standard `[[Double]]` matrices and `[Int]` label arrays directly. `Panel` simply keeps columns named and rows aligned — use it when that organization helps, skip it when raw arrays are simpler.

### Charting Panel data with Swift Charts

Swift Charts iterates data and emits one mark per row, and the columns of a Panel slot in directly. The chart-side code asks for two things: an iterable collection and a stable identifier per element. Panel provides both — the row count is known, and each column reads as a parallel `[Double]`:

```swift
import Charts
import Quiver
import SwiftUI

struct WorkoutTrendChart: View {
    let workouts: Panel

    var body: some View {
        Chart {
            ForEach(0..<workouts.rowCount, id: \.self) { row in
                LineMark(
                    x: .value("Week", workouts["week"][row]),
                    y: .value("Heart Rate", workouts["heartRate"][row])
                )
            }
        }
    }
}
```

For categorical aggregations — total revenue per region, mean response time per endpoint, count of events per day — the natural starting point is `groupedData(by:using:)` on a `[Double]` column, which returns sorted `(category, value)` tuples that map straight to a `BarMark`:

```swift
let sales: [Double]   = [120.0, 95.0, 140.0, 110.0, 85.0, 130.0]
let regions: [String] = ["North", "South", "North", "South", "South", "North"]

let chartData = sales.groupedData(by: regions, using: .sum)
// [(category: "North", value: 390.0), (category: "South", value: 290.0)]
```

The grouping happens once, in Quiver. The chart receives sorted, labeled tuples and renders them — no `Dictionary` to flatten on the chart side, no second pass to deduplicate categories. The full catalog of chart-ready transformations — stacked series, percentile ranks, scaled-to-range outputs, downsampled signals — is documented in <doc:Data-Visualization>.

### Design scope

`Panel` is intentionally focused on numeric columnar data for ML workflows. It is a value type with a fixed schema — columns are defined at creation and all values are `Double`. This focused design keeps `Panel` lightweight and predictable, optimized for the split-scale-train-evaluate cycle that classification workflows require.

Panels conform to Swift's `Equatable` protocol. Two panels are equal when they have the same column names in the same order and the same data in every column. This is useful for verifying that a filtering or splitting operation produced the expected result.

