# Panel Workflows

Applied patterns for using Panel in a machine learning pipeline.

## Overview

Once a panel exists and its columns are accessible, the same value drives the split-scale-train-evaluate cycle that classification workflows require. Every step below operates on a `Panel` directly or on the matrices and label arrays a panel produces, and each step is designed to keep rows aligned without manual bookkeeping.

The progression matches a realistic ML run. We split the data, look at what we have, summarize each column, fit a pipeline, predict, and visualize the result. Every transition is one method call on the panel.

### Splitting data for evaluation

Split a `Panel` into two subsets with a single call. The pattern shows up in both modeling (training and testing sets for machine learning) and statistics (holdout samples for cross-validation studies). All columns are split atomically: the same rows go to each subset across every column, so features and labels stay aligned without manual coordination:

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

The `print` function shows the structure. The `shape` returns dimensions as a named tuple, matching the matrix API. Finally, `head` displays row data in tabular format and by default shows up to 10 rows. Pass a count to limit the output:

```swift
print(employees.head(n: 3))
//        age    income  score
// 0     25.0   50000.0  88.0
// 1     30.0   62000.0  92.0
// 2     35.0   75000.0  85.0
```

The fourth inspection, `summary()`, returns a typed snapshot of per-column statistics and gets its own section below.

> Experiment: **The Quiver Notebook** is the right place to catch data-quality issues before they propagate. Call `head` after every load or filter step and watch the tabular output update as the panel changes. A column of unexpected zeros is the kind of issue that quietly breaks downstream models if it slips past the eye. See <doc:Quiver-Notebook>.

### Typed column summaries

Calling `summary()` on a panel returns a `PanelSummary`, a frozen description of every named column at the moment the snapshot was taken. The panel underneath can mutate, get filtered, or be discarded; the summary still describes the shape the columns had when the call was made. That makes the snapshot the natural checkpoint type for an ML workflow: a training-time snapshot and an inference-time snapshot can be compared field for field, because neither one drifts.

The same `PanelSummary` serves a human reader and downstream code. Print it for the formatted table; reach into a single column for one number:

```swift
import Quiver

let quiz = Panel([
    ("score", [60.0, 70.0, 80.0, 90.0, 100.0]),
    ("attempts", [1.0, 2.0, 1.0, 1.0, 3.0])
])

let snapshot = quiz.summary()

// View 1 — formatted table for a person reading the console
print(snapshot)
// column      count    mean      std   min    max
// ---------------------------------------------
// score           5    80.0  15.8114  60.0  100.0
// attempts        5  1.6000   0.8944   1.0    3.0

// View 2 — property access for downstream code
snapshot.columns["score"]?.mean       // 80.0
snapshot.columns["score"]?.std        // 15.8114
snapshot.columns["attempts"]?.median  // 1.0
snapshot.columns["attempts"]?.iqr     // 1.0
```

Each column inside the snapshot is a `ColumnSummary` with nine fields: `count`, `mean`, `std`, `min`, `q1`, `median`, `q3`, `max`, and `iqr`. The inner value is the same type that `[Double].summary()` returns on a single array, so a question asked of an array reads the same way when asked of a named column.

Two formatters shape the snapshot for the places a report tends to land. The `markdownTable()` method renders a multi-column table (statistics down the side, column names across the top) that pastes cleanly into a PR comment or a stakeholder report. The `csvRows()` method writes one row per column, which moves a snapshot into a spreadsheet without an export step:

```swift
print(snapshot.markdownTable())
// | Statistic | score | attempts |
// | --- | --- | --- |
// | count | 5 | 5 |
// | mean | 80.0 | 1.6000 |
// | std | 15.8114 | 0.8944 |
// | min | 60.0 | 1.0 |
// | max | 100.0 | 3.0 |

print(snapshot.csvRows())
// column,count,mean,std,min,max
// score,5,80.0,15.811388300841896,60.0,100.0
// attempts,5,1.6,0.8944271909999159,1.0,3.0
```

Both deliverables surface count, mean, std, min, and max. The full nine-field view, including the quartiles, lives on the inner `ColumnSummary`, reachable through `snapshot.columns[name]`.

A `Panel` cannot be constructed with zero columns: the initializer requires at least one. A panel can end up with zero rows, however, after a filter removes every match. Calling `summary()` on a zero-row panel returns a `PanelSummary` whose `columnNames` is intact and whose `columns` dictionary still has an entry for every column. Each entry reports `count: 0` with zeros across the remaining fields. This is a deliberate departure from `[Double].summary()`, which returns `nil` for an empty array. A zero-row panel still has structure, since the columns exist and have names, so the snapshot preserves it. To distinguish "no rows" from "rows with a mean of zero," check `snapshot.columns[name]?.count == 0` rather than reading the mean.

### Exploring data

Once we know the shape of the panel, the next questions are usually about its contents. Which values appear in a column, how often, which rows rank highest by some measure, and how the columns relate to each other. Four methods on `Panel` cover that exploration directly:

```swift
import Quiver

let monthly = Panel([
    ("revenue",      [12500.0, 9800.0, 15200.0, 11000.0, 8500.0, 14000.0]),
    ("customers",    [320.0, 285.0, 410.0, 305.0, 250.0, 380.0]),
    ("satisfaction", [4.2, 3.8, 4.5, 4.1, 3.5, 4.3]),
    ("region",       [1.0, 2.0, 1.0, 3.0, 2.0, 1.0])  // 1=North, 2=South, 3=West
])

// Distinct values in a column — useful for spotting unexpected categories
monthly.unique(column: "region")           // [1.0, 2.0, 3.0]

// Counts of each value — the categorical version of summary()
monthly.valueCounts(column: "region")
// [(value: 1.0, count: 3), (value: 2.0, count: 2), (value: 3.0, count: 1)]

// Rank rows by any column without losing alignment across the other columns
let topRevenue = monthly.sortedBy(column: "revenue", ascending: false)

// Pairwise Pearson correlations across every numeric column
let result = monthly.correlationMatrix()
// result.columns: ["revenue", "customers", "satisfaction", "region"]
// result.matrix[0][1]  // 0.9846 — revenue and customers move together
// result.matrix[0][2]  // 0.9685 — revenue and satisfaction move together
```

The `unique` and `valueCounts` methods answer "what's in this column?", the second question after `head()`. The `sortedBy` method ranks rows while keeping every column aligned with the row that owns it, so the top-revenue month also carries its customer count, satisfaction score, and region in the same slot. The `correlationMatrix` method computes Pearson correlations across every pair of numeric columns; see <doc:Correlation> for the math, the diagonal-and-symmetry guarantees, and the NaN-on-constant-column contract.

### Classification pipeline

The `Panel` type integrates directly with Quiver's [classification](<doc:Machine-Learning-Primer>) workflow. A typical pipeline scales features, fits a classifier on training data, and evaluates predictions, all while keeping columns aligned. Calling `Pipeline.fit` bundles the scaler and classifier into a single value, so the scaler trained on the training set is the exact scaler applied at predict time:

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

The `Pipeline.fit` call takes it from there: it fits a `StandardScaler` on the raw features, applies it, trains the `GaussianNaiveBayes` model on the scaled data, and returns the two as one bundled value. The `predict` call applies the stored scaler before running the model, which is what keeps every prediction in the same coordinate system the model was trained on. For the full Pipeline surface, see <doc:Pipeline>.

> Tip: The `Panel` type is a convenience, not a requirement. Every Quiver classifier accepts standard `[[Double]]` matrices and `[Int]` label arrays directly. A `Panel` simply keeps columns named and rows aligned. Use it when that organization helps, skip it when raw arrays are simpler.

### Charting Panel data with Swift Charts

Swift Charts iterates data and emits one mark per row, and the columns of a Panel slot in directly. The chart-side code asks for two things: an iterable collection and a stable identifier per element. Panel provides both. The row count is known, and each column reads as a parallel `[Double]`:

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

For categorical aggregations (total revenue per region, mean response time per endpoint, count of events per day), the natural starting point is `groupedData(by:using:)` on a `[Double]` column, which returns sorted `(category, value)` tuples that map straight to a `BarMark`:

```swift
let sales: [Double]   = [120.0, 95.0, 140.0, 110.0, 85.0, 130.0]
let regions: [String] = ["North", "South", "North", "South", "South", "North"]

let chartData = sales.groupedData(by: regions, using: .sum)
// [(category: "North", value: 390.0), (category: "South", value: 290.0)]
```

The grouping happens once, in Quiver. The chart receives sorted, labeled tuples and renders them: no `Dictionary` to flatten on the chart side, no second pass to deduplicate categories. The full catalog of chart-ready transformations (stacked series, percentile ranks, scaled-to-range outputs, downsampled signals) is documented in <doc:Data-Visualization>.

### Scope

These workflows are intentionally scoped to numeric columnar data for ML. The split-scale-train-evaluate cycle above is what `Panel` is optimized for; everything on this page is a thin shape adapter over that cycle. When the work outgrows that scope (categorical encoding, time-series resampling, multi-table joins), drop down to plain arrays and matrices, do the transformation, and wrap the result back into a `Panel` for the next stage.
