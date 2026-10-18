# Working with Panels

A Quiver type that organizes named columns of numeric data into a single container.

## Overview

A Panel is a table of named columns. Each column holds an array of numbers. Naming the columns means the data can be read, split, and filtered by name instead of by position, the way a spreadsheet or a database table works.

> Note: This page covers Panel as a data structure: how to construct one, access columns, convert to the shapes models need, filter rows, and iterate rows. For applied workflows such as train/test splits, summaries, classification pipelines, and charting, see <doc:Panel-Workflows>.

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

A `Panel` can also be created from an existing matrix by providing column names. This is useful when we already have data in `[[Double]]` form (from a computation, a file import, or another Quiver operation) and want to add column labels:

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

Sometimes the data already lives in a plain `[Double]` and we want the Panel surface (typed summaries, head printing, charting) without writing the literal constructor:

```swift
import Quiver

let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]

// One-shot inspection — the default name is fine
print(scores.toPanel().summary())

// Named — preferred when the column appears in later prints, charts, or summaries
let panel = scores.toPanel("scores")
print(panel.head())
```

This is the bridge between any `[Double]` and the Panel surface. A column of test scores, a list of sensor readings, an array of probabilities, the output of `.standardized()` or `.cumulativeSum()`: anything that is already a numeric array becomes addressable by name and ready for descriptive statistics, summaries, filtering, and charting.

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
employees["income"].median()              // 57500.0 — robust to skew
employees["income"].standardDeviation()   // 10801.23 — typical spread
employees["income"].standardized()        // z-scores, useful for feature scaling
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

Quiver's models (`LinearRegression`, `GaussianNaiveBayes`, `KNearestNeighbors`, and `KMeans`) all accept `[[Double]]` and `[Double]` or `[Int]` directly. None of them accept Panel. This is a deliberate design choice: models stay simple and decoupled from how data is organized. Panel handles the naming and alignment; the extraction step above converts to the shapes models expect.

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

When a filter matches no rows, the resulting panel keeps its column schema but reports a row count of zero. The columns still exist, just empty. See <doc:Panel-Workflows> for how `summary()` and downstream operations behave on a zero-row panel.

### Iterating rows

A panel is organized by column, which is the shape statistics and models want. Some work happens one record at a time instead: feeding sensor samples to a scoring model, finding the moment a signal crosses a threshold, or filling a gap from the previous reading. A panel conforms to Swift's `Sequence` protocol, so a `for`-`in` loop walks it one row at a time:

```swift
import Quiver

let run = Panel([
    ("time", [0.0, 5.0, 10.0, 15.0]),
    ("heartRate", [142.0, 151.0, 158.0, 163.0]),
    ("grade", [0.0, 2.5, 6.0, 5.5])
])

for moment in run {
    print(moment["time"], moment["heartRate"])
}
// 0.0 142.0
// 5.0 151.0
// 10.0 158.0
// 15.0 163.0
```

Each step yields a `Panel.Row`. Its subscript takes a column name and returns a `Double`, the same way the panel's column subscript returns a `[Double]`, so a row reads with no unwrapping. A column name the panel doesn't have stops with a message naming the column. Printing a row lists its values in column order:

```swift
for moment in run where moment["grade"] > 5.0 {
    print(moment)
    break
}
// Row: time 10.0, heartRate 158.0, grade 6.0
```

When the position matters, `enumerated()` pairs each row with its index:

```swift
for (index, moment) in run.enumerated() where moment["heartRate"] > 160.0 {
    print("row \(index) crosses 160 bpm")
}
// row 3 crosses 160 bpm
```

#### How the conformance works

`Sequence` asks a type for one thing: an iterator that hands back the next element until there are none left. A `for`-`in` loop is shorthand for that exchange. The loop over `run` above is the same as asking the panel for its iterator and calling `next()` until it returns `nil`:

```swift
var iterator = run.makeIterator()
while let moment = iterator.next() {
    print(moment["time"])
}
// 0.0
// 5.0
// 10.0
// 15.0
```

The panel's iterator reads each column once when the loop starts, then builds one `Panel.Row` per step from those columns, in row order. A panel with zero rows, such as the result of a filter that matched nothing, ends the loop before the first step.

Because the conformance is the standard one, everything Swift provides for sequences works on a panel. Collecting the rows into an array gives indexed access to individual records:

```swift
let rows = Array(run)
print(rows.count)       // 4
print(rows[2]["grade"]) // 6.0
```

The panel reports its row count as its `underestimatedCount`, so building that array allocates storage once rather than growing it row by row.

Rows are read-only snapshots. Changing data still means building a new panel from its columns, and keeping a subset of rows still goes through `filtered(where:)`, which returns a `Panel`. The stdlib `filter` also works on a panel, but it returns an array of rows rather than a panel, so `filtered(where:)` remains the way to narrow a panel for the next step.

### Design

The `Panel` type is a value type with a fixed schema: columns are defined at creation and all values are `Double`. The schema is set once at construction and never mutates, which is what lets every downstream operation rely on column alignment without defensive checks.

Panels conform to Swift's `Equatable` and `Sequence` protocols. Two panels are equal when they have the same column names in the same order and the same data in every column. This is useful for verifying that a filtering or splitting operation produced the expected result. Iterating a panel yields its rows in order, from the first row to the last.

Column order is preserved. Swift dictionaries are unordered, so a panel stores its columns under both a name-keyed map for constant-time lookup and an ordered array for canonical iteration. The order columns were declared in is the order `head` and `summary` render, the order `toMatrix` extracts, the order a printed row lists its values, and the order any reproducible report should follow. When code needs to walk every column deterministically, it walks the declaration order rather than iterating the underlying dictionary.

### Next steps

With the structure covered, the applied workflows live in <doc:Panel-Workflows>: splitting for machine learning, inspecting data with `head` and typed `summary` values, running a classification pipeline end-to-end, and feeding Panel columns into Swift Charts. For pairwise and matrix-wide Pearson correlation across columns, see <doc:Correlation>.
