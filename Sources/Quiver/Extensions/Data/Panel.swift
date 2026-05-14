// Copyright 2025 Wayne W Bishop. All rights reserved.
//
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language governing
// permissions and limitations under the License.

import Foundation

// MARK: - Panel

/// A Quiver type that organizes named columns of numeric data into a single container.
///
/// `Panel` is provided by the Quiver framework to give names to columns of `[Double]`
/// data while keeping rows aligned. It does not replace Quiver's array and matrix
/// operations — it organizes them. Each column is a standard `[Double]` that supports
/// all existing Quiver vector operations like `.mean()`, `.standardDeviation()`, `.isGreaterThan(_:)`,
/// and boolean masking.
///
/// Use `Panel` when working with multi-feature datasets where column identity
/// matters — for example, building feature matrices for classifiers, splitting
/// data for training and testing, or computing per-column statistics.
///
/// This is a value type. Once created, columns cannot be added or removed.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let data = Panel([
///     ("age", [25.0, 30.0, 35.0, 28.0]),
///     ("income", [50000.0, 60000.0, 75000.0, 55000.0]),
///     ("score", [0.8, 0.6, 0.9, 0.7])
/// ])
///
/// data["income"].mean()          // 60000.0
/// data["score"].standardized()   // z-scores
///
/// let features = data.toMatrix(columns: ["age", "income"])
/// // [[25.0, 50000.0], [30.0, 60000.0], [35.0, 75000.0], [28.0, 55000.0]]
/// ```
public struct Panel: CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        "Panel: \(columnNames.count) columns, \(rowCount) rows"
    }

    // MARK: - Storage

    /// Per-column data, keyed by column name.
    private let storage: [String: [Double]]

    /// Column names in insertion order.
    public let columnNames: [String]

    /// Number of rows (shared across all columns).
    public let rowCount: Int

    /// The dimensions of the panel as a named tuple.
    ///
    /// Returns the same `(rows: Int, columns: Int)` format as matrix `.shape`,
    /// so developers who learn `.shape` on a `[[Double]]` can use the same
    /// API on Panel without switching conventions.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("age", [25.0, 30.0, 35.0]),
    ///     ("score", [88.0, 92.0, 85.0])
    /// ])
    /// data.shape  // (rows: 3, columns: 2)
    /// ```
    public var shape: (rows: Int, columns: Int) {
        (rows: rowCount, columns: columnNames.count)
    }

    // MARK: - Initializers

    /// Creates a panel from an ordered list of named columns.
    ///
    /// All columns must have the same number of elements. Column order is
    /// preserved as given.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("height", [5.8, 6.1, 5.5]),
    ///     ("weight", [160.0, 185.0, 140.0])
    /// ])
    /// ```
    ///
    /// - Parameter columns: An array of (name, values) pairs defining each column.
    public init(_ columns: [(String, [Double])]) {
        precondition(!columns.isEmpty, "Panel must have at least one column")

        let names = columns.map { $0.0 }
        precondition(Set(names).count == names.count, "Column names must be unique")

        let expectedCount = columns[0].1.count
        precondition(expectedCount > 0, "Columns must not be empty")

        var dict: [String: [Double]] = [:]
        for (name, values) in columns {
            precondition(values.count == expectedCount,
                "All columns must have the same number of rows")
            dict[name] = values
        }

        self.storage = dict
        self.columnNames = names
        self.rowCount = expectedCount
    }

    /// Creates a panel from a dictionary of named columns.
    ///
    /// Column order is determined by sorting the dictionary keys alphabetically.
    /// For explicit ordering, use the array-of-tuples initializer instead.
    ///
    /// - Parameter columns: A dictionary mapping column names to value arrays.
    public init(_ columns: [String: [Double]]) {
        let sorted = columns.keys.sorted().map { key in
            guard let values = columns[key] else {
                preconditionFailure("Column '\(key)' missing from dictionary")
            }
            return (key, values)
        }
        self.init(sorted)
    }

    /// Creates a panel from a matrix with column names.
    ///
    /// Each row of the matrix becomes a row in the panel, with values distributed
    /// across the named columns. The number of column names must match the number
    /// of columns in the matrix.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let matrix: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    /// let data = Panel(matrix: matrix, columns: ["x", "y"])
    /// data["x"]  // [1.0, 3.0, 5.0]
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: A 2D array where each inner array is one row.
    ///   - columns: Column names, one per matrix column.
    public init(matrix: [[Double]], columns: [String]) {
        precondition(!matrix.isEmpty, "Matrix must not be empty")
        precondition(!columns.isEmpty, "Column names must not be empty")
        precondition(matrix[0].count == columns.count,
            "Number of column names must match the number of matrix columns")

        var pairs: [(String, [Double])] = []
        for (c, name) in columns.enumerated() {
            let columnValues = matrix.map { $0[c] }
            pairs.append((name, columnValues))
        }

        self.init(pairs)
    }

    // MARK: - Column Access

    /// Returns the values for the named column as a Quiver vector.
    ///
    /// The returned array supports all Quiver vector operations — `.mean()`,
    /// `.standardDeviation()`, `.isGreaterThan(_:)`, boolean masking, and so on.
    ///
    /// - Parameter column: The column name to look up.
    /// - Returns: The column's values as `[Double]`.
    public subscript(column: String) -> [Double] {
        guard let values = storage[column] else {
            preconditionFailure("Column '\(column)' does not exist in panel")
        }
        return values
    }

    /// Returns the named column as integer labels for classification.
    ///
    /// Converts each `Double` value to `Int` by truncation. Use this to extract
    /// a label column for classifiers like `GaussianNaiveBayes`.
    ///
    /// - Parameter column: The column name to look up.
    /// - Returns: The column's values as `[Int]`.
    public func labels(_ column: String) -> [Int] {
        return self[column].map { Int($0) }
    }

    // MARK: - Conversion

    /// Extracts selected columns as a matrix.
    ///
    /// Columns appear in the order specified. If no columns are given, all
    /// columns are included in their insertion order.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("a", [1.0, 4.0]),
    ///     ("b", [2.0, 5.0]),
    ///     ("c", [3.0, 6.0])
    /// ])
    /// data.toMatrix(columns: ["c", "a"])
    /// // [[3.0, 1.0], [6.0, 4.0]]
    /// ```
    ///
    /// - Parameter columns: Column names to include. Defaults to nil (all columns).
    /// - Returns: A 2D array where each inner array is one row.
    public func toMatrix(columns: [String]? = nil) -> [[Double]] {
        let selected = columns ?? columnNames

        // Gather the column arrays in the requested order
        let columnArrays = selected.map { self[$0] }

        // Transpose from column-major to row-major
        var matrix: [[Double]] = []
        for r in 0..<rowCount {
            var row: [Double] = []
            for col in columnArrays {
                row.append(col[r])
            }
            matrix.append(row)
        }

        return matrix
    }

    // MARK: - Filtering

    /// Returns a new panel containing only the rows where the mask is true.
    ///
    /// This integrates with Quiver's boolean comparison operations. Create a mask
    /// using `.isGreaterThan(_:)`, `.isLessThan(_:)`, or `.and(_:)` on any column,
    /// then pass it here to filter all columns simultaneously.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("value", [10.0, 20.0, 30.0, 40.0]),
    ///     ("label", [0.0, 1.0, 0.0, 1.0])
    /// ])
    /// let mask = data["value"].isGreaterThan(15.0)
    /// let filtered = data.filtered(where: mask)
    /// // filtered["value"] == [20.0, 30.0, 40.0]
    /// ```
    ///
    /// - Parameter mask: A boolean array with one element per row.
    /// - Returns: A new `Panel` with the filtered rows.
    public func filtered(where mask: [Bool]) -> Panel {
        precondition(mask.count == rowCount,
            "Mask length (\(mask.count)) must match row count (\(rowCount))")

        let pairs = columnNames.map { name in
            (name, column(name).masked(by: mask))
        }

        // Handle the case where filtering removes all rows
        if pairs[0].1.isEmpty {
            return Panel(emptyWithColumns: columnNames)
        }

        return Panel(pairs)
    }

    // MARK: - Splitting

    /// Splits the panel into training and testing subsets by rows.
    ///
    /// All columns are split atomically — the same rows go to training and testing
    /// across every column. Uses the same seeded-shuffle algorithm as
    /// `Array.trainTestSplit(testRatio:seed:)`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("feature", [1.0, 2.0, 3.0, 4.0, 5.0,
    ///                  6.0, 7.0, 8.0, 9.0, 10.0]),
    ///     ("label", [0.0, 1.0, 0.0, 1.0, 0.0,
    ///                1.0, 0.0, 1.0, 0.0, 1.0])
    /// ])
    /// let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
    /// // train.rowCount == 8, test.rowCount == 2
    /// ```
    ///
    /// - Parameters:
    ///   - testRatio: Fraction of rows for the test set (between 0 and 1, exclusive).
    ///   - seed: A UInt64 seed for reproducible shuffling.
    /// - Returns: A named tuple of `(train: Panel, test: Panel)`.
    public func trainTestSplit(testRatio: Double, seed: UInt64) -> (train: Panel, test: Panel) {
        precondition(rowCount > 1, "Panel must have at least 2 rows to split")
        precondition(testRatio > 0.0 && testRatio < 1.0,
            "Test ratio must be between 0 and 1 (exclusive)")

        // Generate shuffled indices using the same seeded RNG as Quiver's array split
        var rng = SeededRandomNumberGenerator(seed: seed)
        let shuffledIndices = (0..<rowCount).shuffled(using: &rng)

        let testCount = Int(ceil(Double(rowCount) * testRatio))
        let testIndices = Array(shuffledIndices[..<testCount])
        let trainIndices = Array(shuffledIndices[testCount...])

        // Build train and test panels by indexing into each column
        let trainPairs = columnNames.map { name in
            let col = column(name)
            return (name, trainIndices.map { col[$0] })
        }
        let testPairs = columnNames.map { name in
            let col = column(name)
            return (name, testIndices.map { col[$0] })
        }

        return (train: Panel(trainPairs), test: Panel(testPairs))
    }

    // MARK: - Display

    /// Returns the first rows of the panel as a formatted table.
    ///
    /// Displays data in a space-delimited tabular format with column headers
    /// right-aligned above their values, and a row index on the left.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = Panel([
    ///     ("age", [25.0, 30.0, 35.0, 28.0]),
    ///     ("income", [50000.0, 62000.0, 75000.0, 58000.0]),
    ///     ("score", [88.0, 92.0, 85.0, 91.0])
    /// ])
    /// print(data.head(n: 3))
    /// //        age    income  score
    /// // 0     25.0   50000.0   88.0
    /// // 1     30.0   62000.0   92.0
    /// // 2     35.0   75000.0   85.0
    /// ```
    ///
    /// - Parameter n: Number of rows to display. Defaults to 10.
    /// - Returns: A formatted string showing the first `n` rows in tabular form.
    public func head(n: Int = 10) -> String {
        let displayRows = Swift.min(n, rowCount)
        guard displayRows > 0 else { return "(empty Panel)" }

        // Format each value: drop ".0" for whole numbers, otherwise 1 decimal
        func format(_ value: Double) -> String {
            if value == value.rounded(.towardZero) && !value.isNaN && !value.isInfinite {
                let intVal = Int(value)
                return "\(intVal).0"
            }
            return String(format: "%.1f", value)
        }

        // Build the index column
        let indexStrings = (0..<displayRows).map { "\($0)" }
        let indexWidth = Swift.max(indexStrings.last?.count ?? 1, 1)

        // Build formatted value strings per column and compute widths
        var columnStrings: [[String]] = []
        var columnWidths: [Int] = []

        for name in columnNames {
            let col = column(name)
            let values = (0..<displayRows).map { format(col[$0]) }
            let width = Swift.max(name.count, values.map { $0.count }.max() ?? 0)
            columnStrings.append(values)
            columnWidths.append(width)
        }

        // Build header line
        let indexPad = String(repeating: " ", count: indexWidth)
        let headerParts = zip(columnNames, columnWidths).map { name, width in
            name.padding(toLength: width, withPad: " ", startingAt: 0)
                .replacingOccurrences(of: name,
                    with: String(repeating: " ", count: width - name.count) + name)
        }
        var lines = [indexPad + "  " + headerParts.joined(separator: "  ")]

        // Build data rows
        for r in 0..<displayRows {
            let indexLabel = indexStrings[r].padding(toLength: indexWidth, withPad: " ", startingAt: 0)
            let valueParts = (0..<columnNames.count).map { c in
                let cellValue = columnStrings[c][r]
                let width = columnWidths[c]
                return String(repeating: " ", count: width - cellValue.count) + cellValue
            }
            lines.append(indexLabel + "  " + valueParts.joined(separator: "  "))
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Description

    /// Returns a typed snapshot of per-column descriptive statistics.
    ///
    /// For each column, builds a `ColumnSummary` covering count, mean, standard deviation,
    /// quartiles, min, max, and iqr. The returned value conforms to `CustomStringConvertible`,
    /// so `print(panel.summary())` produces the same formatted table as the previous
    /// `String`-returning version. Programmatic callers can read individual values from
    /// `panel.summary().columns["name"]?.mean`.
    public func summary() -> PanelSummary {
        var columnSummaries: [String: ColumnSummary] = [:]
        for name in columnNames {
            let col = column(name)
            // Build a ColumnSummary directly so empty columns still appear in the table
            // with zeros, matching the previous `String`-returning behavior.
            let mean = col.mean() ?? 0.0
            let std = col.standardDeviation() ?? 0.0
            let q = col.quartiles()
            columnSummaries[name] = ColumnSummary(
                count: col.count,
                mean: mean,
                std: std,
                min: q?.min ?? 0.0,
                q1: q?.q1 ?? 0.0,
                median: q?.median ?? 0.0,
                q3: q?.q3 ?? 0.0,
                max: q?.max ?? 0.0,
                iqr: q?.iqr ?? 0.0
            )
        }
        return PanelSummary(columnNames: columnNames, columns: columnSummaries)
    }

    /// Returns a typed snapshot of per-column descriptive statistics.
    ///
    /// `describe` is an alias for ``summary()`` — both return the same
    /// `PanelSummary` value covering count, mean, standard deviation, and the
    /// five-number summary for every column. The alias exists for
    /// discoverability: developers arriving from prior numerical-computing
    /// experience often type `.describe()` first, while developers learning
    /// Quiver from scratch find `.summary()` more natural. Both work.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("age",   [25.0, 30.0, 35.0, 40.0]),
    ///     ("score", [88.0, 92.0, 85.0, 91.0])
    /// ])
    /// print(panel.describe())  // identical output to print(panel.summary())
    /// ```
    public func describe() -> PanelSummary {
        return summary()
    }

    /// Returns the last `n` rows of the panel as a formatted string.
    ///
    /// Mirrors ``head(n:)`` for inspecting the end of a dataset — useful for
    /// time-series data where the most recent observations are at the bottom,
    /// or for verifying that a sort placed the right values at the end.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("day",     [1.0, 2.0, 3.0, 4.0, 5.0]),
    ///     ("revenue", [120.0, 135.0, 142.0, 128.0, 145.0])
    /// ])
    /// print(panel.tail(n: 2))
    /// //    day  revenue
    /// // 3  4.0    128.0
    /// // 4  5.0    145.0
    /// ```
    ///
    /// - Parameter n: Number of rows to display from the end. Defaults to 10.
    /// - Returns: A formatted string showing the last `n` rows in tabular form.
    public func tail(n: Int = 10) -> String {
        let displayRows = Swift.min(n, rowCount)
        guard displayRows > 0 else { return "(empty Panel)" }

        let startRow = rowCount - displayRows

        func format(_ value: Double) -> String {
            if value == value.rounded(.towardZero) && !value.isNaN && !value.isInfinite {
                let intVal = Int(value)
                return "\(intVal).0"
            }
            return String(format: "%.1f", value)
        }

        let indexStrings = (startRow..<rowCount).map { "\($0)" }
        let indexWidth = Swift.max(indexStrings.last?.count ?? 1, 1)

        var columnStrings: [[String]] = []
        var columnWidths: [Int] = []

        for name in columnNames {
            let col = column(name)
            let values = (startRow..<rowCount).map { format(col[$0]) }
            let width = Swift.max(name.count, values.map { $0.count }.max() ?? 0)
            columnStrings.append(values)
            columnWidths.append(width)
        }

        let indexPad = String(repeating: " ", count: indexWidth)
        let headerParts = zip(columnNames, columnWidths).map { name, width in
            String(repeating: " ", count: width - name.count) + name
        }
        var lines = [indexPad + "  " + headerParts.joined(separator: "  ")]

        for r in 0..<displayRows {
            let indexLabel = indexStrings[r].padding(toLength: indexWidth, withPad: " ", startingAt: 0)
            let valueParts = (0..<columnNames.count).map { c in
                let cellValue = columnStrings[c][r]
                let width = columnWidths[c]
                return String(repeating: " ", count: width - cellValue.count) + cellValue
            }
            lines.append(indexLabel + "  " + valueParts.joined(separator: "  "))
        }

        return lines.joined(separator: "\n")
    }

    /// Returns the unique values in a column, sorted ascending.
    ///
    /// Uses exact floating-point equality, so two values that differ by
    /// floating-point noise (e.g., `1.0` and `1.0 + 1e-16`) are treated as
    /// distinct. This is the right behavior for integer-coded categories
    /// (species labels, class IDs, day-of-week values) and is a foot-gun for
    /// continuous measured data — bin continuous data first if the goal is
    /// "how many distinct ranges does this column take."
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("species", [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    /// ])
    /// panel.unique(column: "species")  // [0.0, 1.0, 2.0]
    /// ```
    ///
    /// - Parameter column: The column name to inspect.
    /// - Returns: The unique values in ascending order, or `nil` if the column does not exist.
    public func unique(column: String) -> [Double]? {
        guard let values = storage[column] else { return nil }
        return Array(Set(values)).sorted()
    }

    /// Returns the frequency of each unique value in a column.
    ///
    /// The result is sorted by count descending, with ties broken by value
    /// ascending. The pair shape `(value: Double, count: Int)` is the
    /// chart-ready format for a `BarMark` over categorical data.
    ///
    /// Like ``unique(column:)``, this method uses exact floating-point
    /// equality, so it is intended for integer-coded categories or
    /// integer-valued doubles. Bin continuous data first if needed.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("species", [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    /// ])
    /// panel.valueCounts(column: "species")
    /// // [(value: 0.0, count: 3), (value: 1.0, count: 2), (value: 2.0, count: 1)]
    /// ```
    ///
    /// - Parameter column: The column name to inspect.
    /// - Returns: An array of `(value, count)` pairs sorted by count
    ///   descending, or `nil` if the column does not exist.
    public func valueCounts(column: String) -> [(value: Double, count: Int)]? {
        guard let values = storage[column] else { return nil }
        var counts: [Double: Int] = [:]
        for v in values {
            counts[v, default: 0] += 1
        }
        return counts
            .map { (value: $0.key, count: $0.value) }
            .sorted { lhs, rhs in
                if lhs.count != rhs.count { return lhs.count > rhs.count }
                return lhs.value < rhs.value
            }
    }

    // MARK: - Transformations

    /// Returns a new Panel sorted by the values in one column.
    ///
    /// All columns are reordered together so rows stay aligned — the row that
    /// had the smallest value in the sort column ends up in row 0 of every
    /// column. NaN values sort to the end regardless of the `ascending`
    /// argument, matching the convention used by most numerical libraries.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("name",  [3.0, 1.0, 2.0]),  // placeholder for ID
    ///     ("score", [88.0, 95.0, 72.0])
    /// ])
    /// let topFirst = panel.sortedBy(column: "score", ascending: false)
    /// // row 0 — score 95.0
    /// // row 1 — score 88.0
    /// // row 2 — score 72.0
    /// ```
    ///
    /// Use ``head(n:)`` after sorting to extract the top-N or bottom-N rows.
    ///
    /// - Parameters:
    ///   - column: The column name to sort by.
    ///   - ascending: `true` for smallest-first, `false` for largest-first.
    ///     Defaults to `true`. NaN values always sort to the end.
    /// - Returns: A new Panel with rows reordered by the sort column.
    /// - Note: Calling this method with a column name that does not exist
    ///   triggers a `preconditionFailure`, matching the behavior of the
    ///   subscript accessor.
    public func sortedBy(column: String, ascending: Bool = true) -> Panel {
        let sortColumn = self.column(column)

        // Sort row indices by the value in the sort column. NaN sorts last
        // in both ascending and descending modes.
        let indices = (0..<rowCount).sorted { i, j in
            let a = sortColumn[i]
            let b = sortColumn[j]
            if a.isNaN && b.isNaN { return false }
            if a.isNaN { return false }  // a goes after b
            if b.isNaN { return true }   // a goes before b
            return ascending ? a < b : a > b
        }

        // Reorder every column using the same index permutation.
        var sortedColumns: [(String, [Double])] = []
        for name in columnNames {
            let col = self.column(name)
            sortedColumns.append((name, indices.map { col[$0] }))
        }
        return Panel(sortedColumns)
    }

    /// Returns a new Panel with one column standardized to z-scores.
    ///
    /// Standardization rescales values so the column has mean 0 and standard
    /// deviation 1. Each output value is `(x - mean) / std` for the original
    /// column. The other columns pass through unchanged.
    ///
    /// This is the Panel-level convenience for the array-level
    /// ``Swift/Array/standardized()`` method. Use it for quick exploratory
    /// transforms — comparing two columns on a normalized axis, plotting a
    /// distribution centered on zero. For ML pipelines that need to fit a
    /// scaler on training data and apply the same transform to test data,
    /// reach for ``FeatureScaler`` instead.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("age",   [25.0, 30.0, 35.0, 40.0]),
    ///     ("score", [88.0, 92.0, 85.0, 91.0])
    /// ])
    /// let zPanel = panel.standardized(column: "age")
    /// // age column now has mean 0, std 1; score column unchanged
    /// ```
    ///
    /// - Parameter column: The column name to standardize.
    /// - Returns: A new Panel with the named column replaced by its z-scored values.
    /// - Note: A constant column (zero standard deviation) returns a column
    ///   of zeros for that column, matching the array-level method.
    public func standardized(column: String) -> Panel {
        let original = self.column(column)
        let zScores = original.standardized()

        var newColumns: [(String, [Double])] = []
        for name in columnNames {
            if name == column {
                newColumns.append((name, zScores))
            } else {
                newColumns.append((name, self.column(name)))
            }
        }
        return Panel(newColumns)
    }

    /// Returns the pairwise Pearson correlation matrix with column labels.
    ///
    /// Builds an N-by-N matrix where entry `[i][j]` is the Pearson correlation
    /// between column `i` and column `j`. Diagonal entries are 1.0. The result
    /// pairs the matrix with the column names in the same order, so the labels
    /// are immediately available for printing, charting, or feeding into a
    /// heatmap.
    ///
    /// Pair with ``Swift/Array/heatmapData(labels:)`` or Swift Charts'
    /// `RectangleMark` to visualize the correlation structure of a Panel.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let panel = Panel([
    ///     ("temp",  [70.0, 75.0, 80.0, 85.0, 90.0]),
    ///     ("sales", [40.0, 55.0, 70.0, 88.0, 105.0]),
    ///     ("rain",  [0.0, 1.0, 0.0, 0.0, 0.5])
    /// ])
    /// let result = panel.correlationMatrix()
    /// // result.columns: ["temp", "sales", "rain"]
    /// // result.matrix:  [[1.0, 0.998, ...], [0.998, 1.0, ...], [..., ..., 1.0]]
    /// ```
    ///
    /// Constant columns (zero variance) produce `Double.nan` in their row
    /// and column entries, which carries the "undefined correlation" meaning
    /// without crashing the rest of the matrix.
    ///
    /// - Returns: A named tuple of `(columns, matrix)`. The `columns` array
    ///   matches the order of rows and columns in `matrix`.
    public func correlationMatrix() -> (columns: [String], matrix: [[Double]]) {
        let columnsAsRows = columnNames.map { self.column($0) }
        let matrix = columnsAsRows.correlationMatrix()
        return (columns: columnNames, matrix: matrix)
    }

    // MARK: - Private

    /// Returns the column values for a known column name, trapping if missing.
    private func column(_ name: String) -> [Double] {
        guard let col = storage[name] else {
            preconditionFailure("Column '\(name)' missing from internal storage")
        }
        return col
    }

    /// Creates an empty panel that preserves column names but has zero rows.
    private init(emptyWithColumns names: [String]) {
        var dict: [String: [Double]] = [:]
        for name in names {
            dict[name] = []
        }
        self.storage = dict
        self.columnNames = names
        self.rowCount = 0
    }
}

// MARK: - Array → Panel Bridge

public extension Array where Element == Double {

    /// Wraps the array as a single-column `Panel` for chaining into Panel-level operations.
    ///
    /// The cleanest way to take a flat numeric vector and feed it into the Panel API
    /// without the array-of-tuples ceremony. The default column name `"values"` lets
    /// us call the method without arguments when we just want a quick descriptive
    /// summary; pass an explicit name when the column will be referenced later.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]
    ///
    /// // Default name — fine for one-shot inspection
    /// print(scores.toPanel().summary())
    ///
    /// // Named — preferred when the column appears in later prints or charts
    /// let panel = scores.toPanel("scores")
    /// print(panel.head())
    /// ```
    ///
    /// - Parameter columnName: The name to give the single column. Defaults to `"values"`.
    /// - Returns: A `Panel` with one column containing the array's values.
    func toPanel(_ columnName: String = "values") -> Panel {
        Panel([(columnName, self)])
    }
}
