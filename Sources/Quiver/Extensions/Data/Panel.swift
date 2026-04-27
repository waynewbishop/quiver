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
/// all existing Quiver vector operations like `.mean()`, `.std()`, `.isGreaterThan(_:)`,
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
    /// `.std()`, `.isGreaterThan(_:)`, boolean masking, and so on.
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
            let idx = indexStrings[r].padding(toLength: indexWidth, withPad: " ", startingAt: 0)
            let valueParts = (0..<columnNames.count).map { c in
                let val = columnStrings[c][r]
                let width = columnWidths[c]
                return String(repeating: " ", count: width - val.count) + val
            }
            lines.append(idx + "  " + valueParts.joined(separator: "  "))
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Description

    /// Returns per-column summary statistics as a formatted string.
    ///
    /// For each column, computes count, mean, standard deviation, minimum, and
    /// maximum using existing Quiver vector operations.
    ///
    /// - Returns: A multi-line string with one row of statistics per column.
    public func summary() -> String {

        // Format a value: drop trailing ".0" for whole numbers, otherwise 4 decimals
        func format(_ value: Double) -> String {
            if value == value.rounded(.towardZero) && !value.isNaN && !value.isInfinite {
                return "\(Int(value)).0"
            }
            return String(format: "%.4f", value)
        }

        // Compute stats for each column
        let headers = ["column", "count", "mean", "std", "min", "max"]
        var rows: [[String]] = []

        for name in columnNames {
            let col = column(name)
            let mean = col.mean() ?? 0.0
            let std = col.std() ?? 0.0
            let minVal = col.min() ?? 0.0
            let maxVal = col.max() ?? 0.0
            rows.append([name, "\(col.count)", format(mean), format(std), format(minVal), format(maxVal)])
        }

        // Compute width for each column based on header and data
        var widths = headers.map { $0.count }
        for row in rows {
            for (c, val) in row.enumerated() {
                widths[c] = Swift.max(widths[c], val.count)
            }
        }

        // Build header — first column left-aligned, rest right-aligned
        let headerParts = headers.enumerated().map { c, h in
            c == 0
                ? h.padding(toLength: widths[c], withPad: " ", startingAt: 0)
                : String(repeating: " ", count: widths[c] - h.count) + h
        }
        var lines = [headerParts.joined(separator: "  ")]
        lines.append(String(repeating: "-", count: lines[0].count))

        // Build data rows — first column left-aligned, rest right-aligned
        for row in rows {
            let parts = row.enumerated().map { c, val in
                c == 0
                    ? val.padding(toLength: widths[c], withPad: " ", startingAt: 0)
                    : String(repeating: " ", count: widths[c] - val.count) + val
            }
            lines.append(parts.joined(separator: "  "))
        }

        return lines.joined(separator: "\n")
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
