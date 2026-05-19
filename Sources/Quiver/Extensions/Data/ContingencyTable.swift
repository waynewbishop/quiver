// Copyright 2026 Wayne W Bishop. All rights reserved.
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

/// A labeled two-dimensional table of integer counts.
///
/// `ContingencyTable` pairs raw cell counts with row and column labels and
/// exposes the marginals, joint distribution, and the two conditional
/// distributions that follow from them. The same type is the natural input
/// for the chi-squared test of independence and the natural worked example
/// for conditional probability and Bayes' theorem.
///
/// The failable initializer validates that the count matrix is rectangular,
/// non-empty, and that every cell is non-negative.
///
/// ```swift
/// // Faculty counts: 1 woman + 37 men in Math, 17 women + 20 men in English.
/// let table = ContingencyTable(
///     rowLabels: ["Female", "Male"],
///     columnLabels: ["Math", "English"],
///     counts: [[1, 17], [37, 20]]
/// )!
///
/// print(table.probabilityTable())
/// // | | Math | English | Total |
/// // | --- | ---: | ---: | ---: |
/// // | Female | 0.0133 | 0.2267 | 0.2400 |
/// // | Male | 0.4933 | 0.2667 | 0.7600 |
/// // | Total | 0.5067 | 0.4933 | 1.0000 |
///
/// // P(male | math) — read directly from the column-conditional view.
/// let conditional = table.conditionalByColumn()
/// // conditional[1][0] ≈ 0.974
/// ```
public struct ContingencyTable: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The label for each row, in row order.
    public let rowLabels: [String]

    /// The label for each column, in column order.
    public let columnLabels: [String]

    /// The cell counts as `counts[row][column]`. Rectangular.
    public let counts: [[Int]]

    /// Creates a labeled contingency table, validating shape and values.
    ///
    /// Returns `nil` when `rowLabels.count` does not match the number of
    /// rows in `counts`, `columnLabels.count` does not match the number of
    /// columns in any row, the table is empty, or any cell is negative.
    public init?(
        rowLabels: [String],
        columnLabels: [String],
        counts: [[Int]]
    ) {
        guard !rowLabels.isEmpty,
              !columnLabels.isEmpty,
              counts.count == rowLabels.count else {
            return nil
        }
        for row in counts {
            guard row.count == columnLabels.count else { return nil }
            for cell in row {
                guard cell >= 0 else { return nil }
            }
        }
        self.rowLabels = rowLabels
        self.columnLabels = columnLabels
        self.counts = counts
    }

    // MARK: - Marginals

    /// Returns the total count for each row, in row order.
    public var rowTotals: [Int] {
        return counts.map { $0.reduce(0, +) }
    }

    /// Returns the total count for each column, in column order.
    public var columnTotals: [Int] {
        var totals = [Int](repeating: 0, count: columnLabels.count)
        for row in counts {
            for (c, value) in row.enumerated() {
                totals[c] += value
            }
        }
        return totals
    }

    /// Returns the sum of every cell in the table.
    public var grandTotal: Int {
        return counts.reduce(0) { $0 + $1.reduce(0, +) }
    }

    // MARK: - Distributions

    /// Returns the joint probability distribution P(row, column).
    ///
    /// Each cell of the returned matrix is the corresponding count divided
    /// by ``grandTotal``. The full matrix sums to one. Returns an empty
    /// table when ``grandTotal`` is zero.
    public func jointDistribution() -> [[Double]] {
        let total = grandTotal
        guard total > 0 else {
            return Array(
                repeating: Array(repeating: 0.0, count: columnLabels.count),
                count: rowLabels.count
            )
        }
        let totalDouble = Double(total)
        return counts.map { row in row.map { Double($0) / totalDouble } }
    }

    /// Returns the conditional probability P(row | column).
    ///
    /// Each column of the returned matrix sums to one. Reading
    /// `conditionalByColumn()[r][c]` gives the probability of row `r` given
    /// column `c`. Columns with a zero total contribute a column of zeros.
    public func conditionalByColumn() -> [[Double]] {
        let totals = columnTotals
        var result = Array(
            repeating: Array(repeating: 0.0, count: columnLabels.count),
            count: rowLabels.count
        )
        for c in 0..<columnLabels.count {
            let denominator = totals[c]
            guard denominator > 0 else { continue }
            let denominatorDouble = Double(denominator)
            for r in 0..<rowLabels.count {
                result[r][c] = Double(counts[r][c]) / denominatorDouble
            }
        }
        return result
    }

    /// Returns the conditional probability P(column | row).
    ///
    /// Each row of the returned matrix sums to one. Reading
    /// `conditionalByRow()[r][c]` gives the probability of column `c` given
    /// row `r`. Rows with a zero total contribute a row of zeros.
    public func conditionalByRow() -> [[Double]] {
        let totals = rowTotals
        var result = Array(
            repeating: Array(repeating: 0.0, count: columnLabels.count),
            count: rowLabels.count
        )
        for r in 0..<rowLabels.count {
            let denominator = totals[r]
            guard denominator > 0 else { continue }
            let denominatorDouble = Double(denominator)
            for c in 0..<columnLabels.count {
                result[r][c] = Double(counts[r][c]) / denominatorDouble
            }
        }
        return result
    }

    // MARK: - Printed summaries

    /// Returns a Markdown table of raw counts with marginal totals.
    ///
    /// The first column carries row labels; the rightmost column and bottom
    /// row carry marginal totals. The bottom-right cell is the grand total.
    public func markdownTable() -> String {
        var lines: [String] = []
        var header = "| |"
        var separator = "| --- |"
        for column in columnLabels {
            header += " \(column) |"
            separator += " ---: |"
        }
        header += " Total |"
        separator += " ---: |"
        lines.append(header)
        lines.append(separator)

        let rowMarginals = rowTotals
        for (r, label) in rowLabels.enumerated() {
            var row = "| \(label) |"
            for c in 0..<columnLabels.count {
                row += " \(counts[r][c]) |"
            }
            row += " \(rowMarginals[r]) |"
            lines.append(row)
        }

        var totalRow = "| Total |"
        for value in columnTotals {
            totalRow += " \(value) |"
        }
        totalRow += " \(grandTotal) |"
        lines.append(totalRow)

        return lines.joined(separator: "\n")
    }

    /// Returns a Markdown table of joint probabilities with marginals.
    ///
    /// Each cell shows P(row, column) and the marginals show P(row) and
    /// P(column). The bottom-right cell is `1.0000`. Values are formatted
    /// to four decimal places.
    public func probabilityTable() -> String {
        let joint = jointDistribution()
        let total = grandTotal
        let rowMarginals: [Double]
        let columnMarginalsArray: [Double]
        if total > 0 {
            let totalDouble = Double(total)
            rowMarginals = rowTotals.map { Double($0) / totalDouble }
            columnMarginalsArray = columnTotals.map { Double($0) / totalDouble }
        } else {
            rowMarginals = Array(repeating: 0.0, count: rowLabels.count)
            columnMarginalsArray = Array(repeating: 0.0, count: columnLabels.count)
        }

        var lines: [String] = []
        var header = "| |"
        var separator = "| --- |"
        for column in columnLabels {
            header += " \(column) |"
            separator += " ---: |"
        }
        header += " Total |"
        separator += " ---: |"
        lines.append(header)
        lines.append(separator)

        for (r, label) in rowLabels.enumerated() {
            var row = "| \(label) |"
            for c in 0..<columnLabels.count {
                row += " \(String(format: "%.4f", joint[r][c])) |"
            }
            row += " \(String(format: "%.4f", rowMarginals[r])) |"
            lines.append(row)
        }

        var totalRow = "| Total |"
        for value in columnMarginalsArray {
            totalRow += " \(String(format: "%.4f", value)) |"
        }
        let grand = total > 0 ? 1.0 : 0.0
        totalRow += " \(String(format: "%.4f", grand)) |"
        lines.append(totalRow)

        return lines.joined(separator: "\n")
    }

    /// Returns the cells of the table in row-major order as a single CSV row.
    ///
    /// The values are emitted left-to-right, top-to-bottom. Marginal totals
    /// are not included — call ``markdownTable()`` or ``probabilityTable()``
    /// when a printable layout with marginals is the goal.
    public func csvRow() -> String {
        return counts
            .flatMap { $0 }
            .map { "\($0)" }
            .joined(separator: ",")
    }

    public var description: String {
        // Reproduce a compact, fixed-width text rendering of the raw-count
        // table with marginals. Matches the texture of `PanelSummary`'s
        // description: header, separator, body, totals.
        let rowMarginals = rowTotals
        let columnMarginalsArray = columnTotals
        let grand = grandTotal

        let headers = [""] + columnLabels + ["Total"]
        var rows: [[String]] = []
        for (r, label) in rowLabels.enumerated() {
            var row: [String] = [label]
            for c in 0..<columnLabels.count {
                row.append("\(counts[r][c])")
            }
            row.append("\(rowMarginals[r])")
            rows.append(row)
        }
        var totalRow: [String] = ["Total"]
        for value in columnMarginalsArray {
            totalRow.append("\(value)")
        }
        totalRow.append("\(grand)")
        rows.append(totalRow)

        var widths = headers.map { $0.count }
        for row in rows {
            for (c, cell) in row.enumerated() {
                if cell.count > widths[c] { widths[c] = cell.count }
            }
        }

        func padLeft(_ s: String, to width: Int) -> String {
            let pad = width - s.count
            return pad > 0 ? String(repeating: " ", count: pad) + s : s
        }

        func formatRow(_ row: [String]) -> String {
            return row.enumerated()
                .map { padLeft($0.element, to: widths[$0.offset]) }
                .joined(separator: "  ")
        }

        var lines: [String] = []
        lines.append(formatRow(headers))
        for row in rows {
            lines.append(formatRow(row))
        }
        return lines.joined(separator: "\n")
    }
}
