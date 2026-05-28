//
//  MatrixExpression.swift
//  Quiver
//
//  `asExpression()` for `[[Double]]` and `[[Fraction]]` — the Unicode-display
//  sibling to `asFractions()` on the matrix types. Renders as a bracketed
//  grid with each column right-aligned to its widest entry, so negative
//  signs and decimal points line up.
//
//  Single-row matrices collapse to inline row form `[ a  b  c ]`. Single-
//  column matrices reuse the column-vector form. Multi-row, multi-column
//  matrices use the full bracketed-grid layout.
//

import Foundation

public extension Array where Element == [Double] {

    /// Returns the matrix as a string, with each cell formatted by
    /// `UnicodeMath.formatCell` and each column right-aligned to its widest
    /// entry.
    ///
    /// Sibling to `asFractions(maxDenominator:)` on `[[Double]]`. Both
    /// methods convert the matrix into a display form: `asFractions` exposes
    /// rational structure as `[[Fraction]]`; `asExpression` returns the
    /// matrix as a formatted multi-line string.
    ///
    /// ```swift
    /// let A: [[Double]] = [[3, 1], [-2, 5]]
    /// A.asExpression()
    /// // ⎡  3  1 ⎤
    /// // ⎣ -2  5 ⎦
    /// ```
    ///
    /// Single-row matrices collapse to inline row form
    /// (`[ 1  2  3 ]`). Empty matrices render as `"⟨⟩"`.
    ///
    /// See <doc:Rendering-Math-Primer> for the full family.
    ///
    /// - Returns: The matrix as a Unicode-formatted multi-line string.
    func asExpression() -> String {
        guard let firstRow = first, !firstRow.isEmpty else { return "⟨⟩" }
        let cells = map { row in row.map { UnicodeMath.formatCell($0) } }
        return _matrixRender(cells)
    }
}

public extension Array where Element == [Fraction] {

    /// Returns the matrix of fractions as a string, with each cell rendered
    /// by ``Fraction/asExpression()`` and each column right-aligned to its
    /// widest entry.
    ///
    /// Lets `[[Double]]` chain straight from rational structure to display
    /// form without a closure in the calling code:
    ///
    /// ```swift
    /// let A: [[Double]] = [[3, 1], [-2, 5]]
    /// try A.inverted().asFractions().asExpression()
    /// // ⎡  5/13  -1/13 ⎤
    /// // ⎣ -2/13   3/13 ⎦
    /// ```
    ///
    /// See <doc:Rendering-Math-Primer> for the full family.
    ///
    /// - Returns: The matrix of fractions as a Unicode-formatted string.
    func asExpression() -> String {
        guard let firstRow = first, !firstRow.isEmpty else { return "⟨⟩" }
        let cells = map { row in row.map { $0.asExpression() } }
        return _matrixRender(cells)
    }
}

// MARK: - Internal rendering

/// Assembles a matrix from pre-formatted cells. Computes per-column widths,
/// right-aligns every cell to its column width, joins each row with a
/// two-space gap, and wraps each row in the appropriate Unicode brackets.
internal func _matrixRender(_ cells: [[String]]) -> String {
    let rowCount = cells.count
    let colCount = cells[0].count

    // Per-column width: the widest cell in column j across every row.
    var widths = [Int](repeating: 0, count: colCount)
    for row in cells {
        for (j, cell) in row.enumerated() {
            widths[j] = Swift.max(widths[j], cell.count)
        }
    }

    let paddedRows: [String] = cells.map { row in
        row.enumerated()
            .map { j, cell in String(repeating: " ", count: widths[j] - cell.count) + cell }
            .joined(separator: "  ")
    }

    // Single-row matrix: collapse to inline row form.
    if rowCount == 1 {
        return "[ \(paddedRows[0]) ]"
    }

    var lines: [String] = []
    lines.append("⎡ \(paddedRows.first!) ⎤")
    for line in paddedRows.dropFirst().dropLast() {
        lines.append("⎢ \(line) ⎥")
    }
    lines.append("⎣ \(paddedRows.last!) ⎦")
    return lines.joined(separator: "\n")
}
