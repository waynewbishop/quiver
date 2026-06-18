//
//  ArrayExpression.swift
//  Quiver
//
//  `asExpression()` for `[Double]` and `[Fraction]` — the Unicode-display
//  sibling to `asFractions()`. Both methods convert a vector into a display
//  form: `asFractions` returns rational structure as `[Fraction]`;
//  `asExpression` returns the vector as a formatted string ready for the
//  Notebook output pane.
//
//  Defaults to column form because Quiver teaches linear algebra and
//  textbook convention is that vectors are columns. Pass `.inline` for the
//  angle-bracket form (`⟨3, 4, 5⟩`) used in prose contexts.
//

import Foundation

public extension Array where Element == Double {

    /// Returns the vector as a string, with each entry formatted by the
    /// shared `UnicodeMath.formatCell` rules (NaN, ±∞, small magnitudes,
    /// integers all render predictably).
    ///
    /// Sibling to `asFractions(maxDenominator:)`. Both methods convert
    /// a `[Double]` into a display form: `asFractions` returns rational
    /// structure as `[Fraction]`; `asExpression` returns the vector as a
    /// formatted multi-line string.
    ///
    /// The default `.column` form matches linear-algebra textbook
    /// convention. Pass `.inline` for the angle-bracket form used in prose.
    ///
    /// ```swift
    /// [3.0, 4.0, 5.0].asExpression()
    /// // ⎡ 3 ⎤
    /// // ⎢ 4 ⎥
    /// // ⎣ 5 ⎦
    ///
    /// [3.0, 4.0, 5.0].asExpression(form: .inline)
    /// // "⟨3, 4, 5⟩"
    /// ```
    ///
    /// Single-element vectors render as the scalar — `[5.0].asExpression()`
    /// returns `"5"`. The empty vector renders as `"⟨⟩"` in either form.
    ///
    /// See <doc:Rendering-Math-Primer> for the full family.
    ///
    /// - Parameter form: `.column` (default) for stacked bracket form,
    ///   `.inline` for the angle-bracket form used in prose.
    /// - Returns: The vector as a Unicode-formatted string.
    func asExpression(form: VectorForm = .column) -> String {
        guard !isEmpty else { return "⟨⟩" }
        if count == 1 { return UnicodeMath.formatCell(self[0]) }

        let cells = map { UnicodeMath.formatCell($0) }

        switch form {
        case .inline:
            return "⟨" + cells.joined(separator: ", ") + "⟩"
        case .column:
            return _columnRender(cells)
        }
    }
}

public extension Array where Element == Fraction {

    /// Returns the vector of fractions as a string, with each entry
    /// rendered by ``Fraction/asExpression()`` and right-aligned so
    /// numerators and denominators line up vertically.
    ///
    /// Lets `[Double]` chain straight from rational structure to display
    /// form without a closure in the calling code:
    ///
    /// ```swift
    /// [0.6, 0.75, 0.5].asFractions().asExpression()
    /// // ⎡ 3/5 ⎤
    /// // ⎢ 3/4 ⎥
    /// // ⎣ 1/2 ⎦
    /// ```
    ///
    /// See <doc:Rendering-Math-Primer> for the full family.
    ///
    /// - Parameter form: `.column` (default) for stacked bracket form,
    ///   `.inline` for the angle-bracket form used in prose.
    /// - Returns: The vector of fractions as a Unicode-formatted string.
    func asExpression(form: VectorForm = .column) -> String {
        guard !isEmpty else { return "⟨⟩" }
        if count == 1 { return self[0].asExpression() }

        let cells = map { $0.asExpression() }

        switch form {
        case .inline:
            return "⟨" + cells.joined(separator: ", ") + "⟩"
        case .column:
            return _columnRender(cells)
        }
    }
}

// MARK: - Internal rendering

/// Assembles a column vector from pre-formatted cells, right-aligning them
/// to a common width and choosing the right bracket characters for the
/// first, last, and middle rows.
internal func _columnRender(_ cells: [String]) -> String {
    let padded = UnicodeMath.rightAlign(cells)

    var lines: [String] = []
    lines.append("⎡ \(padded.first!) ⎤")
    for cell in padded.dropFirst().dropLast() {
        lines.append("⎢ \(cell) ⎥")
    }
    if padded.count > 1 {
        lines.append("⎣ \(padded.last!) ⎦")
    }
    return lines.joined(separator: "\n")
}
