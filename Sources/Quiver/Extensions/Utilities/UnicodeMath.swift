//
//  UnicodeMath.swift
//  Quiver
//
//  Shared internal helpers for Unicode mathematical rendering. Used by every
//  type that implements `asExpression()` — `Polynomial`, `Fraction`,
//  `[Double]`, and `[[Double]]` — so the framework's display surfaces format
//  numbers and powers identically wherever they appear.
//
//  The formatter spec follows the conventions established in `ColumnSummary`
//  and `PanelSummary`, extended to handle the edge cases that matter when a
//  value is presented as a mathematical expression rather than a tabular cell:
//  NaN, ±∞, negative zero, sub-1e-3 magnitudes, and the `%.4f`-rounds-to-an-
//  integer trap.
//

import Foundation

/// Reusable Unicode helpers for rendering numbers, superscripts, and aligned
/// columns in mathematical expressions.
///
/// All members are `static` because there is no per-instance state. The enum
/// is `internal` because the helpers are framework-internal infrastructure,
/// not part of Quiver's public surface.
internal enum UnicodeMath {

    // MARK: - Number formatting

    /// Formats a `Double` for display inside a mathematical expression.
    ///
    /// The rules, in order:
    ///
    /// 1. `NaN` renders as `"NaN"`. `+∞` and `-∞` render as `"∞"` and `"-∞"`
    ///    using the Unicode infinity symbol — appropriate in an expression
    ///    context and unambiguous for the reader.
    /// 2. Negative zero is normalized to positive zero so a rendered cell
    ///    never carries a leading minus sign that the underlying value's
    ///    sign-bit alone justifies.
    /// 3. Values with `|x| < 1e-3` (and not exactly zero) render in `%g`
    ///    notation with three significant figures (`"3.00e-05"`). This
    ///    prevents two distinct sub-millisecond coefficients from collapsing
    ///    to the same `"0.0000"` output.
    /// 4. Integer-valued doubles render without a decimal point.
    /// 5. Everything else uses `%.4f` with trailing-zero trimming — but only
    ///    when the trimmed result still parses as a non-integer. A value of
    ///    `0.99999` formats as `"1.0000"` under `%.4f`; trimming would yield
    ///    `"1"`, falsely advertising the value as an integer. In that case
    ///    the un-trimmed string is returned.
    static func formatCell(_ x: Double) -> String {
        if x.isNaN { return "NaN" }
        if x.isInfinite { return x > 0 ? "∞" : "-∞" }

        // Normalize -0.0 so the sign bit alone doesn't produce a leading minus.
        var value = x
        if value == 0.0 { value = 0.0 }

        // Sub-1e-3 magnitudes: switch to %g so distinct small values stay
        // distinguishable in the rendered output.
        if value != 0 && abs(value) < 1e-3 {
            return String(format: "%.3g", value)
        }

        // Integer-valued doubles: render without a decimal point. The
        // `abs(value) < 1e15` guard keeps the `Int` conversion safe for
        // values beyond what Int64 can faithfully represent.
        if value == value.rounded() && abs(value) < 1e15 {
            return String(Int(value))
        }

        // General case: %.4f with trailing-zero trimming, but only when the
        // trimmed output doesn't lie about being an integer.
        let formatted = String(format: "%.4f", value)
        var trimmed = formatted
        while trimmed.last == "0" { trimmed.removeLast() }
        if trimmed.last == "." { trimmed.removeLast() }

        // If trimming produced an integer-shaped string but the underlying
        // value isn't actually an integer, preserve the full %.4f form so the
        // reader sees the decimal places that justify the rounding.
        if trimmed.allSatisfy({ $0.isNumber || $0 == "-" }) && value != value.rounded() {
            return formatted
        }
        return trimmed
    }

    // MARK: - Superscript powers

    /// Returns the Unicode superscript representation of a non-negative
    /// integer power. Multi-digit powers concatenate the per-digit superscript
    /// characters — `10` becomes `"¹⁰"`.
    static func superscript(_ power: Int) -> String {
        let map: [Character: Character] = [
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
        ]
        var result = ""
        for ch in String(power) {
            if let s = map[ch] {
                result.append(s)
            }
        }
        return result
    }

    // MARK: - Column alignment

    /// Right-aligns a column of formatted cells to a common width by
    /// prepending spaces. Used when rendering vectors and matrices so that
    /// negative signs and decimal points line up vertically.
    static func rightAlign(_ cells: [String]) -> [String] {
        let width = cells.map(\.count).max() ?? 0
        return cells.map { String(repeating: " ", count: width - $0.count) + $0 }
    }
}
