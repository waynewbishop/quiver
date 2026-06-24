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

/// A two-measure skewness diagnostic: the moment coefficient, the robust Bowley coefficient,
/// and whether the two agree.
///
/// Returned by `[Double].skewnessReport()`. The moment coefficient (the adjusted
/// Fisher-Pearson value, sample convention) weighs every observation and is sensitive to
/// outliers; the Bowley coefficient is built only from the quartiles and resists them.
/// Reading both, alongside the `agreement`, tells you not just which way the data leans but
/// whether that lean is real or an artifact of a few extreme values.
///
/// The sign of each coefficient gives direction: a positive value means a long tail toward
/// high values (right-skew); a negative value means a long tail toward low values
/// (left-skew). This follows from the definition — the moment coefficient is the third
/// standardized moment, whose sign is the sign of the average cubed deviation from the mean.
public struct SkewnessReport: Equatable, Sendable {

    /// The number of elements summarized.
    public let count: Int

    /// The adjusted Fisher-Pearson moment coefficient (sample convention). Unbounded;
    /// sensitive to outliers. Positive means a long tail toward high values, negative
    /// toward low values.
    public let moment: Double

    /// The Bowley quartile coefficient `(Q3 - 2·Q2 + Q1) / (Q3 - Q1)`. Bounded to
    /// `[-1, 1]`; resistant to outliers because it uses only the quartiles.
    public let bowley: Double

    /// Whether the outlier-sensitive moment measure and the outlier-resistant Bowley
    /// measure agree, conflict by magnitude, or conflict in direction.
    public let agreement: SkewnessAgreement

    public init(count: Int, moment: Double, bowley: Double, agreement: SkewnessAgreement) {
        self.count = count
        self.moment = moment
        self.bowley = bowley
        self.agreement = agreement
    }
}

extension SkewnessReport: CustomStringConvertible {

    /// The "approximately symmetric" band for the unbounded moment coefficient — the
    /// conventional cutoff below which skew is treated as negligible.
    private static let momentSymmetricBand = 0.5

    /// The symmetric band for the bounded Bowley coefficient. Smaller than the moment
    /// band because Bowley is compressed into `[-1, 1]`.
    private static let bowleySymmetricBand = 0.1

    public var description: String {
        let shapeLine = "shape:       \(SkewnessReport.shapePhrase(forMoment: moment))"
        let coefficientLine = "coefficient: \(formatSkew(moment))"
        let crossCheckLine = "cross-check: \(formatSkew(bowley))   \(crossCheckNote)"
        return [shapeLine, coefficientLine, crossCheckLine].joined(separator: "\n")
    }

    /// The plain-language shape phrase derived from the sign of the moment coefficient. The
    /// precise term travels in parentheses so the reader sees both the description and the name.
    private static func shapePhrase(forMoment value: Double) -> String {
        if abs(value) < momentSymmetricBand {
            return "roughly symmetric"
        }
        if value > 0 {
            return "long tail toward high values   (right-skewed)"
        }
        return "long tail toward low values   (left-skewed)"
    }

    /// The corroboration note, attributed to the robust measure. Never resolves a single
    /// direction when the two measures conflict — it names the conflict and prompts
    /// investigation rather than issuing a verdict.
    private var crossCheckNote: String {
        switch agreement {
        case .agree:
            return "\u{2713} a robust measure agrees — the shape looks real"
        case .mixed:
            return "\u{26A0} a few extreme values may be distorting the shape — check your outliers"
        case .direction:
            return "\u{26A0} the robust measure leans the other way — a few extreme values may be distorting this; check your outliers"
        }
    }

    /// Formats a coefficient the way the descriptive-stats surfaces do: whole numbers keep a
    /// single trailing zero, everything else renders to four places.
    private func formatSkew(_ value: Double) -> String {
        if value == value.rounded(.towardZero) && value.isFinite {
            return "\(Int(value)).0"
        }
        return String(format: "%.4f", value)
    }
}

extension SkewnessReport: Codable {
    private enum CodingKeys: String, CodingKey {
        case count, moment, bowley, agreement
    }
}

public extension Array where Element: BinaryFloatingPoint {

    /// Returns the Bowley quartile coefficient of skewness — a robust measure built only from
    /// the quartiles, so a few extreme values cannot distort it.
    ///
    /// Defined as `(Q3 - 2·Q2 + Q1) / (Q3 - Q1)`, the coefficient is bounded to `[-1, 1]`. A
    /// positive value means the upper quartile is stretched away from the median — a long tail
    /// toward high values (right-skew); a negative value means the reverse. Because it uses
    /// only the quartiles, it reports the shape of the central bulk and ignores the tails,
    /// which is exactly what makes it a useful counterweight to the outlier-sensitive
    /// `skewness()`.
    ///
    /// - Returns: The Bowley coefficient, or nil if the array is empty, any element is not
    ///   finite, or the interquartile range is zero (a constant or heavily-tied middle 50%,
    ///   where the measure is undefined).
    func bowleySkewness() -> Element? {
        guard let q = self.quartiles() else { return nil }
        guard q.iqr != 0 else { return nil }
        return (q.q3 - 2 * q.median + q.q1) / q.iqr
    }
}

public extension Array where Element == Double {

    /// Returns a two-measure skewness diagnostic: the outlier-sensitive moment coefficient,
    /// the outlier-resistant Bowley coefficient, and whether the two agree.
    ///
    /// A single skewness number can mislead: the moment coefficient weighs every value, so one
    /// extreme observation can report "strongly skewed" when the bulk of the data is nearly
    /// symmetric. This report computes skewness a second way — the Bowley quartile coefficient,
    /// which ignores the tails — and flags when the two disagree. Agreement corroborates the
    /// shape; disagreement is the signal that a few extreme values are driving the headline
    /// number, and that the right next step is to investigate them.
    ///
    /// The moment coefficient is the bias-corrected sample value (`skewness(bias: false)`); the
    /// Bowley coefficient is `bowleySkewness()`. The `agreement` is decided by comparing the
    /// sign of each measure, with a per-measure symmetric band (each measure has its own
    /// natural scale, so a single threshold would not be meaningful).
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]
    ///
    /// if let report = salaries.skewnessReport() {
    ///     print(report)
    ///     // shape:       long tail toward high values   (right-skewed)
    ///     // coefficient: 2.4109
    ///     // cross-check: -0.1304   ⚠ the robust measure leans the other way — a few extreme values may be distorting this; check your outliers
    /// }
    /// ```
    ///
    /// - Returns: A ``SkewnessReport``, or nil if either underlying measure is undefined —
    ///   fewer than three values or zero spread (moment), or zero interquartile range (Bowley).
    func skewnessReport() -> SkewnessReport? {
        guard let moment = self.skewness(bias: false),
              let bowley = self.bowleySkewness() else { return nil }

        let agreement = SkewnessReport.classifyAgreement(moment: moment, bowley: bowley)
        return SkewnessReport(count: self.count, moment: moment, bowley: bowley, agreement: agreement)
    }
}

extension SkewnessReport {

    /// Classifies how the two measures relate, comparing their *direction* rather than their
    /// magnitudes — the two coefficients live on different scales (moment is unbounded, Bowley
    /// is bounded), so only their signs are comparable. Each measure is reduced to a direction
    /// of -1, 0, or +1 using its own symmetric band, then the two directions are compared.
    static func classifyAgreement(moment: Double, bowley: Double) -> SkewnessAgreement {
        let momentDirection = direction(of: moment, band: momentSymmetricBand)
        let bowleyDirection = direction(of: bowley, band: bowleySymmetricBand)

        if momentDirection == bowleyDirection {
            return .agree
        }
        // Opposite nonzero signs are the strongest disagreement.
        if momentDirection != 0 && bowleyDirection != 0 {
            return .direction
        }
        // Exactly one measure is in its symmetric band while the other is decisively skewed.
        return .mixed
    }

    /// Reduces a coefficient to a direction: 0 inside the symmetric band, otherwise its sign.
    private static func direction(of value: Double, band: Double) -> Int {
        if abs(value) < band { return 0 }
        return value > 0 ? 1 : -1
    }
}
