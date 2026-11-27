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

// MARK: - HypothesisTest

/// The outcome of a hypothesis test: a test statistic, its degrees of freedom, and a p-value.
///
/// Every test in Quiver returns the same type, so the code that reads a result does not
/// change when the test does. The t-tests (`tTest(against:alternative:)`,
/// `welchTTest(against:alternative:)`, `pairedTTest(against:alternative:)`), the proportion
/// tests (`proportionTest(against:alternative:)`), the chi-squared tests
/// (`chiSquaredTest()`) and the analysis of variance (`oneWayANOVA()`) each fill in
/// the fields that apply to them and leave the rest `nil`.
///
/// The p-value answers one question: if the null hypothesis were true, how often would
/// data at least this extreme appear? A small p-value is evidence against the null
/// hypothesis. It is not the probability that the null hypothesis is true, and it does not
/// say whether the effect is large enough to matter.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
///                       255.0, 249.0, 258.0, 244.0, 251.0]
///
/// if let result = sessionSeconds.tTest(against: 240.0) {
///     print(result)
///     // HypothesisTest: one-sample t-test, t = 4.575, df = 9, p = 0.0013
///
///     result.pValue                       // ≈ 0.0013
///     result.rejectsNullHypothesis()      // true, since 0.0013 < 0.05
/// }
/// ```
///
/// > Tip: Pair a test with an interval. `confidenceInterval(level:)` reports how large
/// > the effect plausibly is, while the p-value reports only whether it is distinguishable from none.
public struct HypothesisTest: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The procedure that produced a ``HypothesisTest``.
    public enum Kind: String, Equatable, Codable, Sendable {
        /// A t-test of one sample mean against a hypothesized value.
        case oneSampleT
        /// A t-test of two independent sample means that does not assume equal variances.
        case welchT
        /// A t-test of the mean of paired differences against zero.
        case pairedT
        /// A z-test of one sample proportion against a hypothesized value.
        case oneProportion
        /// A z-test comparing the proportions of two independent samples.
        case twoProportion
        /// A chi-squared test of independence on a contingency table.
        case chiSquaredIndependence
        /// A chi-squared test of observed counts against hypothesized proportions.
        case chiSquaredGoodnessOfFit
        /// An F-test of equal means across two or more groups.
        case oneWayANOVA
    }

    /// The direction of the alternative hypothesis for a test that has one.
    public enum Alternative: String, Equatable, Codable, Sendable {
        /// The true value differs from the hypothesized value in either direction.
        case twoSided
        /// The true value is larger than the hypothesized value.
        case greater
        /// The true value is smaller than the hypothesized value.
        case less
    }

    /// The procedure that produced this result.
    public let kind: Kind

    /// The value of the test statistic: `t`, `z`, `χ²` or `F`, depending on ``kind``.
    public let statistic: Double

    /// The degrees of freedom, or the numerator degrees of freedom for an F-test.
    ///
    /// `nil` for the proportion tests, whose z statistic has no degrees of freedom. A Welch
    /// t-test reports a non-integer value.
    public let degreesOfFreedom: Double?

    /// The denominator degrees of freedom of an F-test, or `nil` for every other test.
    public let denominatorDegreesOfFreedom: Double?

    /// The probability, if the null hypothesis were true, of a statistic at least as extreme as this one.
    public let pValue: Double

    /// The direction of the alternative hypothesis, or `nil` for the chi-squared and F tests.
    ///
    /// Those two tests measure departure from the null hypothesis in a single direction, so a
    /// choice of alternative does not apply.
    public let alternative: Alternative?

    /// Creates a result from its parts.
    ///
    /// The tests in Quiver build their own results; this initializer exists so that a result
    /// can be reconstructed from stored values, for example when decoding an archive.
    public init(
        kind: Kind,
        statistic: Double,
        degreesOfFreedom: Double?,
        denominatorDegreesOfFreedom: Double? = nil,
        pValue: Double,
        alternative: Alternative?
    ) {
        self.kind = kind
        self.statistic = statistic
        self.degreesOfFreedom = degreesOfFreedom
        self.denominatorDegreesOfFreedom = denominatorDegreesOfFreedom
        self.pValue = pValue
        self.alternative = alternative
    }

    /// Returns whether the p-value falls below the chosen significance level.
    ///
    /// A `true` result means the data would be unusual enough, under the null hypothesis,
    /// to reject it at that level. The conventional level is `0.05`: we accept a one in
    /// twenty chance of rejecting a null hypothesis that is in fact true.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let result = HypothesisTest(
    ///     kind: .oneSampleT, statistic: 2.4, degreesOfFreedom: 9,
    ///     pValue: 0.04, alternative: .twoSided
    /// )
    ///
    /// result.rejectsNullHypothesis()         // true at the default 0.05
    /// result.rejectsNullHypothesis(at: 0.01) // false at the stricter 0.01
    /// ```
    ///
    /// - Parameter significance: The significance level, in `(0, 1)`. Defaults to `0.05`.
    /// - Returns: `true` if `pValue` is strictly below `significance`.
    public func rejectsNullHypothesis(at significance: Double = 0.05) -> Bool {
        precondition(significance > 0.0 && significance < 1.0,
            "Significance level must be between 0 and 1")
        return pValue < significance
    }

    // MARK: - CustomStringConvertible

    /// A one-line summary such as `HypothesisTest: Welch t-test, t = -6.612, df = 12.62, p < 0.0001`.
    public var description: String {
        let name = _procedureName
        let statisticText = "\(_statisticSymbol) = \(String(format: "%.3f", statistic))"
        var parts = [name, statisticText]
        if let numerator = degreesOfFreedom {
            if let denominator = denominatorDegreesOfFreedom {
                parts.append("df = (\(_formatDegrees(numerator)), \(_formatDegrees(denominator)))")
            } else {
                parts.append("df = \(_formatDegrees(numerator))")
            }
        }
        parts.append(_formatPValue())
        return "HypothesisTest: " + parts.joined(separator: ", ")
    }

    // MARK: - Private Helpers

    /// The plain-English name of the procedure for the one-line description.
    private var _procedureName: String {
        switch kind {
        case .oneSampleT: return "one-sample t-test"
        case .welchT: return "Welch t-test"
        case .pairedT: return "paired t-test"
        case .oneProportion: return "one-proportion z-test"
        case .twoProportion: return "two-proportion z-test"
        case .chiSquaredIndependence: return "chi-squared test of independence"
        case .chiSquaredGoodnessOfFit: return "chi-squared goodness-of-fit test"
        case .oneWayANOVA: return "one-way ANOVA"
        }
    }

    /// The conventional symbol for the test statistic.
    private var _statisticSymbol: String {
        switch kind {
        case .oneSampleT, .welchT, .pairedT: return "t"
        case .oneProportion, .twoProportion: return "z"
        case .chiSquaredIndependence, .chiSquaredGoodnessOfFit: return "χ²"
        case .oneWayANOVA: return "F"
        }
    }

    /// Formats degrees of freedom as a whole number when they are one, and to two decimals otherwise.
    private func _formatDegrees(_ value: Double) -> String {
        if value == value.rounded() {
            return String(format: "%.0f", value)
        }
        return String(format: "%.2f", value)
    }

    /// Formats the p-value to four decimals, switching to `p < 0.0001` below that resolution.
    private func _formatPValue() -> String {
        if pValue < 0.0001 {
            return "p < 0.0001"
        }
        return "p = \(String(format: "%.4f", pValue))"
    }
}
