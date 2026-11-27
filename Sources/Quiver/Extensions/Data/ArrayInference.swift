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

// MARK: - t-tests and Confidence Intervals

public extension Array where Element == Double {

    /// Runs a one-sample t-test of the mean against a hypothesized value.
    ///
    /// The test asks whether the sample mean is far enough from `hypothesizedMean`, measured
    /// in standard errors, that chance variation in the sample is an unlikely explanation. The
    /// statistic is `t = (mean - hypothesizedMean) / standardError` with `n - 1` degrees of
    /// freedom, and the p-value comes from the t-distribution.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
    ///                       255.0, 249.0, 258.0, 244.0, 251.0]
    ///
    /// // Is the average session different from the 240-second baseline?
    /// if let result = sessionSeconds.tTest(against: 240.0) {
    ///     print(result)
    ///     // HypothesisTest: one-sample t-test, t = 4.575, df = 9, p = 0.0013
    /// }
    ///
    /// // Is it longer than the baseline? Only the upper tail counts.
    /// let longer = sessionSeconds.tTest(against: 240.0, alternative: .greater)
    /// longer?.pValue  // ≈ 0.00067, half the two-sided value
    /// ```
    ///
    /// - Parameters:
    ///   - hypothesizedMean: The population mean under the null hypothesis.
    ///   - alternative: The direction of the alternative hypothesis. Defaults to `.twoSided`.
    /// - Returns: The test result, or `nil` if the array has fewer than two elements, contains a
    ///   value that is not finite, or has no variation (every value identical).
    func tTest(
        against hypothesizedMean: Double,
        alternative: HypothesisTest.Alternative = .twoSided
    ) -> HypothesisTest? {
        return _Inference.oneSampleT(
            sample: self, hypothesizedMean: hypothesizedMean,
            alternative: alternative, kind: .oneSampleT
        )
    }

    /// Runs Welch's t-test comparing the means of two independent samples.
    ///
    /// Welch's test does not assume the two groups share a variance, so it is the safe default
    /// when comparing groups of different sizes or spreads. The statistic is
    /// `t = (mean1 - mean2) / √(s1²/n1 + s2²/n2)`, and the degrees of freedom come from the
    /// Welch-Satterthwaite approximation, so they are usually not a whole number.
    ///
    /// The alternative describes `self` relative to `other`: `.greater` means the mean of this
    /// array is larger than the mean of `other`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let oldRoute = [27.1, 22.4, 25.3, 24.8, 26.0, 23.9, 25.5, 24.1]
    /// let newRoute = [29.0, 31.2, 27.8, 30.5, 28.9, 32.1, 29.7]
    ///
    /// if let result = oldRoute.welchTTest(against: newRoute) {
    ///     print(result)
    ///     // HypothesisTest: Welch t-test, t = -6.612, df = 12.62, p < 0.0001
    ///
    ///     result.rejectsNullHypothesis()  // true
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - other: The second sample.
    ///   - alternative: The direction of the alternative hypothesis about `mean(self) - mean(other)`.
    ///     Defaults to `.twoSided`.
    /// - Returns: The test result, or `nil` if either array has fewer than two elements, contains
    ///   a value that is not finite, or neither array has any variation.
    func welchTTest(
        against other: [Double],
        alternative: HypothesisTest.Alternative = .twoSided
    ) -> HypothesisTest? {
        return _Inference.welchT(first: self, second: other, alternative: alternative)
    }

    /// Runs a paired t-test on two matched measurements of the same subjects.
    ///
    /// Each element of `self` is paired with the element of `other` at the same index, such as
    /// one student's score before and after a course, or two models scored on the same test
    /// split. The test works on the differences `self - other` and asks whether their mean is
    /// distinguishable from zero. Pairing removes the variation between subjects, which makes
    /// the test far more sensitive than comparing the two groups as if they were independent.
    ///
    /// The alternative describes `self` relative to `other`: `.greater` means `self` tends to
    /// be larger than `other`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let before = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]
    /// let after  = [90.0, 75.0, 96.0, 85.0, 89.0, 80.0, 88.0, 94.0]
    ///
    /// // Did scores improve? The one-sided alternative says `after` is larger.
    /// if let result = after.pairedTTest(against: before, alternative: .greater) {
    ///     print(result)
    ///     // HypothesisTest: paired t-test, t = 3.989, df = 7, p = 0.0026
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - other: The matched measurements. Must have the same count as `self`.
    ///   - alternative: The direction of the alternative hypothesis about the mean of `self - other`.
    ///     Defaults to `.twoSided`.
    /// - Returns: The test result, or `nil` if the arrays differ in length, have fewer than two
    ///   pairs, contain a value that is not finite, or the differences are all identical.
    func pairedTTest(
        against other: [Double],
        alternative: HypothesisTest.Alternative = .twoSided
    ) -> HypothesisTest? {
        guard self.count == other.count else { return nil }
        var differences = [Double]()
        differences.reserveCapacity(self.count)
        for index in 0..<self.count {
            differences.append(self[index] - other[index])
        }
        return _Inference.oneSampleT(
            sample: differences, hypothesizedMean: 0.0,
            alternative: alternative, kind: .pairedT
        )
    }

    /// Returns a t-based confidence interval for the population mean.
    ///
    /// The interval is `mean ± tCritical × standardError`, where `tCritical` is the t-distribution
    /// quantile for the chosen coverage and `n - 1` degrees of freedom. Over many repeated samples,
    /// about `level` of the intervals built this way would contain the true mean. That describes
    /// the procedure, not any one interval: it is not a `level` chance that the mean lies inside
    /// the interval we computed.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
    ///                       255.0, 249.0, 258.0, 244.0, 251.0]
    ///
    /// if let interval = sessionSeconds.confidenceInterval(level: 0.95) {
    ///     interval.lower  // ≈ 245.06
    ///     interval.upper  // ≈ 254.94
    /// }
    /// ```
    ///
    /// - Parameter level: The coverage level in `(0, 1)`. Defaults to `0.95`.
    /// - Returns: The interval, or `nil` if the array has fewer than two elements, contains a value
    ///   that is not finite, has no variation, or `level` is outside `(0, 1)`.
    func confidenceInterval(level: Double = 0.95) -> ConfidenceInterval? {
        guard level > 0.0, level < 1.0 else { return nil }
        guard self.count >= 2, self.allSatisfy({ $0.isFinite }) else { return nil }
        guard let mean = self.mean(),
              let standardError = self.standardError(),
              standardError > 0.0,
              let critical = Distributions.t.quantile(
                p: 0.5 + level / 2.0, df: Double(self.count - 1)
              ) else {
            return nil
        }
        let margin = critical * standardError
        return ConfidenceInterval(lower: mean - margin, upper: mean + margin)
    }
}

// MARK: - Analysis of Variance

public extension Array where Element == [Double] {

    /// Runs a one-way analysis of variance on the groups in this array.
    ///
    /// Each inner array is one group of observations. The test asks whether the group means
    /// differ by more than the scatter inside the groups would explain. The F statistic is the
    /// variation between group means divided by the variation within groups, with `k - 1` and
    /// `N - k` degrees of freedom for `k` groups and `N` observations in total.
    ///
    /// A small p-value says at least one group mean differs from the others. It does not say
    /// which one: a follow-up comparison is needed for that.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let methodA = [23.0, 25.0, 21.0, 24.0, 26.0]
    /// let methodB = [30.0, 28.0, 31.0, 29.0, 27.0]
    /// let methodC = [22.0, 20.0, 24.0, 23.0, 21.0]
    ///
    /// let groups = [methodA, methodB, methodC]
    /// if let result = groups.oneWayANOVA() {
    ///     print(result)
    ///     // HypothesisTest: one-way ANOVA, F = 22.782, df = (2, 12), p < 0.0001
    /// }
    /// ```
    ///
    /// - Returns: The test result, or `nil` if there are fewer than two groups, any group is
    ///   empty, a value is not finite, there are no more observations than groups, or the groups
    ///   have no variation inside them.
    func oneWayANOVA() -> HypothesisTest? {
        return _Inference.oneWayANOVA(groups: self)
    }
}

// MARK: - Proportion Tests

public extension Array where Element == Bool {

    /// Runs a z-test of the share of `true` values against a hypothesized proportion.
    ///
    /// Treat each element as one trial and `true` as a success. The test asks whether the
    /// observed share of successes is far enough from `hypothesizedProportion`, measured in
    /// standard errors, to rule out chance. It relies on the normal approximation, which is
    /// reliable when the hypothesized proportion predicts at least about 10 successes and 10
    /// failures. No continuity correction is applied.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // 14 of 20 visitors clicked. Is that different from a fair coin?
    /// let clicks = [Bool](repeating: true, count: 14) + [Bool](repeating: false, count: 6)
    ///
    /// if let result = clicks.proportionTest(against: 0.5) {
    ///     print(result)
    ///     // HypothesisTest: one-proportion z-test, z = 1.789, p = 0.0736
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - hypothesizedProportion: The proportion under the null hypothesis, strictly between 0 and 1.
    ///   - alternative: The direction of the alternative hypothesis. Defaults to `.twoSided`.
    /// - Returns: The test result, or `nil` if the array is empty or the hypothesized proportion
    ///   is not strictly between 0 and 1.
    func proportionTest(
        against hypothesizedProportion: Double,
        alternative: HypothesisTest.Alternative = .twoSided
    ) -> HypothesisTest? {
        return _Inference.oneProportion(
            successes: self.filter { $0 }.count, trials: self.count,
            hypothesizedProportion: hypothesizedProportion, alternative: alternative
        )
    }

    /// Runs a z-test comparing the share of `true` values in two independent samples.
    ///
    /// Under the null hypothesis both samples share one proportion, so the test pools them to
    /// estimate it and measures how far apart the two observed proportions are in standard
    /// errors. It uses the normal approximation, so each sample should contain at least about
    /// 10 successes and 10 failures. The alternative describes `self` relative to `other`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // Variant A converted 30 of 100 visitors, variant B converted 45 of 100.
    /// let variantA = [Bool](repeating: true, count: 30) + [Bool](repeating: false, count: 70)
    /// let variantB = [Bool](repeating: true, count: 45) + [Bool](repeating: false, count: 55)
    ///
    /// if let result = variantA.proportionTest(against: variantB) {
    ///     print(result)
    ///     // HypothesisTest: two-proportion z-test, z = -2.191, p = 0.0285
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - other: The second sample.
    ///   - alternative: The direction of the alternative hypothesis about
    ///     `proportion(self) - proportion(other)`. Defaults to `.twoSided`.
    /// - Returns: The test result, or `nil` if either array is empty or the pooled proportion is
    ///   exactly 0 or 1, which leaves no variation to test against.
    func proportionTest(
        against other: [Bool],
        alternative: HypothesisTest.Alternative = .twoSided
    ) -> HypothesisTest? {
        return _Inference.twoProportion(
            successesFirst: self.filter { $0 }.count, trialsFirst: self.count,
            successesSecond: other.filter { $0 }.count, trialsSecond: other.count,
            alternative: alternative
        )
    }
}

// MARK: - Goodness of Fit

public extension Array where Element == Int {

    /// Runs a chi-squared goodness-of-fit test of observed counts against hypothesized proportions.
    ///
    /// Each element is the number of observations that fell in one category. The test asks
    /// whether those counts depart from the hypothesized shares by more than chance would
    /// produce. The statistic is `χ² = Σ (observed - expected)² / expected` with
    /// `categories - 1` degrees of freedom, where each expected count is the total times the
    /// category's proportion. The approximation is reliable when every expected count is at
    /// least about 5.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // 120 rolls of a die. Is it fair? Equal shares are the default hypothesis.
    /// let rolls = [18, 22, 16, 25, 19, 20]
    ///
    /// if let result = rolls.chiSquaredTest() {
    ///     print(result)
    ///     // HypothesisTest: chi-squared goodness-of-fit test, χ² = 2.500, df = 5, p = 0.7765
    /// }
    ///
    /// // Compare against shares we expect instead of equal ones.
    /// let expected = [0.25, 0.15, 0.25, 0.35]
    /// let orders = [30, 10, 20, 40]
    /// orders.chiSquaredTest(expectedProportions: expected)?.pValue  // ≈ 0.2232
    /// ```
    ///
    /// - Parameter expectedProportions: The hypothesized share of each category, all positive and
    ///   summing to 1. Pass `nil` (the default) for equal shares.
    /// - Returns: The test result, or `nil` if there are fewer than two categories, a count is
    ///   negative, the total is zero, or the proportions do not match the counts or do not sum to 1.
    func chiSquaredTest(expectedProportions: [Double]? = nil) -> HypothesisTest? {
        return _Inference.chiSquaredGoodnessOfFit(
            observed: self, expectedProportions: expectedProportions
        )
    }
}

// MARK: - Independence

public extension ContingencyTable {

    /// Runs a chi-squared test of independence on this table.
    ///
    /// The test asks whether the row variable and the column variable are related, or whether
    /// the pattern in the counts is the sort of scatter we would see if they were independent.
    /// Expected counts come from the margins: each is `rowTotal × columnTotal / grandTotal`. The
    /// statistic is `χ² = Σ (observed - expected)² / expected` with `(rows - 1)(columns - 1)`
    /// degrees of freedom. The approximation is reliable when every expected count is at least
    /// about 5, and no continuity correction is applied to a 2 by 2 table.
    ///
    /// A small p-value says the variables are associated. It does not say how strongly, or
    /// which way the cause runs.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// if let table = ContingencyTable(
    ///     rowLabels: ["Treatment", "Control"],
    ///     columnLabels: ["Improved", "No change"],
    ///     counts: [[30, 20], [18, 32]]
    /// ), let result = table.chiSquaredTest() {
    ///     print(result)
    ///     // HypothesisTest: chi-squared test of independence, χ² = 5.769, df = 1, p = 0.0163
    /// }
    /// ```
    ///
    /// - Returns: The test result, or `nil` if the table has fewer than two rows or columns, or
    ///   any row or column totals zero.
    func chiSquaredTest() -> HypothesisTest? {
        return _Inference.chiSquaredIndependence(counts: counts)
    }
}
