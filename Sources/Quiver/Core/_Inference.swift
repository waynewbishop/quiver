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

// MARK: - Inference Core

/// Internal computation for hypothesis tests and confidence intervals.
///
/// Each function takes plain arrays and returns a finished ``HypothesisTest`` (or the raw
/// numbers an interval needs), or `nil` when the data cannot support the test: too few
/// observations, no variation, or a value that is not finite. The public entry points on
/// `Array` and ``ContingencyTable`` are thin wrappers over these functions.
internal enum _Inference {

    // MARK: - t-tests

    /// Runs a one-sample t-test of the sample mean against a hypothesized mean.
    ///
    /// The statistic is `t = (mean - hypothesizedMean) / standardError` with `n - 1` degrees
    /// of freedom, where the standard error uses the sample standard deviation.
    ///
    /// - Parameters:
    ///   - sample: The observations. Needs at least two, all finite, with some variation.
    ///   - hypothesizedMean: The population mean under the null hypothesis.
    ///   - alternative: The direction of the alternative hypothesis.
    ///   - kind: The kind to record on the result. A paired test passes `.pairedT`.
    /// - Returns: The test result, or `nil` if the sample cannot support a t-test.
    static func oneSampleT(
        sample: [Double],
        hypothesizedMean: Double,
        alternative: HypothesisTest.Alternative,
        kind: HypothesisTest.Kind
    ) -> HypothesisTest? {
        guard hypothesizedMean.isFinite else { return nil }
        guard sample.count >= 2, sample.allSatisfy({ $0.isFinite }) else { return nil }
        guard let mean = sample.mean(),
              let standardError = sample.standardError(),
              standardError > 0.0 else {
            return nil
        }
        let statistic = (mean - hypothesizedMean) / standardError
        let degrees = Double(sample.count - 1)
        guard let pValue = symmetricPValue(
            statistic: statistic, alternative: alternative, degrees: degrees, distribution: .t
        ) else {
            return nil
        }
        return HypothesisTest(
            kind: kind, statistic: statistic, degreesOfFreedom: degrees,
            pValue: pValue, alternative: alternative
        )
    }

    /// Runs Welch's t-test of two independent sample means, without assuming equal variances.
    ///
    /// The statistic is `t = (mean1 - mean2) / √(s1²/n1 + s2²/n2)`. The degrees of freedom come
    /// from the Welch-Satterthwaite approximation, which is generally not a whole number.
    ///
    /// - Parameters:
    ///   - first: The first sample. Needs at least two finite observations.
    ///   - second: The second sample. Needs at least two finite observations.
    ///   - alternative: The direction of the alternative hypothesis about `mean1 - mean2`.
    /// - Returns: The test result, or `nil` if either sample is too small or neither varies.
    static func welchT(
        first: [Double],
        second: [Double],
        alternative: HypothesisTest.Alternative
    ) -> HypothesisTest? {
        guard first.count >= 2, second.count >= 2 else { return nil }
        guard first.allSatisfy({ $0.isFinite }), second.allSatisfy({ $0.isFinite }) else { return nil }
        guard let meanFirst = first.mean(), let meanSecond = second.mean(),
              let varianceFirst = first.variance(), let varianceSecond = second.variance() else {
            return nil
        }
        let countFirst = Double(first.count)
        let countSecond = Double(second.count)

        // Squared standard error of each mean, and of their difference.
        let termFirst = varianceFirst / countFirst
        let termSecond = varianceSecond / countSecond
        let combined = termFirst + termSecond
        guard combined > 0.0, combined.isFinite else { return nil }

        let statistic = (meanFirst - meanSecond) / combined.squareRoot()

        // Welch-Satterthwaite degrees of freedom.
        let denominator = termFirst * termFirst / (countFirst - 1.0)
            + termSecond * termSecond / (countSecond - 1.0)
        guard denominator > 0.0 else { return nil }
        let degrees = combined * combined / denominator

        guard let pValue = symmetricPValue(
            statistic: statistic, alternative: alternative, degrees: degrees, distribution: .t
        ) else {
            return nil
        }
        return HypothesisTest(
            kind: .welchT, statistic: statistic, degreesOfFreedom: degrees,
            pValue: pValue, alternative: alternative
        )
    }

    // MARK: - Proportion tests

    /// Runs a z-test of one sample proportion against a hypothesized proportion.
    ///
    /// The statistic is `z = (p̂ - p0) / √(p0(1 - p0)/n)`, which relies on the normal
    /// approximation to the binomial.
    ///
    /// - Parameters:
    ///   - successes: The number of successes observed.
    ///   - trials: The number of trials. Must be positive.
    ///   - hypothesizedProportion: The proportion under the null hypothesis, strictly between 0 and 1.
    ///   - alternative: The direction of the alternative hypothesis.
    /// - Returns: The test result, or `nil` for invalid counts or an out-of-range hypothesis.
    static func oneProportion(
        successes: Int,
        trials: Int,
        hypothesizedProportion: Double,
        alternative: HypothesisTest.Alternative
    ) -> HypothesisTest? {
        guard trials > 0, successes >= 0, successes <= trials else { return nil }
        guard hypothesizedProportion > 0.0, hypothesizedProportion < 1.0 else { return nil }

        let trialsDouble = Double(trials)
        let observed = Double(successes) / trialsDouble
        let standardError = (hypothesizedProportion * (1.0 - hypothesizedProportion) / trialsDouble)
            .squareRoot()
        let statistic = (observed - hypothesizedProportion) / standardError
        guard let pValue = symmetricPValue(
            statistic: statistic, alternative: alternative, degrees: 0.0, distribution: .normal
        ) else {
            return nil
        }
        return HypothesisTest(
            kind: .oneProportion, statistic: statistic, degreesOfFreedom: nil,
            pValue: pValue, alternative: alternative
        )
    }

    /// Runs a pooled z-test comparing the proportions of two independent samples.
    ///
    /// Under the null hypothesis both samples share one proportion, estimated by pooling them:
    /// `z = (p̂1 - p̂2) / √(p̂(1 - p̂)(1/n1 + 1/n2))`.
    ///
    /// - Parameters:
    ///   - successesFirst: Successes in the first sample.
    ///   - trialsFirst: Trials in the first sample. Must be positive.
    ///   - successesSecond: Successes in the second sample.
    ///   - trialsSecond: Trials in the second sample. Must be positive.
    ///   - alternative: The direction of the alternative hypothesis about `p̂1 - p̂2`.
    /// - Returns: The test result, or `nil` for invalid counts or a pooled proportion of exactly 0 or 1.
    static func twoProportion(
        successesFirst: Int,
        trialsFirst: Int,
        successesSecond: Int,
        trialsSecond: Int,
        alternative: HypothesisTest.Alternative
    ) -> HypothesisTest? {
        guard trialsFirst > 0, successesFirst >= 0, successesFirst <= trialsFirst else { return nil }
        guard trialsSecond > 0, successesSecond >= 0, successesSecond <= trialsSecond else { return nil }

        let trialsFirstDouble = Double(trialsFirst)
        let trialsSecondDouble = Double(trialsSecond)
        let proportionFirst = Double(successesFirst) / trialsFirstDouble
        let proportionSecond = Double(successesSecond) / trialsSecondDouble
        let pooled = Double(successesFirst + successesSecond) / (trialsFirstDouble + trialsSecondDouble)

        // A pooled proportion of 0 or 1 means neither sample varies, so there is no spread to test against.
        guard pooled > 0.0, pooled < 1.0 else { return nil }
        let standardError = (pooled * (1.0 - pooled) * (1.0 / trialsFirstDouble + 1.0 / trialsSecondDouble))
            .squareRoot()
        let statistic = (proportionFirst - proportionSecond) / standardError
        guard let pValue = symmetricPValue(
            statistic: statistic, alternative: alternative, degrees: 0.0, distribution: .normal
        ) else {
            return nil
        }
        return HypothesisTest(
            kind: .twoProportion, statistic: statistic, degreesOfFreedom: nil,
            pValue: pValue, alternative: alternative
        )
    }

    // MARK: - Chi-squared tests

    /// Runs a chi-squared test of independence on a table of counts.
    ///
    /// Expected counts come from the margins: `expected = rowTotal × columnTotal / grandTotal`.
    /// The statistic is `χ² = Σ (observed - expected)² / expected` with `(rows - 1)(columns - 1)`
    /// degrees of freedom. No continuity correction is applied.
    ///
    /// - Parameter counts: The observed counts, one inner array per row. Needs at least two
    ///   rows and two columns, and no row or column may total zero.
    /// - Returns: The test result, or `nil` if the table is too small or has an empty margin.
    static func chiSquaredIndependence(counts: [[Int]]) -> HypothesisTest? {
        guard counts.count >= 2, let columnCount = counts.first?.count, columnCount >= 2 else {
            return nil
        }
        guard counts.allSatisfy({ $0.count == columnCount }) else { return nil }

        var rowTotals = [Double](repeating: 0.0, count: counts.count)
        var columnTotals = [Double](repeating: 0.0, count: columnCount)
        var grandTotal = 0.0
        for (r, row) in counts.enumerated() {
            for (c, cell) in row.enumerated() {
                guard cell >= 0 else { return nil }
                let value = Double(cell)
                rowTotals[r] += value
                columnTotals[c] += value
                grandTotal += value
            }
        }
        guard grandTotal > 0.0 else { return nil }
        guard rowTotals.allSatisfy({ $0 > 0.0 }), columnTotals.allSatisfy({ $0 > 0.0 }) else {
            return nil
        }

        var statistic = 0.0
        for r in 0..<counts.count {
            for c in 0..<columnCount {
                let expected = rowTotals[r] * columnTotals[c] / grandTotal
                let difference = Double(counts[r][c]) - expected
                statistic += difference * difference / expected
            }
        }
        let degrees = Double((counts.count - 1) * (columnCount - 1))
        guard let pValue = chiSquaredUpperTail(statistic: statistic, degrees: degrees) else {
            return nil
        }
        return HypothesisTest(
            kind: .chiSquaredIndependence, statistic: statistic, degreesOfFreedom: degrees,
            pValue: pValue, alternative: nil
        )
    }

    /// Runs a chi-squared goodness-of-fit test of observed counts against hypothesized proportions.
    ///
    /// Each expected count is `total × proportion`. The statistic is
    /// `χ² = Σ (observed - expected)² / expected` with `categories - 1` degrees of freedom.
    ///
    /// - Parameters:
    ///   - observed: The observed count in each category. Needs at least two categories and a positive total.
    ///   - expectedProportions: The hypothesized share of each category, positive and summing to 1.
    ///     Pass `nil` for equal shares.
    /// - Returns: The test result, or `nil` for invalid counts or proportions.
    static func chiSquaredGoodnessOfFit(
        observed: [Int],
        expectedProportions: [Double]?
    ) -> HypothesisTest? {
        guard observed.count >= 2, observed.allSatisfy({ $0 >= 0 }) else { return nil }
        let total = Double(observed.reduce(0, +))
        guard total > 0.0 else { return nil }

        let proportions: [Double]
        if let supplied = expectedProportions {
            guard supplied.count == observed.count else { return nil }
            guard supplied.allSatisfy({ $0 > 0.0 && $0.isFinite }) else { return nil }
            let sum = supplied.reduce(0.0, +)
            guard abs(sum - 1.0) < 1e-9 else { return nil }
            proportions = supplied
        } else {
            proportions = [Double](repeating: 1.0 / Double(observed.count), count: observed.count)
        }

        var statistic = 0.0
        for (index, count) in observed.enumerated() {
            let expected = total * proportions[index]
            let difference = Double(count) - expected
            statistic += difference * difference / expected
        }
        let degrees = Double(observed.count - 1)
        guard let pValue = chiSquaredUpperTail(statistic: statistic, degrees: degrees) else {
            return nil
        }
        return HypothesisTest(
            kind: .chiSquaredGoodnessOfFit, statistic: statistic, degreesOfFreedom: degrees,
            pValue: pValue, alternative: nil
        )
    }

    // MARK: - Analysis of variance

    /// Runs a one-way analysis of variance across two or more groups.
    ///
    /// The F statistic is the variation between group means divided by the variation within
    /// groups: `F = (SSB / (k - 1)) / (SSW / (N - k))`, with `k - 1` and `N - k` degrees of freedom.
    ///
    /// - Parameter groups: One array of observations per group. Needs at least two groups, none
    ///   empty, all values finite, more observations in total than groups, and some variation
    ///   within the groups.
    /// - Returns: The test result, or `nil` if the groups cannot support the test.
    static func oneWayANOVA(groups: [[Double]]) -> HypothesisTest? {
        guard groups.count >= 2 else { return nil }
        guard groups.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ $0.isFinite }) }) else { return nil }

        let groupCount = groups.count
        let total = groups.reduce(0) { $0 + $1.count }
        guard total > groupCount else { return nil }

        var grandSum = 0.0
        for group in groups {
            grandSum += group.reduce(0.0, +)
        }
        let grandMean = grandSum / Double(total)

        var sumSquaresBetween = 0.0
        var sumSquaresWithin = 0.0
        for group in groups {
            let groupMean = group.reduce(0.0, +) / Double(group.count)
            let offset = groupMean - grandMean
            sumSquaresBetween += Double(group.count) * offset * offset
            for value in group {
                let deviation = value - groupMean
                sumSquaresWithin += deviation * deviation
            }
        }

        let degreesBetween = Double(groupCount - 1)
        let degreesWithin = Double(total - groupCount)
        let meanSquareWithin = sumSquaresWithin / degreesWithin

        // Without variation inside the groups the ratio is undefined (or infinite).
        guard meanSquareWithin > 0.0, meanSquareWithin.isFinite else { return nil }
        let statistic = (sumSquaresBetween / degreesBetween) / meanSquareWithin
        guard statistic.isFinite,
              let pValue = Distributions._fUpperTail(
                x: statistic, df1: degreesBetween, df2: degreesWithin
              ) else {
            return nil
        }
        return HypothesisTest(
            kind: .oneWayANOVA, statistic: statistic, degreesOfFreedom: degreesBetween,
            denominatorDegreesOfFreedom: degreesWithin, pValue: pValue, alternative: nil
        )
    }

    // MARK: - Tails

    /// The symmetric reference distributions used by the t and z tests.
    enum SymmetricDistribution {
        case t
        case normal
    }

    /// Converts a statistic from a symmetric distribution into a p-value for the chosen alternative.
    ///
    /// Each branch reads a lower tail directly rather than computing `1 - cdf`, so very small
    /// p-values keep their precision: two-sided is `2·P(T ≤ -|t|)`, "greater" is `P(T ≤ -t)`,
    /// and "less" is `P(T ≤ t)`.
    ///
    /// - Parameters:
    ///   - statistic: The test statistic.
    ///   - alternative: The direction of the alternative hypothesis.
    ///   - degrees: The degrees of freedom, used only for the t-distribution.
    ///   - distribution: The reference distribution.
    /// - Returns: The p-value in `[0, 1]`, or `nil` if the distribution function fails.
    static func symmetricPValue(
        statistic: Double,
        alternative: HypothesisTest.Alternative,
        degrees: Double,
        distribution: SymmetricDistribution
    ) -> Double? {
        guard statistic.isFinite else { return nil }
        let lowerTailPoint: Double
        switch alternative {
        case .twoSided: lowerTailPoint = -abs(statistic)
        case .greater: lowerTailPoint = -statistic
        case .less: lowerTailPoint = statistic
        }

        let lowerTail: Double?
        switch distribution {
        case .t:
            lowerTail = Distributions.t.cdf(x: lowerTailPoint, df: degrees)
        case .normal:
            lowerTail = Distributions.normal.cdf(x: lowerTailPoint, mean: 0.0, standardDeviation: 1.0)
        }
        guard let tail = lowerTail else { return nil }
        let value = alternative == .twoSided ? 2.0 * tail : tail
        return Swift.min(1.0, Swift.max(0.0, value))
    }

    /// Returns the upper-tail probability of a chi-squared statistic: `P(χ² ≥ statistic)`.
    ///
    /// - Parameters:
    ///   - statistic: The chi-squared statistic.
    ///   - degrees: The degrees of freedom. Must be positive.
    /// - Returns: The upper-tail probability, or `nil` if the computation fails.
    static func chiSquaredUpperTail(statistic: Double, degrees: Double) -> Double? {
        guard statistic.isFinite, degrees > 0.0 else { return nil }
        guard let tail = Distributions._regularizedIncompleteGammaQ(a: 0.5 * degrees, x: 0.5 * statistic) else {
            return nil
        }
        return Swift.min(1.0, Swift.max(0.0, tail))
    }
}
