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

import XCTest
@testable import Quiver

/// Tests for `HypothesisTest` and the tests and intervals that produce it.
///
/// Reference values were computed independently with SciPy (`ttest_1samp`, `ttest_ind` with
/// `equal_var=False`, `ttest_rel`, `f_oneway`, `chisquare`, `chi2_contingency` without a
/// continuity correction, and `t.interval`) and agree to better than 1e-9. The proportion
/// z-tests were checked against the closed-form normal approximation.
final class HypothesisTestTests: XCTestCase {

    private let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                                  255.0, 249.0, 258.0, 244.0, 251.0]
    private let oldRoute = [27.1, 22.4, 25.3, 24.8, 26.0, 23.9, 25.5, 24.1]
    private let newRoute = [29.0, 31.2, 27.8, 30.5, 28.9, 32.1, 29.7]
    private let before = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]
    private let after = [90.0, 75.0, 96.0, 85.0, 89.0, 80.0, 88.0, 94.0]

    /// Asserts agreement to a relative tolerance, which is the right test for tiny p-values.
    private func assertRelative(
        _ value: Double?, _ expected: Double, tolerance: Double = 1e-9,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard let value = value else {
            XCTFail("Expected \(expected) but got nil", file: file, line: line)
            return
        }
        XCTAssertEqual(value / expected, 1.0, accuracy: tolerance, file: file, line: line)
    }

    // MARK: - One-sample t-test

    func testOneSampleTMatchesReference() throws {
        let result = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))
        XCTAssertEqual(result.kind, .oneSampleT)
        XCTAssertEqual(result.statistic, 4.57495710997814, accuracy: 1e-9)
        XCTAssertEqual(result.degreesOfFreedom, 9.0)
        XCTAssertNil(result.denominatorDegreesOfFreedom)
        XCTAssertEqual(result.alternative, .twoSided)
        assertRelative(result.pValue, 0.0013374020355815164)
    }

    func testOneSampleTOneSidedAlternatives() throws {
        let greater = try XCTUnwrap(sessionSeconds.tTest(against: 240.0, alternative: .greater))
        let less = try XCTUnwrap(sessionSeconds.tTest(against: 240.0, alternative: .less))
        assertRelative(greater.pValue, 0.0006687010177907582)
        assertRelative(less.pValue, 0.9993312989822093)
        XCTAssertEqual(greater.alternative, .greater)
        XCTAssertEqual(less.alternative, .less)
        // The two one-sided p-values partition the probability.
        XCTAssertEqual(greater.pValue + less.pValue, 1.0, accuracy: 1e-12)
    }

    func testOneSampleTNegativeStatistic() throws {
        let result = try XCTUnwrap(sessionSeconds.tTest(against: 260.0))
        XCTAssertLessThan(result.statistic, 0.0)
        let mirrored = try XCTUnwrap(sessionSeconds.tTest(against: 260.0, alternative: .less))
        // Below the hypothesized mean, the lower tail is the small one.
        XCTAssertEqual(mirrored.pValue, result.pValue / 2.0, accuracy: 1e-12)
    }

    func testOneSampleTAtTheSampleMeanIsInconclusive() throws {
        let result = try XCTUnwrap(sessionSeconds.tTest(against: 250.0))
        XCTAssertEqual(result.statistic, 0.0, accuracy: 1e-12)
        XCTAssertEqual(result.pValue, 1.0, accuracy: 1e-12)
    }

    func testOneSampleTReturnsNilWhenUntestable() {
        XCTAssertNil([Double]().tTest(against: 0.0))
        XCTAssertNil([5.0].tTest(against: 0.0))
        XCTAssertNil([4.0, 4.0, 4.0].tTest(against: 3.0), "no variation means no standard error")
        XCTAssertNil([1.0, 2.0, .nan].tTest(against: 0.0))
        XCTAssertNil([1.0, 2.0, .infinity].tTest(against: 0.0))
        XCTAssertNil([1.0, 2.0, 3.0].tTest(against: .nan))
    }

    func testOneSampleTFarTailIsTinyButPositive() throws {
        // Reading the lower tail directly keeps a very small p-value from rounding to zero.
        let tight = [10.0, 10.1, 9.9, 10.05, 9.95]
        let extreme = try XCTUnwrap(tight.tTest(against: 12.0))
        XCTAssertLessThan(extreme.pValue, 1e-4)
        XCTAssertGreaterThan(extreme.pValue, 0.0)
    }

    // MARK: - Welch t-test

    func testWelchMatchesReference() throws {
        let result = try XCTUnwrap(oldRoute.welchTTest(against: newRoute))
        XCTAssertEqual(result.kind, .welchT)
        XCTAssertEqual(result.statistic, -6.61199755107702, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(result.degreesOfFreedom), 12.621617316761721, accuracy: 1e-9)
        assertRelative(result.pValue, 1.9475015572747795e-05)
    }

    func testWelchOneSidedAlternatives() throws {
        let less = try XCTUnwrap(oldRoute.welchTTest(against: newRoute, alternative: .less))
        let greater = try XCTUnwrap(oldRoute.welchTTest(against: newRoute, alternative: .greater))
        assertRelative(less.pValue, 9.737507786373898e-06)
        assertRelative(greater.pValue, 0.9999902624922137)
    }

    func testWelchIsAntisymmetricInTheTwoSamples() throws {
        let forward = try XCTUnwrap(oldRoute.welchTTest(against: newRoute))
        let reverse = try XCTUnwrap(newRoute.welchTTest(against: oldRoute))
        XCTAssertEqual(forward.statistic, -reverse.statistic, accuracy: 1e-12)
        XCTAssertEqual(forward.pValue, reverse.pValue, accuracy: 1e-15)
        XCTAssertEqual(try XCTUnwrap(forward.degreesOfFreedom),
                       try XCTUnwrap(reverse.degreesOfFreedom), accuracy: 1e-12)
    }

    func testWelchWithEqualSizesAndVariancesHasWholeDegreesOfFreedom() throws {
        let first = [1.0, 2.0, 3.0, 4.0]
        let second = [2.0, 3.0, 4.0, 5.0]
        let result = try XCTUnwrap(first.welchTTest(against: second))
        // Equal n and equal variance: Welch degrees of freedom reduce to n1 + n2 - 2.
        XCTAssertEqual(try XCTUnwrap(result.degreesOfFreedom), 6.0, accuracy: 1e-12)
    }

    func testWelchReturnsNilWhenUntestable() {
        XCTAssertNil([1.0].welchTTest(against: [1.0, 2.0, 3.0]))
        XCTAssertNil([1.0, 2.0, 3.0].welchTTest(against: [5.0]))
        XCTAssertNil([Double]().welchTTest(against: [1.0, 2.0]))
        XCTAssertNil([2.0, 2.0].welchTTest(against: [3.0, 3.0]), "neither sample varies")
        XCTAssertNil([1.0, 2.0, .nan].welchTTest(against: [1.0, 2.0, 3.0]))
    }

    func testWelchWorksWhenOnlyOneSampleVaries() throws {
        let result = try XCTUnwrap([5.0, 5.0, 5.0].welchTTest(against: [1.0, 2.0, 3.0, 4.0]))
        XCTAssertGreaterThan(result.statistic, 0.0)
        XCTAssertTrue(result.pValue > 0.0 && result.pValue < 1.0)
    }

    // MARK: - Paired t-test

    func testPairedMatchesReference() throws {
        let result = try XCTUnwrap(after.pairedTTest(against: before))
        XCTAssertEqual(result.kind, .pairedT)
        XCTAssertEqual(result.statistic, 3.9886201760873283, accuracy: 1e-9)
        XCTAssertEqual(result.degreesOfFreedom, 7.0)
        assertRelative(result.pValue, 0.005265691029161741)
    }

    func testPairedOneSidedAlternative() throws {
        let greater = try XCTUnwrap(after.pairedTTest(against: before, alternative: .greater))
        assertRelative(greater.pValue, 0.0026328455145808705)
    }

    func testPairedEqualsOneSampleTestOnTheDifferences() throws {
        var differences = [Double]()
        for index in 0..<after.count {
            differences.append(after[index] - before[index])
        }
        let paired = try XCTUnwrap(after.pairedTTest(against: before))
        let oneSample = try XCTUnwrap(differences.tTest(against: 0.0))
        XCTAssertEqual(paired.statistic, oneSample.statistic, accuracy: 1e-12)
        XCTAssertEqual(paired.pValue, oneSample.pValue, accuracy: 1e-15)
    }

    func testPairedReturnsNilWhenUntestable() {
        XCTAssertNil([1.0, 2.0, 3.0].pairedTTest(against: [1.0, 2.0]), "unequal lengths")
        XCTAssertNil([1.0].pairedTTest(against: [2.0]), "one pair")
        XCTAssertNil([1.0, 2.0, 3.0].pairedTTest(against: [2.0, 3.0, 4.0]), "constant difference")
    }

    // MARK: - Confidence interval

    func testConfidenceIntervalMatchesReference() throws {
        let ninetyFive = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.95))
        XCTAssertEqual(ninetyFive.lower, 245.05534782421378, accuracy: 1e-6)
        XCTAssertEqual(ninetyFive.upper, 254.94465217578622, accuracy: 1e-6)

        let ninety = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.90))
        XCTAssertEqual(ninety.lower, 245.99315821200125, accuracy: 1e-6)
        XCTAssertEqual(ninety.upper, 254.00684178799875, accuracy: 1e-6)

        let ninetyNine = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.99))
        XCTAssertEqual(ninetyNine.lower, 242.8964677406393, accuracy: 1e-6)
        XCTAssertEqual(ninetyNine.upper, 257.1035322593607, accuracy: 1e-6)
    }

    func testConfidenceIntervalDefaultsToNinetyFivePercentAndIsCentered() throws {
        let interval = try XCTUnwrap(sessionSeconds.confidenceInterval())
        let explicit = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.95))
        XCTAssertEqual(interval, explicit)
        XCTAssertEqual((interval.lower + interval.upper) / 2.0, 250.0, accuracy: 1e-9)
    }

    func testHigherCoverageGivesAWiderInterval() throws {
        let narrow = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.80))
        let wide = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.99))
        XCTAssertGreaterThan(wide.upper - wide.lower, narrow.upper - narrow.lower)
    }

    func testIntervalAgreesWithTheTwoSidedTest() throws {
        // A hypothesized mean outside the 95% interval is rejected at 0.05, and one inside is not.
        let interval = try XCTUnwrap(sessionSeconds.confidenceInterval(level: 0.95))
        let outside = try XCTUnwrap(sessionSeconds.tTest(against: interval.lower - 0.5))
        let inside = try XCTUnwrap(sessionSeconds.tTest(against: interval.lower + 0.5))
        XCTAssertTrue(outside.rejectsNullHypothesis(at: 0.05))
        XCTAssertFalse(inside.rejectsNullHypothesis(at: 0.05))
    }

    func testConfidenceIntervalReturnsNilWhenUndefined() {
        XCTAssertNil([Double]().confidenceInterval())
        XCTAssertNil([5.0].confidenceInterval())
        XCTAssertNil([3.0, 3.0, 3.0].confidenceInterval())
        XCTAssertNil(sessionSeconds.confidenceInterval(level: 0.0))
        XCTAssertNil(sessionSeconds.confidenceInterval(level: 1.0))
        XCTAssertNil(sessionSeconds.confidenceInterval(level: 1.5))
        XCTAssertNil([1.0, 2.0, .nan].confidenceInterval())
    }

    // MARK: - Proportion tests

    func testOneProportionMatchesReference() throws {
        let clicks = [Bool](repeating: true, count: 14) + [Bool](repeating: false, count: 6)
        let result = try XCTUnwrap(clicks.proportionTest(against: 0.5))
        XCTAssertEqual(result.kind, .oneProportion)
        XCTAssertEqual(result.statistic, 1.7888543819998315, accuracy: 1e-12)
        XCTAssertNil(result.degreesOfFreedom)
        assertRelative(result.pValue, 0.0736382701203027)

        let greater = try XCTUnwrap(clicks.proportionTest(against: 0.5, alternative: .greater))
        let less = try XCTUnwrap(clicks.proportionTest(against: 0.5, alternative: .less))
        assertRelative(greater.pValue, 0.03681913506015135)
        assertRelative(less.pValue, 0.9631808649398487)
    }

    func testTwoProportionMatchesReference() throws {
        let variantA = [Bool](repeating: true, count: 30) + [Bool](repeating: false, count: 70)
        let variantB = [Bool](repeating: true, count: 45) + [Bool](repeating: false, count: 55)
        let result = try XCTUnwrap(variantA.proportionTest(against: variantB))
        XCTAssertEqual(result.kind, .twoProportion)
        XCTAssertEqual(result.statistic, -2.1908902300206647, accuracy: 1e-12)
        assertRelative(result.pValue, 0.028459736916310555)

        let less = try XCTUnwrap(variantA.proportionTest(against: variantB, alternative: .less))
        assertRelative(less.pValue, 0.014229868458155277)
    }

    func testTwoProportionIsAntisymmetric() throws {
        let first = [Bool](repeating: true, count: 12) + [Bool](repeating: false, count: 18)
        let second = [Bool](repeating: true, count: 20) + [Bool](repeating: false, count: 10)
        let forward = try XCTUnwrap(first.proportionTest(against: second))
        let reverse = try XCTUnwrap(second.proportionTest(against: first))
        XCTAssertEqual(forward.statistic, -reverse.statistic, accuracy: 1e-12)
        XCTAssertEqual(forward.pValue, reverse.pValue, accuracy: 1e-15)
    }

    func testProportionTestReturnsNilWhenUntestable() {
        let some = [true, false, true, true]
        XCTAssertNil([Bool]().proportionTest(against: 0.5))
        XCTAssertNil(some.proportionTest(against: 0.0))
        XCTAssertNil(some.proportionTest(against: 1.0))
        XCTAssertNil(some.proportionTest(against: -0.2))
        XCTAssertNil(some.proportionTest(against: 1.5))
        XCTAssertNil([Bool]().proportionTest(against: some))
        XCTAssertNil(some.proportionTest(against: [Bool]()))
        XCTAssertNil([true, true].proportionTest(against: [true, true, true]), "pooled proportion of 1")
        XCTAssertNil([false, false].proportionTest(against: [false]), "pooled proportion of 0")
    }

    // MARK: - Chi-squared goodness of fit

    func testGoodnessOfFitMatchesReference() throws {
        let result = try XCTUnwrap([18, 22, 16, 25, 19, 20].chiSquaredTest())
        XCTAssertEqual(result.kind, .chiSquaredGoodnessOfFit)
        XCTAssertEqual(result.statistic, 2.5, accuracy: 1e-12)
        XCTAssertEqual(result.degreesOfFreedom, 5.0)
        XCTAssertNil(result.alternative)
        assertRelative(result.pValue, 0.7764950711233227)
    }

    func testGoodnessOfFitWithSuppliedProportions() throws {
        let result = try XCTUnwrap([30, 10, 20, 40].chiSquaredTest(expectedProportions: [0.25, 0.15, 0.25, 0.35]))
        XCTAssertEqual(result.statistic, 4.380952380952381, accuracy: 1e-12)
        assertRelative(result.pValue, 0.22315805595650048)
    }

    func testGoodnessOfFitPerfectFitHasZeroStatistic() throws {
        let result = try XCTUnwrap([20, 20, 20, 20].chiSquaredTest())
        XCTAssertEqual(result.statistic, 0.0, accuracy: 1e-12)
        XCTAssertEqual(result.pValue, 1.0, accuracy: 1e-12)
    }

    func testGoodnessOfFitFarTailKeepsRelativePrecision() throws {
        // Two categories, 120 versus 0: chi-squared is 120 with df = 1, so p is far below 1e-20.
        let result = try XCTUnwrap([120, 0].chiSquaredTest())
        XCTAssertEqual(result.statistic, 120.0, accuracy: 1e-9)
        XCTAssertLessThan(result.pValue, 1e-20)
        XCTAssertGreaterThan(result.pValue, 0.0)
    }

    func testGoodnessOfFitReturnsNilWhenUntestable() {
        XCTAssertNil([Int]().chiSquaredTest())
        XCTAssertNil([5].chiSquaredTest(), "a single category")
        XCTAssertNil([0, 0, 0].chiSquaredTest(), "no observations")
        XCTAssertNil([5, -1, 3].chiSquaredTest())
        XCTAssertNil([5, 5, 5].chiSquaredTest(expectedProportions: [0.5, 0.5]), "wrong count")
        XCTAssertNil([5, 5].chiSquaredTest(expectedProportions: [0.5, 0.6]), "does not sum to 1")
        XCTAssertNil([5, 5].chiSquaredTest(expectedProportions: [1.0, 0.0]), "a zero proportion")
        XCTAssertNil([5, 5].chiSquaredTest(expectedProportions: [1.5, -0.5]))
    }

    // MARK: - Chi-squared independence

    func testIndependenceMatchesReference() throws {
        let table = try XCTUnwrap(ContingencyTable(
            rowLabels: ["Treatment", "Control"],
            columnLabels: ["Improved", "No change"],
            counts: [[30, 20], [18, 32]]
        ))
        let result = try XCTUnwrap(table.chiSquaredTest())
        XCTAssertEqual(result.kind, .chiSquaredIndependence)
        XCTAssertEqual(result.statistic, 5.76923076923077, accuracy: 1e-12)
        XCTAssertEqual(result.degreesOfFreedom, 1.0)
        assertRelative(result.pValue, 0.01630917187775496)
    }

    func testIndependenceOnALargerTable() throws {
        let table = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A", "B"],
            columnLabels: ["Low", "Medium", "High"],
            counts: [[20, 15, 25], [30, 35, 15]]
        ))
        let result = try XCTUnwrap(table.chiSquaredTest())
        XCTAssertEqual(result.statistic, 9.84375, accuracy: 1e-12)
        XCTAssertEqual(result.degreesOfFreedom, 2.0)
        assertRelative(result.pValue, 0.007285457798938987)
    }

    func testIndependentTableHasZeroStatistic() throws {
        let table = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A", "B"], columnLabels: ["X", "Y"], counts: [[10, 20], [20, 40]]
        ))
        let result = try XCTUnwrap(table.chiSquaredTest())
        XCTAssertEqual(result.statistic, 0.0, accuracy: 1e-12)
        XCTAssertEqual(result.pValue, 1.0, accuracy: 1e-12)
    }

    func testIndependenceReturnsNilWhenUntestable() throws {
        let oneRow = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A"], columnLabels: ["X", "Y"], counts: [[5, 5]]
        ))
        XCTAssertNil(oneRow.chiSquaredTest())

        let oneColumn = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A", "B"], columnLabels: ["X"], counts: [[5], [5]]
        ))
        XCTAssertNil(oneColumn.chiSquaredTest())

        let emptyRow = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A", "B"], columnLabels: ["X", "Y"], counts: [[5, 5], [0, 0]]
        ))
        XCTAssertNil(emptyRow.chiSquaredTest(), "a zero margin gives an expected count of zero")

        let emptyColumn = try XCTUnwrap(ContingencyTable(
            rowLabels: ["A", "B"], columnLabels: ["X", "Y"], counts: [[5, 0], [5, 0]]
        ))
        XCTAssertNil(emptyColumn.chiSquaredTest())
    }

    // MARK: - One-way ANOVA

    func testANOVAMatchesReference() throws {
        let groups = [[23.0, 25.0, 21.0, 24.0, 26.0],
                      [30.0, 28.0, 31.0, 29.0, 27.0],
                      [22.0, 20.0, 24.0, 23.0, 21.0]]
        let result = try XCTUnwrap(groups.oneWayANOVA())
        XCTAssertEqual(result.kind, .oneWayANOVA)
        XCTAssertEqual(result.statistic, 22.781609195402293, accuracy: 1e-9)
        XCTAssertEqual(result.degreesOfFreedom, 2.0)
        XCTAssertEqual(result.denominatorDegreesOfFreedom, 12.0)
        XCTAssertNil(result.alternative)
        assertRelative(result.pValue, 8.207616781675594e-05)
    }

    func testANOVAWithUnequalGroupSizes() throws {
        let result = try XCTUnwrap([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5, 7.5]].oneWayANOVA())
        XCTAssertEqual(result.statistic, 1.2244897959183674, accuracy: 1e-12)
        XCTAssertEqual(result.degreesOfFreedom, 1.0)
        XCTAssertEqual(result.denominatorDegreesOfFreedom, 5.0)
        assertRelative(result.pValue, 0.31885766977837715)
    }

    func testANOVAWithTwoGroupsMatchesPooledTSquared() throws {
        // With two groups, F equals the square of the pooled-variance t statistic.
        let first = [4.0, 5.0, 6.0]
        let second = [4.5, 5.5, 6.5, 7.5]
        let anova = try XCTUnwrap([first, second].oneWayANOVA())

        let meanFirst = try XCTUnwrap(first.mean())
        let meanSecond = try XCTUnwrap(second.mean())
        let varianceFirst = try XCTUnwrap(first.variance())
        let varianceSecond = try XCTUnwrap(second.variance())
        let pooled = (2.0 * varianceFirst + 3.0 * varianceSecond) / 5.0
        let standardError = (pooled * (1.0 / 3.0 + 1.0 / 4.0)).squareRoot()
        let t = (meanFirst - meanSecond) / standardError
        XCTAssertEqual(anova.statistic, t * t, accuracy: 1e-12)
    }

    func testANOVAIdenticalMeansHasZeroStatistic() throws {
        let result = try XCTUnwrap([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]].oneWayANOVA())
        XCTAssertEqual(result.statistic, 0.0, accuracy: 1e-12)
        XCTAssertEqual(result.pValue, 1.0, accuracy: 1e-12)
    }

    func testANOVAFarTailKeepsRelativePrecision() throws {
        // Reference: F.sf(500, 3, 20) from SciPy is 5.720958909263184e-19, computed directly
        // rather than as 1 - cdf, which would round to zero.
        let tail = try XCTUnwrap(Distributions._fUpperTail(x: 500.0, df1: 3.0, df2: 20.0))
        XCTAssertEqual(tail / 5.720958909263184e-19, 1.0, accuracy: 1e-9)
    }

    func testANOVAReturnsNilWhenUntestable() {
        XCTAssertNil([[1.0, 2.0, 3.0]].oneWayANOVA(), "a single group")
        XCTAssertNil([[Double]]().oneWayANOVA())
        XCTAssertNil([[1.0, 2.0], []].oneWayANOVA(), "an empty group")
        XCTAssertNil([[1.0], [2.0]].oneWayANOVA(), "no more observations than groups")
        XCTAssertNil([[2.0, 2.0], [3.0, 3.0]].oneWayANOVA(), "no variation within groups")
        XCTAssertNil([[1.0, 2.0, .nan], [1.0, 2.0, 3.0]].oneWayANOVA())
    }

    // MARK: - Chi-squared and F tails

    func testChiSquaredUpperTailMatchesReference() throws {
        let moderate = try XCTUnwrap(_Inference.chiSquaredUpperTail(statistic: 50.0, degrees: 5.0))
        XCTAssertEqual(moderate / 1.3857973367009573e-09, 1.0, accuracy: 1e-9)
        // Closed form at df = 2: P(X >= x) = exp(-x / 2).
        let closedForm = try XCTUnwrap(_Inference.chiSquaredUpperTail(statistic: 120.0, degrees: 2.0))
        XCTAssertEqual(closedForm / Foundation.exp(-60.0), 1.0, accuracy: 1e-12)
        XCTAssertEqual(closedForm / 8.756510762696494e-27, 1.0, accuracy: 1e-9)
    }

    func testUpperTailsAreComplementsOfTheCDFs() throws {
        let chiTail = try XCTUnwrap(Distributions._regularizedIncompleteGammaQ(a: 2.5, x: 3.0))
        let chiCDF = try XCTUnwrap(Distributions._regularizedIncompleteGammaP(a: 2.5, x: 3.0))
        XCTAssertEqual(chiTail + chiCDF, 1.0, accuracy: 1e-13)

        let fTail = try XCTUnwrap(Distributions._fUpperTail(x: 2.5, df1: 4.0, df2: 12.0))
        let fCDF = try XCTUnwrap(Distributions.f.cdf(x: 2.5, df1: 4.0, df2: 12.0))
        XCTAssertEqual(fTail + fCDF, 1.0, accuracy: 1e-13)
    }

    // MARK: - HypothesisTest value type

    func testRejectsNullHypothesis() throws {
        let result = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))
        XCTAssertTrue(result.rejectsNullHypothesis())
        XCTAssertTrue(result.rejectsNullHypothesis(at: 0.01))
        XCTAssertFalse(result.rejectsNullHypothesis(at: 0.001))
    }

    func testRejectsNullHypothesisUsesStrictInequality() {
        let result = HypothesisTest(
            kind: .oneSampleT, statistic: 2.0, degreesOfFreedom: 9, pValue: 0.05, alternative: .twoSided
        )
        XCTAssertFalse(result.rejectsNullHypothesis(at: 0.05))
    }

    func testEquatable() throws {
        let first = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))
        let same = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))
        let different = try XCTUnwrap(sessionSeconds.tTest(against: 245.0))
        XCTAssertEqual(first, same)
        XCTAssertNotEqual(first, different)
    }

    func testCodableRoundTrip() throws {
        let results: [HypothesisTest] = [
            try XCTUnwrap(sessionSeconds.tTest(against: 240.0)),
            try XCTUnwrap(oldRoute.welchTTest(against: newRoute)),
            try XCTUnwrap([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5, 7.5]].oneWayANOVA()),
            try XCTUnwrap([18, 22, 16, 25, 19, 20].chiSquaredTest()),
            try XCTUnwrap(([true, false, true, true] + [Bool](repeating: false, count: 6)).proportionTest(against: 0.5))
        ]
        for original in results {
            let data = try JSONEncoder().encode(original)
            let decoded = try JSONDecoder().decode(HypothesisTest.self, from: data)
            XCTAssertEqual(decoded, original)
        }
    }

    func testSendableAcrossTaskBoundary() async throws {
        let result = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))

        // Sending a result into a Task requires Sendable conformance.
        let kind = await Task { result.kind }.value
        XCTAssertEqual(kind, .oneSampleT)
    }

    // MARK: - CustomStringConvertible

    func testDescriptionForEachKind() throws {
        let oneSample = try XCTUnwrap(sessionSeconds.tTest(against: 240.0))
        XCTAssertEqual(oneSample.description,
            "HypothesisTest: one-sample t-test, t = 4.575, df = 9, p = 0.0013")

        let welch = try XCTUnwrap(oldRoute.welchTTest(against: newRoute))
        XCTAssertEqual(welch.description,
            "HypothesisTest: Welch t-test, t = -6.612, df = 12.62, p < 0.0001")

        let paired = try XCTUnwrap(after.pairedTTest(against: before, alternative: .greater))
        XCTAssertEqual(paired.description,
            "HypothesisTest: paired t-test, t = 3.989, df = 7, p = 0.0026")

        let clicks = [Bool](repeating: true, count: 14) + [Bool](repeating: false, count: 6)
        let oneProportion = try XCTUnwrap(clicks.proportionTest(against: 0.5))
        XCTAssertEqual(oneProportion.description,
            "HypothesisTest: one-proportion z-test, z = 1.789, p = 0.0736")

        let variantA = [Bool](repeating: true, count: 30) + [Bool](repeating: false, count: 70)
        let variantB = [Bool](repeating: true, count: 45) + [Bool](repeating: false, count: 55)
        let twoProportion = try XCTUnwrap(variantA.proportionTest(against: variantB))
        XCTAssertEqual(twoProportion.description,
            "HypothesisTest: two-proportion z-test, z = -2.191, p = 0.0285")

        let goodness = try XCTUnwrap([18, 22, 16, 25, 19, 20].chiSquaredTest())
        XCTAssertEqual(goodness.description,
            "HypothesisTest: chi-squared goodness-of-fit test, χ² = 2.500, df = 5, p = 0.7765")

        let table = try XCTUnwrap(ContingencyTable(
            rowLabels: ["Treatment", "Control"], columnLabels: ["Improved", "No change"],
            counts: [[30, 20], [18, 32]]
        ))
        let independence = try XCTUnwrap(table.chiSquaredTest())
        XCTAssertEqual(independence.description,
            "HypothesisTest: chi-squared test of independence, χ² = 5.769, df = 1, p = 0.0163")

        let groups = [[23.0, 25.0, 21.0, 24.0, 26.0],
                      [30.0, 28.0, 31.0, 29.0, 27.0],
                      [22.0, 20.0, 24.0, 23.0, 21.0]]
        let anova = try XCTUnwrap(groups.oneWayANOVA())
        XCTAssertEqual(anova.description,
            "HypothesisTest: one-way ANOVA, F = 22.782, df = (2, 12), p < 0.0001")
    }
}
