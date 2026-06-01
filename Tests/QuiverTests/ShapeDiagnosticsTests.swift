import XCTest
@testable import Quiver

/// SciPy parity tests for `skewness()` and `kurtosis()`.
///
/// Reference values produced by `scipy.stats.skew` and `scipy.stats.kurtosis` (SciPy 1.x)
/// with `fisher=True` and `bias=True` defaults. Pinned to 1e-10 for the deterministic cases.
final class ShapeDiagnosticsTests: XCTestCase {

    // MARK: - skewness()

    func testSkewnessOnPerfectlySymmetricSmallSample() {
        // scipy.stats.skew([4, 7, 2, 9, 3, 5, 8, 6], bias=True) == 0.0
        let x = [4.0, 7.0, 2.0, 9.0, 3.0, 5.0, 8.0, 6.0]
        let result = x.skewness()!
        XCTAssertEqual(result, 0.0, accuracy: 1e-12)
    }

    func testSkewnessOnSymmetricArrayIsZero() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let result = x.skewness()!
        XCTAssertEqual(result, 0.0, accuracy: 1e-12)
    }

    func testSkewnessOnHeavyRightSkew() {
        // scipy.stats.skew([1, 1, 1, 1, 2, 10], bias=True) == 1.7419977554012613
        let x = [1.0, 1.0, 1.0, 1.0, 2.0, 10.0]
        let result = x.skewness()!
        XCTAssertEqual(result, 1.7419977554012613, accuracy: 1e-10)
    }

    func testSkewnessOnHeavyLeftSkew() {
        // mirror of the right-skew case
        let x = [10.0, 10.0, 10.0, 10.0, 9.0, 1.0]
        let result = x.skewness()!
        XCTAssertEqual(result, -1.7419977554012613, accuracy: 1e-10)
    }

    func testSkewnessUnbiasedAdjustment() {
        // For the symmetric array, both biased and unbiased should be 0
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let unbiased = x.skewness(bias: false)!
        XCTAssertEqual(unbiased, 0.0, accuracy: 1e-12)
    }

    func testSkewnessReturnsNilForTooFewElements() {
        XCTAssertNil([Double]().skewness())
        XCTAssertNil([1.0].skewness())
        XCTAssertNil([1.0, 2.0].skewness())
    }

    func testSkewnessReturnsNilForZeroVariance() {
        XCTAssertNil([5.0, 5.0, 5.0, 5.0].skewness())
    }

    func testSkewnessReturnsNilForNonFiniteInput() {
        XCTAssertNil([1.0, 2.0, .nan, 4.0].skewness())
        XCTAssertNil([1.0, 2.0, .infinity, 4.0].skewness())
    }

    // MARK: - kurtosis()

    func testKurtosisOnPlatykurticSmallSample() {
        // scipy.stats.kurtosis([4, 7, 2, 9, 3, 5, 8, 6], fisher=True, bias=True) == -1.2380952380952381
        let x = [4.0, 7.0, 2.0, 9.0, 3.0, 5.0, 8.0, 6.0]
        let result = x.kurtosis()!
        XCTAssertEqual(result, -1.2380952380952381, accuracy: 1e-10)
    }

    func testKurtosisOnUniformLikeArray() {
        // scipy.stats.kurtosis(list(range(1, 11)), fisher=True, bias=True) == -1.2242424242424244
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let result = x.kurtosis()!
        XCTAssertEqual(result, -1.2242424242424244, accuracy: 1e-10)
    }

    func testKurtosisOnHeavyTailed() {
        // scipy.stats.kurtosis([1, 1, 1, 1, 2, 10], fisher=True, bias=True) == 1.1089129529362767
        let x = [1.0, 1.0, 1.0, 1.0, 2.0, 10.0]
        let result = x.kurtosis()!
        XCTAssertEqual(result, 1.1089129529362767, accuracy: 1e-10)
    }

    func testKurtosisReturnsNilForTooFewElements() {
        XCTAssertNil([Double]().kurtosis())
        XCTAssertNil([1.0].kurtosis())
        XCTAssertNil([1.0, 2.0].kurtosis())
        XCTAssertNil([1.0, 2.0, 3.0].kurtosis())
    }

    func testKurtosisReturnsNilForZeroVariance() {
        XCTAssertNil([5.0, 5.0, 5.0, 5.0].kurtosis())
    }

    func testKurtosisReturnsNilForNonFiniteInput() {
        XCTAssertNil([1.0, 2.0, .nan, 4.0, 5.0].kurtosis())
        XCTAssertNil([1.0, 2.0, .infinity, 4.0, 5.0].kurtosis())
    }

    // MARK: - Sign and direction sanity

    func testSkewnessSignMatchesTailDirection() {
        let rightTailed = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        let leftTailed = [5.0, 5.0, 5.0, 5.0, 5.0, 1.0]
        XCTAssertGreaterThan(rightTailed.skewness()!, 0)
        XCTAssertLessThan(leftTailed.skewness()!, 0)
    }

    // MARK: - bowleySkewness()

    func testBowleySkewnessOnSalariesMatchesReference() {
        // (Q3 - 2*Q2 + Q1)/(Q3 - Q1) with linear (R-7) quartiles
        // Q1=55.75, Q2=59.0, Q3=61.5 -> (61.5 - 118.0 + 55.75)/5.75 = -0.13043478...
        let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]
        XCTAssertEqual(salaries.bowleySkewness()!, -0.1304347826, accuracy: 1e-10)
    }

    func testBowleySkewnessSignMatchesQuartileDirection() {
        // Upper quartile stretched away from the median -> positive (right).
        let rightLeaning = [1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 40.0, 60.0]
        XCTAssertEqual(rightLeaning.bowleySkewness()!, 0.8426966292, accuracy: 1e-10)
        // Reflection (negation) flips the sign.
        let leftLeaning = rightLeaning.map { -$0 }
        XCTAssertEqual(leftLeaning.bowleySkewness()!, -0.8426966292, accuracy: 1e-10)
    }

    func testBowleySkewnessIsBoundedByOne() {
        // The coefficient is mathematically confined to [-1, 1] for any input.
        var generator = SeededRandomNumberGenerator(seed: 42)
        for _ in 0..<1000 {
            let n = Int.random(in: 4...50, using: &generator)
            let sample = (0..<n).map { _ in Double.random(in: -100...100, using: &generator) }
            guard let bowley = sample.bowleySkewness() else { continue }
            XCTAssertLessThanOrEqual(abs(bowley), 1.0)
        }
    }

    func testBowleySkewnessReturnsNilForEmpty() {
        XCTAssertNil([Double]().bowleySkewness())
    }

    func testBowleySkewnessReturnsNilForZeroIQR() {
        // A constant array, and an array with a constant middle 50%, both have IQR 0.
        XCTAssertNil([5.0, 5.0, 5.0, 5.0].bowleySkewness())
        XCTAssertNil([1.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 12.0].bowleySkewness())
    }

    func testBowleySkewnessReturnsNilForNonFiniteInput() {
        XCTAssertNil([1.0, 2.0, .nan, 4.0].bowleySkewness())
        XCTAssertNil([1.0, 2.0, .infinity, 4.0].bowleySkewness())
    }

    // MARK: - skewnessReport()

    func testSkewnessReportOnSalariesFlagsDirectionDisagreement() {
        // The canonical outlier case: moment says strongly right (+2.41), the robust
        // measure says slightly left (-0.13). Opposite directions -> .direction.
        let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]
        let report = salaries.skewnessReport()!
        XCTAssertEqual(report.count, 6)
        XCTAssertEqual(report.moment, 2.4109114412, accuracy: 1e-10)
        XCTAssertEqual(report.bowley, -0.1304347826, accuracy: 1e-10)
        XCTAssertEqual(report.agreement, .direction)
    }

    func testSkewnessReportOnSymmetricDataAgrees() {
        // Both measures fall inside their symmetric bands (|moment| < 0.5, |bowley| < 0.1).
        let sensorNoise = [-0.3, -0.1, 0.0, 0.05, 0.1, -0.05, 0.2, -0.2]
        let report = sensorNoise.skewnessReport()!
        XCTAssertEqual(report.moment, -0.2645200285, accuracy: 1e-10)
        XCTAssertEqual(report.bowley, -0.0666666667, accuracy: 1e-10)
        XCTAssertEqual(report.agreement, .agree)
    }

    func testSkewnessReportOnCleanSkewAgrees() {
        // Both measures decisively right, same direction -> .agree.
        let cleanRight = [1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 40.0, 60.0]
        let report = cleanRight.skewnessReport()!
        XCTAssertGreaterThan(report.moment, 0.5)
        XCTAssertGreaterThan(report.bowley, 0.1)
        XCTAssertEqual(report.agreement, .agree)
    }

    func testSkewnessReportReturnsNilWhenMomentUndefined() {
        // Fewer than three values: the moment coefficient is undefined.
        XCTAssertNil([1.0, 2.0].skewnessReport())
        // Zero spread: skewness(bias:) returns nil.
        XCTAssertNil([5.0, 5.0, 5.0, 5.0].skewnessReport())
    }

    func testSkewnessReportReturnsNilForNonFiniteInput() {
        XCTAssertNil([1.0, 2.0, .nan, 4.0, 5.0].skewnessReport())
    }

    // MARK: - SkewnessReport conformances

    func testSkewnessReportDescriptionContainsShapeAndCoefficients() {
        let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]
        let description = salaries.skewnessReport()!.description
        XCTAssertTrue(description.contains("long tail toward high values"))
        XCTAssertTrue(description.contains("right-skewed"))
        XCTAssertTrue(description.contains("2.4109"))
        XCTAssertTrue(description.contains("-0.1304"))
        // The direction-disagreement note prompts an outlier check, never a single verdict.
        XCTAssertTrue(description.contains("check your outliers"))
    }

    func testSkewnessReportDescriptionReportsSymmetric() {
        let sensorNoise = [-0.3, -0.1, 0.0, 0.05, 0.1, -0.05, 0.2, -0.2]
        let description = sensorNoise.skewnessReport()!.description
        XCTAssertTrue(description.contains("roughly symmetric"))
        XCTAssertTrue(description.contains("a robust measure agrees"))
    }

    func testSkewnessReportEquatable() {
        let a = SkewnessReport(count: 6, moment: 2.41, bowley: -0.13, agreement: .direction)
        let b = SkewnessReport(count: 6, moment: 2.41, bowley: -0.13, agreement: .direction)
        let c = SkewnessReport(count: 6, moment: 2.41, bowley: -0.13, agreement: .agree)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testSkewnessReportCodableRoundTrip() throws {
        let original = SkewnessReport(count: 6, moment: 2.41, bowley: -0.13, agreement: .direction)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SkewnessReport.self, from: data)
        XCTAssertEqual(original, decoded)
    }

    func testSkewnessAgreementClassification() {
        // Same nonzero direction -> agree.
        XCTAssertEqual(SkewnessReport.classifyAgreement(moment: 1.2, bowley: 0.4), .agree)
        // Both inside their symmetric bands -> agree.
        XCTAssertEqual(SkewnessReport.classifyAgreement(moment: 0.2, bowley: 0.05), .agree)
        // Opposite nonzero directions -> direction.
        XCTAssertEqual(SkewnessReport.classifyAgreement(moment: 2.41, bowley: -0.13), .direction)
        // One symmetric, the other decisively skewed -> mixed.
        XCTAssertEqual(SkewnessReport.classifyAgreement(moment: 1.5, bowley: 0.02), .mixed)
    }
}
