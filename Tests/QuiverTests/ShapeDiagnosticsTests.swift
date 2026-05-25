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
}
