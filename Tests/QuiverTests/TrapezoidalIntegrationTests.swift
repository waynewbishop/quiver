import XCTest
@testable import Quiver

final class TrapezoidalIntegrationTests: XCTestCase {

    // MARK: - trapezoidalIntegral

    func testTotalAreaUnderConstantRate() throws {
        // 10 samples at 250 W, dt = 1s → 250 × 9 = 2250 J (nine trapezoids of width 1).
        let power = [Double](repeating: 250.0, count: 10)
        let total = try XCTUnwrap(power.trapezoidalIntegral(dt: 1.0))
        XCTAssertEqual(total, 2250.0, accuracy: 1e-12)
    }

    func testTotalAreaOnLinearRamp() throws {
        // Samples 0, 1, 2, ..., 9 with dt = 1.
        // Exact trapezoidal sum: 0.5·(0+1) + 0.5·(1+2) + ... + 0.5·(8+9) = 40.5
        let ramp = (0...9).map { Double($0) }
        let total = try XCTUnwrap(ramp.trapezoidalIntegral(dt: 1.0))
        XCTAssertEqual(total, 40.5, accuracy: 1e-12)
    }

    func testIntegralOfSineMatchesAnalyticAnswer() throws {
        // ∫₀^π sin(x) dx = 2.0 exactly.
        // 1,001 samples over [0, π] with dt = π/1000.
        let n = 1001
        let dt = Double.pi / Double(n - 1)
        let samples = (0..<n).map { Foundation.sin(Double($0) * dt) }
        let total = try XCTUnwrap(samples.trapezoidalIntegral(dt: dt))
        // Trapezoidal rule on 1,001 samples is accurate to ~1e-6.
        XCTAssertEqual(total, 2.0, accuracy: 1e-5)
    }

    func testReturnsNilForTooFewSamples() {
        XCTAssertNil([Double]().trapezoidalIntegral(dt: 1.0))
        XCTAssertNil([5.0].trapezoidalIntegral(dt: 1.0))
    }

    // MARK: - cumulativeTrapezoidal

    func testCumulativeStartsAtZero() {
        let power = [Double](repeating: 100.0, count: 5)
        let curve = power.cumulativeTrapezoidal(dt: 1.0)
        XCTAssertEqual(curve[0], 0.0, accuracy: 1e-12)
    }

    func testCumulativeMatchesRunningSum() {
        // 4 samples, all 200, dt = 1 → cumulative [0, 200, 400, 600].
        let power = [200.0, 200.0, 200.0, 200.0]
        let curve = power.cumulativeTrapezoidal(dt: 1.0)
        XCTAssertEqual(curve, [0.0, 200.0, 400.0, 600.0])
    }

    func testCumulativeEndsAtTrapezoidalTotal() throws {
        // Last element of cumulative must equal trapezoidalIntegral.
        let ramp = (0...9).map { Double($0) }
        let curve = ramp.cumulativeTrapezoidal(dt: 1.0)
        let last = try XCTUnwrap(curve.last)
        XCTAssertEqual(last, 40.5, accuracy: 1e-12)
    }

    func testCumulativeEqualsRecomputedTrapezoidalAtEveryStep() throws {
        // The cumulative curve at index i should equal the trapezoidal integral
        // of the prefix self[0...i].
        let ramp = (1...20).map { Double($0) }
        let curve = ramp.cumulativeTrapezoidal(dt: 0.5)
        for i in 1..<ramp.count {
            let prefix = Array(ramp[0...i])
            let prefixTotal = try XCTUnwrap(prefix.trapezoidalIntegral(dt: 0.5))
            XCTAssertEqual(curve[i], prefixTotal, accuracy: 1e-12,
                "cumulative[\(i)] should match trapezoidalIntegral of prefix")
        }
    }

    func testCumulativeEmptyReturnsEmpty() {
        XCTAssertEqual([Double]().cumulativeTrapezoidal(dt: 1.0), [])
    }

    func testCumulativeSingleElementReturnsSingleZero() {
        XCTAssertEqual([5.0].cumulativeTrapezoidal(dt: 1.0), [0.0])
    }
}
