import XCTest
@testable import Quiver

final class JacobianTests: XCTestCase {

    // MARK: - Correctness vs. analytic Jacobians

    // f(l, w) = [area, perimeter] = [l·w, 2(l+w)].
    // Analytic J at (3, 2) = [[w, l], [2, 2]] = [[2, 3], [2, 2]].
    func testAreaPerimeterJacobianMatchesAnalytic() {
        func plotMetrics(_ v: [Double]) -> [Double] {
            let length = v[0]
            let width = v[1]
            let area = length * width
            let perimeter = 2 * (length + width)
            return [area, perimeter]
        }

        let j = jacobian(of: plotMetrics, at: [3.0, 2.0])

        XCTAssertEqual(j.count, 2)       // m = 2 outputs
        XCTAssertEqual(j[0].count, 2)    // n = 2 inputs
        XCTAssertEqual(j[0][0], 2.0, accuracy: 1e-9)   // ∂area/∂length = width
        XCTAssertEqual(j[0][1], 3.0, accuracy: 1e-9)   // ∂area/∂width  = length
        XCTAssertEqual(j[1][0], 2.0, accuracy: 1e-9)   // ∂perimeter/∂length
        XCTAssertEqual(j[1][1], 2.0, accuracy: 1e-9)   // ∂perimeter/∂width
    }

    // f(x, y) = [x², x·y].  Analytic J = [[2x, 0], [y, x]].
    // At (2, 3): [[4, 0], [3, 2]].
    func testQuadraticJacobianMatchesAnalytic() {
        func f(_ v: [Double]) -> [Double] {
            let x = v[0]
            let y = v[1]
            return [x * x, x * y]
        }

        let j = jacobian(of: f, at: [2.0, 3.0])

        XCTAssertEqual(j[0][0], 4.0, accuracy: 1e-9)   // ∂(x²)/∂x = 2x
        XCTAssertEqual(j[0][1], 0.0, accuracy: 1e-9)   // ∂(x²)/∂y = 0
        XCTAssertEqual(j[1][0], 3.0, accuracy: 1e-9)   // ∂(xy)/∂x = y
        XCTAssertEqual(j[1][1], 2.0, accuracy: 1e-9)   // ∂(xy)/∂y = x
    }

    // MARK: - Shape handling

    // Non-square: f: ℝ³ → ℝ² must produce a 2×3 Jacobian.
    // f(a, b, c) = [a + 2b + 3c, a·b·c]. J = [[1, 2, 3], [bc, ac, ab]].
    // At (1, 2, 3): [[1, 2, 3], [6, 3, 2]].
    func testNonSquareJacobianShapeAndValues() {
        func f(_ v: [Double]) -> [Double] {
            let a = v[0]
            let b = v[1]
            let c = v[2]
            return [a + 2 * b + 3 * c, a * b * c]
        }

        let j = jacobian(of: f, at: [1.0, 2.0, 3.0])

        XCTAssertEqual(j.count, 2)       // m = 2 rows (outputs)
        XCTAssertEqual(j[0].count, 3)    // n = 3 columns (inputs)
        XCTAssertEqual(j[0][0], 1.0, accuracy: 1e-9)   // ∂(a+2b+3c)/∂a = 1
        XCTAssertEqual(j[0][1], 2.0, accuracy: 1e-9)   // ∂/∂b = 2
        XCTAssertEqual(j[0][2], 3.0, accuracy: 1e-9)   // ∂/∂c = 3
        XCTAssertEqual(j[1][0], 6.0, accuracy: 1e-9)   // bc = 2·3
        XCTAssertEqual(j[1][1], 3.0, accuracy: 1e-9)   // ac = 1·3
        XCTAssertEqual(j[1][2], 2.0, accuracy: 1e-9)   // ab = 1·2
    }

    // Single input, single output: f(x) = x³, J = [[3x²]]. At x = 2: [[12]].
    func testSingleInputSingleOutput() {
        func cube(_ v: [Double]) -> [Double] { [v[0] * v[0] * v[0]] }

        let j = jacobian(of: cube, at: [2.0])

        XCTAssertEqual(j.count, 1)
        XCTAssertEqual(j[0].count, 1)
        XCTAssertEqual(j[0][0], 12.0, accuracy: 1e-8)
    }

    // Single input, many outputs: f(t) = [t, t², t³]. J = [[1], [2t], [3t²]].
    // At t = 2: [[1], [4], [12]] — a 3×1 column.
    func testSingleInputManyOutputs() {
        func curve(_ v: [Double]) -> [Double] {
            let t = v[0]
            return [t, t * t, t * t * t]
        }

        let j = jacobian(of: curve, at: [2.0])

        XCTAssertEqual(j.count, 3)       // 3 outputs
        XCTAssertEqual(j[0].count, 1)    // 1 input
        XCTAssertEqual(j[0][0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(j[1][0], 4.0, accuracy: 1e-8)
        XCTAssertEqual(j[2][0], 12.0, accuracy: 1e-8)
    }

    // MARK: - Linear-map property

    // Near x, a small input change δ satisfies J·δ ≈ f(x+δ) − f(x) to second order.
    func testJacobianApproximatesLocalChange() {
        func f(_ v: [Double]) -> [Double] {
            let x = v[0]
            let y = v[1]
            return [x * x + y, x * y]
        }

        let x = [1.5, 2.0]
        let delta = [1e-4, -2e-4]

        let j = jacobian(of: f, at: x)
        // predicted = J·δ  (transformedBy computes matrix · vector)
        let predicted = delta.transformedBy(j)

        // actual = f(x+δ) − f(x)
        let xPlus = [x[0] + delta[0], x[1] + delta[1]]
        let actual = f(xPlus).subtract(f(x))

        // Agreement is limited by the O(‖δ‖²) truncation of the linear model.
        XCTAssertEqual(predicted[0], actual[0], accuracy: 1e-7)
        XCTAssertEqual(predicted[1], actual[1], accuracy: 1e-7)
    }

    // MARK: - Step size

    // A linear function has a constant Jacobian, so a well-chosen step recovers it
    // essentially exactly (no truncation error to trade against).
    // f(x, y) = [2x + y, x − 3y]. J = [[2, 1], [1, −3]] everywhere.
    func testLinearFunctionRecoveredAtDefaultStep() {
        func f(_ v: [Double]) -> [Double] {
            let x = v[0]
            let y = v[1]
            return [2 * x + y, x - 3 * y]
        }

        for step in [1e-3, 6e-6] {
            let j = jacobian(of: f, at: [4.0, 5.0], step: step)
            XCTAssertEqual(j[0][0], 2.0, accuracy: 1e-9)
            XCTAssertEqual(j[0][1], 1.0, accuracy: 1e-9)
            XCTAssertEqual(j[1][0], 1.0, accuracy: 1e-9)
            XCTAssertEqual(j[1][1], -3.0, accuracy: 1e-9)
        }
    }

    // Numerical-literacy guard: too small a step magnifies floating-point round-off,
    // so even a linear function loses accuracy at step 1e-8. This is why the default
    // sits near the cube root of machine epsilon rather than as small as possible.
    func testTooSmallStepMagnifiesRoundoff() {
        func f(_ v: [Double]) -> [Double] {
            let x = v[0]
            let y = v[1]
            return [2 * x + y, x - 3 * y]
        }

        let good = jacobian(of: f, at: [4.0, 5.0], step: 6e-6)[0][0]
        let tiny = jacobian(of: f, at: [4.0, 5.0], step: 1e-8)[0][0]

        // The default step nails 2.0; the too-small step visibly drifts off it.
        XCTAssertEqual(good, 2.0, accuracy: 1e-9)
        XCTAssertGreaterThan(abs(tiny - 2.0), 1e-9)
    }
}
