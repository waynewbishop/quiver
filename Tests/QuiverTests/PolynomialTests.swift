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
import Foundation
@testable import Quiver

final class PolynomialTests: XCTestCase {

    // MARK: - Construction and degree

    func testInitAndDegree() {
        XCTAssertEqual(Polynomial([1, 3, 2]).degree, 2)
        XCTAssertEqual(Polynomial([5]).degree, 0)
        XCTAssertEqual(Polynomial([0]).degree, 0)
        XCTAssertEqual(Polynomial([0, 0, 3]).degree, 2)
        XCTAssertEqual(Polynomial([1, 2, 0]).degree, 1)

        // Empty array becomes the zero polynomial
        XCTAssertEqual(Polynomial([]).coefficients, [0])
        XCTAssertEqual(Polynomial([]).degree, 0)
    }

    // MARK: - Evaluation

    // Covers scalar evaluation, array evaluation, and consistency between paths
    func testEvaluate() {
        // 2x² + 3x + 1 — scalar evaluation across positive and negative inputs
        let p = Polynomial([1, 3, 2])
        XCTAssertEqual(p(0), 1, accuracy: 1e-12)
        XCTAssertEqual(p(1), 6, accuracy: 1e-12)
        XCTAssertEqual(p(2), 15, accuracy: 1e-12)
        XCTAssertEqual(p(-1), 0, accuracy: 1e-12)
        XCTAssertEqual(p(-2), 3, accuracy: 1e-12)

        // Array evaluation produces the same per-input values
        let arrayResult = p([0, 1, 2, 3])
        XCTAssertEqual(arrayResult.count, 4)
        XCTAssertEqual(arrayResult[0], 1, accuracy: 1e-12)
        XCTAssertEqual(arrayResult[1], 6, accuracy: 1e-12)
        XCTAssertEqual(arrayResult[2], 15, accuracy: 1e-12)
        XCTAssertEqual(arrayResult[3], 28, accuracy: 1e-12)

        // Array path and scalar path agree across mixed inputs
        let q = Polynomial([1, -3, 2, -1])  // -x³ + 2x² - 3x + 1
        let xs: [Double] = [-2, -1, 0, 1, 2, 3]
        let arrayPath = q(xs)
        let scalarPath = xs.map { q($0) }
        for (a, b) in zip(arrayPath, scalarPath) {
            XCTAssertEqual(a, b, accuracy: 1e-12)
        }
    }

    // MARK: - Derivative

    // Covers derivatives of constant, linear, and quadratic polynomials
    func testDerivative() {
        // Constant — derivative is zero
        let constant = Polynomial([7]).derivative()
        XCTAssertEqual(constant.coefficients, [0])
        XCTAssertEqual(constant.degree, 0)

        // Linear: 3x + 2 → 3
        XCTAssertEqual(Polynomial([2, 3]).derivative().coefficients, [3])

        // Quadratic: 2x² + 3x + 1 → 4x + 3
        XCTAssertEqual(Polynomial([1, 3, 2]).derivative().coefficients, [3, 4])
    }

    // MARK: - Trim

    func testTrimmedRemovesTrailingZeros() {
        // Two polynomials representing the same function, different storage
        let p1 = Polynomial([1, 2])
        let p2 = Polynomial([1, 2, 0, 0])
        XCTAssertNotEqual(p1, p2, "raw equality is structural, so trailing zeros differ")
        XCTAssertEqual(p1.trimmed(), p2.trimmed(), "trim canonicalizes both to the same form")

        // Zero polynomial canonicalizes to [0], not []
        XCTAssertEqual(Polynomial([0, 0, 0]).trimmed(), Polynomial([0]))

        // Already-trimmed polynomial is unchanged
        XCTAssertEqual(Polynomial([1, 2, 3]).trimmed(), Polynomial([1, 2, 3]))
    }

    // MARK: - Arithmetic

    func testAddition() {
        // (2x² + 3x + 1) + (-2x² - 3x + 4) = 5
        let p = Polynomial([1, 3, 2])
        let q = Polynomial([4, -3, -2])
        let sum = p + q
        XCTAssertEqual(sum.coefficients, [5])
        XCTAssertEqual(sum.degree, 0)

        // Different-length addition: (x + 1) + (x² + x) = x² + 2x + 1
        let r = Polynomial([1, 1])
        let s = Polynomial([0, 1, 1])
        let rs = r + s
        XCTAssertEqual(rs.coefficients, [1, 2, 1])
    }

    func testMultiplication() {
        // (x + 1)(x - 1) = x² - 1
        let p = Polynomial([1, 1])
        let q = Polynomial([-1, 1])
        let product = p * q
        XCTAssertEqual(product.coefficients.count, 3)
        XCTAssertEqual(product.coefficients[0], -1, accuracy: 1e-12)
        XCTAssertEqual(product.coefficients[1], 0, accuracy: 1e-12)
        XCTAssertEqual(product.coefficients[2], 1, accuracy: 1e-12)

        // (2x + 3)(x + 4) = 2x² + 11x + 12
        let r = Polynomial([3, 2])
        let s = Polynomial([4, 1])
        let rs = r * s
        XCTAssertEqual(rs.coefficients[0], 12, accuracy: 1e-12)
        XCTAssertEqual(rs.coefficients[1], 11, accuracy: 1e-12)
        XCTAssertEqual(rs.coefficients[2], 2, accuracy: 1e-12)
    }

    func testScalarMultiplication() {
        let p = Polynomial([1, 3, 2])
        let scaled = 3.0 * p
        XCTAssertEqual(scaled.coefficients, [3, 9, 6])

        // Multiplying by 0 produces the zero polynomial
        let zeroed = 0.0 * p
        XCTAssertEqual(zeroed.coefficients, [0])
        XCTAssertEqual(zeroed.degree, 0)
    }

    // MARK: - Equatable

    func testEquatable() {
        let a = Polynomial([1, 3, 2])
        let b = Polynomial([1, 3, 2])
        let c = Polynomial([1, 3, 3])
        let d = Polynomial([1, 3])

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertNotEqual(a, d)
    }

    // MARK: - Codable

    func testCodableRoundTrip() throws {
        let original = Polynomial([1.5, -3.25, 2.0])
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Polynomial.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    // MARK: - Description

    func testDescription() {
        XCTAssertEqual(String(describing: Polynomial([1, 3, 2])), "2x² + 3x + 1")
        XCTAssertEqual(String(describing: Polynomial([0, 1])), "x")
        XCTAssertEqual(String(describing: Polynomial([5])), "5")
        XCTAssertEqual(String(describing: Polynomial([0])), "0")
        XCTAssertEqual(String(describing: Polynomial([1, -3, 2])), "2x² - 3x + 1")
        XCTAssertEqual(String(describing: Polynomial([0, -1])), "-x")

        // Higher-degree term with coefficient 1 drops the "1"
        XCTAssertEqual(String(describing: Polynomial([0, 0, 1])), "x²")

        // Multi-digit superscript: degree 10 with coefficient 1
        var coeffs = [Double](repeating: 0, count: 11)
        coeffs[10] = 1
        XCTAssertEqual(String(describing: Polynomial(coeffs)), "x¹⁰")

        // Mixed signs and zeros: x³ - 2x + 5
        XCTAssertEqual(String(describing: Polynomial([5, -2, 0, 1])), "x³ - 2x + 5")
    }

    // MARK: - Zero polynomial

    func testZeroPolynomial() {
        let zero = Polynomial([0])
        XCTAssertEqual(zero.degree, 0)
        XCTAssertEqual(zero(5), 0, accuracy: 1e-12)
        XCTAssertEqual(zero([1, 2, 3]), [0, 0, 0])
        XCTAssertEqual(String(describing: zero), "0")
        XCTAssertEqual(zero.derivative().coefficients, [0])

        // Adding a polynomial to the zero polynomial returns the polynomial
        let p = Polynomial([1, 3, 2])
        XCTAssertEqual((zero + p).coefficients, p.coefficients)
        XCTAssertEqual((p + zero).coefficients, p.coefficients)
    }
}
