// Copyright 2025 Wayne W Bishop. All rights reserved.
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

final class FractionTests: XCTestCase {

    // MARK: - Basic Conversions

    // Whole numbers, simple fractions, negatives, and the matrix-inverse
    // thirteenths case all render correctly via Fraction.description.
    func testFractionDescription() {
        // Whole numbers
        XCTAssertEqual(Fraction(2.0).description, "2")
        XCTAssertEqual(Fraction(-3.0).description, "-3")
        XCTAssertEqual(Fraction(0.0).description, "0")
        XCTAssertEqual(Fraction(1.0).description, "1")

        // Simple positive fractions
        XCTAssertEqual(Fraction(0.5).description, "1/2")
        XCTAssertEqual(Fraction(0.25).description, "1/4")
        XCTAssertEqual(Fraction(0.2).description, "1/5")
        XCTAssertEqual(Fraction(0.75).description, "3/4")
        XCTAssertEqual(Fraction(1.0 / 3.0).description, "1/3")

        // Negative fractions
        XCTAssertEqual(Fraction(-0.5).description, "-1/2")
        XCTAssertEqual(Fraction(-0.2).description, "-1/5")
        XCTAssertEqual(Fraction(-1.0 / 3.0).description, "-1/3")

        // Thirteenths — the matrix inverse use case where the determinant is 13
        XCTAssertEqual(Fraction(5.0 / 13.0).description, "5/13")
        XCTAssertEqual(Fraction(-1.0 / 13.0).description, "-1/13")
        XCTAssertEqual(Fraction(-2.0 / 13.0).description, "-2/13")
        XCTAssertEqual(Fraction(3.0 / 13.0).description, "3/13")
    }

    func testValueProperty() {
        let f = Fraction(0.2)
        XCTAssertEqual(f.value, 0.2, accuracy: 1e-10)
        XCTAssertEqual(f.numerator, 1)
        XCTAssertEqual(f.denominator, 5)
    }

    // MARK: - Double Extension

    func testDoubleAsFraction() {
        let x = 0.384615384615
        let f = x.asFraction()
        XCTAssertEqual(f.description, "5/13")
        XCTAssertEqual(f.value, 5.0 / 13.0, accuracy: 1e-9)
    }

    // MARK: - Vector Extension

    func testVectorAsFractions() {
        let normalized = [0.6, 0.8]
        let fractions = normalized.asFractions()

        XCTAssertEqual(fractions[0].description, "3/5")
        XCTAssertEqual(fractions[1].description, "4/5")
    }

    // MARK: - Matrix Extension

    // Matrix inverse rendered as fractions — covers the determinant-13 case,
    // the identity (no-op inverse), and a diagonal scaling matrix.
    func testMatrixInverseAsFractions() throws {
        // Determinant 13 case
        let a = try [[3.0, 1.0], [2.0, 5.0]].inverted().asFractions()
        XCTAssertEqual(a[0][0].description, "5/13")
        XCTAssertEqual(a[0][1].description, "-1/13")
        XCTAssertEqual(a[1][0].description, "-2/13")
        XCTAssertEqual(a[1][1].description, "3/13")

        // Identity inverts to itself
        let identity = try [[1.0, 0.0], [0.0, 1.0]].inverted().asFractions()
        XCTAssertEqual(identity[0][0].description, "1")
        XCTAssertEqual(identity[0][1].description, "0")
        XCTAssertEqual(identity[1][0].description, "0")
        XCTAssertEqual(identity[1][1].description, "1")

        // Diagonal scaling matrix
        let scale = try [[2.0, 0.0], [0.0, 3.0]].inverted().asFractions()
        XCTAssertEqual(scale[0][0].description, "1/2")
        XCTAssertEqual(scale[1][1].description, "1/3")
    }

    // MARK: - Determinant as Fraction

    func testDeterminantAsFraction() {
        let A = [[3.0, 1.0],
                 [2.0, 5.0]]
        let det = A.determinant
        XCTAssertEqual(det.asFraction().description, "13")
    }

    // MARK: - Edge Cases

    func testMaxDenominator() {
        // Pi cannot be expressed as a clean fraction
        let pi = Double.pi
        let f = pi.asFraction(maxDenominator: 100)

        // Should find a reasonable approximation (e.g., 355/113 or 22/7)
        XCTAssertEqual(f.value, pi, accuracy: 0.01)
        XCTAssertLessThanOrEqual(f.denominator, 100)
    }

    func testExplicitConstruction() {
        let f = Fraction(numerator: 6, denominator: 4)
        // Should reduce to 3/2
        XCTAssertEqual(f.numerator, 3)
        XCTAssertEqual(f.denominator, 2)
        XCTAssertEqual(f.description, "3/2")
    }

    // MARK: - asExpression()

    // asExpression returns the same string as description for every shape
    // of fraction the type produces.
    func testAsExpression() {
        XCTAssertEqual(Fraction(2.0).asExpression(), "2")
        XCTAssertEqual(Fraction(-3.0).asExpression(), "-3")
        XCTAssertEqual(Fraction(0.0).asExpression(), "0")
        XCTAssertEqual((0.5).asFraction().asExpression(), "1/2")
        XCTAssertEqual((-3.0 / 4.0).asFraction().asExpression(), "-3/4")

        // description forwards to asExpression — the two always agree.
        let samples = [Fraction(2.0), Fraction(0.0), (5.0 / 13.0).asFraction()]
        for f in samples {
            XCTAssertEqual(f.asExpression(), f.description)
        }
    }
}
