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

final class PolyfitTests: XCTestCase {

    // MARK: - Degree 1 matches LinearRegression

    func testPolyfitDegreeOneMatchesLinearRegression() throws {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.1, 3.9, 6.1, 8.0, 9.8]

        guard let poly = [Double].polyfit(x: x, y: y, degree: 1) else {
            XCTFail("polyfit returned nil"); return
        }

        let model = try LinearRegression.fit(features: x, targets: y, intercept: true)

        // poly.coefficients should equal [intercept, slope] from LR
        XCTAssertEqual(poly.coefficients.count, 2)
        XCTAssertEqual(poly.coefficients[0], model.coefficients[0], accuracy: 1e-9)
        XCTAssertEqual(poly.coefficients[1], model.coefficients[1], accuracy: 1e-9)
    }

    // MARK: - Quadratic reference fit

    func testPolyfitQuadraticReference() {
        // y = 2x² + 3x + 1, evaluated at integer x in [1, 5]
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [6.0, 15.0, 28.0, 45.0, 66.0]

        guard let poly = [Double].polyfit(x: x, y: y, degree: 2) else {
            XCTFail("polyfit returned nil"); return
        }

        XCTAssertEqual(poly.coefficients.count, 3)
        XCTAssertEqual(poly.coefficients[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(poly.coefficients[1], 3.0, accuracy: 1e-6)
        XCTAssertEqual(poly.coefficients[2], 2.0, accuracy: 1e-6)
    }

    // MARK: - Validation paths

    func testPolyfitMismatchedLengthsReturnsNil() {
        let x = [1.0, 2.0, 3.0]
        let y = [1.0, 2.0]
        XCTAssertNil([Double].polyfit(x: x, y: y, degree: 1))
    }

    func testPolyfitInsufficientPointsReturnsNil() {
        // Need more than `degree` points; degree=2 with 2 points should fail.
        let x = [1.0, 2.0]
        let y = [3.0, 5.0]
        XCTAssertNil([Double].polyfit(x: x, y: y, degree: 2))

        // Negative degree is rejected.
        XCTAssertNil([Double].polyfit(x: [1, 2, 3], y: [1, 2, 3], degree: -1))
    }

    // MARK: - Fit-then-evaluate

    func testPolyfitFitThenEvaluate() {
        // y = 2x² + 3x + 1
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [6.0, 15.0, 28.0, 45.0, 66.0]

        guard let poly = [Double].polyfit(x: x, y: y, degree: 2) else {
            XCTFail("polyfit returned nil"); return
        }

        // Evaluate at a held-out point: 2*36 + 3*6 + 1 = 91
        XCTAssertEqual(poly(6.0), 91.0, accuracy: 1e-6)
        XCTAssertEqual(poly(0.0), 1.0, accuracy: 1e-6)

        // Vectorized evaluation must agree with the original y values.
        let recovered = poly(x)
        for (a, b) in zip(recovered, y) {
            XCTAssertEqual(a, b, accuracy: 1e-6)
        }
    }
}
