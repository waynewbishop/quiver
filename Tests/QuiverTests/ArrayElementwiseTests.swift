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

final class ArrayElementwiseTests: XCTestCase {

    let tolerance = 1e-10

    // MARK: - Power

    // Squaring via power(2)
    func testPower() {
        let values = [1.0, 2.0, 3.0, 4.0]
        let result = values.power(2)
        XCTAssertEqual(result, [1.0, 4.0, 9.0, 16.0])
    }

    // Fractional exponent produces square roots
    func testPowerFractional() {
        let values = [4.0, 9.0, 16.0]
        let result = values.power(0.5)
        XCTAssertEqual(result[0], 2.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 3.0, accuracy: tolerance)
        XCTAssertEqual(result[2], 4.0, accuracy: tolerance)
    }

    // MARK: - Trigonometric

    // sin(0) = 0, sin(π/2) = 1
    func testSin() {
        let values = [0.0, Double.pi / 2]
        let result = values.sin()
        XCTAssertEqual(result[0], 0.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 1.0, accuracy: tolerance)
    }

    // cos(0) = 1, cos(π) = -1
    func testCos() {
        let values = [0.0, Double.pi]
        let result = values.cos()
        XCTAssertEqual(result[0], 1.0, accuracy: tolerance)
        XCTAssertEqual(result[1], -1.0, accuracy: tolerance)
    }

    // tan(0) = 0, tan(π/4) = 1
    func testTan() {
        let values = [0.0, Double.pi / 4]
        let result = values.tan()
        XCTAssertEqual(result[0], 0.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 1.0, accuracy: tolerance)
    }

    // MARK: - Logarithmic and Exponential

    // ln(1) = 0, ln(e) = 1
    func testLog() {
        let values = [1.0, M_E]
        let result = values.log()
        XCTAssertEqual(result[0], 0.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 1.0, accuracy: tolerance)
    }

    // log10(1) = 0, log10(100) = 2
    func testLog10() {
        let values = [1.0, 100.0]
        let result = values.log10()
        XCTAssertEqual(result[0], 0.0, accuracy: tolerance)
        XCTAssertEqual(result[1], 2.0, accuracy: tolerance)
    }

    // exp(0) = 1, exp(1) = e
    func testExp() {
        let values = [0.0, 1.0]
        let result = values.exp()
        XCTAssertEqual(result[0], 1.0, accuracy: tolerance)
        XCTAssertEqual(result[1], M_E, accuracy: tolerance)
    }

    // MARK: - Square Root and Square

    // sqrt(4) = 2, sqrt(9) = 3
    func testSqrt() {
        let values = [4.0, 9.0, 25.0]
        let result = values.sqrt()
        XCTAssertEqual(result, [2.0, 3.0, 5.0])
    }

    // square(3) = 9, square(4) = 16
    func testSquare() {
        let values = [2.0, 3.0, 4.0]
        let result = values.square()
        XCTAssertEqual(result, [4.0, 9.0, 16.0])
    }

    // sqrt and square are inverses
    func testSqrtSquareRoundTrip() {
        let values = [1.0, 4.0, 9.0, 16.0]
        let result = values.sqrt().square()
        for (a, b) in zip(values, result) {
            XCTAssertEqual(a, b, accuracy: tolerance)
        }
    }

    // MARK: - Rounding

    // floor rounds down
    func testFloor() {
        let values = [1.2, 2.8, -1.5, 3.0]
        let result = values.floor()
        XCTAssertEqual(result, [1.0, 2.0, -2.0, 3.0])
    }

    // ceil rounds up
    func testCeil() {
        let values = [1.2, 2.8, -1.5, 3.0]
        let result = values.ceil()
        XCTAssertEqual(result, [2.0, 3.0, -1.0, 3.0])
    }

    // round rounds to nearest
    func testRound() {
        let values = [1.2, 2.5, 2.8, -1.5]
        let result = values.round()
        XCTAssertEqual(result, [1.0, 3.0, 3.0, -2.0])
    }

    // MARK: - Empty Arrays

    // All operations return empty arrays for empty input
    func testEmptyArrays() {
        let empty: [Double] = []
        XCTAssertTrue(empty.power(2).isEmpty)
        XCTAssertTrue(empty.sin().isEmpty)
        XCTAssertTrue(empty.cos().isEmpty)
        XCTAssertTrue(empty.tan().isEmpty)
        XCTAssertTrue(empty.log().isEmpty)
        XCTAssertTrue(empty.log10().isEmpty)
        XCTAssertTrue(empty.exp().isEmpty)
        XCTAssertTrue(empty.sqrt().isEmpty)
        XCTAssertTrue(empty.square().isEmpty)
        XCTAssertTrue(empty.floor().isEmpty)
        XCTAssertTrue(empty.ceil().isEmpty)
        XCTAssertTrue(empty.round().isEmpty)
    }

    // MARK: - Float Variants

    // Float versions produce the same results
    func testFloatVariants() {
        let values: [Float] = [4.0, 9.0, 16.0]
        XCTAssertEqual(values.sqrt(), [2.0, 3.0, 4.0])
        XCTAssertEqual(values.square(), [16.0, 81.0, 256.0])
        XCTAssertEqual(values.power(0.5)[0], 2.0, accuracy: 1e-5)
    }
}
