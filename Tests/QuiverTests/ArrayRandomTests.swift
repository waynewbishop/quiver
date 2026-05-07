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

final class ArrayRandomTests: XCTestCase {

    // Covers 1D, 2D, empty, and uniqueness
    func testRandomUniform() {
        let result1D = [Double].random(5)
        XCTAssertEqual(result1D.count, 5)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }

        let result2D = [Double].random(3, 4)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0.0)
                XCTAssertLessThanOrEqual(value, 1.0)
            }
        }

        XCTAssertTrue([Double].random(0).isEmpty)
        XCTAssertNotEqual([Double].random(10), [Double].random(10))
    }

    // MARK: - Custom Range Tests

    // 1D and 2D random doubles in a custom range produce values in bounds
    func testRandomCustomRange() {
        let result1D = [Double].random(100, in: -5.0...5.0)
        XCTAssertEqual(result1D.count, 100)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, -5.0)
            XCTAssertLessThanOrEqual(value, 5.0)
        }

        let result2D = [Double].random(3, 4, in: 10.0...20.0)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 10.0)
                XCTAssertLessThanOrEqual(value, 20.0)
            }
        }
    }

    // MARK: - Normal Distribution Tests

    // Covers 1D default, custom params, 2D, and zero std
    func testRandomNormal() {
        let result1D = [Double].randomNormal(1000)
        XCTAssertEqual(result1D.count, 1000)
        XCTAssertEqual(result1D.mean()!, 0.0, accuracy: 0.2)
        XCTAssertEqual(result1D.std()!, 1.0, accuracy: 0.2)

        let custom = [Double].randomNormal(1000, mean: 5.0, std: 2.0)
        XCTAssertEqual(custom.mean()!, 5.0, accuracy: 0.3)
        XCTAssertEqual(custom.std()!, 2.0, accuracy: 0.3)

        let result2D = [Double].randomNormal(3, 4)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)

        let zeroStd = [Double].randomNormal(10, mean: 3.0, std: 0.0)
        for value in zeroStd { XCTAssertEqual(value, 3.0) }
    }

    // MARK: - Random Integer Tests

    // 1D, 2D, and negative-range int generation all produce values in bounds
    func testRandomInt() {
        let result1D = [Int].random(100, in: 0..<10)
        XCTAssertEqual(result1D.count, 100)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThan(value, 10)
        }

        let result2D = [Int].random(3, 4, in: 1..<7)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 1)
                XCTAssertLessThan(value, 7)
            }
        }

        let negative = [Int].random(50, in: -10..<0)
        for value in negative {
            XCTAssertGreaterThanOrEqual(value, -10)
            XCTAssertLessThan(value, 0)
        }
    }
}
