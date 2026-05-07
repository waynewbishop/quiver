// Copyright 2025 Wayne W Bishop. All rights reserved.
//
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language governing
// permissions and limitations under the License.

import XCTest
@testable import Quiver

final class MatrixArithmeticTests: XCTestCase {

    // MARK: - Matrix Element-wise Operations

    // Element-wise add, subtract, multiply, divide on matrices
    func testMatrixElementwiseArithmetic() {
        XCTAssertEqual(
            [[1.0, 2.0], [3.0, 4.0]].add([[5.0, 6.0], [7.0, 8.0]]),
            [[6.0, 8.0], [10.0, 12.0]]
        )
        XCTAssertEqual(
            [[5.0, 7.0], [9.0, 11.0]].subtract([[1.0, 2.0], [3.0, 4.0]]),
            [[4.0, 5.0], [6.0, 7.0]]
        )
        XCTAssertEqual(
            [[2.0, 3.0], [4.0, 5.0]].multiply([[3.0, 2.0], [1.0, 2.0]]),
            [[6.0, 6.0], [4.0, 10.0]]
        )
        XCTAssertEqual(
            [[6.0, 8.0], [10.0, 12.0]].divide([[2.0, 4.0], [5.0, 3.0]]),
            [[3.0, 2.0], [2.0, 4.0]]
        )
    }

    // MARK: - Matrix Scalar Broadcasting

    // Covers add, subtract, multiply, divide with scalars (both directions)
    func testMatrixScalarOperations() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]]

        // Addition (commutative)
        XCTAssertEqual(matrix + 10.0, [[11.0, 12.0], [13.0, 14.0]])
        XCTAssertEqual(10.0 + matrix, [[11.0, 12.0], [13.0, 14.0]])

        // Subtraction (non-commutative)
        XCTAssertEqual([[11.0, 12.0], [13.0, 14.0]] - 10.0, [[1.0, 2.0], [3.0, 4.0]])
        XCTAssertEqual(10.0 - matrix, [[9.0, 8.0], [7.0, 6.0]])

        // Multiplication (commutative)
        XCTAssertEqual([[2.0, 3.0], [4.0, 5.0]] * 2.0, [[4.0, 6.0], [8.0, 10.0]])
        XCTAssertEqual(2.0 * [[2.0, 3.0], [4.0, 5.0]], [[4.0, 6.0], [8.0, 10.0]])

        // Division (non-commutative)
        XCTAssertEqual([[4.0, 6.0], [8.0, 10.0]] / 2.0, [[2.0, 3.0], [4.0, 5.0]])
        XCTAssertEqual(20.0 / [[2.0, 4.0], [5.0, 10.0]], [[10.0, 5.0], [4.0, 2.0]])
    }

    // MARK: - Real-World Use Cases

    func testDataNormalization() {
        let data = [[100.0, 200.0], [300.0, 400.0]]
        let normalized = (data - 250.0) / 150.0

        let expected = [[-1.0, -1.0/3.0], [1.0/3.0, 1.0]]

        XCTAssertEqual(normalized[0][0], expected[0][0], accuracy: 0.0001)
        XCTAssertEqual(normalized[0][1], expected[0][1], accuracy: 0.0001)
        XCTAssertEqual(normalized[1][0], expected[1][0], accuracy: 0.0001)
        XCTAssertEqual(normalized[1][1], expected[1][1], accuracy: 0.0001)
    }

    // MARK: - Vector Scalar Broadcasting

    // Covers add, subtract, multiply, divide with scalars (both directions)
    func testVectorScalarOperations() {
        XCTAssertEqual([1.0, 2.0, 3.0] + 10.0, [11.0, 12.0, 13.0])
        XCTAssertEqual(10.0 + [1.0, 2.0, 3.0], [11.0, 12.0, 13.0])
        XCTAssertEqual([11.0, 12.0, 13.0] - 10.0, [1.0, 2.0, 3.0])
        XCTAssertEqual(10.0 - [1.0, 2.0, 3.0], [9.0, 8.0, 7.0])
        XCTAssertEqual([2.0, 3.0, 4.0] * 2.0, [4.0, 6.0, 8.0])
        XCTAssertEqual(2.0 * [2.0, 3.0, 4.0], [4.0, 6.0, 8.0])
        XCTAssertEqual([4.0, 6.0, 8.0] / 2.0, [2.0, 3.0, 4.0])
        XCTAssertEqual(20.0 / [2.0, 4.0, 5.0], [10.0, 5.0, 4.0])
    }
}
