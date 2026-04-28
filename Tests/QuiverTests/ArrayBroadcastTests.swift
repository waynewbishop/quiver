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

// Array-to-array operators (e.g., [Double] + [Double]) are intentionally NOT
// overloaded in Quiver. Element-wise array operations go through named methods:
// `add`, `subtract`, `multiply`, `divide` (see ArrayArithmeticTests). Any future
// addition of operator overloads for array-array would need explicit design review.

final class ArrayBroadcastTests: XCTestCase {

    // MARK: - Scalar Broadcasting Tests

    // Covers add, multiply, subtract, divide scalars and empty array
    func testBroadcastScalarOperations() {
        let array = [1.0, 2.0, 3.0]
        XCTAssertEqual(array.broadcast(adding: 5.0), [6.0, 7.0, 8.0])
        XCTAssertEqual(array.broadcast(multiplyingBy: 2.0), [2.0, 4.0, 6.0])
        XCTAssertEqual([5.0, 7.0, 9.0].broadcast(subtracting: 2.0), [3.0, 5.0, 7.0])
        XCTAssertEqual([2.0, 4.0, 6.0].broadcast(dividingBy: 2.0), [1.0, 2.0, 3.0])

        // Empty array
        let empty: [Double] = []
        XCTAssertEqual(empty.broadcast(adding: 5.0), [])
        XCTAssertEqual(empty.broadcast(multiplyingBy: 2.0), [])
    }

    // MARK: - Matrix-Vector Broadcasting Tests

    func testBroadcastAddingToEachRow() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let rowVector = [10.0, 20.0, 30.0]

        let result = matrix.broadcast(addingToEachRow: rowVector)
        XCTAssertEqual(result, [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
    }

    func testBroadcastAddingToEachColumn() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let columnVector = [10.0, 20.0]

        let result = matrix.broadcast(addingToEachColumn: columnVector)
        XCTAssertEqual(result, [[11.0, 12.0, 13.0], [24.0, 25.0, 26.0]])
    }

    func testBroadcastMultiplyingEachRowBy() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let rowVector = [10.0, 20.0, 30.0]

        let result = matrix.broadcast(multiplyingEachRowBy: rowVector)
        XCTAssertEqual(result, [[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]])
    }

    func testBroadcastMultiplyingEachColumnBy() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let columnVector = [10.0, 20.0]

        let result = matrix.broadcast(multiplyingEachColumnBy: columnVector)
        XCTAssertEqual(result, [[10.0, 20.0, 30.0], [80.0, 100.0, 120.0]])
    }

    // MARK: - Custom Broadcasting Tests

    func testBroadcastWithCustomOperation() {
        let array = [1.0, 2.0, 3.0]
        let result = array.broadcast(with: 2.0) { base, exponent in
            pow(base, exponent)
        }
        XCTAssertEqual(result, [1.0, 4.0, 9.0])
    }

    func testBroadcastWithRowVectorCustomOperation() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let rowVector = [10.0, 20.0, 30.0]

        let result = matrix.broadcast(withRowVector: rowVector) { matrixElement, vectorElement in
            matrixElement / vectorElement
        }
        XCTAssertEqual(result, [[0.1, 0.1, 0.1], [0.4, 0.25, 0.2]])
    }

    func testBroadcastWithColumnVectorCustomOperation() {
        let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let columnVector = [10.0, 20.0]

        let result = matrix.broadcast(withColumnVector: columnVector) { matrixElement, vectorElement in
            (matrixElement + vectorElement) * 2.0
        }
        XCTAssertEqual(result, [[22.0, 24.0, 26.0], [48.0, 50.0, 52.0]])
    }

    // MARK: - Complex Broadcast Combinations

    func testComplexBroadcastOperation() {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0]

        let result = array
            .broadcast(multiplyingBy: 2.0)
            .broadcast(adding: 1.0)
            .broadcast(dividingBy: 2.0)

        XCTAssertEqual(result, [1.5, 2.5, 3.5, 4.5, 5.5])
    }

    func testMatrixComplexBroadcastOperation() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]]
        let rowVector = [10.0, 20.0]
        let columnVector = [100.0, 200.0]

        let intermediate = matrix.broadcast(addingToEachRow: rowVector)
        let result = intermediate.broadcast(multiplyingEachColumnBy: columnVector)

        XCTAssertEqual(result, [[1100.0, 2200.0], [2600.0, 4800.0]])
    }

    // MARK: - Operator-Form Scalar Broadcasting

    // Locks in the supported operator surface: scalar-on-array works in both directions
    func testOperatorScalarBroadcasting() {
        let v = [1.0, 2.0, 3.0]
        XCTAssertEqual(v + 10.0, [11.0, 12.0, 13.0])
        XCTAssertEqual(10.0 + v, [11.0, 12.0, 13.0])
        XCTAssertEqual(v - 1.0, [0.0, 1.0, 2.0])
        XCTAssertEqual(v * 2.0, [2.0, 4.0, 6.0])
        XCTAssertEqual(v / 2.0, [0.5, 1.0, 1.5])

        let m = [[1.0, 2.0], [3.0, 4.0]]
        XCTAssertEqual(m + 10.0, [[11.0, 12.0], [13.0, 14.0]])
        XCTAssertEqual(m * 2.0, [[2.0, 4.0], [6.0, 8.0]])
    }

    // Operator and method forms must produce identical results — protects against drift
    func testOperatorMatchesMethodForm() {
        let v = [1.0, 2.0, 3.0]
        XCTAssertEqual(v + 5.0, v.broadcast(adding: 5.0))
        XCTAssertEqual(v - 5.0, v.broadcast(subtracting: 5.0))
        XCTAssertEqual(v * 5.0, v.broadcast(multiplyingBy: 5.0))
        XCTAssertEqual(v / 5.0, v.broadcast(dividingBy: 5.0))
    }
}
