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

final class MatrixSolveTests: XCTestCase {

    // MARK: - Simple 2x2

    func testSolveSimple2x2() {
        // 2x +  y = 5
        //  x + 3y = 10
        // x = 1, y = 3
        let A: [[Double]] = [
            [2, 1],
            [1, 3]
        ]
        let b = [5.0, 10.0]
        guard let x = A.solve(b) else {
            XCTFail("solve returned nil"); return
        }
        XCTAssertEqual(x.count, 2)
        XCTAssertEqual(x[0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(x[1], 3.0, accuracy: 1e-9)

        // Verify: A · x should equal b
        XCTAssertEqual(2 * x[0] + 1 * x[1], 5.0, accuracy: 1e-9)
        XCTAssertEqual(1 * x[0] + 3 * x[1], 10.0, accuracy: 1e-9)
    }

    // MARK: - Identity

    func testSolveIdentity() {
        let I: [[Double]] = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        let b = [4.0, -2.0, 7.5]
        guard let x = I.solve(b) else {
            XCTFail("solve returned nil"); return
        }
        XCTAssertEqual(x.count, 3)
        XCTAssertEqual(x[0], 4.0, accuracy: 1e-12)
        XCTAssertEqual(x[1], -2.0, accuracy: 1e-12)
        XCTAssertEqual(x[2], 7.5, accuracy: 1e-12)
    }

    // MARK: - Singular matrix returns nil

    func testSolveSingularReturnsNil() {
        // Rows are scalar multiples — the matrix is singular.
        let A: [[Double]] = [
            [1, 2],
            [2, 4]
        ]
        let b = [3.0, 6.0]
        XCTAssertNil(A.solve(b))
    }

    // MARK: - Dimension mismatch returns nil

    func testSolveDimensionMismatchReturnsNil() {
        let A: [[Double]] = [
            [1, 0],
            [0, 1]
        ]
        let b = [1.0, 2.0, 3.0]
        XCTAssertNil(A.solve(b))
    }

    // MARK: - Non-square returns nil

    func testSolveNonSquareReturnsNil() {
        // 2 rows, 3 columns — not square.
        let A: [[Double]] = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        let b = [1.0, 2.0]
        XCTAssertNil(A.solve(b))

        // Inconsistent row lengths
        let B: [[Double]] = [
            [1, 2],
            [3, 4, 5]
        ]
        XCTAssertNil(B.solve([1, 2]))
    }
}
