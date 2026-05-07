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

final class ArrayGenerationTests: XCTestCase {

    // 1D generation methods produce arrays of the expected length and value
    func testGeneration1D() {
        XCTAssertEqual([Double].zeros(3), [0.0, 0.0, 0.0])
        XCTAssertEqual([Double].ones(3), [1.0, 1.0, 1.0])
        XCTAssertEqual([Double].full(3, value: 7.5), [7.5, 7.5, 7.5])
        XCTAssertEqual([Double].linspace(start: 0, end: 10, count: 5), [0.0, 2.5, 5.0, 7.5, 10.0])
        XCTAssertEqual([Double].arange(0, 10, step: 2.5), [0.0, 2.5, 5.0, 7.5])
    }

    // 2D generation methods produce matrices of the expected shape and value
    func testGeneration2D() {
        XCTAssertEqual([Int].zeros(2, 3), [[0, 0, 0], [0, 0, 0]])
        XCTAssertEqual([Int].ones(2, 2), [[1, 1], [1, 1]])
        XCTAssertEqual([Int].full(2, 2, value: 7), [[7, 7], [7, 7]])
        XCTAssertEqual(
            [Double].diag([1.0, 2.0, 3.0]),
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
    }
}
