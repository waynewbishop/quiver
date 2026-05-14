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

final class ArrayArithmeticTests: XCTestCase {

    // Element-wise add, subtract, multiply, divide produce the expected vectors
    func testElementwiseArithmetic() {
        XCTAssertEqual([1.0, 2.0, 3.0].add([4.0, 5.0, 6.0]),      [5.0, 7.0, 9.0])
        XCTAssertEqual([5.0, 7.0, 9.0].subtract([1.0, 2.0, 3.0]), [4.0, 5.0, 6.0])
        XCTAssertEqual([2.0, 3.0, 4.0].multiply([3.0, 2.0, 1.0]), [6.0, 6.0, 4.0])
        XCTAssertEqual([6.0, 8.0, 10.0].divide([2.0, 4.0, 5.0]),  [3.0, 2.0, 2.0])
    }
}
