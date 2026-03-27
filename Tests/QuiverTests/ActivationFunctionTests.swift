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

final class ActivationFunctionTests: XCTestCase {

    // Known values and sum-to-one property
    func testSoftmaxBasic() {
        let logits = [2.0, 1.0, 0.1]
        let probs = logits.softMax()
        XCTAssertEqual(probs[0], 0.6590, accuracy: 0.001)
        XCTAssertEqual(probs[1], 0.2424, accuracy: 0.001)
        XCTAssertEqual(probs[2], 0.0986, accuracy: 0.001)
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-9)

        // Also verify with different input
        let probs2 = [1.0, 2.0, 3.0, 4.0, 5.0].softMax()
        XCTAssertEqual(probs2.reduce(0, +), 1.0, accuracy: 1e-9)
    }

    // Uniform input → uniform output
    func testSoftmaxUniform() {
        let logits = [3.0, 3.0, 3.0, 3.0]
        let probs = logits.softMax()
        for p in probs {
            XCTAssertEqual(p, 0.25, accuracy: 1e-9)
        }
    }

    // Large values should not overflow (numerical stability)
    func testSoftmaxNumericalStability() {
        let logits = [1000.0, 1001.0, 1002.0]
        let probs = logits.softMax()
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-9)
        XCTAssertGreaterThan(probs[2], probs[1])
        XCTAssertGreaterThan(probs[1], probs[0])
    }

    // Single element → probability 1.0
    func testSoftmaxSingleElement() {
        let probs = [42.0].softMax()
        XCTAssertEqual(probs, [1.0])
    }

    // MARK: - Sigmoid Tests

    // σ(0) = 0.5 exactly
    func testSigmoidZero() {
        let result = [0.0].sigmoid()
        XCTAssertEqual(result[0], 0.5, accuracy: 1e-9)
    }

    // Known values — hand-verified against Python scipy.special.expit
    func testSigmoidKnownValues() {
        let logits = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let result = logits.sigmoid()
        XCTAssertEqual(result[0], 0.1192, accuracy: 0.001)
        XCTAssertEqual(result[1], 0.2689, accuracy: 0.001)
        XCTAssertEqual(result[2], 0.5000, accuracy: 0.001)
        XCTAssertEqual(result[3], 0.7311, accuracy: 0.001)
        XCTAssertEqual(result[4], 0.8808, accuracy: 0.001)
    }

    // Symmetry: σ(x) + σ(-x) = 1.0
    func testSigmoidSymmetry() {
        let values = [0.5, 1.0, 3.0, 10.0]
        let positive = values.sigmoid()
        let negative = values.map { -$0 }.sigmoid()
        for i in 0..<values.count {
            XCTAssertEqual(positive[i] + negative[i], 1.0, accuracy: 1e-9)
        }
    }

    // Large values should saturate without overflow
    func testSigmoidSaturation() {
        let result = [-1000.0, 1000.0].sigmoid()
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[1], 1.0, accuracy: 1e-9)
    }
}
