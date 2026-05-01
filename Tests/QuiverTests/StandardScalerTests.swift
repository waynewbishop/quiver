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
@testable import Quiver

final class StandardScalerTests: XCTestCase {

    // Fitted means and standard deviations match expected column statistics
    func testFitStatistics() {
        // Column 0: [1, 2, 3] → mean 2, population std sqrt(2/3) ≈ 0.8165
        // Column 1: [10, 20, 30] → mean 20, population std sqrt(200/3) ≈ 8.1650
        let features: [[Double]] = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ]

        let scaler = StandardScaler.fit(features: features)

        XCTAssertEqual(scaler.featureCount, 2)
        XCTAssertEqual(scaler.means[0], 2.0, accuracy: 1e-9)
        XCTAssertEqual(scaler.means[1], 20.0, accuracy: 1e-9)
        XCTAssertEqual(scaler.stds[0], (2.0 / 3.0).squareRoot(), accuracy: 1e-9)
        XCTAssertEqual(scaler.stds[1], (200.0 / 3.0).squareRoot(), accuracy: 1e-9)
    }

    // Transforming the training data produces zero mean and unit variance per column
    func testTransformProducesZeroMeanUnitVariance() {
        let features: [[Double]] = [
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0]
        ]

        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)

        // Column-wise mean should be ~0
        let col0Mean = scaled.map { $0[0] }.reduce(0, +) / Double(scaled.count)
        let col1Mean = scaled.map { $0[1] }.reduce(0, +) / Double(scaled.count)
        XCTAssertEqual(col0Mean, 0.0, accuracy: 1e-9)
        XCTAssertEqual(col1Mean, 0.0, accuracy: 1e-9)

        // Column-wise population variance should be ~1
        let col0Var = scaled.map { pow($0[0] - col0Mean, 2.0) }.reduce(0, +) / Double(scaled.count)
        let col1Var = scaled.map { pow($0[1] - col1Mean, 2.0) }.reduce(0, +) / Double(scaled.count)
        XCTAssertEqual(col0Var, 1.0, accuracy: 1e-9)
        XCTAssertEqual(col1Var, 1.0, accuracy: 1e-9)
    }

    // Test data is transformed using training statistics, not its own
    func testTrainTestSeparation() {
        // Training mean for column 0 is 2.0, population std = sqrt(2/3) ≈ 0.8165
        let train: [[Double]] = [[1.0], [2.0], [3.0]]
        let trainStd = (2.0 / 3.0).squareRoot()
        // Test value of 2.0 should z-score to exactly 0.0 using the training mean
        let test: [[Double]] = [[2.0]]

        let scaler = StandardScaler.fit(features: train)
        let scaledTest = scaler.transform(test)

        XCTAssertEqual(scaledTest[0][0], 0.0, accuracy: 1e-9)

        // Test value of 3.0 should z-score to (3-2)/trainStd
        let scaledAbove = scaler.transform([[3.0]])
        XCTAssertEqual(scaledAbove[0][0], 1.0 / trainStd, accuracy: 1e-9)
    }

    // A constant column (zero std) maps to 0.0 instead of producing NaN
    func testConstantColumn() {
        let features: [[Double]] = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0]
        ]

        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)

        // Column 0 is constant — all values map to 0.0
        XCTAssertEqual(scaled[0][0], 0.0)
        XCTAssertEqual(scaled[1][0], 0.0)
        XCTAssertEqual(scaled[2][0], 0.0)

        // Column 1: values [10, 20, 30] → mean 20, population std = sqrt(200/3)
        // First value (10 - 20) / sqrt(200/3) = -10 / sqrt(200/3)
        let col1Std = (200.0 / 3.0).squareRoot()
        XCTAssertEqual(scaled[0][1], -10.0 / col1Std, accuracy: 1e-9)
    }

    // Values outside the training range scale proportionally
    func testOutOfRangeValues() {
        let train: [[Double]] = [[0.0], [10.0]]
        let scaler = StandardScaler.fit(features: train)

        // Mean = 5.0, std = 5.0
        // -5 should z-score to -2.0
        let scaled = scaler.transform([[-5.0], [20.0]])

        XCTAssertEqual(scaled[0][0], -2.0, accuracy: 1e-9)
        XCTAssertEqual(scaled[1][0], 3.0, accuracy: 1e-9)
    }

    // MARK: - Equatable

    // Same training data produces equal scalers
    func testStandardScalerEquatable() {
        let features: [[Double]] = [[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]

        let scaler1 = StandardScaler.fit(features: features)
        let scaler2 = StandardScaler.fit(features: features)
        XCTAssertEqual(scaler1, scaler2)

        // Different training data produces different scalers
        let scaler3 = StandardScaler.fit(features: [[0.0, 0.0], [10.0, 10.0]])
        XCTAssertNotEqual(scaler1, scaler3)
    }

    // MARK: - Codable

    // Round-trip preserves equality and transformation output
    func testStandardScalerCodable() throws {
        let features: [[Double]] = [
            [1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]
        ]
        let scaler = StandardScaler.fit(features: features)

        let data = try JSONEncoder().encode(scaler)
        let decoded = try JSONDecoder().decode(StandardScaler.self, from: data)
        XCTAssertEqual(scaler, decoded)

        // Transformation output matches after decode
        let testInput: [[Double]] = [[2.5, 250.0]]
        XCTAssertEqual(scaler.transform(testInput), decoded.transform(testInput))
    }
}
