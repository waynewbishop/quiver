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

final class FeatureScalerTests: XCTestCase {

    // Scaling to [0, 1] maps min to 0 and max to 1 per column
    func testBasicScaling() {
        let features: [[Double]] = [
            [100, 0.1],
            [200, 0.5],
            [300, 0.9]
        ]

        let scaler = FeatureScaler.fit(features: features)
        let scaled = scaler.transform(features)

        // Column 0: 100→0, 200→0.5, 300→1
        XCTAssertEqual(scaled[0][0], 0.0, accuracy: 0.001)
        XCTAssertEqual(scaled[1][0], 0.5, accuracy: 0.001)
        XCTAssertEqual(scaled[2][0], 1.0, accuracy: 0.001)

        // Column 1: 0.1→0, 0.5→0.5, 0.9→1
        XCTAssertEqual(scaled[0][1], 0.0, accuracy: 0.001)
        XCTAssertEqual(scaled[1][1], 0.5, accuracy: 0.001)
        XCTAssertEqual(scaled[2][1], 1.0, accuracy: 0.001)
    }

    // Test data uses training statistics, not its own min/max
    func testTrainTestSeparation() {
        let train: [[Double]] = [[0, 10], [100, 50]]
        let test: [[Double]] = [[50, 30]]

        let scaler = FeatureScaler.fit(features: train)
        let scaledTest = scaler.transform(test)

        // Column 0: train range [0, 100], so 50 → 0.5
        // Column 1: train range [10, 50], so 30 → 0.5
        XCTAssertEqual(scaledTest[0][0], 0.5, accuracy: 0.001)
        XCTAssertEqual(scaledTest[0][1], 0.5, accuracy: 0.001)
    }

    // Constant column (zero range) maps to lower bound
    func testConstantColumn() {
        let features: [[Double]] = [[5, 10], [5, 20], [5, 30]]

        let scaler = FeatureScaler.fit(features: features)
        let scaled = scaler.transform(features)

        // Column 0 is constant — all values map to 0.0
        XCTAssertEqual(scaled[0][0], 0.0)
        XCTAssertEqual(scaled[1][0], 0.0)
        XCTAssertEqual(scaled[2][0], 0.0)
    }

    // Custom target range
    func testCustomRange() {
        let features: [[Double]] = [[0], [50], [100]]

        let scaler = FeatureScaler.fit(features: features, range: -1.0...1.0)
        let scaled = scaler.transform(features)

        // 0→-1, 50→0, 100→1
        XCTAssertEqual(scaled[0][0], -1.0, accuracy: 0.001)
        XCTAssertEqual(scaled[1][0], 0.0, accuracy: 0.001)
        XCTAssertEqual(scaled[2][0], 1.0, accuracy: 0.001)
    }

    // Integration: scaler works with GaussianNaiveBayes pipeline
    func testWithNaiveBayes() {
        let features: [[Double]] = [
            [600, 15000, 0.08], [500, 78000, 0.04],
            [850, 11000, 0.45], [780, 5000, 0.50],
            [410, 95000, 0.06], [520, 180000, 0.05],
            [810, 8000, 0.40], [690, 0, 0.48]
        ]
        let labels = [1, 1, 0, 0, 1, 1, 0, 0]

        let (trainX, testX) = features.trainTestSplit(testRatio: 0.25, seed: 42)
        let (trainY, testY) = labels.trainTestSplit(testRatio: 0.25, seed: 42)

        // Fit scaler on training data only
        let scaler = FeatureScaler.fit(features: trainX)

        let model = GaussianNaiveBayes.fit(
            features: scaler.transform(trainX),
            labels: trainY
        )
        let predictions = model.predict(scaler.transform(testX))

        // Just verify it runs without crashing and produces valid labels
        XCTAssertEqual(predictions.count, testY.count)
        for p in predictions {
            XCTAssertTrue(p == 0 || p == 1)
        }
    }

    // MARK: - Equatable

    // FeatureScaler supports == comparison
    func testFeatureScalerEquatable() {
        let features: [[Double]] = [[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]

        let scaler1 = FeatureScaler.fit(features: features)
        let scaler2 = FeatureScaler.fit(features: features)
        XCTAssertEqual(scaler1, scaler2)

        // Different data produces different scalers
        let scaler3 = FeatureScaler.fit(features: [[0.0, 0.0], [10.0, 10.0]])
        XCTAssertNotEqual(scaler1, scaler3)
    }
}
