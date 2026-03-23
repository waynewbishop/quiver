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

final class KNearestNeighborsTests: XCTestCase {

    // Two well-separated clusters — model should achieve perfect accuracy
    func testPerfectSeparation() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1], [0.8, 1.9],
            [8.0, 9.0], [8.5, 8.8], [8.2, 9.1], [7.8, 8.9]
        ]
        let labels = [0, 0, 0, 0, 1, 1, 1, 1]

        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)
        let predictions = model.predict(features)

        XCTAssertEqual(predictions, labels)
    }

    // Verify fitted model stores expected configuration
    func testFitStoresConfiguration() {
        let features: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let labels = [0, 1, 0]

        let model = KNearestNeighbors.fit(
            features: features, labels: labels,
            k: 1, metric: .cosine, weight: .distance
        )

        XCTAssertEqual(model.trainingFeatures.count, 3)
        XCTAssertEqual(model.trainingLabels, [0, 1, 0])
        XCTAssertEqual(model.k, 1)
        XCTAssertEqual(model.featureCount, 2)
    }

    // k=1 should return the label of the single nearest neighbor
    func testKEqualsOne() {
        let features: [[Double]] = [
            [0.0, 0.0], [10.0, 10.0]
        ]
        let labels = [0, 1]

        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 1)

        // Point near origin → class 0
        XCTAssertEqual(model.predict([[0.1, 0.1]]), [0])
        // Point near (10,10) → class 1
        XCTAssertEqual(model.predict([[9.9, 9.9]]), [1])
    }

    // Cosine metric should classify by direction, not magnitude
    func testCosineMetric() {
        // Class 0: vectors pointing "northeast" (positive x and y)
        // Class 1: vectors pointing "southeast" (positive x, negative y)
        let features: [[Double]] = [
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],      // class 0: 45° direction
            [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]     // class 1: -45° direction
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let model = KNearestNeighbors.fit(
            features: features, labels: labels, k: 3, metric: .cosine
        )

        // A large vector in the 45° direction should be class 0 (same direction)
        XCTAssertEqual(model.predict([[100.0, 100.0]]), [0])
        // A large vector in the -45° direction should be class 1
        XCTAssertEqual(model.predict([[100.0, -100.0]]), [1])
    }

    // Distance weighting should favor the closer neighbor when counts tie
    func testDistanceWeighting() {
        // Two class-0 points far away, one class-1 point very close
        let features: [[Double]] = [
            [10.0, 0.0],    // class 0 — far
            [10.0, 1.0],    // class 0 — far
            [0.1, 0.0]      // class 1 — very close
        ]
        let labels = [0, 0, 1]

        let query = [[0.0, 0.0]]

        // Uniform: 2 votes for class 0, 1 vote for class 1 → predicts 0
        let uniformModel = KNearestNeighbors.fit(
            features: features, labels: labels, k: 3, weight: .uniform
        )
        XCTAssertEqual(uniformModel.predict(query), [0])

        // Distance-weighted: class 1 at distance 0.1 gets weight 10.0,
        // class 0 at distances ~10 get weight ~0.1 each → predicts 1
        let weightedModel = KNearestNeighbors.fit(
            features: features, labels: labels, k: 3, weight: .distance
        )
        XCTAssertEqual(weightedModel.predict(query), [1])
    }

    // Full pipeline: trainTestSplit → fit → predict → confusionMatrix
    func testFullPipeline() {
        var features: [[Double]] = []
        var labels: [Int] = []

        // Generate two well-separated clusters
        for _ in 0..<60 {
            features.append([Double.random(in: 0.0...4.0), Double.random(in: 0.0...4.0)])
            labels.append(0)
        }
        for _ in 0..<40 {
            features.append([Double.random(in: 6.0...10.0), Double.random(in: 6.0...10.0)])
            labels.append(1)
        }

        // Split
        let (trainX, testX) = features.trainTestSplit(testRatio: 0.25, seed: 42)
        let (trainY, testY) = labels.trainTestSplit(testRatio: 0.25, seed: 42)

        // Fit and predict
        let model = KNearestNeighbors.fit(features: trainX, labels: trainY, k: 5)
        let predictions = model.predict(testX)

        // Evaluate
        let cm = predictions.confusionMatrix(actual: testY)
        XCTAssertGreaterThan(cm.accuracy, 0.8)
        XCTAssertNotNil(cm.precision)
        XCTAssertNotNil(cm.recall)
    }

    // MARK: - Classify

    // classify() groups inputs by predicted label
    func testClassifyGroupsByLabel() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [5.0, 8.0], [6.0, 9.0], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]
        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)

        let results = model.classify([[2.0, 3.0], [5.0, 7.0], [6.0, 8.0]])

        // Should have two groups (class 0 and class 1)
        XCTAssertEqual(results.count, 2)

        // Groups should be sorted by label
        XCTAssertEqual(results[0].label, 0)
        XCTAssertEqual(results[1].label, 1)

        // Total points across all groups should match input count
        let totalPoints = results.reduce(0) { $0 + $1.count }
        XCTAssertEqual(totalPoints, 3)
    }

    // classify() results conform to Sequence — can iterate points
    func testClassifySequenceConformance() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8],
            [5.0, 8.0], [6.0, 9.0]
        ]
        let labels = [0, 0, 1, 1]
        let model = KNearestNeighbors.fit(features: features, labels: labels, k: 2)

        let results = model.classify([[1.0, 1.5]])
        XCTAssertEqual(results.count, 1)

        // Can iterate over points in the classification group
        var pointCount = 0
        for _ in results[0] {
            pointCount += 1
        }
        XCTAssertEqual(pointCount, results[0].count)
    }
}
