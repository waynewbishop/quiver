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

final class NaiveBayesTests: XCTestCase {

    // Two well-separated clusters — model should achieve perfect accuracy
    func testPerfectSeparation() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1], [0.8, 1.9],
            [8.0, 9.0], [8.5, 8.8], [8.2, 9.1], [7.8, 8.9]
        ]
        let labels = [0, 0, 0, 0, 1, 1, 1, 1]

        let model = GaussianNaiveBayes.fit(features: features, labels: labels)
        let predictions = model.predict(features)

        XCTAssertEqual(predictions, labels)
    }

    // Verify fitted class statistics match expected values
    func testFitClassStats() {
        let features: [[Double]] = [
            [2.0, 4.0], [4.0, 6.0],  // class 0: means [3, 5]
            [10.0, 20.0], [12.0, 22.0]  // class 1: means [11, 21]
        ]
        let labels = [0, 0, 1, 1]

        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        XCTAssertEqual(model.classes.count, 2)
        XCTAssertEqual(model.featureCount, 2)

        let class0 = model.classes[0]
        XCTAssertEqual(class0.label, 0)
        XCTAssertEqual(class0.prior, 0.5)
        XCTAssertEqual(class0.means[0], 3.0, accuracy: 0.001)
        XCTAssertEqual(class0.means[1], 5.0, accuracy: 0.001)
        XCTAssertEqual(class0.count, 2)

        let class1 = model.classes[1]
        XCTAssertEqual(class1.label, 1)
        XCTAssertEqual(class1.means[0], 11.0, accuracy: 0.001)
        XCTAssertEqual(class1.means[1], 21.0, accuracy: 0.001)
    }

    // Mixed-scale features should not produce all-same predictions
    // (reproduces the lecture's unscaled data problem — log-space arithmetic handles it)
    func testMixedScaleFeatures() {
        // Feature 0: small scale (0-1), Feature 1: large scale (100000-300000)
        var features: [[Double]] = []
        var labels: [Int] = []

        // Class 0: low loyalty, high balance
        for _ in 0..<50 {
            features.append([Double.random(in: 0.01...0.15), Double.random(in: 150000...250000)])
            labels.append(0)
        }
        // Class 1: high loyalty, low balance
        for _ in 0..<50 {
            features.append([Double.random(in: 0.35...0.55), Double.random(in: 10000...80000)])
            labels.append(1)
        }

        let model = GaussianNaiveBayes.fit(features: features, labels: labels)
        let predictions = model.predict(features)

        // Model should predict at least some of each class (not all zeros)
        let uniquePredictions = Set(predictions)
        XCTAssertTrue(uniquePredictions.contains(0), "Model should predict class 0")
        XCTAssertTrue(uniquePredictions.contains(1), "Model should predict class 1")

        // Accuracy should be well above random (50%)
        let correct = zip(predictions, labels).filter { $0 == $1 }.count
        let accuracy = Double(correct) / Double(labels.count)
        XCTAssertGreaterThan(accuracy, 0.8)
    }

    // Imbalanced priors — model should still predict minority class when features are clear
    func testImbalancedPriors() {
        // 90% class 0, 10% class 1 — but class 1 features are distinct
        var features: [[Double]] = []
        var labels: [Int] = []

        for _ in 0..<90 {
            features.append([Double.random(in: 0.0...3.0)])
            labels.append(0)
        }
        for _ in 0..<10 {
            features.append([Double.random(in: 10.0...13.0)])
            labels.append(1)
        }

        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        // A clear class-1 sample should be predicted as 1 despite the imbalanced prior
        let prediction = model.predict([[11.0]])
        XCTAssertEqual(prediction, [1])
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
        let model = GaussianNaiveBayes.fit(features: trainX, labels: trainY)
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
            [1.0, 2.0], [2.0, 3.0], [1.5, 2.5],
            [8.0, 9.0], [9.0, 8.0], [8.5, 8.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]
        let model = GaussianNaiveBayes.fit(features: features, labels: labels)

        let results = model.classify([[1.0, 1.5], [8.0, 8.0], [9.0, 9.0]])

        // Should have two groups
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].label, 0)
        XCTAssertEqual(results[1].label, 1)

        // Total points across groups matches input count
        let totalPoints = results.reduce(0) { $0 + $1.count }
        XCTAssertEqual(totalPoints, 3)
    }

    // Same training data produces equal models
    func testNaiveBayesEquatable() {
        let features: [[Double]] = [[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 8.0]]
        let labels = [0, 0, 1, 1]
        let model1 = GaussianNaiveBayes.fit(features: features, labels: labels)
        let model2 = GaussianNaiveBayes.fit(features: features, labels: labels)
        XCTAssertEqual(model1, model2)
    }
}
