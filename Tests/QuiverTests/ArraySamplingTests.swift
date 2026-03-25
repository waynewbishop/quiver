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

final class ArraySamplingTests: XCTestCase {

    // Split preserves all elements with correct partition sizes
    func testBasicPartition() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let (train, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)

        XCTAssertEqual(train.count, 8)
        XCTAssertEqual(test.count, 2)

        // Verify no elements lost or duplicated
        var all = train
        all.append(contentsOf: test)
        XCTAssertEqual(Set(all), Set(data))
    }

    // Same seed produces identical results every time
    func testReproducibility() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        let first = data.trainTestSplit(testRatio: 0.2, seed: 42)
        let second = data.trainTestSplit(testRatio: 0.2, seed: 42)

        XCTAssertEqual(first.train, second.train)
        XCTAssertEqual(first.test, second.test)
    }

    // Same seed on paired arrays keeps elements aligned
    func testPairedArrayConsistency() {
        let features = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        let labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        let (trainX, testX) = features.trainTestSplit(testRatio: 0.2, seed: 7)
        let (trainY, testY) = labels.trainTestSplit(testRatio: 0.2, seed: 7)

        XCTAssertEqual(trainX.count, trainY.count)
        XCTAssertEqual(testX.count, testY.count)

        // Verify alignment — each label should match its feature's original index
        for i in 0..<trainX.count {
            let featureIndex = Int(trainX[i] / 10.0) - 1
            let expectedLabel = labels[featureIndex]
            XCTAssertEqual(trainY[i], expectedLabel)
        }
    }

    // Different seeds produce different splits
    func testDifferentSeedsDiffer() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        let first = data.trainTestSplit(testRatio: 0.2, seed: 42)
        let second = data.trainTestSplit(testRatio: 0.2, seed: 99)

        XCTAssertNotEqual(first.train, second.train)
    }

    // MARK: - Stratified Split

    // Stratified split preserves class ratios in both partitions
    func testStratifiedPreservesRatio() {
        let features = Array(0..<100)
        // 80 class-0, 20 class-1 (imbalanced)
        var labels = Array(repeating: 0, count: 80)
        labels.append(contentsOf: Array(repeating: 1, count: 20))

        let split = features.stratifiedSplit(labels: labels, testRatio: 0.2, seed: 42)

        // Total counts correct
        XCTAssertEqual(split.trainFeatures.count + split.testFeatures.count, 100)
        XCTAssertEqual(split.trainLabels.count + split.testLabels.count, 100)

        // Test set should have proportional representation
        let testClass0 = split.testLabels.filter { $0 == 0 }.count
        let testClass1 = split.testLabels.filter { $0 == 1 }.count

        // 80 * 0.2 = 16, 20 * 0.2 = 4
        XCTAssertEqual(testClass0, 16)
        XCTAssertEqual(testClass1, 4)
    }

    // Stratified split is reproducible with same seed
    func testStratifiedReproducibility() {
        let features: [[Double]] = [[1], [2], [3], [4], [5], [6], [7], [8]]
        let labels = [0, 0, 0, 0, 1, 1, 1, 1]

        let first = features.stratifiedSplit(labels: labels, testRatio: 0.25, seed: 42)
        let second = features.stratifiedSplit(labels: labels, testRatio: 0.25, seed: 42)

        XCTAssertEqual(first.trainLabels, second.trainLabels)
        XCTAssertEqual(first.testLabels, second.testLabels)
    }

    // Stratified split keeps features and labels aligned
    func testStratifiedAlignment() {
        let features: [[Double]] = [[10], [20], [30], [40], [50], [60]]
        let labels = [0, 0, 0, 1, 1, 1]

        let split = features.stratifiedSplit(labels: labels, testRatio: 0.33, seed: 7)

        // Each class should appear in the test set
        XCTAssertTrue(split.testLabels.contains(0))
        XCTAssertTrue(split.testLabels.contains(1))

        // Features and labels must stay aligned
        XCTAssertEqual(split.trainFeatures.count, split.trainLabels.count)
        XCTAssertEqual(split.testFeatures.count, split.testLabels.count)
    }

    // Various ratios produce correct partition sizes
    func testRatioSizes() {
        let data = Array(0..<100)

        let split10 = data.trainTestSplit(testRatio: 0.1, seed: 1)
        XCTAssertEqual(split10.test.count, 10)
        XCTAssertEqual(split10.train.count, 90)

        let split50 = data.trainTestSplit(testRatio: 0.5, seed: 1)
        XCTAssertEqual(split50.test.count, 50)
        XCTAssertEqual(split50.train.count, 50)

        let split90 = data.trainTestSplit(testRatio: 0.9, seed: 1)
        XCTAssertEqual(split90.test.count, 90)
        XCTAssertEqual(split90.train.count, 10)
    }

    // MARK: - Oversample

    // Oversampling balances classes to match the majority count
    func testOversampleBalancesBinaryClasses() {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [1.2, 2.1],
            [7.0, 8.0], [7.5, 8.5]
        ]
        let labels = [0, 0, 0, 0, 1, 1]

        let (balanced, balancedLabels) = features.oversample(labels: labels)

        // Both classes should now have 4 samples
        let class0 = balancedLabels.filter { $0 == 0 }.count
        let class1 = balancedLabels.filter { $0 == 1 }.count
        XCTAssertEqual(class0, 4)
        XCTAssertEqual(class1, 4)
        XCTAssertEqual(balanced.count, balancedLabels.count)
    }

    // Oversampling handles multi-class data
    func testOversampleMultiClass() {
        let features: [[Double]] = [
            [1.0, 1.0], [1.5, 1.5], [1.2, 1.2],
            [5.0, 5.0], [5.5, 5.5],
            [9.0, 9.0]
        ]
        let labels = [0, 0, 0, 1, 1, 2]

        let (balanced, balancedLabels) = features.oversample(labels: labels)

        // Total should be 9 (3 classes × 3 each)
        XCTAssertEqual(balanced.count, 9)

        // All classes should match the largest count (3)
        let class0 = balancedLabels.filter { $0 == 0 }.count
        let class1 = balancedLabels.filter { $0 == 1 }.count
        let class2 = balancedLabels.filter { $0 == 2 }.count
        XCTAssertEqual(class0, 3)
        XCTAssertEqual(class1, 3)
        XCTAssertEqual(class2, 3)
    }

    // Already balanced data returns unchanged
    func testOversampleAlreadyBalanced() {
        let features: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        let labels = [0, 0, 1, 1]

        let (balanced, balancedLabels) = features.oversample(labels: labels)

        XCTAssertEqual(balanced.count, 4)
        XCTAssertEqual(balancedLabels, labels)
    }

    // Synthetic points have the correct dimensionality
    func testOversampleDimensions() {
        let features: [[Double]] = [
            [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0],
            [8.0, 9.0, 10.0]
        ]
        let labels = [0, 0, 0, 1]

        let (balanced, _) = features.oversample(labels: labels)

        // Every row should have 3 columns
        for row in balanced {
            XCTAssertEqual(row.count, 3)
        }
    }
}
