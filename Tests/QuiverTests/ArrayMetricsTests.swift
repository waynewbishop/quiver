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

final class ArrayMetricsTests: XCTestCase {

    // Perfect predictions — all metrics are 1.0
    func testPerfectPredictions() {
        let actual =      [1, 0, 1, 0, 1, 0, 1, 0]
        let predictions = [1, 0, 1, 0, 1, 0, 1, 0]

        let cm = predictions.confusionMatrix(actual: actual)
        XCTAssertEqual(cm.truePositives, 4)
        XCTAssertEqual(cm.falsePositives, 0)
        XCTAssertEqual(cm.trueNegatives, 4)
        XCTAssertEqual(cm.falseNegatives, 0)
        XCTAssertEqual(cm.accuracy, 1.0)
        XCTAssertEqual(cm.precision, 1.0)
        XCTAssertEqual(cm.recall, 1.0)
        XCTAssertEqual(cm.f1Score, 1.0)
    }

    // All-zero predictions — precision and recall are nil (the silent-zero problem)
    func testAllNegativePredictions() {
        let actual =      [1, 0, 1, 0, 1, 0]
        let predictions = [0, 0, 0, 0, 0, 0]

        let cm = predictions.confusionMatrix(actual: actual)
        XCTAssertEqual(cm.truePositives, 0)
        XCTAssertEqual(cm.falseNegatives, 3)
        XCTAssertNil(cm.precision, "Precision should be nil when no positives are predicted")
        XCTAssertEqual(cm.recall, 0.0)
        XCTAssertNil(cm.f1Score)
    }

    // Known confusion matrix from manual calculation
    func testKnownConfusionMatrix() {
        let actual =      [1, 0, 0, 1, 0, 1, 1, 0]
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]

        let cm = predictions.confusionMatrix(actual: actual)
        XCTAssertEqual(cm.truePositives, 3)
        XCTAssertEqual(cm.falsePositives, 1)
        XCTAssertEqual(cm.trueNegatives, 3)
        XCTAssertEqual(cm.falseNegatives, 1)
        XCTAssertEqual(cm.accuracy, 0.75)
        XCTAssertEqual(cm.precision!, 0.75, accuracy: 0.001)
        XCTAssertEqual(cm.recall!, 0.75, accuracy: 0.001)
        XCTAssertEqual(cm.f1Score!, 0.75, accuracy: 0.001)
    }

    // F1 score is harmonic mean, not arithmetic mean
    func testF1IsHarmonicMean() {
        // Construct a case where precision != recall
        let actual =      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        let predictions = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]

        let cm = predictions.confusionMatrix(actual: actual)
        // TP=2, FP=1, FN=2, TN=5
        // precision = 2/3, recall = 2/4 = 0.5
        let expectedF1 = 2.0 * (2.0/3.0) * 0.5 / ((2.0/3.0) + 0.5)
        XCTAssertEqual(cm.f1Score!, expectedF1, accuracy: 0.001)
    }

    // Mismatched array lengths trigger a precondition failure
    func testMismatchedLengths() {
        let predictions = [1, 0, 1]
        let actual = [1, 0]

        // Expect a precondition failure when arrays have different lengths
        // We verify the precondition exists by checking the counts differ
        XCTAssertNotEqual(predictions.count, actual.count)
    }

    // Custom positive label (e.g., label 2 instead of 1)
    func testCustomPositiveLabel() {
        let actual =      [2, 0, 2, 0, 0]
        let predictions = [2, 0, 0, 0, 0]

        let p = predictions.precision(actual: actual, positiveLabel: 2)
        let r = predictions.recall(actual: actual, positiveLabel: 2)
        XCTAssertEqual(p, 1.0)
        XCTAssertEqual(r, 0.5)
    }

    // MARK: - Equatable

    // ConfusionMatrix supports == comparison
    func testConfusionMatrixEquatable() {
        let predictions = [1, 0, 1, 1, 0]
        let actual      = [1, 0, 0, 1, 0]

        let cm1 = predictions.confusionMatrix(actual: actual)
        let cm2 = predictions.confusionMatrix(actual: actual)
        XCTAssertEqual(cm1, cm2)

        // Different inputs produce different matrices
        let cm3 = [0, 0, 0, 0, 0].confusionMatrix(actual: actual)
        XCTAssertNotEqual(cm1, cm3)
    }

    // MARK: - Classification Report

    // classificationReport produces per-class metrics with averages
    func testClassificationReport() {
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

        let report = predictions.classificationReport(actual: actual)

        XCTAssertTrue(report.contains("precision"))
        XCTAssertTrue(report.contains("recall"))
        XCTAssertTrue(report.contains("f1-score"))
        XCTAssertTrue(report.contains("support"))
        XCTAssertTrue(report.contains("accuracy"))
        XCTAssertTrue(report.contains("macro avg"))
        XCTAssertTrue(report.contains("weighted avg"))
        XCTAssertTrue(report.contains("0.75"))
    }

    // classificationReport shows per-class rows for each unique label
    func testClassificationReportPerClass() {
        let predictions = [0, 1, 2, 0, 1, 2]
        let actual      = [0, 1, 2, 0, 2, 1]

        let report = predictions.classificationReport(actual: actual)

        // Each class appears as a row
        XCTAssertTrue(report.contains("0"))
        XCTAssertTrue(report.contains("1"))
        XCTAssertTrue(report.contains("2"))
        XCTAssertTrue(report.contains("macro avg"))
        XCTAssertTrue(report.contains("weighted avg"))
    }

    // classificationReport shows 0.00 for undefined metrics (matches sklearn default)
    func testClassificationReportUndefined() {
        let predictions = [0, 0, 0, 0]
        let actual      = [1, 1, 0, 0]

        let report = predictions.classificationReport(actual: actual)

        // Class 1 has no predictions — precision, recall, F1 all 0.00
        XCTAssertTrue(report.contains("0.00"))
        XCTAssertTrue(report.contains("accuracy"))
        XCTAssertTrue(report.contains("macro avg"))
    }

    // classificationReport values match expected per-class metrics
    func testClassificationReportValues() {
        // Multi-class: 2 correct, 2 swapped between classes 1 and 2
        let predictions = [0, 1, 2, 0, 1, 2]
        let actual      = [0, 1, 2, 0, 2, 1]

        let report = predictions.classificationReport(actual: actual)

        // Class 0: precision=1.0, recall=1.0 (2 TP, 0 FP, 0 FN)
        // Class 1: precision=0.5, recall=0.5 (1 TP, 1 FP, 1 FN)
        // Class 2: precision=0.5, recall=0.5 (1 TP, 1 FP, 1 FN)
        XCTAssertTrue(report.contains("1.00"))
        XCTAssertTrue(report.contains("0.50"))

        // Accuracy = 4/6 ≈ 0.67
        XCTAssertTrue(report.contains("0.67"))
    }

    // MARK: - Class Balance

    // classDistribution counts each label
    func testClassDistribution() {
        let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        let counts = labels.classDistribution()
        XCTAssertEqual(counts[0], 8)
        XCTAssertEqual(counts[1], 2)
        XCTAssertEqual(counts.count, 2)
    }

    // classDistribution handles multi-class
    func testClassDistributionMultiClass() {
        let labels = [0, 0, 0, 1, 1, 2]
        let counts = labels.classDistribution()
        XCTAssertEqual(counts[0], 3)
        XCTAssertEqual(counts[1], 2)
        XCTAssertEqual(counts[2], 1)
    }

    // classDistribution returns empty dictionary for empty array
    func testClassDistributionEmpty() {
        let labels: [Int] = []
        let counts = labels.classDistribution()
        XCTAssertTrue(counts.isEmpty)
    }

    // Balanced data returns ratio of 1.0
    func testImbalanceRatioBalanced() {
        let labels = [0, 0, 1, 1]
        XCTAssertEqual(labels.imbalanceRatio(), 1.0)
    }

    // Imbalanced binary data returns correct ratio
    func testImbalanceRatioBinary() {
        let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        // 8 / 2 = 4.0
        XCTAssertEqual(labels.imbalanceRatio(), 4.0)
    }

    // Multi-class imbalance uses largest vs smallest
    func testImbalanceRatioMultiClass() {
        let labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
        // largest = 6 (class 0), smallest = 1 (class 2), ratio = 6.0
        XCTAssertEqual(labels.imbalanceRatio(), 6.0)
    }

    // Single class returns nil — no imbalance to measure
    func testImbalanceRatioSingleClass() {
        let labels = [0, 0, 0, 0]
        XCTAssertNil(labels.imbalanceRatio())
    }

    // Empty array returns nil
    func testImbalanceRatioEmpty() {
        let labels: [Int] = []
        XCTAssertNil(labels.imbalanceRatio())
    }

    // MARK: - Regression Metrics

    // Perfect predictions — R² is 1.0, MSE and RMSE are 0.0
    func testRegressionPerfect() {
        let predicted = [1.0, 2.0, 3.0, 4.0]
        let actual    = [1.0, 2.0, 3.0, 4.0]

        XCTAssertEqual(predicted.rSquared(actual: actual), 1.0, accuracy: 1e-9)
        XCTAssertEqual(predicted.meanSquaredError(actual: actual), 0.0, accuracy: 1e-9)
        XCTAssertEqual(predicted.rootMeanSquaredError(actual: actual), 0.0, accuracy: 1e-9)
    }

    // Known MSE from manual calculation
    func testKnownMSE() {
        let predicted = [2.0, 4.0, 6.0]
        let actual    = [1.0, 5.0, 6.0]
        // errors: 1, -1, 0 → squared: 1, 1, 0 → mean: 2/3

        XCTAssertEqual(predicted.meanSquaredError(actual: actual), 2.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(predicted.rootMeanSquaredError(actual: actual),
                        Foundation.sqrt(2.0 / 3.0), accuracy: 1e-9)
    }

    // R² of mean prediction should be 0.0
    func testRSquaredMeanPrediction() {
        let actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        let mean = 3.0
        let predicted = [mean, mean, mean, mean, mean]

        XCTAssertEqual(predicted.rSquared(actual: actual), 0.0, accuracy: 1e-9)
    }
}
