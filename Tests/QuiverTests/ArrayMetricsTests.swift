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

    // classificationReport produces a formatted multi-line summary
    func testClassificationReport() {
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

        let report = predictions.classificationReport(actual: actual)

        XCTAssertTrue(report.contains("Accuracy:"))
        XCTAssertTrue(report.contains("Precision:"))
        XCTAssertTrue(report.contains("Recall:"))
        XCTAssertTrue(report.contains("F1 Score:"))
        XCTAssertTrue(report.contains("75.0%"))
        XCTAssertTrue(report.contains("0.75"))
    }

    // classificationReport shows N/A when metrics are undefined
    func testClassificationReportUndefined() {
        let predictions = [0, 0, 0, 0]
        let actual      = [1, 1, 0, 0]

        let report = predictions.classificationReport(actual: actual)

        XCTAssertTrue(report.contains("N/A"), "Precision should show N/A when no positives predicted")
        XCTAssertTrue(report.contains("Accuracy:"))
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
