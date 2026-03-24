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

import Foundation

// MARK: - Classification Metrics

public extension Array where Element == Int {

    /// Builds a confusion matrix comparing these predictions against actual labels.
    ///
    /// The calling array is treated as predicted labels and the `actual` parameter
    /// provides the ground truth. This design uses Swift's labeled parameters to
    /// prevent the argument-swap bugs common in positional APIs — the predictions
    /// are always `self` and the ground truth is always `actual:`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
    /// let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
    ///
    /// let cm = predictions.confusionMatrix(actual: actual)
    /// // cm.truePositives  = 3
    /// // cm.falsePositives = 1
    /// // cm.trueNegatives  = 3
    /// // cm.falseNegatives = 1
    /// // cm.precision      = Optional(0.75)
    /// // cm.recall         = Optional(0.75)
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: A ``ConfusionMatrix`` containing the four outcome counts and derived metrics.
    func confusionMatrix(actual: [Int], positiveLabel: Int = 1) -> ConfusionMatrix {
        return _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
    }

    /// Fraction of predictions that match the actual labels: (TP + TN) / total.
    ///
    /// Accuracy can be misleading on imbalanced datasets. If 95% of samples belong to
    /// one class, a model that always predicts that class achieves 95% accuracy while
    /// providing no useful discrimination. Use ``precision(actual:positiveLabel:)`` and
    /// ``recall(actual:positiveLabel:)`` for a more complete picture.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 1, 0, 0]
    /// let actual      = [1, 0, 0, 0, 0]
    /// let acc = predictions.accuracy(actual: actual)  // 0.8
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: The accuracy as a value between 0 and 1.
    func accuracy(actual: [Int], positiveLabel: Int = 1) -> Double {
        let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
        return cm.accuracy
    }

    /// Of all predicted positives, what fraction were correct: TP / (TP + FP).
    ///
    /// Returns `nil` when the model predicted no positives at all, making the denominator
    /// zero. This surfaces the problem at the type level — callers must handle the `nil`
    /// case explicitly rather than silently receiving 0.0.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [0, 0, 0, 0, 0]  // predicted no positives
    /// let actual      = [1, 0, 1, 0, 0]
    /// let p = predictions.precision(actual: actual)  // nil — undefined
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: Precision as a value between 0 and 1, or `nil` if undefined.
    func precision(actual: [Int], positiveLabel: Int = 1) -> Double? {
        let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
        return cm.precision
    }

    /// Of all actual positives, what fraction did the model catch: TP / (TP + FN).
    ///
    /// Returns `nil` when there are no actual positives in the data. High recall is
    /// critical in scenarios where missing a positive case is costly — malware detection,
    /// medical screening, or customer churn prediction.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 0, 1, 0, 0, 1, 0]
    /// let actual      = [1, 0, 1, 1, 0, 1, 1, 0]
    /// let r = predictions.recall(actual: actual)  // Optional(0.6)
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: Recall as a value between 0 and 1, or `nil` if undefined.
    func recall(actual: [Int], positiveLabel: Int = 1) -> Double? {
        let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
        return cm.recall
    }

    /// Harmonic mean of precision and recall: 2 * P * R / (P + R).
    ///
    /// Returns `nil` when either precision or recall is undefined, or when both are zero.
    /// The harmonic mean penalizes extreme imbalances between precision and recall more
    /// heavily than an arithmetic mean would.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
    /// let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
    /// let f1 = predictions.f1Score(actual: actual)  // Optional(0.75)
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: F1 score as a value between 0 and 1, or `nil` if undefined.
    func f1Score(actual: [Int], positiveLabel: Int = 1) -> Double? {
        let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
        return cm.f1Score
    }

    /// Returns a formatted summary of classification metrics comparing these predictions
    /// against actual labels.
    ///
    /// Computes accuracy, precision, recall, and F1 score in one call and returns them
    /// as a readable multi-line string. Metrics that are undefined (e.g., precision when
    /// no positives were predicted) display as "N/A".
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
    /// let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
    /// print(predictions.classificationReport(actual: actual))
    /// // Accuracy:  75.0%
    /// // Precision: 0.75
    /// // Recall:    0.75
    /// // F1 Score:  0.75
    /// ```
    ///
    /// - Parameters:
    ///   - actual: The ground truth class labels, one per sample.
    ///   - positiveLabel: The label value representing the positive class. Defaults to `1`.
    /// - Returns: A formatted string containing accuracy, precision, recall, and F1 score.
    func classificationReport(actual: [Int], positiveLabel: Int = 1) -> String {
        let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: positiveLabel)
        let accLine = "Accuracy:  \(String(format: "%.1f", cm.accuracy * 100))%"
        let precLine = "Precision: \(cm.precision.map { String(format: "%.2f", $0) } ?? "N/A")"
        let recLine = "Recall:    \(cm.recall.map { String(format: "%.2f", $0) } ?? "N/A")"
        let f1Line = "F1 Score:  \(cm.f1Score.map { String(format: "%.2f", $0) } ?? "N/A")"
        return [accLine, precLine, recLine, f1Line].joined(separator: "\n")
    }
}

// MARK: - Regression Metrics

public extension Array where Element == Double {

    /// Coefficient of determination R² comparing these predicted values against actuals.
    ///
    /// R² = 1 − SS_res / SS_tot measures how well predictions explain the variance
    /// in the actual data. A value of 1.0 means perfect prediction; 0.0 means the
    /// model explains no more variance than predicting the mean.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predicted = [2.1, 4.0, 5.9, 8.1]
    /// let actual    = [2.0, 4.0, 6.0, 8.0]
    /// let r2 = predicted.rSquared(actual: actual)  // ≈ 0.999
    /// ```
    ///
    /// - Parameter actual: The ground truth target values, one per sample.
    /// - Returns: R² as a value between 0 and 1 (can be negative for very poor models).
    func rSquared(actual: [Double]) -> Double {
        precondition(count == actual.count,
            "Predicted and actual arrays must have the same length")
        return _Regression.rSquared(predicted: self, actual: actual)
    }

    /// Mean squared error comparing these predicted values against actuals.
    ///
    /// MSE = Σ(yᵢ − ŷᵢ)² / n averages the squared differences between predicted
    /// and actual values. Lower is better; zero means perfect prediction.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predicted = [2.5, 4.0, 6.0]
    /// let actual    = [2.0, 4.0, 6.0]
    /// let mse = predicted.meanSquaredError(actual: actual)  // ≈ 0.083
    /// ```
    ///
    /// - Parameter actual: The ground truth target values, one per sample.
    /// - Returns: The mean squared error (always non-negative).
    func meanSquaredError(actual: [Double]) -> Double {
        precondition(count == actual.count,
            "Predicted and actual arrays must have the same length")
        return _Regression.meanSquaredError(predicted: self, actual: actual)
    }

    /// Root mean squared error comparing these predicted values against actuals.
    ///
    /// RMSE = √MSE provides an error metric in the same units as the target variable,
    /// making it more interpretable than MSE for reporting.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predicted = [2.5, 4.0, 6.0]
    /// let actual    = [2.0, 4.0, 6.0]
    /// let rmse = predicted.rootMeanSquaredError(actual: actual)  // ≈ 0.289
    /// ```
    ///
    /// - Parameter actual: The ground truth target values, one per sample.
    /// - Returns: The root mean squared error (always non-negative).
    func rootMeanSquaredError(actual: [Double]) -> Double {
        return Foundation.sqrt(meanSquaredError(actual: actual))
    }
}
