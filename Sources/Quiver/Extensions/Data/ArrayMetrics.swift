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

    /// Returns a formatted per-class classification report comparing these predictions
    /// against actual labels.
    ///
    /// For each class, computes precision, recall, F1 score, and support (sample count).
    /// Includes overall accuracy, macro average (unweighted mean across classes), and
    /// weighted average (weighted by support). Undefined metrics display as 0.00 in
    /// the report. The individual `precision()`, `recall()`, and `f1Score()` methods
    /// still return `nil` for programmatic access to undefined states.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
    /// let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
    /// print(predictions.classificationReport(actual: actual))
    /// //               precision    recall  f1-score   support
    /// //
    /// //            0       0.75      0.75      0.75         4
    /// //            1       0.75      0.75      0.75         4
    /// //
    /// //     accuracy                           0.75         8
    /// //    macro avg       0.75      0.75      0.75         8
    /// // weighted avg       0.75      0.75      0.75         8
    /// ```
    ///
    /// - Parameter actual: The ground truth class labels, one per sample.
    /// - Returns: A `ClassificationReport` with per-class metrics, accuracy, and macro/weighted averages.
    func classificationReport(actual: [Int]) -> ClassificationReport {
        precondition(count == actual.count,
            "Predicted and actual arrays must have the same length")

        let total = count
        let classes = Array(Set(actual + self)).sorted()

        // Compute per-class metrics using one-vs-rest
        var perClassDict: [Int: ClassMetrics] = [:]
        for cls in classes {
            let cm = _Metrics.confusionMatrix(predicted: self, actual: actual, positiveLabel: cls)
            let support = actual.filter { $0 == cls }.count
            perClassDict[cls] = ClassMetrics(
                label: cls,
                precision: cm.precision,
                recall: cm.recall,
                f1Score: cm.f1Score,
                support: support
            )
        }

        // Compute overall accuracy
        let correct = zip(self, actual).filter { $0.0 == $0.1 }.count
        let accuracy = total > 0 ? Double(correct) / Double(total) : 0.0

        // Macro average — unweighted mean of per-class metrics
        let definedP = classes.compactMap { perClassDict[$0]?.precision }
        let definedR = classes.compactMap { perClassDict[$0]?.recall }
        let definedF = classes.compactMap { perClassDict[$0]?.f1Score }
        let macroP = definedP.isEmpty ? nil : definedP.reduce(0, +) / Double(definedP.count)
        let macroR = definedR.isEmpty ? nil : definedR.reduce(0, +) / Double(definedR.count)
        let macroF = definedF.isEmpty ? nil : definedF.reduce(0, +) / Double(definedF.count)

        // Weighted average — weighted by support
        let totalSupport = classes.compactMap { perClassDict[$0]?.support }.reduce(0, +)
        let weightedP: Double? = totalSupport > 0
            ? classes.compactMap { cls -> Double? in
                guard let m = perClassDict[cls], let p = m.precision else { return nil }
                return p * Double(m.support)
            }.reduce(0, +) / Double(totalSupport)
            : nil
        let weightedR: Double? = totalSupport > 0
            ? classes.compactMap { cls -> Double? in
                guard let m = perClassDict[cls], let r = m.recall else { return nil }
                return r * Double(m.support)
            }.reduce(0, +) / Double(totalSupport)
            : nil
        let weightedF: Double? = totalSupport > 0
            ? classes.compactMap { cls -> Double? in
                guard let m = perClassDict[cls], let f = m.f1Score else { return nil }
                return f * Double(m.support)
            }.reduce(0, +) / Double(totalSupport)
            : nil

        let macroAverage = ClassMetrics(label: -1, precision: macroP, recall: macroR, f1Score: macroF, support: totalSupport)
        let weightedAverage = ClassMetrics(label: -1, precision: weightedP, recall: weightedR, f1Score: weightedF, support: totalSupport)

        return ClassificationReport(
            perClass: perClassDict,
            classOrder: classes,
            accuracy: accuracy,
            macroAverage: macroAverage,
            weightedAverage: weightedAverage,
            totalSupport: totalSupport
        )
    }

    // MARK: - Class Balance

    /// Returns the number of samples per class as a dictionary.
    ///
    /// Counts how many times each unique label appears in the array. This is
    /// the starting point for understanding class balance before training a
    /// classifier. Models trained on imbalanced data tend to predict the larger
    /// class and ignore the smaller one — knowing the distribution up front
    /// lets developers decide whether to oversample before fitting.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    /// let counts = labels.classDistribution()
    /// // 8 zeros, 2 ones (Dictionary order is not guaranteed)
    /// ```
    ///
    /// - Returns: A dictionary mapping each unique label to its sample count.
    ///   Returns an empty dictionary if the array is empty.
    func classDistribution() -> [Int: Int] {
        // Count occurrences of each label
        var counts: [Int: Int] = [:]
        for label in self {
            counts[label, default: 0] += 1
        }
        return counts
    }

    /// Returns the ratio of the largest class to the smallest class.
    ///
    /// A ratio of 1.0 means all classes have exactly the same number of samples
    /// — perfectly balanced. A ratio of 4.0 means the largest class has four
    /// times as many samples as the smallest. The higher the ratio, the greater
    /// the imbalance and the more likely a model will ignore the smaller class.
    ///
    /// Returns `nil` for empty arrays or arrays with only one class, since
    /// imbalance is only meaningful when comparing two or more classes.
    ///
    /// Unlike most ML libraries, which leave imbalance detection to manual
    /// inspection, this method gives developers a single value they can branch
    /// on — setting their own threshold based on the domain:
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ///
    /// // Check imbalance before training — threshold is up to the developer
    /// if let ratio = labels.imbalanceRatio(), ratio > 3.0 {
    ///     let (balanced, balancedLabels) = features.oversample(labels: labels)
    ///     // train on balanced data
    /// }
    /// ```
    ///
    /// Common thresholds: ratios above 2.0 warrant attention, above 3.0
    /// suggest oversampling, and above 5.0 indicate severe imbalance where
    /// evaluation metrics like accuracy become unreliable.
    ///
    /// - Returns: The ratio of the largest class count to the smallest, or `nil`
    ///   if the array is empty or contains fewer than two classes.
    func imbalanceRatio() -> Double? {
        let counts = classDistribution()

        // Imbalance requires at least two classes to compare
        guard counts.count >= 2 else { return nil }

        let values = counts.values
        guard let largest = values.max(), let smallest = values.min(),
              smallest > 0 else { return nil }

        // Ratio of 1.0 = balanced, higher = more imbalanced
        return Double(largest) / Double(smallest)
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
