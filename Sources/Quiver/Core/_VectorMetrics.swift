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

// MARK: - Confusion Matrix

/// Captures the four outcomes of a binary classifier — true positives, false positives,
/// true negatives, and false negatives — and derives standard evaluation metrics from them.
///
/// All metrics are computed from these four counts. Precision and recall return `nil`
/// when their denominators are zero, which surfaces undefined results at the type level
/// rather than silently returning zero.
public struct ConfusionMatrix: CustomStringConvertible, Equatable {

    public var description: String {
        "TP: \(truePositives)  FP: \(falsePositives)  TN: \(trueNegatives)  FN: \(falseNegatives)  (accuracy: \(String(format: "%.1f", accuracy * 100))%)"
    }

    /// Positive samples the model correctly identified as positive.
    public let truePositives: Int

    /// Negative samples the model incorrectly identified as positive.
    public let falsePositives: Int

    /// Negative samples the model correctly identified as negative.
    public let trueNegatives: Int

    /// Positive samples the model incorrectly identified as negative.
    public let falseNegatives: Int

    /// Fraction of all predictions that were correct: (TP + TN) / total.
    ///
    /// Accuracy can be misleading on imbalanced datasets. A model that always predicts
    /// the majority class achieves high accuracy while catching none of the minority class.
    public var accuracy: Double {
        let total = truePositives + falsePositives + trueNegatives + falseNegatives
        guard total > 0 else { return 0.0 }
        return Double(truePositives + trueNegatives) / Double(total)
    }

    /// Of all predicted positives, what fraction were actually positive: TP / (TP + FP).
    ///
    /// Returns `nil` when the model predicted no positives at all, making the metric
    /// undefined. This prevents the silent zero-return behavior found in some ML
    /// libraries, where a precision of 0.0 could mean either "all predictions were wrong"
    /// or "no predictions were made" — two very different situations.
    public var precision: Double? {
        let denominator = truePositives + falsePositives
        guard denominator > 0 else { return nil }
        return Double(truePositives) / Double(denominator)
    }

    /// Of all actual positives, what fraction did the model catch: TP / (TP + FN).
    ///
    /// Returns `nil` when there are no actual positives in the data, making the metric
    /// undefined. High recall means the model misses few positive cases — critical in
    /// scenarios like malware detection or medical screening where false negatives are costly.
    public var recall: Double? {
        let denominator = truePositives + falseNegatives
        guard denominator > 0 else { return nil }
        return Double(truePositives) / Double(denominator)
    }

    /// Harmonic mean of precision and recall: 2 * P * R / (P + R).
    ///
    /// Returns `nil` when either precision or recall is undefined, or when both are zero.
    /// The harmonic mean penalizes extreme imbalances — a model with 100% precision but
    /// 1% recall scores only 0.02, not 0.505 as an arithmetic mean would suggest.
    public var f1Score: Double? {
        guard let p = precision, let r = recall else { return nil }
        let sum = p + r
        guard sum > 0 else { return nil }
        return 2.0 * p * r / sum
    }
}

// MARK: - Internal Metrics Computation

/// Internal namespace for classification metric calculations.
///
/// Separated from `_Vector` because metrics operate on integer label arrays,
/// not on numeric vectors requiring arithmetic.
internal enum _Metrics {

    /// Builds a confusion matrix by comparing predicted labels against actual labels.
    ///
    /// Counts each prediction into one of four buckets based on whether the predicted
    /// and actual labels match the designated positive label.
    ///
    /// - Parameters:
    ///   - predicted: The model's predicted class labels.
    ///   - actual: The ground truth class labels.
    ///   - positiveLabel: Which label value represents the positive class.
    /// - Returns: A `ConfusionMatrix` with the four outcome counts.
    static func confusionMatrix(predicted: [Int], actual: [Int], positiveLabel: Int) -> ConfusionMatrix {
        precondition(predicted.count == actual.count,
            "Predicted and actual arrays must have the same length")

        var tp = 0, fp = 0, tn = 0, fn = 0

        for i in 0..<predicted.count {
            let predictedPositive = predicted[i] == positiveLabel
            let actualPositive = actual[i] == positiveLabel

            if predictedPositive && actualPositive {
                tp += 1
            } else if predictedPositive && !actualPositive {
                fp += 1
            } else if !predictedPositive && actualPositive {
                fn += 1
            } else {
                tn += 1
            }
        }

        return ConfusionMatrix(
            truePositives: tp,
            falsePositives: fp,
            trueNegatives: tn,
            falseNegatives: fn
        )
    }
}
