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

import Foundation

// MARK: - Bernoulli Naive Bayes

/// A trained Bernoulli Naive Bayes classifier for yes/no features.
///
/// Bernoulli Naive Bayes treats every feature as a binary event — present (`1.0`)
/// or absent (`0.0`) — such as whether a word ends in "-tion" or whether an email
/// contains the word "free". During training, the model learns the probability
/// that each feature is present within each class, along with class prior
/// probabilities. During prediction, it applies Bayes' theorem and returns the
/// most likely class. Both presence and absence count as evidence: a feature that
/// is usually present in a class lowers that class's score when it is missing.
///
/// Feature probabilities are smoothed with additive (Laplace) smoothing, so a
/// feature never seen in a class receives a small but non-zero probability rather
/// than ruling the class out entirely. Any number of classes is supported.
///
/// This is a value type — once created via ``fit(features:labels:alpha:)``, the
/// model is immutable. There is no separate "unfitted" state, which eliminates
/// the common bug of calling predict before fit.
///
/// Example:
/// ```swift
/// import Quiver
///
/// // Features: [ends in "-tion", ends in "-eau"]
/// // Labels: 0 = masculine, 1 = feminine
/// let features: [[Double]] = [
///     [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]
/// ]
/// let labels = [1, 1, 0, 0]
///
/// let model = BernoulliNaiveBayes.fit(features: features, labels: labels)
/// let predictions = model.predict([[1.0, 0.0], [0.0, 1.0]])
/// // [1, 0]
/// ```
public struct BernoulliNaiveBayes: Classifier, Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let classLabel = classes.count == 1 ? "class" : "classes"
        let featureLabel = featureCount == 1 ? "feature" : "features"
        let alphaFormatted = String(format: "%g", alpha)
        return "BernoulliNaiveBayes: \(classes.count) \(classLabel), \(featureCount) \(featureLabel), alpha: \(alphaFormatted)"
    }

    /// Probabilities learned from the training data for a single class.
    ///
    /// Contains the prior probability (relative frequency), the smoothed
    /// probability that each feature is present, and the number of training
    /// samples belonging to this class.
    public struct ClassStats: Codable, CustomStringConvertible, Equatable, Sendable {

        public var description: String {
            let probabilitiesFormatted = featureProbabilities
                .map { String(format: "%.2f", $0) }
                .joined(separator: ", ")
            return "Class \(label): prior \(String(format: "%.1f", prior * 100))%, feature probabilities [\(probabilitiesFormatted)], \(count) samples"
        }

        /// The class label this entry represents.
        public let label: Int
        /// Prior probability P(class), computed as count / totalSamples.
        public let prior: Double
        /// Smoothed probability P(feature = 1 | class) for each feature.
        ///
        /// Computed as (presentCount + alpha) / (count + 2·alpha), where
        /// presentCount is the number of samples in this class with the
        /// feature set to `1.0`. Always strictly between 0 and 1.
        public let featureProbabilities: [Double]
        /// Number of training samples belonging to this class.
        public let count: Int
    }

    /// The learned probabilities for each class, one entry per unique label,
    /// sorted by ascending label.
    public let classes: [ClassStats]

    /// Number of features the model was trained on.
    public let featureCount: Int

    /// The additive smoothing strength used during fitting.
    public let alpha: Double

    /// Fits a Bernoulli Naive Bayes model to the given binary training data.
    ///
    /// Counts how often each feature is present within each class and converts
    /// the counts to smoothed probabilities:
    ///
    /// P(feature = 1 | class) = (presentCount + alpha) / (classCount + 2·alpha)
    ///
    /// The default `alpha` of `1.0` is Laplace smoothing — equivalent to adding
    /// one imaginary sample with the feature present and one with it absent to
    /// every class. Smaller values trust the observed counts more closely.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a
    ///     feature. Every value must be exactly `0.0` or `1.0`.
    ///   - labels: 1D array of integer class labels, one per sample.
    ///   - alpha: Additive smoothing strength. Must be positive and finite.
    /// - Complexity: O(*n*·*f*) where *n* is the number of samples and *f* is
    ///   the feature count. Counts per-class feature presence in a single pass.
    /// - Returns: A trained ``BernoulliNaiveBayes`` model.
    public static func fit(features: [[Double]], labels: [Int], alpha: Double = 1.0) -> BernoulliNaiveBayes {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == labels.count,
            "Features and labels must have the same number of samples")
        precondition(alpha > 0.0 && alpha.isFinite,
            "Smoothing alpha must be positive and finite, got \(alpha)")

        let featureCount = features[0].count
        for sample in features {
            precondition(sample.count == featureCount,
                "Every sample must have \(featureCount) features, found \(sample.count)")
            _preconditionBinary(sample)
        }

        let totalSamples = Double(features.count)

        // Count feature presence per class, one entry per label in ascending order
        let classStats = _NaiveBayes.sampleIndicesGroupedByLabel(labels).map { (label, indices) -> ClassStats in
            let count = indices.count

            var presentCounts = [Double](repeating: 0.0, count: featureCount)
            for sampleIndex in indices {
                for f in 0..<featureCount {
                    presentCounts[f] += features[sampleIndex][f]
                }
            }

            // P(xⱼ = 1 | c) = (Nⱼ + α) / (N_c + 2α)
            let denominator = Double(count) + 2.0 * alpha
            var featureProbabilities = [Double](repeating: 0.0, count: featureCount)
            for f in 0..<featureCount {
                featureProbabilities[f] = (presentCounts[f] + alpha) / denominator
            }

            return ClassStats(
                label: label,
                prior: Double(count) / totalSamples,
                featureProbabilities: featureProbabilities,
                count: count
            )
        }

        return BernoulliNaiveBayes(classes: classStats, featureCount: featureCount, alpha: alpha)
    }

    /// Predicts class labels for one or more samples.
    ///
    /// For each sample, computes log P(class) + Σ log P(feature_i | class) for
    /// every class and returns the class with the highest log-probability. Working
    /// in log-space avoids the floating-point underflow that occurs when multiplying
    /// many small probabilities together.
    ///
    /// - Complexity: O(*q*·*c*·*f*) where *q* is the number of query samples,
    ///   *c* is the number of classes, and *f* is the feature count.
    /// - Parameter features: 2D array where each row is a sample to classify.
    ///   Every value must be exactly `0.0` or `1.0`.
    /// - Returns: An array of predicted class labels, one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        let labels = classes.map { $0.label }
        return predictLogProbabilities(features).map { logJointProbabilities in
            _NaiveBayes.mostProbableLabel(
                logJointProbabilities: logJointProbabilities,
                labels: labels
            )
        }
    }

    /// Returns log-probabilities for each class for each sample.
    ///
    /// The returned array has one row per sample, with each row containing
    /// the unnormalized log-probability for each class (in the same order
    /// as the ``classes`` array).
    ///
    /// - Complexity: O(*q*·*c*·*f*) where *q* is the number of query samples,
    ///   *c* is the number of classes, and *f* is the feature count.
    /// - Parameter features: 2D array where each row is a sample to classify.
    ///   Every value must be exactly `0.0` or `1.0`.
    /// - Returns: 2D array of log-probabilities, shape [samples × classes].
    public func predictLogProbabilities(_ features: [[Double]]) -> [[Double]] {
        return features.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            Self._preconditionBinary(sample)
            return classes.map { classStats in
                _logProbability(sample: sample, classStats: classStats)
            }
        }
    }

    /// Returns normalized posterior class probability estimates for each sample.
    ///
    /// Applies the softmax transform to the unnormalized log-probabilities from
    /// ``predictLogProbabilities(_:)``, producing a probability distribution
    /// across classes that sums to 1.0 for each row. Normalization does not make
    /// the values calibrated; Naive Bayes is often overconfident because it
    /// treats correlated features as independent evidence.
    ///
    /// The output ordering matches the ``classes`` array. For row `i` and class
    /// index `c`, `result[i][c]` is the model's estimated `P(class = classes[c] | features[i])`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let model = BernoulliNaiveBayes.fit(features: trainX, labels: trainY)
    /// let probabilities = model.predictProbabilities([[1.0, 0.0]])
    /// // probabilities[0] sums to 1.0 across classes
    /// ```
    ///
    /// - Complexity: O(*q*·*c*·*f*) where *q* is the number of query samples,
    ///   *c* is the number of classes, and *f* is the feature count.
    /// - Parameter features: 2D array where each row is a sample to classify.
    ///   Every value must be exactly `0.0` or `1.0`.
    /// - Returns: 2D array of class probabilities, shape [samples × classes].
    ///   Each row sums to 1.0.
    public func predictProbabilities(_ features: [[Double]]) -> [[Double]] {
        return predictLogProbabilities(features).map { logProbabilities in
            logProbabilities.softMax()
        }
    }

    // MARK: - Private Helpers

    /// Traps when any value in the sample is not exactly 0.0 or 1.0.
    private static func _preconditionBinary(_ sample: [Double]) {
        for value in sample {
            precondition(value == 0.0 || value == 1.0,
                "BernoulliNaiveBayes features must be 0.0 or 1.0, found \(value)")
        }
    }

    /// Computes log P(class) + Σ log P(x_i | class) for a sample and class.
    ///
    /// A present feature contributes log p and an absent feature contributes
    /// log(1 − p), where p is the class's smoothed feature probability. Smoothing
    /// keeps every p strictly inside (0, 1), so both logarithms are finite.
    private func _logProbability(sample: [Double], classStats: ClassStats) -> Double {
        var logProbability = Foundation.log(classStats.prior)

        for f in 0..<featureCount {
            let p = classStats.featureProbabilities[f]
            if sample[f] == 1.0 {
                logProbability += Foundation.log(p)
            } else {
                // log1p(−p) = log(1 − p), accurate when p is small
                logProbability += Foundation.log1p(-p)
            }
        }

        return logProbability
    }
}
