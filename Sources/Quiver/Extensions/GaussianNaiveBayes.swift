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

// MARK: - Gaussian Naive Bayes

/// A trained Gaussian Naive Bayes classifier.
///
/// Gaussian Naive Bayes assumes that the features within each class follow a normal
/// (Gaussian) distribution. During training, the model learns the mean and variance
/// of each feature for each class, along with class prior probabilities. During
/// prediction, it applies Bayes' theorem to compute the most likely class for
/// each new sample.
///
/// This is a value type — once created via ``fit(features:labels:)``, the model is
/// immutable. There is no separate "unfitted" state, which eliminates the common
/// bug of calling predict before fit.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [
///     [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]
/// ]
/// let labels = [0, 0, 1, 1]
///
/// let model = GaussianNaiveBayes.fit(features: features, labels: labels)
/// let predictions = model.predict([[2.0, 2.5], [5.5, 7.0]])
/// // [0, 1]
/// ```
public struct GaussianNaiveBayes: Classifier, CustomStringConvertible, Equatable {

    public var description: String {
        "GaussianNaiveBayes: \(classes.count) classes, \(featureCount) features"
    }

    /// Statistics learned from the training data for a single class.
    ///
    /// Contains the prior probability (relative frequency), per-feature mean and
    /// variance, and the number of training samples belonging to this class.
    public struct ClassStats: CustomStringConvertible, Equatable {

        public var description: String {
            let meansStr = means.map { String(format: "%.2f", $0) }.joined(separator: ", ")
            return "Class \(label): prior \(String(format: "%.1f", prior * 100))%, means [\(meansStr)], \(count) samples"
        }

        /// The class label this entry represents.
        public let label: Int
        /// Prior probability P(class), computed as count / totalSamples.
        public let prior: Double
        /// Mean of each feature for samples in this class.
        public let means: [Double]
        /// Variance of each feature for samples in this class.
        public let variances: [Double]
        /// Number of training samples belonging to this class.
        public let count: Int
    }

    /// The learned statistics for each class, one entry per unique label.
    public let classes: [ClassStats]

    /// Number of features the model was trained on.
    public let featureCount: Int

    /// Fits a Gaussian Naive Bayes model to the given training data.
    ///
    /// Computes the mean and variance of each feature for each class, plus the
    /// class prior probabilities. The returned model is ready for prediction
    /// immediately — there is no separate unfitted state.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - labels: 1D array of integer class labels, one per sample.
    /// - Complexity: O(*n*·*f*) where *n* is the number of samples and *f* is
    ///   the feature count. Computes per-class statistics in a single pass.
    /// - Returns: A trained ``GaussianNaiveBayes`` model.
    public static func fit(features: [[Double]], labels: [Int]) -> GaussianNaiveBayes {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == labels.count,
            "Features and labels must have the same number of samples")

        let featureCount = features[0].count
        let totalSamples = Double(features.count)

        // Group sample indices by class label
        var classIndices: [Int: [Int]] = [:]
        for (i, label) in labels.enumerated() {
            classIndices[label, default: []].append(i)
        }

        // Compute per-class statistics
        let classStats = classIndices.map { (label, indices) -> ClassStats in
            let count = indices.count
            let prior = Double(count) / totalSamples

            var means = [Double](repeating: 0.0, count: featureCount)
            var variances = [Double](repeating: 0.0, count: featureCount)

            // Compute mean for each feature
            for idx in indices {
                for f in 0..<featureCount {
                    means[f] += features[idx][f]
                }
            }
            for f in 0..<featureCount {
                means[f] /= Double(count)
            }

            // Compute variance for each feature
            for idx in indices {
                for f in 0..<featureCount {
                    let diff = features[idx][f] - means[f]
                    variances[f] += diff * diff
                }
            }
            for f in 0..<featureCount {
                variances[f] /= Double(count)
                // Add a small floor to prevent division by zero when a feature
                // is constant within a class
                variances[f] = Swift.max(variances[f], 1e-9)
            }

            return ClassStats(label: label, prior: prior, means: means,
                              variances: variances, count: count)
        }.sorted { $0.label < $1.label }

        return GaussianNaiveBayes(classes: classStats, featureCount: featureCount)
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
    /// - Returns: An array of predicted class labels, one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        return features.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            return _predictSingle(sample)
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
    /// - Returns: 2D array of log-probabilities, shape [samples × classes].
    public func predictLogProbabilities(_ features: [[Double]]) -> [[Double]] {
        return features.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            return classes.map { classStats in
                _logProbability(sample: sample, classStats: classStats)
            }
        }
    }

    // MARK: - Private Helpers

    /// Predicts the most likely class for a single sample.
    private func _predictSingle(_ sample: [Double]) -> Int {
        var bestLabel = classes[0].label
        var bestLogProb = -Double.infinity

        for classStats in classes {
            let logProb = _logProbability(sample: sample, classStats: classStats)
            if logProb > bestLogProb {
                bestLogProb = logProb
                bestLabel = classStats.label
            }
        }

        return bestLabel
    }

    /// Computes log P(class) + Σ log P(x_i | class) for a sample and class.
    private func _logProbability(sample: [Double], classStats: ClassStats) -> Double {
        var logProb = Foundation.log(classStats.prior)

        for f in 0..<featureCount {
            logProb += _logGaussianPDF(
                x: sample[f],
                mean: classStats.means[f],
                variance: classStats.variances[f]
            )
        }

        return logProb
    }

    /// Log of the Gaussian probability density function.
    ///
    /// log P(x | μ, σ²) = -0.5 * [log(2π) + log(σ²) + (x - μ)² / σ²]
    ///
    /// Working in log-space avoids underflow when feature values are far
    /// from the class mean, which would produce extremely small PDF values
    /// that round to zero in standard floating-point arithmetic.
    private func _logGaussianPDF(x: Double, mean: Double, variance: Double) -> Double {
        let diff = x - mean
        return -0.5 * (Foundation.log(2.0 * .pi) + Foundation.log(variance) + (diff * diff) / variance)
    }
}
