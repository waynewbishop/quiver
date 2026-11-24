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

// MARK: - Internal Naive Bayes Computation

/// Internal namespace for the steps every Naive Bayes classifier shares.
///
/// Each Naive Bayes variant scores a sample against class c as
/// log P(c) + Σ log P(xⱼ | c) and predicts the class with the highest score.
/// The variants differ only in how P(xⱼ | c) is modeled — a normal density
/// for continuous features, a Bernoulli probability for binary ones. The
/// grouping of training samples by class and the selection of the winning
/// class are identical, so they live here once.
internal enum _NaiveBayes {

    /// Groups training sample indices by their class label.
    ///
    /// Indices within each group keep their original ascending order, so
    /// per-class sums accumulate in the same order as the training data.
    ///
    /// - Parameter labels: 1D array of integer class labels, one per sample.
    /// - Returns: One entry per unique label, sorted by ascending label, each
    ///   pairing the label with the indices of the samples that carry it.
    static func sampleIndicesGroupedByLabel(_ labels: [Int]) -> [(label: Int, indices: [Int])] {
        var indicesByLabel: [Int: [Int]] = [:]
        for (sampleIndex, label) in labels.enumerated() {
            indicesByLabel[label, default: []].append(sampleIndex)
        }
        return indicesByLabel
            .map { (label: $0.key, indices: $0.value) }
            .sorted { $0.label < $1.label }
    }

    /// Returns the label whose log joint probability is highest.
    ///
    /// Ties resolve to the earliest label in `labels`. When every score is
    /// negative infinity, the first label is returned.
    ///
    /// - Parameters:
    ///   - logJointProbabilities: log P(c) + Σ log P(xⱼ | c) for each class,
    ///     ordered to match `labels`.
    ///   - labels: The class labels, one per score. Must not be empty.
    /// - Returns: The label with the maximum log joint probability.
    static func mostProbableLabel(logJointProbabilities: [Double], labels: [Int]) -> Int {
        precondition(!labels.isEmpty, "A Naive Bayes model must have at least one class")
        precondition(logJointProbabilities.count == labels.count,
            "Expected one log probability per class")

        var bestLabel = labels[0]
        var bestLogProbability = -Double.infinity
        for classIndex in 0..<labels.count where logJointProbabilities[classIndex] > bestLogProbability {
            bestLogProbability = logJointProbabilities[classIndex]
            bestLabel = labels[classIndex]
        }
        return bestLabel
    }
}
