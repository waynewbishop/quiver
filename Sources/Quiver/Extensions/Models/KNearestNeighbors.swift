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

// MARK: - Distance Metric

/// The distance metric used to compare samples in K-Nearest Neighbors.
///
/// Euclidean distance measures straight-line distance between points and works
/// well when features have similar scales. Cosine distance measures the angle
/// between vectors and works well for text embeddings and high-dimensional data
/// where magnitude is less meaningful than direction.
public enum DistanceMetric: Codable, Equatable {

    /// Euclidean distance: √Σ(aᵢ − bᵢ)².
    ///
    /// Sensitive to feature scale — consider using ``FeatureScaler`` before
    /// fitting when features have different units or magnitudes.
    case euclidean

    /// Cosine distance: 1 − cosine similarity.
    ///
    /// Scale-invariant — vectors pointing in the same direction have distance
    /// 0 regardless of their magnitude. Preferred for text embeddings, TF-IDF
    /// vectors, and other high-dimensional sparse data.
    case cosine
}

// MARK: - Vote Weighting

/// The weighting strategy for neighbor votes in K-Nearest Neighbors.
///
/// Uniform weighting gives each neighbor one vote. Distance weighting gives
/// closer neighbors more influence, which can improve accuracy when the
/// decision boundary is near the query point.
public enum VoteWeight: Codable, Equatable {

    /// Each neighbor gets one vote regardless of distance.
    case uniform

    /// Closer neighbors get more influence: weight = 1 / distance.
    ///
    /// When a neighbor has distance 0 (exact match), the model predicts
    /// that neighbor's label immediately without voting.
    case distance
}

// MARK: - K-Nearest Neighbors

/// A trained K-Nearest Neighbors classifier.
///
/// Nearest Neighbors is a "lazy learning" algorithm — it stores the training data and defers
/// all computation to prediction time. For each new sample, it finds the `k`
/// closest training points and predicts the most common label among them.
///
/// This is a value type — once created via ``fit(features:labels:k:metric:weight:)``,
/// the model is immutable. There is no separate "unfitted" state, which eliminates
/// the common bug of calling predict before fit.
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
/// let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)
/// let predictions = model.predict([[2.0, 2.5], [5.5, 7.0]])
/// // [0, 1]
/// ```
public struct KNearestNeighbors: Classifier, Codable, CustomStringConvertible, Equatable {

    public var description: String {
        "KNearestNeighbors: k=\(k), \(metric), \(trainingFeatures.count) training points, \(featureCount) features"
    }

    /// The training feature vectors, stored for distance computation at prediction time.
    public let trainingFeatures: [[Double]]

    /// The training labels, one per feature vector.
    public let trainingLabels: [Int]

    /// The number of neighbors to consider for each prediction.
    public let k: Int

    /// The distance metric used to compare samples.
    public let metric: DistanceMetric

    /// The weighting strategy for neighbor votes.
    public let weight: VoteWeight

    /// Number of features the model was trained on.
    public let featureCount: Int

    /// Fits a K-Nearest Neighbors model to the given training data.
    ///
    /// Nearest Neighbors is a lazy learner — this method simply stores the training data
    /// for use during prediction. No computation happens at fit time, which
    /// makes fitting instantaneous regardless of dataset size.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - labels: 1D array of integer class labels, one per sample.
    ///   - k: Number of neighbors to consider. Defaults to 3. Odd values avoid ties
    ///     in binary classification.
    ///   - metric: Distance metric for comparing samples. Defaults to ``DistanceMetric/euclidean``.
    ///   - weight: How to weight neighbor votes. Defaults to ``VoteWeight/uniform``.
    /// - Returns: A trained ``KNearestNeighbors`` model.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        k: Int = 3,
        metric: DistanceMetric = .euclidean,
        weight: VoteWeight = .uniform
    ) -> KNearestNeighbors {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == labels.count,
            "Features and labels must have the same number of samples")
        precondition(k > 0, "k must be positive")
        precondition(k <= features.count,
            "k (\(k)) cannot exceed the number of training samples (\(features.count))")

        return KNearestNeighbors(
            trainingFeatures: features,
            trainingLabels: labels,
            k: k,
            metric: metric,
            weight: weight,
            featureCount: features[0].count
        )
    }

    /// Predicts class labels for one or more samples.
    ///
    /// For each sample, computes the distance to every training point, selects
    /// the `k` nearest neighbors, and returns the most common label among them.
    /// With distance weighting, closer neighbors contribute more to the vote.
    ///
    /// - Complexity: O(*q*·*t*·*d*) where *q* is the number of query samples,
    ///   *t* is the number of training samples, and *d* is the feature count.
    ///   Scale the training set to the most relevant samples when working with
    ///   large datasets.
    /// - Parameter features: 2D array where each row is a sample to classify.
    /// - Returns: An array of predicted class labels, one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        return features.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            return _predictSingle(sample)
        }
    }

    // MARK: - Private Helpers

    /// Predicts the label for a single sample.
    private func _predictSingle(_ sample: [Double]) -> Int {

        // Compute distance from sample to every training point
        var distances = [(index: Int, distance: Double)]()
        distances.reserveCapacity(trainingFeatures.count)

        for i in 0..<trainingFeatures.count {
            let d = _distance(sample, trainingFeatures[i])
            distances.append((index: i, distance: d))
        }

        // Find the k nearest neighbors
        distances.sort { $0.distance < $1.distance }
        let neighbors = distances.prefix(k)

        // Vote among neighbors
        switch weight {
        case .uniform:
            return _majorityVote(neighbors)
        case .distance:
            return _distanceWeightedVote(neighbors)
        }
    }

    /// Computes the distance between two samples using the configured metric.
    private func _distance(_ a: [Double], _ b: [Double]) -> Double {
        switch metric {
        case .euclidean:
            var sum = 0.0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return Foundation.sqrt(sum)
        case .cosine:
            // Performance: Inlines cosine distance to avoid _Vector wrapper overhead
            // per prediction. Same formula as 1.0 - cosineOfAngle(with:).
            var dot = 0.0, magA = 0.0, magB = 0.0
            for i in 0..<a.count {
                dot += a[i] * b[i]
                magA += a[i] * a[i]
                magB += b[i] * b[i]
            }
            let denom = Foundation.sqrt(magA) * Foundation.sqrt(magB)
            return denom > 0 ? 1.0 - dot / denom : 1.0
        }
    }

    /// Returns the most common label among neighbors (one vote each).
    private func _majorityVote(_ neighbors: ArraySlice<(index: Int, distance: Double)>) -> Int {
        var counts: [Int: Int] = [:]
        for neighbor in neighbors {
            let label = trainingLabels[neighbor.index]
            counts[label, default: 0] += 1
        }
        guard let winner = counts.max(by: { $0.value < $1.value }) else {
            preconditionFailure("Vote counts must not be empty")
        }
        return winner.key
    }

    /// Returns the label with the highest total weight (weight = 1/distance).
    private func _distanceWeightedVote(_ neighbors: ArraySlice<(index: Int, distance: Double)>) -> Int {

        // If any neighbor has distance 0, predict its label immediately
        for neighbor in neighbors {
            if neighbor.distance == 0.0 {
                return trainingLabels[neighbor.index]
            }
        }

        var weights: [Int: Double] = [:]
        for neighbor in neighbors {
            let label = trainingLabels[neighbor.index]
            weights[label, default: 0.0] += 1.0 / neighbor.distance
        }
        guard let winner = weights.max(by: { $0.value < $1.value }) else {
            preconditionFailure("Weighted votes must not be empty")
        }
        return winner.key
    }
}
