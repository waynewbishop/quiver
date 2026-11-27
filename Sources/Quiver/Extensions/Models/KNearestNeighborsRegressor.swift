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

// MARK: - K-Nearest Neighbors Regression

/// A regression model that predicts a value by averaging the targets of the nearest training points.
///
/// `KNearestNeighborsRegressor` is the regression counterpart of ``KNearestNeighbors``. To predict
/// for a new sample, it finds the `k` training samples closest to it and returns the average of
/// their targets. Nothing is learned in advance and no equation is fitted: the training data
/// itself is the model. That makes the predictions easy to explain ("the three most similar
/// houses sold for these prices") and lets the model follow curves that a straight line cannot.
///
/// Because the model is built on distance, features measured on very different scales should be
/// standardized first, or the feature with the largest numbers will decide every prediction.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let sqft   = [[1000.0], [1500.0], [2000.0], [2500.0], [3000.0]]
/// let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]
///
/// let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
/// print(model)
/// // KNearestNeighborsRegressor: k=2, euclidean, 5 training points, 1 features
///
/// // The two nearest sizes to 1,800 sq ft are 1,500 and 2,000, so the prediction is their average.
/// model.predict([[1800.0]])  // [230000.0]
/// ```
///
/// > Tip: Weight the neighbors by distance with `weight: .distance` so that closer houses count
/// > for more than distant ones.
public struct KNearestNeighborsRegressor: Regressor, Codable, CustomStringConvertible, Equatable, Sendable {

    /// A one-line summary such as `KNearestNeighborsRegressor: k=3, euclidean, 40 training points, 2 features`.
    public var description: String {
        "KNearestNeighborsRegressor: k=\(k), \(metric), \(trainingFeatures.count) training points, \(featureCount) features"
    }

    /// The training samples, one row per sample.
    public let trainingFeatures: [[Double]]

    /// The target value of each training sample, in the same order as ``trainingFeatures``.
    public let trainingTargets: [Double]

    /// The number of neighbors averaged for each prediction.
    public let k: Int

    /// The distance metric used to find the nearest neighbors.
    public let metric: DistanceMetric

    /// How the neighbors' targets are combined: an equal average, or weighted by closeness.
    public let weight: VoteWeight

    /// The number of features in each training sample.
    public let featureCount: Int

    /// Fits a nearest-neighbors regression model to the training data.
    ///
    /// Fitting stores the training data; the work happens at prediction time. The returned model
    /// is immutable, and there is no unfitted state to misuse.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let features = [[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [5.0, 4.0], [3.0, 3.0]]
    /// let targets  = [10.0, 12.0, 30.0, 32.0, 20.0]
    ///
    /// let model = KNearestNeighborsRegressor.fit(features: features, targets: targets, k: 3)
    /// model.predict([[2.5, 2.5]])  // [14.0]
    /// ```
    ///
    /// - Parameters:
    ///   - features: The training samples. Each inner array is one sample.
    ///   - targets: The continuous target for each sample. Must have the same count as `features`.
    ///   - k: The number of neighbors to average. Defaults to 3. Must be positive and no larger than
    ///     the number of training samples.
    ///   - metric: The distance metric. Defaults to `.euclidean`.
    ///   - weight: Whether neighbors count equally (`.uniform`, the default) or in proportion to
    ///     `1 / distance` (`.distance`). When a neighbor sits at distance 0, the prediction is the
    ///     average of the targets of the exact matches.
    /// - Returns: A fitted model.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        k: Int = 3,
        metric: DistanceMetric = .euclidean,
        weight: VoteWeight = .uniform
    ) -> KNearestNeighborsRegressor {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples")
        precondition(k > 0, "k must be positive")
        precondition(k <= features.count,
            "k (\(k)) cannot exceed the number of training samples (\(features.count))")

        return KNearestNeighborsRegressor(
            trainingFeatures: features,
            trainingTargets: targets,
            k: k,
            metric: metric,
            weight: weight,
            featureCount: features[0].count
        )
    }

    /// Predicts a continuous value for each sample.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let features = [[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [5.0, 4.0], [3.0, 3.0]]
    /// let targets  = [10.0, 12.0, 30.0, 32.0, 20.0]
    ///
    /// let model = KNearestNeighborsRegressor.fit(features: features, targets: targets, k: 3)
    /// model.predict([[2.5, 2.5], [4.5, 4.5]])  // ≈ [14.0, 27.33]
    /// ```
    ///
    /// - Parameter features: The samples to predict. Each must have `featureCount` features.
    /// - Returns: One predicted value per sample.
    public func predict(_ features: [[Double]]) -> [Double] {
        return features.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            return _predictSingle(sample)
        }
    }

    // MARK: - Private Helpers

    /// Predicts the target for a single sample.
    private func _predictSingle(_ sample: [Double]) -> Double {
        let neighbors = _NearestNeighbors.nearest(
            to: sample, in: trainingFeatures, metric: metric, k: k
        )
        switch weight {
        case .uniform:
            return _uniformAverage(neighbors)
        case .distance:
            return _distanceWeightedAverage(neighbors)
        }
    }

    /// Returns the plain average of the neighbors' targets.
    private func _uniformAverage(_ neighbors: ArraySlice<_NearestNeighbors.Neighbor>) -> Double {
        var total = 0.0
        for neighbor in neighbors {
            total += trainingTargets[neighbor.index]
        }
        return total / Double(neighbors.count)
    }

    /// Returns the average of the neighbors' targets weighted by `1 / distance`.
    ///
    /// A neighbor at distance 0 would carry infinite weight, so when any exist the prediction is
    /// the plain average of just those exact matches.
    private func _distanceWeightedAverage(_ neighbors: ArraySlice<_NearestNeighbors.Neighbor>) -> Double {
        var exactTotal = 0.0
        var exactCount = 0
        for neighbor in neighbors where neighbor.distance == 0.0 {
            exactTotal += trainingTargets[neighbor.index]
            exactCount += 1
        }
        if exactCount > 0 {
            return exactTotal / Double(exactCount)
        }

        var weightedTotal = 0.0
        var weightSum = 0.0
        for neighbor in neighbors {
            let neighborWeight = 1.0 / neighbor.distance
            weightedTotal += neighborWeight * trainingTargets[neighbor.index]
            weightSum += neighborWeight
        }
        return weightedTotal / weightSum
    }
}
