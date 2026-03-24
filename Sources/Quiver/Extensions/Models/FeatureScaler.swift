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

// MARK: - Feature Scaler

/// A column-wise min-max scaler for feature matrices.
///
/// In machine learning, features often have very different scales — for example,
/// credit scores range from 300 to 850 while loyalty ratios range from 0 to 1.
/// Many algorithms (Nearest Neighbors, logistic regression, Naive Bayes, K-Means) perform
/// poorly when features are on different scales because larger values dominate
/// distance and probability calculations.
///
/// `FeatureScaler` learns the minimum and maximum of each feature column from
/// training data, then uses those statistics to scale any matrix to a target
/// range. Crucially, the same statistics are applied to both training and test
/// data, which prevents test data from influencing the scaling — a subtle but
/// common source of data leakage in ML pipelines.
///
/// This is a value type. Once created via `fit(features:range:)`, the scaler
/// is immutable and can be safely reused across multiple transforms.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let trainX: [[Double]] = [
///     [619, 15000, 0.08], [502, 78000, 0.04],
///     [850, 11000, 0.12], [720, 98000, 0.18]
/// ]
/// let testX: [[Double]] = [[680, 62000, 0.28]]
///
/// // Fit on training data, transform both sets
/// let scaler = FeatureScaler.fit(features: trainX)
/// let scaledTrain = scaler.transform(trainX)
/// let scaledTest = scaler.transform(testX)
/// ```
public struct FeatureScaler: CustomStringConvertible, Equatable {

    public var description: String {
        "FeatureScaler: \(featureCount) features, range \(range.lowerBound)...\(range.upperBound)"
    }

    /// Per-column minimum values learned from the training data.
    public let minimums: [Double]

    /// Per-column maximum values learned from the training data.
    public let maximums: [Double]

    /// The target range that scaled values are mapped to.
    public let range: ClosedRange<Double>

    /// Number of feature columns the scaler was fitted on.
    public let featureCount: Int

    /// Learns column-wise min and max values from a feature matrix.
    ///
    /// Scans each column independently to find its minimum and maximum values.
    /// These statistics are stored and used by `transform(_:)` to scale new data
    /// to the target range.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - range: The target range for scaled values. Defaults to `0.0...1.0`.
    /// - Returns: A fitted `FeatureScaler` ready to transform data.
    public static func fit(features: [[Double]], range: ClosedRange<Double> = 0.0...1.0) -> FeatureScaler {
        precondition(!features.isEmpty, "Features array must not be empty")

        let featureCount = features[0].count
        var minimums = [Double](repeating: Double.infinity, count: featureCount)
        var maximums = [Double](repeating: -Double.infinity, count: featureCount)

        // Find min and max for each column
        for row in features {
            for f in 0..<featureCount {
                if row[f] < minimums[f] { minimums[f] = row[f] }
                if row[f] > maximums[f] { maximums[f] = row[f] }
            }
        }

        return FeatureScaler(
            minimums: minimums,
            maximums: maximums,
            range: range,
            featureCount: featureCount
        )
    }

    /// Scales a feature matrix using the learned column statistics.
    ///
    /// Each column is scaled independently using the min and max values learned
    /// during fitting. Values in the input may fall outside the training range —
    /// they will be scaled proportionally but may land outside the target range.
    ///
    /// If a column had zero range in the training data (all values identical),
    /// that column is mapped to the lower bound of the target range.
    ///
    /// - Parameter features: 2D array to scale, with the same number of columns
    ///   as the training data.
    /// - Returns: A scaled copy of the input matrix.
    public func transform(_ features: [[Double]]) -> [[Double]] {
        let targetRange = range.upperBound - range.lowerBound

        return features.map { row in
            precondition(row.count == featureCount,
                "Row has \(row.count) features, scaler expects \(featureCount)")

            var scaled = [Double](repeating: 0.0, count: featureCount)
            for f in 0..<featureCount {
                let dataRange = maximums[f] - minimums[f]
                if dataRange == 0.0 {
                    // Constant column — map to lower bound
                    scaled[f] = range.lowerBound
                } else {
                    scaled[f] = ((row[f] - minimums[f]) / dataRange) * targetRange + range.lowerBound
                }
            }
            return scaled
        }
    }
}
