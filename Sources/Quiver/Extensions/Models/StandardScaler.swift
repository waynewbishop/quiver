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

// MARK: - Standard Scaler

/// A column-wise z-score scaler for feature matrices.
///
/// `StandardScaler` learns the mean and standard deviation of each feature
/// column from training data, then transforms values using the formula
/// `(value - mean) / std`. The result is a distribution centered at zero
/// with unit variance — typically falling between -3 and 3 for normally
/// distributed data.
///
/// Compared to `FeatureScaler` (min-max scaling), `StandardScaler` is more
/// robust to outliers. A single extreme value will not compress the rest
/// of the data into a narrow range, because the formula uses mean and
/// standard deviation rather than minimum and maximum.
///
/// This is a value type. Once created via `fit(features:)`, the scaler is
/// immutable and can be safely reused across multiple transforms.
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
/// let scaler = StandardScaler.fit(features: trainX)
/// let scaledTrain = scaler.transform(trainX)
/// let scaledTest = scaler.transform(testX)
/// ```
public struct StandardScaler: Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        "StandardScaler: \(featureCount) features"
    }

    /// Per-column mean values learned from the training data.
    public let means: [Double]

    /// Per-column standard deviation values learned from the training data.
    public let stds: [Double]

    /// Number of feature columns the scaler was fitted on.
    public let featureCount: Int

    /// Learns column-wise mean and standard deviation from a feature matrix.
    ///
    /// Scans each column independently to compute its mean and population
    /// standard deviation. These statistics are stored and used by
    /// `transform(_:)` to z-score new data.
    ///
    /// - Parameter features: 2D array where each row is a sample and each column is a feature.
    /// - Returns: A fitted `StandardScaler` ready to transform data.
    public static func fit(features: [[Double]]) -> StandardScaler {
        precondition(!features.isEmpty, "Features array must not be empty")

        let featureCount = features[0].count
        let rowCount = Double(features.count)
        var means = [Double](repeating: 0.0, count: featureCount)
        var stds = [Double](repeating: 0.0, count: featureCount)

        // Compute mean per column
        for row in features {
            for f in 0..<featureCount {
                means[f] += row[f]
            }
        }
        for f in 0..<featureCount {
            means[f] /= rowCount
        }

        // Compute population standard deviation per column (ddof=0)
        for row in features {
            for f in 0..<featureCount {
                let diff = row[f] - means[f]
                stds[f] += diff * diff
            }
        }
        for f in 0..<featureCount {
            stds[f] = (stds[f] / rowCount).squareRoot()
        }

        return StandardScaler(
            means: means,
            stds: stds,
            featureCount: featureCount
        )
    }

    /// Scales a feature matrix using the learned column statistics.
    ///
    /// Each column is z-scored independently using the mean and standard
    /// deviation learned during fitting. Values in the input may fall
    /// outside the training distribution — they will be scaled
    /// proportionally and may produce z-scores beyond the typical range.
    ///
    /// If a column had zero standard deviation in the training data
    /// (all values identical), that column is mapped to zero.
    ///
    /// - Parameter features: 2D array to scale, with the same number of columns
    ///   as the training data.
    /// - Returns: A z-scored copy of the input matrix.
    public func transform(_ features: [[Double]]) -> [[Double]] {
        return features.map { row in
            precondition(row.count == featureCount,
                "Row has \(row.count) features, scaler expects \(featureCount)")

            var scaled = [Double](repeating: 0.0, count: featureCount)
            for f in 0..<featureCount {
                if stds[f] == 0.0 {
                    // Constant column — map to zero
                    scaled[f] = 0.0
                } else {
                    scaled[f] = (row[f] - means[f]) / stds[f]
                }
            }
            return scaled
        }
    }
}
