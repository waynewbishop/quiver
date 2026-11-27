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

// MARK: - Nearest-Neighbor Search

/// Internal nearest-neighbor search shared by ``KNearestNeighbors`` and ``KNearestNeighborsRegressor``.
///
/// Both models find the `k` training points closest to a sample and then combine what those
/// points carry: the classifier votes on their labels, the regressor averages their targets.
/// The search is identical, so it lives here once.
internal enum _NearestNeighbors {

    /// A training point's position in the training set and its distance from the query sample.
    typealias Neighbor = (index: Int, distance: Double)

    /// Computes the distance between two samples using the given metric.
    ///
    /// - Parameters:
    ///   - a: The first sample.
    ///   - b: The second sample. Must have the same count as `a`.
    ///   - metric: The distance metric.
    /// - Returns: The distance between `a` and `b`.
    static func distance(_ a: [Double], _ b: [Double], metric: DistanceMetric) -> Double {
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
        case .manhattan:
            var sum = 0.0
            for i in 0..<a.count {
                sum += abs(a[i] - b[i])
            }
            return sum
        case .chebyshev:
            var maxDiff = 0.0
            for i in 0..<a.count {
                let diff = abs(a[i] - b[i])
                if diff > maxDiff { maxDiff = diff }
            }
            return maxDiff
        case .squaredEuclidean:
            var sum = 0.0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sum
        case .minkowski(let p):
            var sum = 0.0
            for i in 0..<a.count {
                sum += Foundation.pow(abs(a[i] - b[i]), p)
            }
            return Foundation.pow(sum, 1.0 / p)
        }
    }

    /// Finds the `k` training points nearest a sample, nearest first.
    ///
    /// Equal distances fall back to training order, so the neighbor list, and every tie-break
    /// that reads it, is the same on every run.
    ///
    /// - Parameters:
    ///   - sample: The query sample.
    ///   - trainingFeatures: The training samples, each with the same count as `sample`.
    ///   - metric: The distance metric.
    ///   - k: How many neighbors to keep. Must not exceed the number of training samples.
    /// - Returns: The `k` nearest neighbors as `(index, distance)` pairs, nearest first.
    static func nearest(
        to sample: [Double],
        in trainingFeatures: [[Double]],
        metric: DistanceMetric,
        k: Int
    ) -> ArraySlice<Neighbor> {
        var distances = [Neighbor]()
        distances.reserveCapacity(trainingFeatures.count)

        for i in 0..<trainingFeatures.count {
            let d = distance(sample, trainingFeatures[i], metric: metric)
            distances.append((index: i, distance: d))
        }

        distances.sort { ($0.distance, $0.index) < ($1.distance, $1.index) }
        return distances.prefix(k)
    }
}
