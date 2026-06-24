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

// MARK: - Cluster

/// A single cluster from a K-Means model, containing a centroid and its assigned data points.
///
/// Each cluster holds the centroid position and the data points assigned to it.
/// Conforms to `Sequence` so you can iterate directly over the points:
///
/// ```swift
/// let clusters = model.clusters(from: data)
/// for cluster in clusters {
///     print("Center: \(cluster.centroid), size: \(cluster.count)")
///     for point in cluster {
///         print(point)
///     }
/// }
/// ```
public struct Cluster: Codable, Sequence, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let center = centroid.map { String(format: "%.2f", $0) }.joined(separator: ", ")
        return "Cluster: center [\(center)], \(count) \(count == 1 ? "point" : "points")"
    }

    /// The centroid position for this cluster.
    public let centroid: [Double]

    /// The data points assigned to this cluster.
    public let points: [[Double]]

    /// The number of data points in this cluster.
    public var count: Int { points.count }

    /// Returns an iterator over the data points in this cluster.
    public func makeIterator() -> IndexingIterator<[[Double]]> {
        return points.makeIterator()
    }
}
