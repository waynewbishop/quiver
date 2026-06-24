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

// MARK: - Classification Result

/// A group of data points sharing the same predicted class label.
///
/// `Classification` conforms to `Sequence` — the same Swift protocol that
/// powers `for-in` loops. Iterating a `Classification` gives you its
/// data points directly, just like iterating an `Array`. This mirrors
/// the ``Cluster`` pattern in K-Means, giving classification results
/// the same familiar iteration model that iOS developers already use.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [[1, 2], [2, 3], [5, 8], [6, 9]]
/// let labels = [0, 0, 1, 1]
/// let model = KNearestNeighbors.fit(features: features, labels: labels, k: 3)
///
/// let results = model.classify([[2, 2], [5, 7]])
/// for group in results {
///     print("Class \(group.label): \(group.count) points")
///     for point in group {
///         print("  \(point)")
///     }
/// }
/// ```
public struct Classification: Codable, Sequence, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        "Class \(label): \(count) \(count == 1 ? "point" : "points")"
    }

    /// The predicted class label for this group.
    public let label: Int

    /// The input data points assigned to this class.
    public let points: [[Double]]

    /// The number of data points assigned to this class.
    public var count: Int { points.count }

    /// Returns an iterator over the data points in this classification group.
    public func makeIterator() -> IndexingIterator<[[Double]]> {
        return points.makeIterator()
    }
}
