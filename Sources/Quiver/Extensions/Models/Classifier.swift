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
public struct Classification: Codable, Sequence, CustomStringConvertible, Equatable {

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

// MARK: - Classifier Protocol

/// A protocol for supervised classification models that predict discrete class labels.
///
/// Types conforming to `Classifier` provide a ``predict(_:)`` method that returns
/// integer class labels. The protocol provides a default ``classify(_:)`` implementation
/// that groups inputs by predicted label, returning structured ``Classification`` results.
///
/// Both ``KNearestNeighbors`` and ``GaussianNaiveBayes`` conform to this protocol.
/// ``predict(_:)`` returns raw labels for evaluation pipelines and remains
/// fully available — ``classify(_:)`` is an additional convenience, not a replacement.
public protocol Classifier {
    /// Predicts class labels for the given feature vectors.
    /// - Parameter features: 2D array where each row is a sample to classify.
    /// - Returns: An array of predicted class labels, one per sample.
    func predict(_ features: [[Double]]) -> [Int]
}

extension Classifier {

    /// Groups input features by their predicted class label.
    ///
    /// This method calls ``predict(_:)`` internally, then organizes the inputs
    /// by predicted label into ``Classification`` groups — each with a label,
    /// the points assigned to that class, and a count.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let model = KNearestNeighbors.fit(
    ///     features: [[1, 2], [2, 3], [5, 8], [6, 9]],
    ///     labels: [0, 0, 1, 1], k: 3
    /// )
    ///
    /// let results = model.classify([[2, 2], [5, 7], [6, 8]])
    /// for group in results {
    ///     print("Class \(group.label): \(group.count) points")
    /// }
    /// // Class 0: 1 points
    /// // Class 1: 2 points
    /// ```
    ///
    /// - Parameter features: 2D array where each row is a sample to classify.
    /// - Returns: An array of ``Classification`` groups, one per unique predicted label, sorted by label.
    public func classify(_ features: [[Double]]) -> [Classification] {
        let labels = predict(features)

        // Group inputs by predicted label
        var groups: [Int: [[Double]]] = [:]
        for (feature, label) in Swift.zip(features, labels) {
            groups[label, default: []].append(feature)
        }

        // Return sorted by label for consistent ordering
        return groups.keys.sorted().map { label in
            Classification(label: label, points: groups[label]!)
        }
    }
}
