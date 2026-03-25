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
public struct Cluster: Codable, Sequence, CustomStringConvertible, Equatable {

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

// MARK: - K-Means Clustering

/// A trained K-Means clustering model.
///
/// K-Means partitions data into `k` clusters by iteratively assigning each point
/// to the nearest centroid and recomputing centroids as the mean of assigned points.
/// The algorithm converges when centroids stop moving or the maximum iteration count
/// is reached.
///
/// This is a value type — once created via ``fit(data:k:maxIterations:seed:)``,
/// the model is immutable. There is no separate "unfitted" state, which eliminates
/// the common bug of calling predict before fit.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let data: [[Double]] = [
///     [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
///     [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
/// ]
///
/// let model = KMeans.fit(data: data, k: 2, seed: 42)
/// print(model.labels)       // [0, 0, 0, 1, 1, 1]
/// print(model.centroids)    // cluster centers
/// print(model.inertia)      // sum of squared distances
/// ```
public struct KMeans: Codable, CustomStringConvertible, Equatable {

    public var description: String {
        "KMeans: \(centroids.count) clusters, \(labels.count) points, converged in \(iterations) iterations (inertia: \(String(format: "%.2f", inertia)))"
    }

    /// The final centroid positions after convergence.
    public let centroids: [[Double]]

    /// The cluster label assigned to each training sample (0 to k-1).
    public let labels: [Int]

    /// Sum of squared distances from each point to its assigned centroid.
    public let inertia: Double

    /// The number of iterations the algorithm ran before stopping.
    public let iterations: Int

    /// Number of features the model was trained on.
    public let featureCount: Int

    /// Fits a K-Means model to the given data.
    ///
    /// Initializes centroids by randomly selecting `k` distinct data points, then
    /// iterates: assign each point to its nearest centroid, recompute centroids as
    /// the mean of assigned points. Stops when centroids no longer change or
    /// `maxIterations` is reached.
    ///
    /// - Parameters:
    ///   - data: 2D array where each row is a sample and each column is a feature.
    ///   - k: Number of clusters to form.
    ///   - maxIterations: Maximum number of assign-update cycles. Defaults to 100.
    ///   - seed: Random seed for reproducible centroid initialization. Defaults to nil (random).
    /// - Complexity: O(*n*·*k*·*d*·*i*) where *n* is the number of samples,
    ///   *k* is the cluster count, *d* is the feature count, and *i* is the
    ///   number of iterations until convergence. Performs well for datasets
    ///   up to low tens of thousands.
    /// - Returns: A trained ``KMeans`` model with final centroids and cluster assignments.
    public static func fit(
        data: [[Double]],
        k: Int,
        maxIterations: Int = 100,
        seed: UInt64? = nil
    ) -> KMeans {
        precondition(!data.isEmpty, "Data array must not be empty")
        precondition(k > 0, "k must be positive")
        precondition(k <= data.count,
            "k (\(k)) cannot exceed the number of samples (\(data.count))")

        let featureCount = data[0].count

        // Initialize centroids by selecting k random distinct samples
        var centroids = _initializeCentroids(data: data, k: k, seed: seed)
        var labels = [Int](repeating: 0, count: data.count)
        var iterationCount = 0

        for _ in 0..<maxIterations {
            iterationCount += 1

            // Assign each point to its nearest centroid
            let newLabels = _assignLabels(data: data, centroids: centroids)

            // Recompute centroids as the mean of assigned points
            let newCentroids = _recomputeCentroids(
                data: data, labels: newLabels, k: k, featureCount: featureCount)

            // Check convergence — centroids didn't move
            if newCentroids == centroids {
                labels = newLabels
                centroids = newCentroids
                break
            }

            labels = newLabels
            centroids = newCentroids
        }

        // Compute inertia (sum of squared distances to assigned centroid)
        let inertia = _computeInertia(data: data, labels: labels, centroids: centroids)

        return KMeans(
            centroids: centroids,
            labels: labels,
            inertia: inertia,
            iterations: iterationCount,
            featureCount: featureCount
        )
    }

    /// Fits multiple K-Means models with different seeds and returns the best one.
    ///
    /// Random centroid initialization can produce poor clusterings due to local minima.
    /// This method runs the algorithm `attempts` times with different seeds and returns
    /// the model with the lowest inertia (tightest clusters).
    ///
    /// - Parameters:
    ///   - data: 2D array where each row is a sample and each column is a feature.
    ///   - k: Number of clusters to form.
    ///   - maxIterations: Maximum number of assign-update cycles per attempt. Defaults to 100.
    ///   - attempts: Number of times to run the algorithm with different seeds. Defaults to 10.
    /// - Complexity: O(*a*·*n*·*k*·*d*·*i*) where *a* is the number of attempts.
    ///   Runs ``fit(data:k:maxIterations:seed:)`` multiple times and returns the
    ///   model with lowest inertia. Reduce `attempts` for faster results on
    ///   large datasets.
    /// - Returns: The ``KMeans`` model with the lowest inertia across all attempts.
    public static func bestFit(
        data: [[Double]],
        k: Int,
        maxIterations: Int = 100,
        attempts: Int = 10
    ) -> KMeans {
        precondition(attempts > 0, "attempts must be positive")
        let results = (0..<attempts)
            .map { fit(data: data, k: k, maxIterations: maxIterations, seed: UInt64($0)) }
        guard let best = results.min(by: { $0.inertia < $1.inertia }) else {
            preconditionFailure("bestFit requires at least one attempt")
        }
        return best
    }

    /// Computes inertia for a range of k values (elbow method).
    ///
    /// Runs K-Means for each value of k and returns the inertia at each step.
    /// Plot the result against the k values to find the "elbow" — the point where
    /// adding more clusters stops providing meaningful improvement.
    ///
    /// ```swift
    /// let data: [[Double]] = [
    ///     [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
    ///     [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
    /// ]
    ///
    /// let kRange = Array(1...6)
    /// let inertias = KMeans.elbowMethod(data: data, kRange: kRange, seed: 42)
    /// // kRange and inertias are parallel arrays ready for charting
    /// ```
    ///
    /// - Parameters:
    ///   - data: 2D array where each row is a sample and each column is a feature.
    ///   - kRange: The k values to evaluate.
    ///   - maxIterations: Maximum iterations per fit. Defaults to 100.
    ///   - seed: Random seed for reproducible results. Defaults to nil.
    /// - Complexity: O(|*kRange*|·*n*·*k*·*d*·*i*). Runs ``fit(data:k:maxIterations:seed:)``
    ///   once per value in `kRange`. Use a narrow range when dataset size is large.
    /// - Returns: An array of inertia values, one per k value.
    public static func elbowMethod(
        data: [[Double]],
        kRange: [Int],
        maxIterations: Int = 100,
        seed: UInt64? = nil
    ) -> [Double] {
        return kRange.map { k in
            fit(data: data, k: k, maxIterations: maxIterations, seed: seed).inertia
        }
    }

    /// Groups the original training data into clusters with their centroids.
    ///
    /// This method pairs each centroid with the data points assigned to it,
    /// returning an array of ``Cluster`` values that conform to `Sequence`.
    /// Each cluster can be iterated directly to access its points.
    ///
    /// ```swift
    /// let model = KMeans.fit(data: data, k: 3, seed: 42)
    /// let clusters = model.clusters(from: data)
    ///
    /// for cluster in clusters {
    ///     print("Center: \(cluster.centroid), size: \(cluster.count)")
    /// }
    /// ```
    ///
    /// - Complexity: O(*n*·*k*·*d*) where *n* is the number of samples, *k* is
    ///   the cluster count, and *d* is the feature count.
    /// - Parameter data: The same data used for fitting (or any data with matching feature count).
    /// - Returns: An array of ``Cluster`` values, one per centroid, in centroid order.
    public func clusters(from data: [[Double]]) -> [Cluster] {
        let k = centroids.count
        var grouped = [[[Double]]](repeating: [], count: k)
        let assignedLabels = predict(data)
        for i in 0..<data.count {
            grouped[assignedLabels[i]].append(data[i])
        }
        return (0..<k).map { Cluster(centroid: centroids[$0], points: grouped[$0]) }
    }

    /// Assigns cluster labels to new data points based on the trained centroids.
    ///
    /// For each sample, computes the distance to every centroid and assigns the
    /// label of the nearest one. The centroids are not updated.
    ///
    /// - Complexity: O(*n*·*k*·*d*) where *n* is the number of samples, *k* is
    ///   the cluster count, and *d* is the feature count.
    /// - Parameter data: 2D array where each row is a sample to assign.
    /// - Returns: An array of cluster labels (0 to k-1), one per sample.
    public func predict(_ data: [[Double]]) -> [Int] {
        return data.map { sample in
            precondition(sample.count == featureCount,
                "Sample has \(sample.count) features, model expects \(featureCount)")
            return _nearestCentroid(sample, centroids: centroids)
        }
    }

    // MARK: - Private Helpers

    /// Selects k distinct samples as initial centroids.
    private static func _initializeCentroids(
        data: [[Double]], k: Int, seed: UInt64?
    ) -> [[Double]] {
        var indices = Array(0..<data.count)

        // Fisher-Yates shuffle with optional seeded RNG
        if let seed = seed {
            var rng = _SeededRNG(seed: seed)
            for i in stride(from: indices.count - 1, through: 1, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                indices.swapAt(i, j)
            }
        } else {
            indices.shuffle()
        }

        return Array(indices.prefix(k).map { data[$0] })
    }

    /// Assigns each data point to its nearest centroid.
    private static func _assignLabels(
        data: [[Double]], centroids: [[Double]]
    ) -> [Int] {
        return data.map { sample in
            _nearestCentroidStatic(sample, centroids: centroids)
        }
    }

    /// Returns the index of the nearest centroid for a sample (static version).
    private static func _nearestCentroidStatic(
        _ sample: [Double], centroids: [[Double]]
    ) -> Int {
        var bestIndex = 0
        var bestDistance = Double.infinity
        for i in 0..<centroids.count {
            // Performance: Inlines squared Euclidean distance to avoid the temporary
            // difference array that distance(to:) allocates. Also skips sqrt since
            // we only need relative ordering. Called n * k times per iteration.
            var sum = 0.0
            let centroid = centroids[i]
            for d in 0..<sample.count {
                let diff = sample[d] - centroid[d]
                sum += diff * diff
            }
            // Compare squared distances — avoids sqrt per comparison
            if sum < bestDistance {
                bestDistance = sum
                bestIndex = i
            }
        }
        return bestIndex
    }

    /// Returns the index of the nearest centroid for a sample (instance version).
    private func _nearestCentroid(
        _ sample: [Double], centroids: [[Double]]
    ) -> Int {
        return KMeans._nearestCentroidStatic(sample, centroids: centroids)
    }

    /// Recomputes centroids as the mean of points assigned to each cluster.
    private static func _recomputeCentroids(
        data: [[Double]], labels: [Int], k: Int, featureCount: Int
    ) -> [[Double]] {
        var centroids = [[Double]](repeating: [Double](repeating: 0.0, count: featureCount), count: k)
        var counts = [Int](repeating: 0, count: k)

        for i in 0..<data.count {
            let cluster = labels[i]
            counts[cluster] += 1
            for j in 0..<featureCount {
                centroids[cluster][j] += data[i][j]
            }
        }

        // Divide by count to get the mean. If a cluster has no assigned points,
        // its centroid stays at the origin. This can happen when k exceeds the
        // number of natural groupings or when initialization places two centroids
        // in the same dense region. The empty cluster will typically attract points
        // in subsequent iterations as other centroids shift away.
        for c in 0..<k {
            if counts[c] > 0 {
                for j in 0..<featureCount {
                    centroids[c][j] /= Double(counts[c])
                }
            }
        }

        return centroids
    }

    /// Computes inertia: sum of squared Euclidean distances to assigned centroids.
    private static func _computeInertia(
        data: [[Double]], labels: [Int], centroids: [[Double]]
    ) -> Double {
        // Performance: Inlines squared distance — same rationale as _nearestCentroidStatic.
        var inertia = 0.0
        for i in 0..<data.count {
            let centroid = centroids[labels[i]]
            let sample = data[i]
            for d in 0..<sample.count {
                let diff = sample[d] - centroid[d]
                inertia += diff * diff
            }
        }
        return inertia
    }
}

// MARK: - Seeded Random Number Generator

/// A simple linear congruential generator for reproducible centroid initialization.
private struct _SeededRNG {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        // LCG constants from Knuth's MMIX
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
