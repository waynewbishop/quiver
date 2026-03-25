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

// MARK: - Train/Test Split

public extension Array {

    /// Splits the array into training and testing subsets using a seeded shuffle
    /// for reproducible results.
    ///
    /// The seed parameter guarantees that the same array with the same seed always
    /// produces the same split. This enables paired arrays (features and labels) to
    /// be split consistently by using the same seed for both calls.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let features: [[Double]] = [
    ///     [1400, 3], [1600, 3], [1700, 2], [1875, 3], [1100, 2],
    ///     [1550, 2], [2350, 4], [2450, 4], [1425, 3], [1700, 3]
    /// ]
    ///
    /// let prices: [Double] = [245000, 312000, 279000, 308000, 199000,
    ///                          219000, 405000, 324000, 319000, 255000]
    ///
    /// // Same seed produces the same shuffle order
    /// let (trainFeatures, testFeatures) = features.trainTestSplit(testRatio: 0.2, seed: 42)
    /// let (trainPrices, testPrices) = prices.trainTestSplit(testRatio: 0.2, seed: 42)
    /// ```
    ///
    /// - Parameters:
    ///   - testRatio: The fraction of elements to place in the test set (between 0 and 1, exclusive).
    ///   - seed: A UInt64 seed for the random number generator, ensuring reproducibility.
    /// - Returns: A named tuple of `(train: [Element], test: [Element])`.
    func trainTestSplit(testRatio: Double, seed: UInt64) -> (train: [Element], test: [Element]) {
        return _Sampling.trainTestSplit(self, testRatio: testRatio, seed: seed)
    }

    /// Splits paired feature and label arrays into training and testing subsets,
    /// preserving the class distribution from the labels in both sets.
    ///
    /// Standard random splitting can produce skewed partitions when classes are
    /// imbalanced. For example, if only 5% of samples are positive, a random
    /// 80/20 split might put all positive samples in one partition. Stratified
    /// splitting prevents this by splitting each class independently, so both
    /// sets reflect the original class ratios.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let features: [[Double]] = [
    ///     [619, 15000], [502, 78000], [699, 0],
    ///     [850, 11000], [645, 125000], [720, 98000],
    ///     [410, 45000], [780, 0], [590, 175000], [680, 62000]
    /// ]
    /// let labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    ///
    /// let split = features.stratifiedSplit(labels: labels, testRatio: 0.2, seed: 42)
    /// // split.trainFeatures, split.testFeatures, split.trainLabels, split.testLabels
    /// // Both partitions preserve the 50/50 class ratio
    /// ```
    ///
    /// - Parameters:
    ///   - labels: The class labels for each element, same length as the array.
    ///   - testRatio: The fraction of elements for the test set (between 0 and 1, exclusive).
    ///   - seed: A UInt64 seed for the random number generator, ensuring reproducibility.
    /// - Returns: A named tuple of `(trainFeatures, testFeatures, trainLabels, testLabels)`.
    func stratifiedSplit<L: Hashable>(
        labels: [L],
        testRatio: Double,
        seed: UInt64
    ) -> (trainFeatures: [Element], testFeatures: [Element], trainLabels: [L], testLabels: [L]) {
        return _Sampling.stratifiedSplit(
            features: self, labels: labels,
            testRatio: testRatio, seed: seed
        )
    }
}

// MARK: - Oversampling

public extension Array where Element == [Double] {

    /// Balances an imbalanced dataset by generating synthetic samples for smaller classes.
    ///
    /// When one class has far fewer samples than another, models tend to predict
    /// the larger class and ignore the smaller one. This method identifies every
    /// class below the largest count and generates new samples by interpolating
    /// between existing points in vector space — placing each synthetic point
    /// on the line between two real samples from the same class.
    ///
    /// The smaller class is detected automatically. For multi-class data,
    /// every class with fewer samples than the largest class is oversampled independently.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let features: [[Double]] = [
    ///     [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [1.2, 2.1],
    ///     [7.0, 8.0], [7.5, 8.5]
    /// ]
    /// let labels = [0, 0, 0, 0, 1, 1]
    ///
    /// let (balanced, balancedLabels) = features.oversample(labels: labels)
    /// // balanced.count == 8 (4 original class 0 + 2 original class 1 + 2 synthetic class 1)
    /// // balancedLabels == [0, 0, 0, 0, 1, 1, 1, 1]
    /// ```
    ///
    /// - Parameter labels: The class label for each row, same length as the array.
    /// - Returns: A tuple of `(features, labels)` with all classes balanced to the majority count.
    func oversample(labels: [Int]) -> (features: [[Double]], labels: [Int]) {
        precondition(self.count == labels.count,
            "Features and labels must have the same number of samples")
        precondition(!self.isEmpty, "Features array must not be empty")

        // Group indices by class
        var classBuckets: [Int: [Int]] = [:]
        for i in 0..<labels.count {
            classBuckets[labels[i], default: []].append(i)
        }

        // Find the majority count
        let majorityCount = classBuckets.values.map { $0.count }.max() ?? 0

        // Start with all original data
        var resultFeatures = self
        var resultLabels = labels

        // Oversample each class below the majority count
        for (label, indices) in classBuckets {
            let needed = majorityCount - indices.count
            if needed <= 0 { continue }

            // Gather existing points for this class
            let points = indices.map { self[$0] }

            // Generate synthetic points by interpolating between class members
            for i in 0..<needed {
                let base = points[i % points.count]
                let neighbor = points[(i + 1) % points.count]
                let diff = neighbor.subtract(base)
                let t = Double.random(in: 0.0...1.0)
                let point = base.add(diff.broadcast(multiplyingBy: t))
                resultFeatures.append(point)
                resultLabels.append(label)
            }
        }

        return (features: resultFeatures, labels: resultLabels)
    }
}
