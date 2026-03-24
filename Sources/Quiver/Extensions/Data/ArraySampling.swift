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
