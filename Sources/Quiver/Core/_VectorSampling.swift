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

// MARK: - Sampling Operations

/// Internal namespace for sampling operations that work on any element type.
///
/// This is separate from `_Vector` because sampling does not require a `Numeric`
/// constraint — splitting works on arrays of any type.
internal enum _Sampling {

    /// Draws a single sample of `count` elements from `elements`, with or without replacement.
    ///
    /// With replacement, each draw picks an independent uniform index, so a value may
    /// appear more than once and `count` may exceed `elements.count`. Without replacement,
    /// the indices are shuffled with the seeded generator and the first `count` are taken,
    /// so every drawn value is distinct and `count` may not exceed `elements.count`.
    ///
    /// - Parameters:
    ///   - elements: The source array to draw from.
    ///   - count: The number of elements to draw.
    ///   - replace: Whether a value may be selected more than once.
    ///   - generator: The random number generator to draw from, passed `inout`.
    /// - Returns: An array of `count` drawn elements, or an empty array when `elements`
    ///   is empty or `count <= 0`.
    static func sample<T, G: RandomNumberGenerator>(
        _ elements: [T],
        count: Int,
        replace: Bool,
        using generator: inout G
    ) -> [T] {
        precondition(replace || count <= elements.count,
            "Cannot draw \(count) elements without replacement from \(elements.count) elements")

        guard !elements.isEmpty, count > 0 else { return [] }

        let n = elements.count

        if replace {
            // Draw count independent uniform indices; a value may repeat.
            var result: [T] = []
            result.reserveCapacity(count)
            for _ in 0..<count {
                let index = Int.random(in: 0..<n, using: &generator)
                result.append(elements[index])
            }
            return result
        }

        // Without replacement: shuffle the indices and take the first count.
        let shuffledIndices = (0..<n).shuffled(using: &generator)
        return shuffledIndices[..<count].map { elements[$0] }
    }

    /// Splits array elements into training and testing partitions using a seeded shuffle.
    ///
    /// The algorithm shuffles an index array using the seeded generator, then slices
    /// the shuffled indices at the split point.
    ///
    /// - Parameters:
    ///   - elements: The source array to split.
    ///   - testRatio: The fraction of elements for the test set (between 0 and 1, exclusive).
    ///   - seed: A UInt64 seed for reproducible shuffling.
    /// - Returns: A named tuple of `(train: [T], test: [T])`.
    static func trainTestSplit<T>(_ elements: [T], testRatio: Double, seed: UInt64) -> (train: [T], test: [T]) {
        precondition(!elements.isEmpty, "Array must not be empty")
        precondition(testRatio > 0.0 && testRatio < 1.0,
            "Test ratio must be between 0 and 1 (exclusive)")

        // Create a seeded generator and shuffle indices using native Swift
        var rng = SeededRandomNumberGenerator(seed: seed)
        let shuffledIndices = (0..<elements.count).shuffled(using: &rng)

        // Split at the computed boundary
        let testCount = Int(ceil(Double(elements.count) * testRatio))
        let test = shuffledIndices[..<testCount].map { elements[$0] }
        let train = shuffledIndices[testCount...].map { elements[$0] }

        return (train: train, test: test)
    }

    /// Splits paired features and labels into training and testing partitions,
    /// preserving the class distribution from the labels array in both sets.
    ///
    /// Groups elements by their label, shuffles each group independently using the
    /// seeded generator, then takes the specified ratio from each group for the test set.
    /// This guarantees that rare classes appear in both partitions proportionally.
    ///
    /// - Parameters:
    ///   - features: The feature array to split (rows of data).
    ///   - labels: The label array, same length as features.
    ///   - testRatio: The fraction of elements for the test set (between 0 and 1, exclusive).
    ///   - seed: A UInt64 seed for reproducible shuffling.
    /// - Returns: A named tuple of `(trainFeatures: [T], testFeatures: [T], trainLabels: [L], testLabels: [L])`.
    static func stratifiedSplit<T, L: Hashable>(
        features: [T],
        labels: [L],
        testRatio: Double,
        seed: UInt64
    ) -> (trainFeatures: [T], testFeatures: [T], trainLabels: [L], testLabels: [L]) {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == labels.count,
            "Features and labels must have the same length")
        precondition(testRatio > 0.0 && testRatio < 1.0,
            "Test ratio must be between 0 and 1 (exclusive)")

        // Group indices by label
        var groups: [L: [Int]] = [:]
        for (i, label) in labels.enumerated() {
            groups[label, default: []].append(i)
        }

        var rng = SeededRandomNumberGenerator(seed: seed)
        var trainIndices: [Int] = []
        var testIndices: [Int] = []

        // Process groups in sorted order for deterministic results
        let sortedKeys = groups.keys.sorted { "\($0)" < "\($1)" }

        for key in sortedKeys {
            guard var indices = groups[key], !indices.isEmpty else {
                continue
            }
            indices.shuffle(using: &rng)

            // At least 1 in test per class (if class has 2+ samples)
            let testCount = max(1, Int(ceil(Double(indices.count) * testRatio)))
            let actualTestCount = indices.count > 1 ? min(testCount, indices.count - 1) : 0

            testIndices.append(contentsOf: indices[..<actualTestCount])
            trainIndices.append(contentsOf: indices[actualTestCount...])
        }

        let trainFeatures = trainIndices.map { features[$0] }
        let testFeatures = testIndices.map { features[$0] }
        let trainLabels = trainIndices.map { labels[$0] }
        let testLabels = testIndices.map { labels[$0] }

        return (trainFeatures: trainFeatures, testFeatures: testFeatures,
                trainLabels: trainLabels, testLabels: testLabels)
    }

    /// Partitions `count` sample indices into `k` train/validation folds via a
    /// seeded shuffle.
    ///
    /// Shuffles the index range `0..<count` once with the seeded generator, then
    /// divides the shuffled order into `k` contiguous, near-equal blocks. Each
    /// block serves as one fold's validation set; that fold's training set is
    /// every index not in the block. Across the returned folds, every index
    /// appears in exactly one validation set — complete coverage with no overlap.
    /// When `count` is not divisible by `k`, the first `count % k` folds receive
    /// one extra validation index, so fold sizes differ by at most one.
    ///
    /// This returns index sets, not sliced data, so the caller controls what
    /// happens inside each fold — fitting a scaler on the training indices alone,
    /// for instance, rather than leaking validation statistics into the fit.
    ///
    /// - Parameters:
    ///   - count: The number of sample indices to partition. Must be positive.
    ///   - k: The number of folds. Must be at least 2 and at most `count`.
    ///   - seed: A UInt64 seed for reproducible shuffling.
    /// - Returns: An array of `k` named tuples `(train: [Int], validation: [Int])`,
    ///   each holding sample indices into a parallel data array.
    static func kFoldIndices(count: Int, k: Int, seed: UInt64) -> [(train: [Int], validation: [Int])] {
        precondition(count > 0, "Sample count must be positive, got \(count)")
        precondition(k >= 2, "Fold count k must be at least 2, got \(k)")
        precondition(k <= count,
            "Fold count k (\(k)) cannot exceed the sample count (\(count))")

        // Shuffle the index range once; the shuffle order drives every fold.
        var rng = SeededRandomNumberGenerator(seed: seed)
        let shuffled = (0..<count).shuffled(using: &rng)

        // Base block size, with the first `remainder` folds taking one extra so
        // the validation sizes differ by at most one when count % k != 0.
        let baseSize = count / k
        let remainder = count % k

        var folds: [(train: [Int], validation: [Int])] = []
        folds.reserveCapacity(k)

        var start = 0
        for fold in 0..<k {
            let foldSize = baseSize + (fold < remainder ? 1 : 0)
            let end = start + foldSize

            // The contiguous block [start, end) is this fold's validation set;
            // everything outside it is the training set.
            let validation = Array(shuffled[start..<end])
            var train = [Int]()
            train.reserveCapacity(count - foldSize)
            train.append(contentsOf: shuffled[0..<start])
            train.append(contentsOf: shuffled[end..<count])

            folds.append((train: train, validation: validation))
            start = end
        }

        return folds
    }
}
