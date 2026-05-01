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

// MARK: - Seeded Random Number Generator

/// A deterministic random number generator that produces the same sequence for a given seed.
///
/// Swift's built-in `SystemRandomNumberGenerator` is intentionally non-reproducible.
/// Apple provides the `RandomNumberGenerator` protocol so we can plug in our own
/// when reproducibility is required — such as generating identical train/test splits
/// across multiple calls.
///
/// Uses the xorshift64 algorithm: three XOR-shift operations on a `UInt64` value.
/// Same seed always produces the same sequence of numbers.
internal struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    /// Creates a generator seeded with the given value.
    init(seed: UInt64) {
        // A zero state would produce all zeros, so we default to 1
        self.state = seed == 0 ? 1 : seed
    }

    /// Returns the next random UInt64 by scrambling the internal state.
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// MARK: - Sampling Operations

/// Internal namespace for sampling operations that work on any element type.
///
/// This is separate from `_Vector` because sampling does not require a `Numeric`
/// constraint — splitting works on arrays of any type.
internal enum _Sampling {

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
}
