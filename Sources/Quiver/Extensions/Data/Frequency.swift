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

// MARK: - Relative Frequency for Double Arrays

public extension Array where Element == Double {

    /// Returns the relative frequency of `value` in the array.
    ///
    /// Counts how many elements equal `value` and divides by the total count, producing
    /// a number between 0 and 1. This is the empirical probability of drawing `value`
    /// when sampling uniformly at random from the array. The result is the building block
    /// for class priors in Naive Bayes and for any probability question framed as
    /// "what fraction of the data looks like this?"
    ///
    /// Frequencies are computed as `count / total` with no Bessel correction — these are
    /// population frequencies for the data we have, not estimates of an underlying sample.
    /// An empty array returns `0.0` rather than `nil` so that the result is always a
    /// finite probability and never crashes on a missing denominator.
    ///
    /// `NaN` values never compare equal to anything, including themselves. A `NaN` in
    /// the array contributes to the total but cannot be matched by `value`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let rolls = [1.0, 2.0, 1.0, 3.0]
    /// let pOne = rolls.probability(of: 1.0)  // 0.5
    /// ```
    ///
    /// - Parameter value: The value whose relative frequency to compute.
    /// - Returns: The fraction of elements equal to `value`, in the range `[0, 1]`.
    ///   Returns `0.0` for an empty array.
    func probability(of value: Double) -> Double {
        guard !self.isEmpty else { return 0.0 }
        let matches = self.reduce(into: 0) { count, element in
            if element == value { count += 1 }
        }
        return Double(matches) / Double(self.count)
    }

    /// Returns a dictionary mapping each unique value to its relative frequency.
    ///
    /// Builds a complete frequency table over the array. Each key is a unique value that
    /// appears in the data, and each associated value is the fraction of elements equal
    /// to that key. The resulting frequencies sum to `1.0` within floating-point tolerance,
    /// making the table a valid empirical probability distribution.
    ///
    /// This is the natural input to a class-prior table for Naive Bayes, to a histogram,
    /// or to any downstream model that consumes a categorical distribution. As with
    /// ``probability(of:)``, frequencies are computed as `count / total` with no Bessel
    /// correction — these are population frequencies, not sample estimates.
    ///
    /// `NaN` values do not appear as keys. `NaN` is not equal to itself in Swift's
    /// floating-point comparison, so dictionary insertion treats every `NaN` as a
    /// separate key. To keep the table well-defined, callers should filter `NaN` out
    /// of their data before computing a distribution over it.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let labels = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
    /// let dist = labels.frequencyDistribution()
    /// // [1.0: 0.5, 2.0: 0.333..., 3.0: 0.166...]
    /// ```
    ///
    /// - Returns: A dictionary mapping each unique value to its relative frequency.
    ///   Returns an empty dictionary for an empty array.
    func frequencyDistribution() -> [Double: Double] {
        guard !self.isEmpty else { return [:] }
        var counts: [Double: Int] = [:]
        for element in self {
            counts[element, default: 0] += 1
        }
        let total = Double(self.count)
        var distribution: [Double: Double] = [:]
        distribution.reserveCapacity(counts.count)
        for (key, count) in counts {
            distribution[key] = Double(count) / total
        }
        return distribution
    }
}

// MARK: - Distinct Values for Hashable & Comparable Arrays

public extension Array where Element: Hashable & Comparable {

    /// Returns the unique values in the array, sorted in ascending order.
    ///
    /// Removes duplicates and orders the result deterministically — calling `distinct()`
    /// twice on the same input always produces the same output, which `Array(Set(self))`
    /// does not guarantee. This determinism matters for tests, for snapshot comparisons,
    /// and for any pipeline that compares results across runs.
    ///
    /// The method is generic over `Hashable & Comparable`, so it works on `[Double]`,
    /// `[Int]`, `[String]`, and any custom type that conforms to both protocols. The
    /// `Hashable` constraint enables fast deduplication via a hash set, and the
    /// `Comparable` constraint provides the natural ascending order used in the output.
    ///
    /// The name `distinct()` rather than `unique()` is chosen to avoid collision with
    /// the swift-algorithms package, which exposes `uniqued()` on `Sequence`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [3.0, 1.0, 2.0, 1.0]
    /// scores.distinct()  // [1.0, 2.0, 3.0]
    ///
    /// let words = ["beta", "alpha", "alpha", "gamma"]
    /// words.distinct()   // ["alpha", "beta", "gamma"]
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    ///   Deduplication is linear; the trailing sort dominates.
    /// - Returns: The unique elements in ascending order.
    ///   Returns an empty array for an empty input.
    func distinct() -> [Element] {
        guard !self.isEmpty else { return [] }
        return Array(Set(self)).sorted()
    }

    /// Returns each unique value paired with the number of times it appears.
    ///
    /// Produces a frequency table as `(value, count)` tuples sorted in ascending order
    /// by value. This is the integer-count companion to ``Array/frequencyDistribution()``
    /// — the same shape of data, but raw counts instead of relative frequencies. The
    /// ascending-by-value sort makes the output deterministic and easy to read in
    /// tabular form.
    ///
    /// The tuple labels `(value:, count:)` are part of the API. Callers can pattern-match
    /// or access fields by label rather than by tuple index, which keeps call sites
    /// self-documenting.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let rolls = [3.0, 1.0, 2.0, 1.0]
    /// let table = rolls.distinctCounts()
    /// // [(value: 1.0, count: 2), (value: 2.0, count: 1), (value: 3.0, count: 1)]
    ///
    /// for entry in table {
    ///     print("\(entry.value) appeared \(entry.count) time(s)")
    /// }
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of unique elements.
    ///   Counting is linear; the trailing sort dominates.
    /// - Returns: An array of `(value, count)` tuples sorted in ascending order by value.
    ///   Returns an empty array for an empty input.
    func distinctCounts() -> [(value: Element, count: Int)] {
        guard !self.isEmpty else { return [] }
        var counts: [Element: Int] = [:]
        for element in self {
            counts[element, default: 0] += 1
        }
        return counts
            .map { (value: $0.key, count: $0.value) }
            .sorted { $0.value < $1.value }
    }
}
