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

    /// Draws a single random sample of `count` elements from the array.
    ///
    /// Unlike the sampling-distribution methods, which draw many samples internally
    /// and return only a statistic of each, this returns the drawn elements themselves
    /// so they can be inspected, printed, or summarized directly. It is the atomic
    /// draw those methods are built on.
    ///
    /// With replacement, each draw picks independently, so a value may appear more
    /// than once and `count` may exceed the array's length. Without replacement, every
    /// drawn value is distinct, so `count` may not exceed the array's length —
    /// requesting more triggers a precondition failure. ``trainTestSplit(testRatio:seed:)``
    /// is the task-specific form of a without-replacement draw, partitioned for model
    /// evaluation rather than returned as a single sample.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let literacyRates = [73.4, 81.2, 64.9, 90.1, 55.3, 78.6, 88.0, 69.5]
    ///
    /// // Draw 5 districts with replacement, then take the sample mean as a point estimate.
    /// let drawn = literacyRates.sample(5, replace: true, seed: 31208)
    /// let estimate = drawn.mean()
    /// ```
    ///
    /// - Parameters:
    ///   - count: The number of elements to draw. A `count` of `0` returns an empty array.
    ///   - replace: Whether a value may be selected more than once. Defaults to `true`.
    ///   - seed: A `UInt64` seed for reproducible sampling. A seed of `0` is remapped
    ///     to `1` internally, so `seed: 0` and `seed: 1` produce the same sample —
    ///     start a seed sweep at `1` to avoid the duplicate.
    /// - Returns: An array of `count` drawn elements. Empty if the array is empty or
    ///   `count <= 0`.
    func sample(_ count: Int, replace: Bool = true, seed: UInt64) -> [Element] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        return _Sampling.sample(self, count: count, replace: replace, using: &generator)
    }

    /// Draws a single random sample of `count` elements from the array, drawing from
    /// a caller-supplied random number generator.
    ///
    /// Identical to ``sample(_:replace:seed:)`` except that the caller threads in an
    /// existing generator, mirroring the standard library's `Array.shuffled(using:)`.
    /// Pass a `SeededRandomNumberGenerator` to reuse one seeded stream across several
    /// draws, or a `SystemRandomNumberGenerator` for unseeded sampling.
    ///
    /// - Parameters:
    ///   - count: The number of elements to draw. A `count` of `0` returns an empty array.
    ///   - replace: Whether a value may be selected more than once. Defaults to `true`.
    ///   - generator: The random number generator to draw from, passed `inout`.
    /// - Returns: An array of `count` drawn elements. Empty if the array is empty or
    ///   `count <= 0`.
    func sample(
        _ count: Int,
        replace: Bool = true,
        using generator: inout some RandomNumberGenerator
    ) -> [Element] {
        return _Sampling.sample(self, count: count, replace: replace, using: &generator)
    }

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

// MARK: - Resampling

public extension Array where Element == Double {

    /// Returns the resampled distribution of a statistic.
    ///
    /// Resamples the array with replacement many times, applies a statistic to
    /// each resample, and returns the resulting distribution of values. Pair
    /// the result with ``percentileCI(level:)`` to turn the distribution into
    /// a percentile-based confidence interval. This is the technique known in
    /// the statistics literature as the *bootstrap*.
    ///
    /// The resampling uses Quiver's seeded generator, so the same `seed` always
    /// produces the same distribution. The closure receives a fresh resample
    /// each iteration and returns a single statistic.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]
    ///
    /// // Resampled distribution of the median
    /// let medians = scores.resampled(iterations: 1000, seed: 42) { resample in
    ///     resample.median() ?? 0.0
    /// }
    /// let ci = medians.percentileCI(level: 0.95)
    /// ```
    ///
    /// - Parameters:
    ///   - iterations: Number of resamples to draw. Defaults to 1,000.
    ///   - seed: A `UInt64` seed for reproducible resampling.
    ///   - statistic: A closure that takes a resample and returns a single value.
    /// - Returns: An array of length `iterations` containing the statistic computed
    ///   on each resample. Empty if the input array is empty or `iterations <= 0`.
    func resampled(
        iterations: Int = 1000,
        seed: UInt64,
        statistic: ([Double]) -> Double
    ) -> [Double] {
        guard !self.isEmpty, iterations > 0 else { return [] }

        var rng = SeededRandomNumberGenerator(seed: seed)
        let n = self.count
        var results: [Double] = []
        results.reserveCapacity(iterations)

        var resample = [Double](repeating: 0.0, count: n)
        for _ in 0..<iterations {
            // Draw n indices with replacement, build the resample, apply the statistic
            for i in 0..<n {
                let index = Int.random(in: 0..<n, using: &rng)
                resample[i] = self[index]
            }
            results.append(statistic(resample))
        }
        return results
    }

    /// Shared sampling-distribution engine used by the named convenience methods.
    ///
    /// Draws `iterations` samples of size `sampleSize` with replacement from `self`,
    /// applies `statistic` to each sample, and returns the resulting array. The
    /// engine is generic over the random number generator so a caller can either
    /// seed one internally or thread an existing generator through the loop.
    ///
    /// `sampleSize` is allowed to exceed `self.count`; sampling with replacement
    /// makes that mathematically legal.
    fileprivate func _samplingDistribution<G: RandomNumberGenerator>(
        sampleSize: Int,
        iterations: Int,
        using generator: inout G,
        statistic: ([Double]) -> Double
    ) -> [Double] {
        guard !self.isEmpty, iterations > 0, sampleSize > 0 else { return [] }

        let n = self.count
        var results: [Double] = []
        results.reserveCapacity(iterations)

        var sample = [Double](repeating: 0.0, count: sampleSize)
        for _ in 0..<iterations {
            for i in 0..<sampleSize {
                let index = Int.random(in: 0..<n, using: &generator)
                sample[i] = self[index]
            }
            results.append(statistic(sample))
        }
        return results
    }

    /// Returns the sampling distribution of an arbitrary statistic.
    ///
    /// Draws `iterations` samples of size `sampleSize` with replacement from `self`,
    /// applies `statistic` to each sample, and returns the resulting array. This is
    /// the general engine behind the named convenience methods — `mean`, `median`,
    /// and standard deviation are three instances of it. Reach for it directly when
    /// the statistic is something the named methods do not cover, such as a
    /// percentile or the range, at a sample size of your choosing.
    ///
    /// Unlike ``resampled(iterations:seed:statistic:)``, which fixes the sample
    /// size to the population size `n` (the bootstrap), this method lets the
    /// caller pick any `sampleSize`. Sampling is with replacement, so a
    /// `sampleSize` larger than `self.count` is allowed.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let population = [Double].randomNormal(10_000, mean: 50, standardDeviation: 8)
    ///
    /// // Sampling distribution of the 90th percentile at sample size 40
    /// let p90s = population.samplingDistribution(sampleSize: 40, iterations: 1000, seed: 42) { sample in
    ///     sample.percentile(90.0) ?? .nan
    /// }
    /// p90s.mean()  // the average 90th-percentile estimate across samples
    /// ```
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - seed: A `UInt64` seed for reproducible sampling. A seed of `0` is
    ///     remapped to `1` internally, so `seed: 0` and `seed: 1` produce the
    ///     same distribution — start a seed sweep at `1` to avoid the duplicate.
    ///   - statistic: A closure that takes a sample and returns a single value.
    /// - Returns: An array of length `iterations` containing the statistic computed
    ///   on each sample. Empty if `self` is empty, `iterations <= 0`, or
    ///   `sampleSize <= 0`.
    func samplingDistribution(
        sampleSize: Int,
        iterations: Int = 1000,
        seed: UInt64,
        statistic: ([Double]) -> Double
    ) -> [Double] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator,
            statistic: statistic
        )
    }

    /// Returns the sampling distribution of an arbitrary statistic, drawing from
    /// a caller-supplied random number generator.
    ///
    /// Identical to ``samplingDistribution(sampleSize:iterations:seed:statistic:)``
    /// except that the caller threads in an existing generator, mirroring the
    /// standard library's `Array.shuffled(using:)`. Pass a
    /// `SeededRandomNumberGenerator` to reuse one seeded stream across a pipeline,
    /// or a `SystemRandomNumberGenerator` for unseeded sampling.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - generator: The random number generator to draw from, passed `inout`.
    ///   - statistic: A closure that takes a sample and returns a single value.
    /// - Returns: An array of length `iterations` containing the statistic computed
    ///   on each sample. Empty if `self` is empty, `iterations <= 0`, or
    ///   `sampleSize <= 0`.
    func samplingDistribution(
        sampleSize: Int,
        iterations: Int = 1000,
        using generator: inout some RandomNumberGenerator,
        statistic: ([Double]) -> Double
    ) -> [Double] {
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator,
            statistic: statistic
        )
    }

    /// Returns the sampling distribution of the sample mean.
    ///
    /// Draws `iterations` samples of size `sampleSize` with replacement from `self`,
    /// computes the mean of each sample, and returns the resulting array of sample
    /// means. The returned array is itself a `[Double]`, so every Quiver statistical
    /// operation works on the sampling distribution directly — `mean()` gives the
    /// average of the sample means, `standardDeviation()` gives the empirical
    /// standard error.
    ///
    /// This is the Central Limit Theorem in code. Even when the source population
    /// is skewed, the distribution of sample means is approximately normal once
    /// the sample size is large enough.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let population = [Double].randomExponential(10_000, rate: 0.5)
    /// let sampleMeans = population.samplingDistributionOfMean(
    ///     sampleSize: 50, iterations: 1000, seed: 42
    /// )
    /// sampleMeans.mean()              // ≈ 2.0
    /// sampleMeans.standardDeviation() // ≈ 0.28
    /// ```
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - seed: A `UInt64` seed for reproducible sampling. A seed of `0` is
    ///     remapped to `1` internally, so `seed: 0` and `seed: 1` produce the
    ///     same distribution — start a seed sweep at `1` to avoid the duplicate.
    /// - Returns: An array of length `iterations` containing the sample means.
    ///   Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfMean(
        sampleSize: Int,
        iterations: Int = 1000,
        seed: UInt64
    ) -> [Double] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.mean() ?? .nan }
    }

    /// Returns the sampling distribution of the sample mean, drawing from a
    /// caller-supplied random number generator.
    ///
    /// Identical to ``samplingDistributionOfMean(sampleSize:iterations:seed:)``
    /// except that the caller threads in an existing generator, mirroring the
    /// standard library's `Array.shuffled(using:)`. Pass a
    /// `SeededRandomNumberGenerator` to reuse one seeded stream across a pipeline,
    /// or a `SystemRandomNumberGenerator` for unseeded sampling.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - generator: The random number generator to draw from, passed `inout`.
    /// - Returns: An array of length `iterations` containing the sample means.
    ///   Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfMean(
        sampleSize: Int,
        iterations: Int = 1000,
        using generator: inout some RandomNumberGenerator
    ) -> [Double] {
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.mean() ?? .nan }
    }

    /// Returns the sampling distribution of the sample median.
    ///
    /// Draws `iterations` samples of size `sampleSize` with replacement from `self`,
    /// computes the median of each sample, and returns the resulting array. The
    /// median's sampling distribution typically has a wider spread than the
    /// mean's at the same sample size — quantifiable evidence that the mean is
    /// the more statistically efficient summary on symmetric data.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - seed: A `UInt64` seed for reproducible sampling. A seed of `0` is
    ///     remapped to `1` internally, so `seed: 0` and `seed: 1` produce the
    ///     same distribution — start a seed sweep at `1` to avoid the duplicate.
    /// - Returns: An array of length `iterations` containing the sample medians.
    ///   Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfMedian(
        sampleSize: Int,
        iterations: Int = 1000,
        seed: UInt64
    ) -> [Double] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.median() ?? .nan }
    }

    /// Returns the sampling distribution of the sample median, drawing from a
    /// caller-supplied random number generator.
    ///
    /// Identical to ``samplingDistributionOfMedian(sampleSize:iterations:seed:)``
    /// except that the caller threads in an existing generator, mirroring the
    /// standard library's `Array.shuffled(using:)`. Pass a
    /// `SeededRandomNumberGenerator` to reuse one seeded stream across a pipeline,
    /// or a `SystemRandomNumberGenerator` for unseeded sampling.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - generator: The random number generator to draw from, passed `inout`.
    /// - Returns: An array of length `iterations` containing the sample medians.
    ///   Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfMedian(
        sampleSize: Int,
        iterations: Int = 1000,
        using generator: inout some RandomNumberGenerator
    ) -> [Double] {
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.median() ?? .nan }
    }

    /// Returns the sampling distribution of the sample standard deviation.
    ///
    /// Draws `iterations` samples of size `sampleSize` with replacement from `self`,
    /// computes the sample standard deviation (`ddof: 1`) of each, and returns
    /// the resulting array. Reads as the spread of an estimator: how much the
    /// sample standard deviation varies from one sample to the next.
    ///
    /// > Note: The sample standard deviation is undefined for `sampleSize: 1` —
    /// > it divides by `n − 1 = 0` — so each such entry is `.nan` rather than a
    /// > fabricated `0.0`. A single `.nan` propagates through any aggregate of the
    /// > result: `mean()`, `standardDeviation()`, and histogramming all return or
    /// > render `.nan` once one entry is non-finite. Use a `sampleSize` of at least
    /// > 2, or filter the result with `filter { $0.isFinite }` before aggregating.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - seed: A `UInt64` seed for reproducible sampling. A seed of `0` is
    ///     remapped to `1` internally, so `seed: 0` and `seed: 1` produce the
    ///     same distribution — start a seed sweep at `1` to avoid the duplicate.
    /// - Returns: An array of length `iterations` containing the sample standard
    ///   deviations. Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfStandardDeviation(
        sampleSize: Int,
        iterations: Int = 1000,
        seed: UInt64
    ) -> [Double] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.standardDeviation() ?? .nan }
    }

    /// Returns the sampling distribution of the sample standard deviation,
    /// drawing from a caller-supplied random number generator.
    ///
    /// Identical to
    /// ``samplingDistributionOfStandardDeviation(sampleSize:iterations:seed:)``
    /// except that the caller threads in an existing generator, mirroring the
    /// standard library's `Array.shuffled(using:)`. Pass a
    /// `SeededRandomNumberGenerator` to reuse one seeded stream across a pipeline,
    /// or a `SystemRandomNumberGenerator` for unseeded sampling.
    ///
    /// > Note: As with the seeded overload, `sampleSize: 1` yields `.nan` entries
    /// > because the sample standard deviation is undefined for a single
    /// > observation. Filter with `filter { $0.isFinite }` before aggregating.
    ///
    /// - Parameters:
    ///   - sampleSize: Size of each sample drawn from `self` (with replacement).
    ///   - iterations: Number of samples to draw. Defaults to 1,000.
    ///   - generator: The random number generator to draw from, passed `inout`.
    /// - Returns: An array of length `iterations` containing the sample standard
    ///   deviations. Empty if `self` is empty, `iterations <= 0`, or `sampleSize <= 0`.
    func samplingDistributionOfStandardDeviation(
        sampleSize: Int,
        iterations: Int = 1000,
        using generator: inout some RandomNumberGenerator
    ) -> [Double] {
        return _samplingDistribution(
            sampleSize: sampleSize,
            iterations: iterations,
            using: &generator
        ) { sample in sample.standardDeviation() ?? .nan }
    }
}
