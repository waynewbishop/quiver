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

// MARK: - Basic Statistics for Numeric Types

public extension Array where Element: Numeric {
    /// Returns the sum of all elements in the array.
    ///
    /// Adds every element together using the `+` operator. Returns the additive identity
    /// (zero) for an empty array. Works with any `Numeric` type including `Int`, `Double`,
    /// and `Float`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let revenue = [1200.0, 950.0, 1800.0, 1100.0]
    /// let total = revenue.sum()  // 5050.0
    /// ```
    ///
    /// - Returns: The sum of all elements, or zero for an empty array
    func sum() -> Element {
        let vector = _Vector(elements: self)
        return vector.sum()
    }
    
    /// Returns the product of all elements in the array.
    ///
    /// Multiplies every element together using the `*` operator. This is useful for
    /// computing factorials, compound growth rates, and probability products where
    /// independent events are multiplied together.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let growthFactors = [1.05, 1.03, 0.98, 1.07]
    /// let compoundGrowth = growthFactors.product()  // 1.1367
    /// ```
    ///
    /// - Returns: The product of all elements
    func product() -> Element {
        let vector = _Vector(elements: self)
        return vector.product()
    }
    
    /// Returns the running total at each position in the array.
    ///
    /// Each element in the result is the sum of all elements from the start of the array
    /// up to and including that position. This is useful for tracking running balances,
    /// computing empirical distribution functions, and converting individual values into
    /// cumulative totals.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let dailySales = [120.0, 95.0, 140.0, 110.0, 130.0]
    /// let runningTotal = dailySales.cumulativeSum()
    /// // [120.0, 215.0, 355.0, 465.0, 595.0]
    /// ```
    ///
    /// - Returns: Array of running totals with the same length as the input
    func cumulativeSum() -> [Element] {
        let vector = _Vector(elements: self)
        return vector.cumulativeSum().elements
    }
    
    /// Returns the running product at each position in the array.
    ///
    /// Each element in the result is the product of all elements from the start of the
    /// array up to and including that position. This is useful for computing compound
    /// growth over time and factorial sequences.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let monthlyGrowth = [1.05, 1.03, 0.98, 1.07]
    /// let cumulativeGrowth = monthlyGrowth.cumulativeProduct()
    /// // [1.05, 1.0815, 1.05987, 1.13406]
    /// ```
    ///
    /// - Returns: Array of running products with the same length as the input
    func cumulativeProduct() -> [Element] {
        let vector = _Vector(elements: self)
        return vector.cumulativeProduct().elements
    }
}

// MARK: - Operations requiring comparison

public extension Array where Element: Numeric & Comparable {
    /// Returns the index of the smallest element in the array.
    ///
    /// Finds the position of the minimum value rather than the value itself. This is
    /// useful when the index carries meaning — for example, identifying which day had
    /// the lowest temperature or which feature has the smallest weight.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [88.0, 72.0, 95.0, 81.0]
    /// let worstIndex = scores.argMin()  // 1 (72.0 is at index 1)
    /// ```
    ///
    /// - Returns: The index of the minimum element, or nil if the array is empty
    func argMin() -> Int? {
        let vector = _Vector(elements: self)
        return vector.minWithIndex()?.index
    }
    
    /// Returns the index of the largest element in the array.
    ///
    /// Finds the position of the maximum value rather than the value itself. This is
    /// useful for identifying which category, feature, or time period produced the
    /// highest result.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [88.0, 72.0, 95.0, 81.0]
    /// let bestIndex = scores.argMax()  // 2 (95.0 is at index 2)
    /// ```
    ///
    /// - Returns: The index of the maximum element, or nil if the array is empty
    func argMax() -> Int? {
        let vector = _Vector(elements: self)
        return vector.maxWithIndex()?.index
    }
}

// MARK: - Additional Statistics for FloatingPoint Types

public extension Array where Element: FloatingPoint {
    /// Returns the arithmetic mean (average) of all elements in the array.
    ///
    /// The mean is the sum of all values divided by the count. It represents the center
    /// of the distribution but is sensitive to outliers — a single extreme value can shift
    /// the mean significantly. Compare with `median()` to detect skewed distributions.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let prices = [29.99, 49.99, 19.99, 39.99, 24.99]
    /// let avgPrice = prices.mean()  // 32.99
    /// ```
    ///
    /// - Returns: The arithmetic mean, or nil if the array is empty
    func mean() -> Element? {
        let vector = _Vector(elements: self)
        return vector.mean()
    }
    
    /// Returns the median (middle value) of the array.
    ///
    /// The median is the value that separates the upper half from the lower half of the
    /// data. Unlike the mean, it is resistant to outliers — a single extreme value does
    /// not shift it. For an even number of elements, the median is the average of the
    /// two middle values.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let responseTimes = [12.0, 15.0, 14.0, 450.0, 13.0]
    /// let median = responseTimes.median()  // 14.0 (unaffected by the 450 outlier)
    /// let mean = responseTimes.mean()      // 100.8 (pulled up by the outlier)
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    ///   Sorts the array internally.
    /// - Returns: The median value, or nil if the array is empty
    func median() -> Element? {
        let vector = _Vector(elements: self)
        return vector.median()
    }
    
    /// Returns the variance of all elements in the array.
    ///
    /// Variance measures how far values spread from the mean. It is computed as the
    /// average of squared deviations from the mean. A variance of zero means all values
    /// are identical. Variance is the square of the standard deviation.
    ///
    /// The `ddof` parameter controls whether to compute sample variance (dividing by
    /// `n - 1`, the default) or population variance (dividing by `n`). Use sample variance
    /// when the data is a subset drawn from a larger population; use population variance
    /// when the data represents the entire set.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [4.0, 7.0, 2.0, 9.0, 3.0]
    ///
    /// let sampleVar = data.variance()         // 8.5 (sample, ddof: 1)
    /// let popVar = data.variance(ddof: 0)     // 6.8 (population, ddof: 0)
    /// ```
    ///
    /// - Parameter ddof: Delta Degrees of Freedom — 1 for sample (default), 0 for population
    /// - Returns: The variance, or nil if the array has fewer elements than `ddof + 1`
    func variance(ddof: Int = 1) -> Element? {
        let vector = _Vector(elements: self)
        return vector.variance(ddof: ddof)
    }

    /// Returns the standard deviation of all elements in the array.
    ///
    /// Standard deviation measures spread in the same units as the original data (unlike
    /// variance, which is in squared units). It is the square root of the variance. A low
    /// standard deviation means values cluster near the mean; a high one means they are
    /// scattered. Standard deviation is the key input to z-score standardization and
    /// outlier detection.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [4.0, 7.0, 2.0, 9.0, 3.0]
    ///
    /// let sampleStd = data.standardDeviation()         // 2.92 (sample, ddof: 1)
    /// let popStd = data.standardDeviation(ddof: 0)     // 2.61 (population, ddof: 0)
    /// ```
    ///
    /// - Parameter ddof: Delta Degrees of Freedom — 1 for sample (default), 0 for population
    /// - Returns: The standard deviation, or nil if the array has fewer elements than `ddof + 1`
    func standardDeviation(ddof: Int = 1) -> Element? {
        let vector = _Vector(elements: self)
        return vector.standardDeviation(ddof: ddof)
    }

    /// Returns the standard error of the mean for the elements in the array.
    ///
    /// The standard error estimates how much the sample mean would vary if the data were
    /// resampled from the same population. It is the standard deviation divided by the
    /// square root of the sample size. Standard error shrinks as the sample grows, which
    /// is why larger samples produce more precise estimates of the mean.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [4.0, 7.0, 2.0, 9.0, 3.0]
    ///
    /// let se = data.standardError()        // 1.30 (sample, ddof: 1)
    /// ```
    ///
    /// - Parameter ddof: Delta Degrees of Freedom — 1 for sample (default), 0 for population
    /// - Returns: The standard error, or nil if the array has fewer elements than `ddof + 1`
    func standardError(ddof: Int = 1) -> Element? {
        let vector = _Vector(elements: self)
        return vector.standardError(ddof: ddof)
    }

    /// Returns the sample skewness of the array — a single number describing how lopsided
    /// the distribution is.
    ///
    /// Skewness is the standardized third moment of the data: `(1/n) * Σ((x - mean)/std)^3`,
    /// using the population standard deviation inside the ratio so the result is a pure shape
    /// measure independent of scale. The sign tells you which way the distribution leans —
    /// positive means a long right tail (mean pulled above the median), negative means a long
    /// left tail (mean pulled below the median). The magnitude tells you how much: a rule of
    /// thumb from intro statistics is that absolute values above 1 indicate substantial skew,
    /// below 0.5 indicate near-symmetry, and the range in between is ambiguous and worth
    /// pairing with a histogram and ``kurtosis(bias:)`` before concluding.
    ///
    /// The default returns the biased moment ratio `g1` — the form most intro-stats
    /// textbooks introduce first. Pass `bias: false` for the Adjusted Fisher-Pearson
    /// estimator `G1 = g1 * sqrt(n(n-1))/(n-2)`, which corrects the downward bias of
    /// `g1` on small samples.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let symmetric = [1.0, 2.0, 3.0, 4.0, 5.0]
    /// symmetric.skewness()                  // 0.0
    ///
    /// let rightSkewed = [1.0, 1.0, 1.0, 1.0, 2.0, 10.0]
    /// rightSkewed.skewness()                // ~1.455 (long right tail)
    /// ```
    ///
    /// - Parameter bias: When `true` (default), returns the biased moment ratio `g1`. When
    ///   `false`, returns the bias-corrected `G1` (requires `n >= 3`).
    /// - Returns: The skewness, or nil if the array has fewer than 3 elements, the standard
    ///   deviation is zero, or any element is not finite.
    func skewness(bias: Bool = true) -> Element? {
        guard self.count >= 3 else { return nil }
        guard self.allSatisfy({ $0.isFinite }) else { return nil }

        let vector = _Vector(elements: self)
        guard let g1 = vector.standardizedMoment(order: 3, ddof: 0) else { return nil }

        if bias {
            return g1
        }
        let n = Element(self.count)
        let correction = (n * (n - 1)).squareRoot() / (n - 2)
        return g1 * correction
    }

    /// Returns the sample excess kurtosis of the array — a single number describing how heavy
    /// the tails of the distribution are relative to a normal.
    ///
    /// Kurtosis is the standardized fourth moment: `(1/n) * Σ((x - mean)/std)^4`. Quiver
    /// returns the *excess* form (Fisher's definition), subtracting 3 so that a normal
    /// distribution has kurtosis 0. Positive excess kurtosis indicates heavier tails than a
    /// normal — more values out in the extremes than the bell curve predicts. Negative excess
    /// kurtosis indicates lighter tails, closer to a uniform distribution. Values within
    /// roughly ±0.5 of zero are consistent with a normal model for most practical purposes;
    /// values beyond ±1 are a strong signal that the normal is the wrong candidate
    /// distribution.
    ///
    /// Fisher's excess kurtosis is the default — the form OpenIntro Statistics and most
    /// modern textbooks use, where a normal distribution has kurtosis zero. The older
    /// Pearson convention (which leaves the `-3` off, so a normal has kurtosis 3) is
    /// not used here. The default returns the biased moment ratio `g2`. Pass
    /// `bias: false` for the sample-size-corrected estimator (requires `n >= 4`).
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let uniformLike = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    /// uniformLike.kurtosis()                // ~-1.224 (light tails, platykurtic)
    ///
    /// let heavyTailed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    /// heavyTailed.kurtosis()                // ~3.0 (heavy right tail)
    /// ```
    ///
    /// - Parameter bias: When `true` (default), returns the biased moment ratio `g2`. When
    ///   `false`, returns the bias-corrected estimator (requires `n >= 4`).
    /// - Returns: The excess kurtosis (normal distribution has value 0), or nil if the array
    ///   has fewer than 4 elements, the standard deviation is zero, or any element is not finite.
    func kurtosis(bias: Bool = true) -> Element? {
        guard self.count >= 4 else { return nil }
        guard self.allSatisfy({ $0.isFinite }) else { return nil }

        let vector = _Vector(elements: self)
        guard let m4Ratio = vector.standardizedMoment(order: 4, ddof: 0) else { return nil }

        let g2 = m4Ratio - 3
        if bias {
            return g2
        }
        let n = Element(self.count)
        let factor = (n - 1) / ((n - 2) * (n - 3))
        return factor * ((n + 1) * g2 + 6)
    }

    /// Detects outliers using the z-score method and returns a boolean mask.
    ///
    /// Each element's z-score is computed as the absolute distance from the mean divided
    /// by the standard deviation. Elements whose z-score exceeds the threshold are marked
    /// `true` in the returned mask. The mask can be passed to `masked(by:)` to extract
    /// the outlier values.
    ///
    /// For performance when processing multiple arrays with the same distribution, pass
    /// pre-calculated `mean` and `standardDeviation` values to avoid recomputing them.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]
    ///
    /// let mask = data.outlierMask(threshold: 2.0)
    /// // [false, false, false, false, false, true, false]
    ///
    /// let outliers = data.masked(by: mask)  // [35.0]
    /// ```
    ///
    /// - Parameters:
    ///   - threshold: Number of standard deviations from the mean to flag as an outlier (default: 2.0)
    ///   - mean: Pre-calculated mean to use instead of computing from the array (optional)
    ///   - standardDeviation: Pre-calculated standard deviation to use instead of computing from the array (optional)
    /// - Returns: Boolean mask where `true` indicates the element at that index is an outlier
    func outlierMask(threshold: Element = 2.0, mean: Element? = nil, standardDeviation: Element? = nil) -> [Bool] {
        guard !self.isEmpty else { return [] }

        let computedMean = mean ?? self.mean() ?? 0
        let computedStd = standardDeviation ?? self.standardDeviation() ?? 1

        return self.map { abs($0 - computedMean) > threshold * computedStd }
    }
}

// MARK: - Empirical Rule

public extension Array where Element == Double {
    /// Returns the observed and expected fractions of values within one, two, and three
    /// sample standard deviations of the mean — the 68-95-99.7 rule applied to your data.
    ///
    /// The empirical rule (also called the 68-95-99.7 rule or the three-sigma rule) states
    /// that for normally distributed data, about 68% of values fall within one standard
    /// deviation of the mean, 95% within two, and 99.7% within three. Comparing the observed
    /// fractions against these expected fractions is the cheapest first-pass check on whether
    /// a dataset is plausibly normal — small deviations suggest normality is a reasonable
    /// working assumption; large deviations say reach for a histogram, a Q-Q plot, or a formal
    /// test before trusting any method that assumes normality. The check is informal by
    /// design: it gives you the numbers and lets you read them, rather than reducing the
    /// question to a single yes-or-no answer that hides where the data deviates.
    ///
    /// The sample standard deviation (ddof = 1) is used, matching the convention of every
    /// other Quiver statistic. The lesson the call is teaching is sampling variation against
    /// a theoretical model, not the model in isolation.
    ///
    /// Returned values:
    /// - **count**: The number of elements summarized
    /// - **within1Sigma / within2Sigma / within3Sigma**: Observed fractions within k standard deviations of the mean
    /// - **expected1Sigma / expected2Sigma / expected3Sigma**: Theoretical Gaussian fractions (0.6827, 0.9545, 0.9973)
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let measurements = [/* sensor readings */]
    ///
    /// if let check = measurements.empiricalRule() {
    ///     print(check)
    ///     // Empirical rule check (n = 395)
    ///     //               actual    expected    diff
    ///     //   within 1σ:  0.664     0.683       -0.019
    ///     //   within 2σ:  0.954     0.954       +0.000
    ///     //   within 3σ:  0.996     0.997       -0.001
    /// }
    /// ```
    ///
    /// - Complexity: O(*n*) — one pass for the mean, one for the standard deviation, three for the bucket counts.
    /// - Returns: An ``EmpiricalRule`` value containing observed and expected fractions, or nil if
    ///   the array has fewer than two elements, the standard deviation is zero, or any element is not finite.
    func empiricalRule() -> EmpiricalRule? {
        guard self.count >= 2 else { return nil }
        guard self.allSatisfy({ $0.isFinite }) else { return nil }
        guard let mean = self.mean(),
              let std = self.standardDeviation(),
              std != 0 else { return nil }

        var c1 = 0, c2 = 0, c3 = 0
        for value in self {
            let distance = abs(value - mean)
            if distance <= std { c1 += 1 }
            if distance <= 2 * std { c2 += 1 }
            if distance <= 3 * std { c3 += 1 }
        }

        let n = Double(self.count)
        return EmpiricalRule(
            count: self.count,
            within1Sigma: Double(c1) / n,
            within2Sigma: Double(c2) / n,
            within3Sigma: Double(c3) / n
        )
    }
}

// MARK: - Vector Array Operations

public extension Array where Element == [Double] {
    /// Computes the element-wise mean across multiple vectors, producing a single vector
    /// where each dimension is the average of the corresponding dimensions across all inputs.
    ///
    /// This operation is fundamental to several machine learning workflows: creating document
    /// vectors by averaging word embeddings, computing cluster centroids in K-means, and
    /// averaging feature vectors for nearest-neighbor classification. All input vectors must
    /// have the same number of dimensions.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let wordEmbeddings = [
    ///     [0.2, 0.5, -0.3, 0.8],   // "swift"
    ///     [0.1, 0.6, 0.2, -0.4]    // "algorithms"
    /// ]
    /// let documentVector = wordEmbeddings.meanVector()
    /// // [0.15, 0.55, -0.05, 0.2]
    /// ```
    ///
    /// - Returns: A vector containing the mean of each dimension, or nil if the array is empty
    ///   or vectors have inconsistent dimensions
    func meanVector() -> [Double]? {
        guard !isEmpty else { return nil }
        guard let first = self.first else { return nil }

        let dimensions = first.count

        // Verify all vectors have the same dimensions
        guard self.allSatisfy({ $0.count == dimensions }) else {
            return nil
        }

        return (0..<dimensions).map { dim in
            self.map { $0[dim] }.mean() ?? 0.0
        }
    }
}

public extension Array where Element == [Float] {
    /// Computes the element-wise mean across multiple vectors, producing a single vector
    /// where each dimension is the average of the corresponding dimensions across all inputs.
    ///
    /// This is the `Float` variant of `meanVector()`. All input vectors must have the same
    /// number of dimensions.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let vectors: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    /// let centroid = vectors.meanVector()
    /// // [4.0, 5.0, 6.0]
    /// ```
    ///
    /// - Returns: A vector containing the mean of each dimension, or nil if the array is empty
    ///   or vectors have inconsistent dimensions
    func meanVector() -> [Float]? {
        guard !isEmpty else { return nil }
        guard let first = self.first else { return nil }

        let dimensions = first.count

        // Verify all vectors have the same dimensions
        guard self.allSatisfy({ $0.count == dimensions }) else {
            return nil
        }

        return (0..<dimensions).map { dim in
            self.map { $0[dim] }.mean() ?? 0.0
        }
    }
}

// MARK: - Confidence Intervals

public extension Array where Element == Double {

    /// Returns the percentile-based confidence interval at the given coverage level.
    ///
    /// Treats the array as the resampled distribution of a statistic and returns
    /// the empirical percentile interval — the lower and upper percentiles that
    /// span `level` of the distribution's mass. For `level = 0.95`, the interval
    /// runs from the 2.5th percentile to the 97.5th percentile.
    ///
    /// This is the simplest of the percentile-based interval methods and the one
    /// most commonly seen in introductory statistics. It is **not** the bias-corrected
    /// and accelerated (BCa) interval, and it is **not** a t-based interval —
    /// those are different (and more complex) constructions. The percentile
    /// interval is appropriate when the resampled distribution looks roughly
    /// symmetric around the original sample statistic.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]
    /// let medians = scores.resampled(iterations: 1000, seed: 42) { resample in
    ///     resample.median() ?? 0.0
    /// }
    /// let interval = medians.percentileCI(level: 0.95)
    /// // (lower, upper) — the 2.5th and 97.5th percentiles of the resampled distribution
    /// ```
    ///
    /// - Parameter level: The coverage level in `(0, 1)`. Defaults to `0.95`.
    /// - Returns: A `(lower, upper)` tuple of percentile bounds, or `nil` if the
    ///   array is empty or `level` is outside `(0, 1)`.
    func percentileCI(level: Double = 0.95) -> (lower: Double, upper: Double)? {
        guard !self.isEmpty else { return nil }
        guard level > 0.0 && level < 1.0 else { return nil }

        let alpha = 1.0 - level
        let lowerPercent = (alpha / 2.0) * 100.0
        let upperPercent = (1.0 - alpha / 2.0) * 100.0

        guard let lower = self.percentile(lowerPercent),
              let upper = self.percentile(upperPercent) else {
            return nil
        }
        return (lower: lower, upper: upper)
    }
}

// MARK: - Mode

public extension Array where Element: Hashable {
    /// Returns the most frequently occurring value(s) in the array.
    ///
    /// The mode is the third measure of central tendency alongside `mean()` and `median()`.
    /// Unlike those measures, mode works on any `Hashable` type — integers, strings, booleans,
    /// or doubles — making it useful for categorical data where averaging is undefined.
    ///
    /// When multiple values tie for the highest frequency, all tied values are returned.
    /// When every value occurs exactly once, the entire array is returned (every value is
    /// tied for first). An empty input returns an empty array.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let diceRolls = [1, 3, 3, 5, 6, 3, 2]
    /// diceRolls.mode()                          // [3]
    ///
    /// let ratings = [4, 5, 4, 3, 5, 4, 5]
    /// ratings.mode()                            // [4, 5]  — bimodal
    ///
    /// let responses = ["yes", "no", "yes", "maybe"]
    /// responses.mode()                          // ["yes"]
    /// ```
    ///
    /// - Complexity: O(*n*) where *n* is the number of elements.
    /// - Returns: An array of all values tied for highest frequency. Empty if input is empty.
    func mode() -> [Element] {
        let counts = Dictionary(grouping: self, by: { $0 }).mapValues(\.count)
        guard let maxCount = counts.values.max() else { return [] }
        return counts.filter { $0.value == maxCount }.map(\.key)
    }
}
