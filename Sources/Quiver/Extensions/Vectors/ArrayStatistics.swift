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
    /// The `ddof` parameter controls whether to compute population variance (dividing
    /// by `n`) or sample variance (dividing by `n - 1`). Use population variance when
    /// the data represents the entire set; use sample variance when it is a subset.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [4.0, 7.0, 2.0, 9.0, 3.0]
    ///
    /// let popVar = data.variance()         // 6.8 (population, ddof: 0)
    /// let sampleVar = data.variance(ddof: 1)  // 8.5 (sample, ddof: 1)
    /// ```
    ///
    /// - Parameter ddof: Delta Degrees of Freedom — 0 for population (default), 1 for sample
    /// - Returns: The variance, or nil if the array has fewer elements than `ddof + 1`
    func variance(ddof: Int = 0) -> Element? {
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
    /// let popStd = data.std()         // 2.61 (population, ddof: 0)
    /// let sampleStd = data.std(ddof: 1)  // 2.92 (sample, ddof: 1)
    /// ```
    ///
    /// - Parameter ddof: Delta Degrees of Freedom — 0 for population (default), 1 for sample
    /// - Returns: The standard deviation, or nil if the array has fewer elements than `ddof + 1`
    func std(ddof: Int = 0) -> Element? {
        let vector = _Vector(elements: self)
        return vector.std(ddof: ddof)
    }

    /// Detects outliers using the z-score method and returns a boolean mask.
    ///
    /// Each element's z-score is computed as the absolute distance from the mean divided
    /// by the standard deviation. Elements whose z-score exceeds the threshold are marked
    /// `true` in the returned mask. The mask can be passed to `masked(by:)` to extract
    /// the outlier values.
    ///
    /// For performance when processing multiple arrays with the same distribution, pass
    /// pre-calculated `mean` and `std` values to avoid recomputing them.
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
    ///   - std: Pre-calculated standard deviation to use instead of computing from the array (optional)
    /// - Returns: Boolean mask where `true` indicates the element at that index is an outlier
    func outlierMask(threshold: Element = 2.0, mean: Element? = nil, std: Element? = nil) -> [Bool] {
        guard !self.isEmpty else { return [] }

        let computedMean = mean ?? self.mean() ?? 0
        let computedStd = std ?? self.std() ?? 1

        return self.map { abs($0 - computedMean) > threshold * computedStd }
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
