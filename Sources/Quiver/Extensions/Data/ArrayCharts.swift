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

// MARK: - Chart Helper Functions

/// Specifies how values within a group or window are combined into a single result.
///
/// Used by `groupBy(_:using:)`, `groupedData(by:using:)`, and `downsample(factor:using:)`
/// to control how multiple values are aggregated.
public enum AggregationMethod {
    /// Sum all values in the group
    case sum
    /// Calculate the arithmetic mean of the group
    case mean
    /// Count the number of values in the group
    case count
    /// Select the minimum value in the group
    case min
    /// Select the maximum value in the group
    case max
}

public extension Array where Element: FloatingPoint {

    // MARK: - Time Series Operations

    /// Calculates a rolling mean (moving average) over a specified window.
    ///
    /// The rolling mean smooths noisy data by replacing each value with the average of its
    /// surrounding values within the window. This is essential for identifying trends in
    /// time-series data by filtering out short-term fluctuations. The result has the same
    /// length as the input, with early values averaged over a smaller partial window.
    ///
    /// Common uses:
    /// - **Stock prices**: Smooth daily volatility to reveal weekly or monthly trends
    /// - **Sensor data**: Filter noise from temperature, pressure, or motion readings
    /// - **Performance metrics**: Identify sustained patterns in latency or throughput
    ///
    /// Example:
    /// ```swift
    /// let dailySales = [120.0, 95.0, 140.0, 110.0, 130.0, 85.0, 150.0]
    ///
    /// let trend = dailySales.rollingMean(window: 3)
    /// // [120.0, 107.5, 118.33, 115.0, 126.67, 108.33, 121.67]
    /// // Each value is the average of itself and up to 2 prior values
    /// ```
    ///
    /// - Parameter window: The number of consecutive elements to average together
    /// - Returns: Array of rolling means with the same length as the input
    func rollingMean(window: Int) -> [Element] {
        guard window > 0 && !isEmpty else { return [] }
        guard window <= count else { return Array(repeating: mean() ?? .zero, count: count) }

        var result: [Element] = []
        result.reserveCapacity(count)

        // Sliding window: maintain running sum instead of recalculating each iteration
        var runningSum = Element.zero

        for i in 0..<count {
            runningSum += self[i]

            // Remove the element that just fell out of the window
            if i >= window {
                runningSum -= self[i - window]
            }

            let windowSize = Swift.min(i + 1, window)
            result.append(runningSum / Element(windowSize))
        }

        return result
    }

    /// Calculates the difference between each element and the element a fixed number of
    /// periods before it.
    ///
    /// Period-over-period differencing isolates how much a value changed between consecutive
    /// observations. This is useful for converting cumulative or absolute measurements into
    /// rates of change, and for removing trends from time-series data before further analysis.
    ///
    /// The result array is shorter than the input by the lag amount, since the first `lag`
    /// elements have no prior value to subtract.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let monthlyRevenue = [50000.0, 52000.0, 48000.0, 55000.0, 60000.0]
    ///
    /// // Month-over-month change
    /// let changes = monthlyRevenue.diff()
    /// // [2000.0, -4000.0, 7000.0, 5000.0]
    ///
    /// // Quarter-over-quarter change (3-period lag)
    /// let quarterly = monthlyRevenue.diff(lag: 3)
    /// // [5000.0, 8000.0]
    /// ```
    ///
    /// - Parameter lag: The number of periods to look back when computing the difference (default: 1)
    /// - Returns: Array of differences with length equal to `count - lag`
    func diff(lag: Int = 1) -> [Element] {
        guard lag > 0 && lag < count else { return [] }

        var result: [Element] = []
        result.reserveCapacity(count - lag)

        for i in lag..<count {
            result.append(self[i] - self[i - lag])
        }

        return result
    }

    /// Calculates the percentage change between each element and the element a fixed number
    /// of periods before it.
    ///
    /// Percentage change normalizes differences by the prior value, making it possible to
    /// compare growth rates across series with different scales. A result of `10.0` means
    /// a 10% increase; `-5.0` means a 5% decrease. If a prior value is zero, the change
    /// for that position is reported as zero to avoid division errors.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let stockPrices = [100.0, 105.0, 102.0, 110.0, 108.0]
    ///
    /// let dailyReturns = stockPrices.percentChange()
    /// // [5.0, -2.857, 7.843, -1.818]
    ///
    /// // Year-over-year comparison with a 4-period lag
    /// let yoy = stockPrices.percentChange(lag: 4)
    /// // [8.0]
    /// ```
    ///
    /// - Parameter lag: The number of periods to look back when computing the change (default: 1)
    /// - Returns: Array of percentage changes with length equal to `count - lag`
    func percentChange(lag: Int = 1) -> [Element] {
        guard lag > 0 && lag < count else { return [] }

        var result: [Element] = []
        result.reserveCapacity(count - lag)

        for i in lag..<count {
            let previous = self[i - lag]
            guard previous != .zero else {
                result.append(.zero)
                continue
            }
            let change = ((self[i] - previous) / previous) * Element(100)
            result.append(change)
        }

        return result
    }

    // MARK: - Distribution Analysis

    /// Divides data into evenly spaced bins and counts the frequency of values in each.
    ///
    /// A histogram reveals the shape of a data distribution — whether values cluster around
    /// a central peak, spread uniformly, or skew toward one end. Each bin spans an equal
    /// range, and the result provides the midpoint and count for every bin, ready for
    /// visualization or further analysis.
    ///
    /// Common uses:
    /// - **Data exploration**: Understand the distribution before applying statistical models
    /// - **Quality control**: Detect bimodal patterns or unexpected outliers
    /// - **Feature engineering**: Identify natural breakpoints for discretizing continuous data
    ///
    /// Example:
    /// ```swift
    /// let scores = [72.0, 85.0, 90.0, 78.0, 92.0, 88.0, 76.0, 95.0, 81.0, 87.0]
    ///
    /// let distribution = scores.histogram(bins: 4)
    /// // [(midpoint: 74.875, count: 3),
    /// //  (midpoint: 80.625, count: 2),
    /// //  (midpoint: 86.375, count: 2),
    /// //  (midpoint: 92.125, count: 3)]
    /// ```
    ///
    /// - Parameter bins: The number of equal-width bins to create
    /// - Returns: Array of tuples containing the midpoint and count for each bin
    func histogram(bins: Int) -> [(midpoint: Element, count: Int)] where Element: BinaryFloatingPoint {
        guard bins > 0 && !isEmpty else { return [] }

        guard let minVal = self.min(), let maxVal = self.max() else {
            return []
        }

        // Handle case where all values are the same
        guard minVal != maxVal else {
            return [(midpoint: minVal, count: count)]
        }

        let binWidth = (maxVal - minVal) / Element(bins)

        // Single pass: assign each element to its bin using arithmetic
        var counts = [Int](repeating: 0, count: bins)
        for value in self {
            var index = Int((value - minVal) / binWidth)
            // Clamp the max value into the last bin
            if index >= bins { index = bins - 1 }
            counts[index] += 1
        }

        // Build result with midpoints
        var result: [(Element, Int)] = []
        for i in 0..<bins {
            let lower = minVal + Element(i) * binWidth
            let midpoint = lower + binWidth / Element(2)
            result.append((midpoint, counts[i]))
        }

        return result
    }

    /// Calculates the five-number summary and interquartile range of the data.
    ///
    /// Quartiles divide sorted data into four equal parts, providing a robust summary
    /// that is resistant to outliers. The interquartile range (IQR) — the distance between
    /// Q1 and Q3 — captures where the middle 50% of values fall, making it a reliable
    /// measure of spread for skewed distributions.
    ///
    /// Returned values:
    /// - **min / max**: The smallest and largest values
    /// - **q1**: The 25th percentile (lower quartile)
    /// - **median**: The 50th percentile (middle value)
    /// - **q3**: The 75th percentile (upper quartile)
    /// - **iqr**: The interquartile range (q3 - q1)
    ///
    /// Example:
    /// ```swift
    /// let responseTimes = [12.0, 15.0, 18.0, 22.0, 25.0, 30.0, 45.0, 120.0]
    ///
    /// if let stats = responseTimes.quartiles() {
    ///     // stats.min     = 12.0
    ///     // stats.q1      = 16.5
    ///     // stats.median  = 23.5
    ///     // stats.q3      = 37.5
    ///     // stats.max     = 120.0
    ///     // stats.iqr     = 21.0
    /// }
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    /// - Returns: A tuple containing the five-number summary and IQR, or nil if the array is empty
    func quartiles() -> (min: Element, q1: Element, median: Element, q3: Element, max: Element, iqr: Element)? where Element: BinaryFloatingPoint {
        guard !isEmpty else { return nil }

        let sorted = self.sorted()
        guard let minVal = sorted.first, let maxVal = sorted.last else {
            return nil
        }

        // Use percentile method for consistency
        let medianValue = percentile(50.0) ?? .zero
        let q1Value = percentile(25.0) ?? .zero
        let q3Value = percentile(75.0) ?? .zero

        let iqr = q3Value - q1Value

        return (min: minVal, q1: q1Value, median: medianValue, q3: q3Value, max: maxVal, iqr: iqr)
    }

    /// Calculates the value at a specific percentile using linear interpolation.
    ///
    /// The percentile indicates the point below which a given percentage of observations
    /// fall. For example, the 90th percentile is the value below which 90% of the data
    /// lies. This method sorts the data and interpolates between adjacent values when the
    /// percentile falls between two observations.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let responseTimes = [12.0, 25.0, 18.0, 45.0, 30.0, 15.0, 22.0, 120.0]
    ///
    /// let p50 = responseTimes.percentile(50.0)   // 23.5 (median)
    /// let p90 = responseTimes.percentile(90.0)   // 71.25
    /// let p99 = responseTimes.percentile(99.0)   // 115.15
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    ///   Sorts the array internally. When computing multiple percentiles,
    ///   consider ``quartiles()`` which sorts once.
    /// - Parameter p: The percentile to calculate, between 0 and 100 inclusive
    /// - Returns: The interpolated value at the given percentile, or nil if the array is empty
    ///   or `p` is outside the valid range
    func percentile(_ p: Double) -> Element? where Element: BinaryFloatingPoint {
        guard !isEmpty else { return nil }
        guard p >= 0 && p <= 100 else { return nil }

        let sorted = self.sorted()
        let indexDouble = (p / 100.0) * Double(sorted.count - 1)
        let lowerIndex = Int(indexDouble)
        let upperIndex = Swift.min(lowerIndex + 1, sorted.count - 1)
        let fraction = Element(indexDouble - Double(lowerIndex))

        return sorted[lowerIndex] + fraction * (sorted[upperIndex] - sorted[lowerIndex])
    }

    /// Calculates the percentile rank of a specific value within the array.
    ///
    /// The percentile rank tells us what percentage of values in the array fall at or
    /// below the given value. A rank of 75.0 means the value is greater than or equal
    /// to 75% of the data. This is the inverse operation of `percentile(_:)`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [60.0, 70.0, 80.0, 90.0, 100.0]
    ///
    /// let rank = scores.percentileRank(of: 80.0)  // 50.0
    /// // 80 is at the 50th percentile (middle of the distribution)
    /// ```
    ///
    /// - Parameter value: The value to find the percentile rank for
    /// - Returns: The percentile rank between 0 and 100
    func percentileRank(of value: Element) -> Element {
        guard !isEmpty else { return .zero }

        // Single pass: count below and equal simultaneously
        var belowCount = 0
        var equalCount = 0
        for element in self {
            if element < value {
                belowCount += 1
            } else if element == value {
                equalCount += 1
            }
        }

        let rank = (Element(belowCount) + Element(equalCount) / Element(2)) / Element(count) * Element(100)
        return rank
    }

    /// Calculates the percentile rank for every value in the array.
    ///
    /// Each element is replaced by its percentile rank — the percentage of values in the
    /// array that fall at or below it. Equal values receive the same rank. This is useful
    /// for normalizing data into a uniform distribution or for comparing positions across
    /// different datasets.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [85.0, 92.0, 78.0, 92.0, 88.0]
    ///
    /// let ranks = scores.percentileRanks()
    /// // [30.0, 80.0, 10.0, 80.0, 50.0]
    /// // Both 92s share the same rank; 78 is at the 10th percentile
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    /// - Returns: Array of percentile ranks (0-100) corresponding to each element
    func percentileRanks() -> [Element] {
        guard !isEmpty else { return [] }

        // Sort once and compute all ranks in a single pass
        let sorted = self.enumerated().sorted { $0.element < $1.element }
        var ranks = [Element](repeating: .zero, count: count)

        var i = 0
        while i < sorted.count {
            // Find the range of equal values
            var j = i
            while j < sorted.count && sorted[j].element == sorted[i].element {
                j += 1
            }

            // All equal values share the same rank
            let belowCount = i
            let equalCount = j - i
            let rank = (Element(belowCount) + Element(equalCount) / Element(2)) / Element(count) * Element(100)

            for k in i..<j {
                ranks[sorted[k].offset] = rank
            }

            i = j
        }

        return ranks
    }

    // MARK: - Normalization and Scaling

    /// Scales values to a target range using min-max normalization.
    ///
    /// Min-max scaling linearly maps the smallest value to the lower bound and the largest
    /// value to the upper bound, preserving the relative spacing between all values. This
    /// is commonly used to normalize features before machine learning, or to map data values
    /// to visual properties like chart mark sizes or color intensity.
    ///
    /// If all values are identical (zero range), every element maps to the lower bound
    /// of the target range.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let revenues = [45000.0, 52000.0, 48000.0, 61000.0, 55000.0]
    ///
    /// // Scale to point sizes for a scatter chart
    /// let sizes = revenues.scaled(to: 10.0...50.0)
    /// // [10.0, 17.5, 13.75, 50.0, 32.5]
    /// ```
    ///
    /// - Parameter range: The target closed range to scale values into
    /// - Returns: Array of scaled values mapped to the target range
    func scaled(to range: ClosedRange<Element>) -> [Element] {
        guard !isEmpty else { return [] }
        guard let minVal = self.min(), let maxVal = self.max() else {
            return []
        }

        let dataRange = maxVal - minVal
        guard dataRange != .zero else {
            return Array(repeating: range.lowerBound, count: count)
        }

        let targetRange = range.upperBound - range.lowerBound
        return map { (($0 - minVal) / dataRange) * targetRange + range.lowerBound }
    }

    /// Converts each value to its percentage of the total sum.
    ///
    /// Each element is divided by the sum of all elements and multiplied by 100,
    /// producing a distribution where all values add up to 100. This is the standard
    /// preparation step for pie charts, donut charts, and proportional bar charts.
    ///
    /// If the total sum is zero, all elements are returned as zero to avoid division errors.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let categorySpend = [3200.0, 1800.0, 950.0, 4050.0]
    ///
    /// let shares = categorySpend.asPercentages()
    /// // [32.0, 18.0, 9.5, 40.5]
    /// ```
    ///
    /// - Returns: Array where each value represents its percentage of the total sum
    func asPercentages() -> [Element] {
        guard !isEmpty else { return [] }

        let total = reduce(Element.zero, +)
        guard total != .zero else {
            return Array(repeating: .zero, count: count)
        }

        return map { ($0 / total) * Element(100) }
    }

    /// Standardizes values using z-score normalization so the result has a mean of 0 and
    /// a standard deviation of 1.
    ///
    /// Each value is transformed by subtracting the mean and dividing by the standard
    /// deviation. This centers the data around zero and expresses each value in units of
    /// standard deviation from the mean, making it possible to compare distributions with
    /// different scales on the same axis.
    ///
    /// If the standard deviation is zero (all values are identical), all elements are
    /// returned as zero.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let temperatures = [72.0, 68.0, 73.0, 70.0, 75.0]
    ///
    /// let zScores = temperatures.standardized()
    /// // [-0.18, -1.63, 0.18, -0.91, 1.27]
    /// // Positive values are above the mean; negative values are below
    /// ```
    ///
    /// - Returns: Array of z-score standardized values, or empty array if standard deviation is zero
    func standardized() -> [Element] {
        guard !isEmpty else { return [] }
        guard let meanVal = mean(), let stdVal = std() else {
            return []
        }

        guard stdVal != .zero else {
            return Array(repeating: .zero, count: count)
        }

        return map { ($0 - meanVal) / stdVal }
    }

    // MARK: - Grouping and Aggregation

    /// Groups values by category labels and aggregates each group using the specified method.
    ///
    /// Each value is paired with its corresponding category label by index position. Values
    /// sharing the same label are collected into a group, and the aggregation method is
    /// applied to produce a single result per category. The categories array must have the
    /// same length as the values array.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let sales = [120.0, 95.0, 140.0, 110.0, 85.0, 130.0]
    /// let regions = ["North", "South", "North", "South", "South", "North"]
    ///
    /// let totalByRegion = sales.groupBy(regions, using: .sum)
    /// // ["North": 390.0, "South": 290.0]
    ///
    /// let avgByRegion = sales.groupBy(regions, using: .mean)
    /// // ["North": 130.0, "South": 96.67]
    /// ```
    ///
    /// - Parameters:
    ///   - categories: Array of category labels with the same length as the values array
    ///   - method: The aggregation method to apply to each group (`.sum`, `.mean`, `.count`, `.min`, `.max`)
    /// - Returns: Dictionary mapping each category label to its aggregated value
    func groupBy(_ categories: [String], using method: AggregationMethod) -> [String: Element] {
        guard categories.count == count else { return [:] }

        var groups: [String: [Element]] = [:]

        for (value, category) in zip(self, categories) {
            groups[category, default: []].append(value)
        }

        var result: [String: Element] = [:]

        for (category, values) in groups {
            switch method {
            case .sum:
                result[category] = values.reduce(Element.zero, +)
            case .mean:
                result[category] = values.mean() ?? .zero
            case .count:
                result[category] = Element(values.count)
            case .min:
                result[category] = values.min() ?? .zero
            case .max:
                result[category] = values.max() ?? .zero
            }
        }

        return result
    }

    /// Groups values by category and returns chart-ready tuples sorted by category name.
    ///
    /// This is a convenience wrapper around `groupBy(_:using:)` that returns the results
    /// as an array of named tuples sorted alphabetically by category. The sorted, tuple-based
    /// output maps directly to Swift Charts mark initializers without additional transformation.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let sales = [120.0, 95.0, 140.0, 110.0, 85.0, 130.0]
    /// let regions = ["North", "South", "North", "South", "South", "North"]
    ///
    /// let chartData = sales.groupedData(by: regions, using: .sum)
    /// // [(category: "North", value: 390.0), (category: "South", value: 290.0)]
    /// ```
    ///
    /// - Parameters:
    ///   - categories: Array of category labels with the same length as the values array
    ///   - method: The aggregation method to apply to each group
    /// - Returns: Array of `(category, value)` tuples sorted alphabetically by category
    func groupedData(by categories: [String], using method: AggregationMethod) -> [(category: String, value: Element)] {
        let grouped = groupBy(categories, using: method)
        return grouped.map { ($0.key, $0.value) }.sorted { $0.category < $1.category }
    }
}

// MARK: - Multi-Series Operations

public extension Array where Element == [Double] {

    /// Stacks multiple data series cumulatively so each series builds on the one below it.
    ///
    /// Stacked area and bar charts require cumulative values where each series represents
    /// the running total of all series beneath it. The first series in the result is
    /// unchanged; the second is the sum of series one and two; the third is the sum of
    /// all three; and so on. All inner arrays must have the same length.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let mobile  = [120.0, 135.0, 150.0, 140.0]
    /// let desktop = [200.0, 190.0, 210.0, 195.0]
    /// let tablet  = [50.0, 55.0, 45.0, 60.0]
    ///
    /// let stacked = [mobile, desktop, tablet].stackedCumulative()
    /// // stacked[0] = [120.0, 135.0, 150.0, 140.0]     (mobile only)
    /// // stacked[1] = [320.0, 325.0, 360.0, 335.0]     (mobile + desktop)
    /// // stacked[2] = [370.0, 380.0, 405.0, 395.0]     (all three)
    /// ```
    ///
    /// - Returns: Array of arrays where each series is the cumulative sum of all preceding series
    func stackedCumulative() -> [[Double]] {
        guard !isEmpty else { return [] }
        guard let firstSeries = first else { return [] }

        var result: [[Double]] = []
        var cumulative = [Double](repeating: 0.0, count: firstSeries.count)

        for series in self {
            guard series.count == firstSeries.count else { continue }

            var stacked: [Double] = []
            for (i, value) in series.enumerated() {
                cumulative[i] += value
                stacked.append(cumulative[i])
            }
            result.append(stacked)
        }

        return result
    }

    /// Converts multiple data series into percentage-based stacking where each time point
    /// sums to 100%.
    ///
    /// At each index position, the values across all series are converted to their percentage
    /// of the total at that position. This shows how each series contributes proportionally
    /// over time, removing the effect of overall growth or decline. Useful for 100% stacked
    /// bar charts and area charts that emphasize composition rather than magnitude.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let mobile  = [120.0, 135.0, 150.0, 140.0]
    /// let desktop = [200.0, 190.0, 210.0, 195.0]
    /// let tablet  = [50.0, 55.0, 45.0, 60.0]
    ///
    /// let percents = [mobile, desktop, tablet].stackedPercentage()
    /// // percents[0] = [32.4, 35.5, 37.0, 35.4]   (mobile %)
    /// // percents[1] = [54.1, 50.0, 51.9, 49.4]   (desktop %)
    /// // percents[2] = [13.5, 14.5, 11.1, 15.2]   (tablet %)
    /// ```
    ///
    /// - Returns: Array of arrays where each value is the percentage contribution at that index
    func stackedPercentage() -> [[Double]] {
        guard !isEmpty else { return [] }
        guard let firstSeries = first else { return [] }
        let seriesCount = firstSeries.count

        var result: [[Double]] = []

        // Calculate totals at each time point
        var totals = [Double](repeating: 0.0, count: seriesCount)
        for series in self {
            guard series.count == seriesCount else { continue }
            for (i, value) in series.enumerated() {
                totals[i] += value
            }
        }

        // Convert to percentages
        for series in self {
            var percentSeries: [Double] = []
            for (i, value) in series.enumerated() {
                if totals[i] != 0 {
                    percentSeries.append((value / totals[i]) * 100.0)
                } else {
                    percentSeries.append(0.0)
                }
            }
            result.append(percentSeries)
        }

        return result
    }

    /// Calculates the Pearson correlation matrix for multiple data series.
    ///
    /// The correlation matrix measures the linear relationship between every pair of series,
    /// with each cell containing a Pearson correlation coefficient. This reveals which
    /// variables move together (positive correlation), move inversely (negative correlation),
    /// or behave independently (near zero). The diagonal is always 1.0 since each series
    /// correlates perfectly with itself.
    ///
    /// Interpretation:
    /// - **1.0**: Perfect positive correlation (both rise and fall together)
    /// - **0.0**: No linear relationship
    /// - **-1.0**: Perfect negative correlation (one rises as the other falls)
    ///
    /// Example:
    /// ```swift
    /// let temperature = [30.0, 32.0, 35.0, 28.0, 33.0]
    /// let iceCream    = [200.0, 220.0, 260.0, 180.0, 230.0]
    /// let hotCocoa    = [150.0, 130.0, 100.0, 170.0, 120.0]
    ///
    /// let matrix = [temperature, iceCream, hotCocoa].correlationMatrix()
    /// // [[1.0,   0.99, -0.99],   // temperature
    /// //  [0.99,  1.0,  -0.98],   // ice cream
    /// //  [-0.99, -0.98, 1.0]]    // hot cocoa
    /// ```
    ///
    /// - Complexity: O(*n*²·*m*) where *n* is the number of series and *m* is
    ///   the series length. Performs well for up to a few hundred series.
    /// - Returns: A 2D array of Pearson correlation coefficients between all series pairs
    func correlationMatrix() -> [[Double]] {
        guard !isEmpty else { return [] }

        let n = count
        var matrix: [[Double]] = []

        for i in 0..<n {
            var row: [Double] = []
            for j in 0..<n {
                if i == j {
                    row.append(1.0)
                } else {
                    row.append(pearsonCorrelation(self[i], self[j]))
                }
            }
            matrix.append(row)
        }

        return matrix
    }

    /// Calculate Pearson correlation between two arrays
    private func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count && !x.isEmpty else { return 0.0 }

        let n = Double(x.count)
        let meanX = x.reduce(0, +) / n
        let meanY = y.reduce(0, +) / n

        var numerator = 0.0
        var denomX = 0.0
        var denomY = 0.0

        for i in 0..<x.count {
            let diffX = x[i] - meanX
            let diffY = y[i] - meanY
            numerator += diffX * diffY
            denomX += diffX * diffX
            denomY += diffY * diffY
        }

        guard denomX > 0 && denomY > 0 else { return 0.0 }

        return numerator / Foundation.sqrt(denomX * denomY)
    }

    /// Computes a correlation matrix and flattens it into labeled tuples for heatmap rendering.
    ///
    /// This method first calculates the Pearson correlation matrix for all series, then
    /// converts the 2D matrix into an array of `(x, y, value)` tuples where `x` and `y`
    /// are the series labels and `value` is the correlation coefficient. The output maps
    /// directly to `RectangleMark` in Swift Charts, with the value driving color intensity.
    ///
    /// The labels array must have the same length as the number of series.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let temperature = [30.0, 32.0, 35.0, 28.0, 33.0]
    /// let iceCream    = [200.0, 220.0, 260.0, 180.0, 230.0]
    /// let hotCocoa    = [150.0, 130.0, 100.0, 170.0, 120.0]
    ///
    /// let labels = ["Temp", "Ice Cream", "Hot Cocoa"]
    /// let heatmap = [temperature, iceCream, hotCocoa].heatmapData(labels: labels)
    /// // [("Temp", "Temp", 1.0), ("Temp", "Ice Cream", 0.99), ...]
    /// ```
    ///
    /// - Complexity: O(*n*²·*m*) where *n* is the number of series and *m* is
    ///   the series length. Computes the full correlation matrix internally.
    /// - Parameter labels: Array of labels for each series, matching the number of inner arrays
    /// - Returns: Array of `(x, y, value)` tuples suitable for heatmap visualization
    func heatmapData(labels: [String]) -> [(x: String, y: String, value: Double)] {
        let matrix = correlationMatrix()
        guard matrix.count == labels.count else { return [] }

        var result: [(String, String, Double)] = []

        for i in 0..<matrix.count {
            for j in 0..<matrix[i].count {
                result.append((labels[i], labels[j], matrix[i][j]))
            }
        }

        return result
    }
}

// MARK: - Downsampling

public extension Array where Element: FloatingPoint {

    /// Reduces the array size by grouping consecutive elements into fixed-size windows and
    /// aggregating each window into a single value.
    ///
    /// Downsampling is essential for rendering large datasets in charts without overwhelming
    /// the renderer. A factor of 6 on hourly data produces 4-hour summaries; a factor of
    /// 24 produces daily summaries. The aggregation method controls how values within each
    /// window are combined — use `.mean` for smoothed trends, `.max` for peak detection,
    /// `.sum` for totals, or `.count` for frequency.
    ///
    /// If the array length is not evenly divisible by the factor, the final window contains
    /// the remaining elements.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let hourlyTemps = [
    ///     18.0, 17.5, 17.0, 16.5, 16.0, 16.5,
    ///     18.0, 20.0, 22.0, 24.0, 25.5, 26.0,
    ///     27.0, 27.5, 27.0, 26.0, 24.5, 23.0,
    ///     21.0, 19.5, 18.5, 18.0, 17.5, 17.0
    /// ]
    ///
    /// // Six-hour averages
    /// let sixHourly = hourlyTemps.downsample(factor: 6, using: .mean)
    /// // [17.6, 22.6, 26.5, 18.6]
    ///
    /// // Six-hour peaks
    /// let peaks = hourlyTemps.downsample(factor: 6, using: .max)
    /// // [18.0, 26.0, 27.5, 21.0]
    /// ```
    ///
    /// - Parameters:
    ///   - factor: The number of consecutive elements to aggregate into each output value
    ///   - method: The aggregation method to apply within each window
    /// - Returns: Array with length equal to `ceil(count / factor)`
    func downsample(factor: Int, using method: AggregationMethod) -> [Element] {
        guard factor > 0 && !isEmpty else { return [] }
        guard factor < count else { return [aggregate(using: method)] }

        var result: [Element] = []
        let chunkCount = (count + factor - 1) / factor
        result.reserveCapacity(chunkCount)

        for i in 0..<chunkCount {
            let start = i * factor
            let end = Swift.min(start + factor, count)
            let chunk = Array(self[start..<end])
            result.append(chunk.aggregate(using: method))
        }

        return result
    }

    /// Aggregate array using specified method
    private func aggregate(using method: AggregationMethod) -> Element {
        switch method {
        case .sum:
            return reduce(Element.zero, +)
        case .mean:
            return mean() ?? .zero
        case .count:
            return Element(count)
        case .min:
            return self.min() ?? .zero
        case .max:
            return self.max() ?? .zero
        }
    }
}
