import Foundation

/// A frozen snapshot of descriptive statistics for a single numerical column.
///
/// Returned by `[Double].summary()` and used as the per-column inner type of
/// `PanelSummary`. All statistics are computed once at construction time and
/// stored as properties — subsequent reads do not recompute. The standard
/// deviation uses population variance (`ddof: 0`), matching the default of
/// `[Double].std()`.
public struct ColumnSummary: Equatable, Codable, Sendable {

    /// The number of values in the source column.
    public let count: Int

    /// The arithmetic mean.
    public let mean: Double

    /// The population standard deviation (`ddof: 0`).
    public let std: Double

    /// The smallest value.
    public let min: Double

    /// The 25th percentile.
    public let q1: Double

    /// The 50th percentile (median).
    public let median: Double

    /// The 75th percentile.
    public let q3: Double

    /// The largest value.
    public let max: Double

    /// The interquartile range, `q3 - q1`.
    public let iqr: Double

    public init(count: Int, mean: Double, std: Double, min: Double, q1: Double, median: Double, q3: Double, max: Double, iqr: Double) {
        self.count = count
        self.mean = mean
        self.std = std
        self.min = min
        self.q1 = q1
        self.median = median
        self.q3 = q3
        self.max = max
        self.iqr = iqr
    }

    /// Renders the summary as a two-column Markdown table suitable for pasting into a deliverable.
    public func markdownTable() -> String {
        var lines = [
            "| Statistic | Value |",
            "| --- | --- |"
        ]
        lines.append("| count | \(count) |")
        lines.append("| mean | \(formatNumber(mean)) |")
        lines.append("| std | \(formatNumber(std)) |")
        lines.append("| min | \(formatNumber(min)) |")
        lines.append("| q1 | \(formatNumber(q1)) |")
        lines.append("| median | \(formatNumber(median)) |")
        lines.append("| q3 | \(formatNumber(q3)) |")
        lines.append("| max | \(formatNumber(max)) |")
        lines.append("| iqr | \(formatNumber(iqr)) |")
        return lines.joined(separator: "\n")
    }

    /// Renders the summary as CSV with one row per statistic.
    public func csvRows() -> String {
        let rows = [
            "statistic,value",
            "count,\(count)",
            "mean,\(mean)",
            "std,\(std)",
            "min,\(min)",
            "q1,\(q1)",
            "median,\(median)",
            "q3,\(q3)",
            "max,\(max)",
            "iqr,\(iqr)"
        ]
        return rows.joined(separator: "\n")
    }
}

extension ColumnSummary: CustomStringConvertible {
    public var description: String {
        let lines = [
            "count:  \(count)",
            "mean:   \(formatNumber(mean))",
            "std:    \(formatNumber(std))",
            "min:    \(formatNumber(min))",
            "q1:     \(formatNumber(q1))",
            "median: \(formatNumber(median))",
            "q3:     \(formatNumber(q3))",
            "max:    \(formatNumber(max))",
            "iqr:    \(formatNumber(iqr))"
        ]
        return lines.joined(separator: "\n")
    }
}

// Drops trailing ".0" for whole numbers, otherwise renders four decimal places.
// Matches the convention used in Panel.summary()'s existing String formatter.
private func formatNumber(_ value: Double) -> String {
    if value == value.rounded(.towardZero) && !value.isNaN && !value.isInfinite {
        return "\(Int(value)).0"
    }
    return String(format: "%.4f", value)
}

public extension Array where Element == Double {

    /// Returns a typed snapshot of descriptive statistics: count, mean, std, min, quartiles, max, and iqr.
    ///
    /// All statistics are computed in a single pass over the data using
    /// existing `[Double]` extensions and stored as properties on the
    /// returned `ColumnSummary`. Subsequent reads are property accesses.
    ///
    /// Returns `nil` for empty arrays, matching the contract of `mean()`.
    ///
    /// Example:
    /// ```swift
    /// let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]
    /// if let stats = scores.summary() {
    ///     print(stats)                  // formatted block
    ///     print(stats.markdownTable())  // pastes cleanly into a PR or report
    ///     stats.mean                    // 78.375
    /// }
    /// ```
    func summary() -> ColumnSummary? {
        guard !isEmpty else { return nil }
        guard let meanValue = mean(),
              let stdValue = std(),
              let q = quartiles() else {
            return nil
        }
        return ColumnSummary(
            count: count,
            mean: meanValue,
            std: stdValue,
            min: q.min,
            q1: q.q1,
            median: q.median,
            q3: q.q3,
            max: q.max,
            iqr: q.iqr
        )
    }
}
