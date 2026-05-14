import Foundation

/// The five-number summary plus the interquartile range for a sorted dataset.
///
/// Returned by `[Double].quartiles()` and equivalent methods. The named
/// properties match the labels of the previous tuple-based return type, so
/// existing call sites that read `q.min`, `q.q1`, `q.median`, `q.q3`,
/// `q.max`, and `q.iqr` continue to work without change.
public struct Quartiles<T: BinaryFloatingPoint & Sendable>: Equatable, Sendable {

    /// The smallest value in the dataset.
    public let min: T

    /// The 25th percentile — the value below which a quarter of the data falls.
    public let q1: T

    /// The 50th percentile — the median.
    public let median: T

    /// The 75th percentile — the value below which three quarters of the data falls.
    public let q3: T

    /// The largest value in the dataset.
    public let max: T

    /// The interquartile range, defined as `q3 - q1`. Captures the spread of the middle 50% of the data.
    public let iqr: T

    public init(min: T, q1: T, median: T, q3: T, max: T, iqr: T) {
        self.min = min
        self.q1 = q1
        self.median = median
        self.q3 = q3
        self.max = max
        self.iqr = iqr
    }
}

extension Quartiles: CustomStringConvertible {
    public var description: String {
        // Padding chosen so labels align in a typical print() output.
        let lines = [
            "min:    \(min)",
            "q1:     \(q1)",
            "median: \(median)",
            "q3:     \(q3)",
            "max:    \(max)",
            "iqr:    \(iqr)"
        ]
        return lines.joined(separator: "\n")
    }
}

extension Quartiles: Codable where T: Codable {}
