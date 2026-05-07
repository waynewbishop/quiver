import Foundation

/// Per-class metrics for a single class label in a classification report.
///
/// Each of `precision`, `recall`, and `f1Score` is `Double?` — `nil` means
/// the metric is undefined for this class (typically because no samples
/// were predicted as this class, making precision's denominator zero).
public struct ClassMetrics: Equatable, Codable, Sendable {

    /// The class label this row describes. For aggregate rows
    /// (macro/weighted average), this is set to the sentinel `-1` and
    /// callers should consult the aggregate accessors on
    /// `ClassificationReport` rather than reading `label`.
    public let label: Int

    /// Precision for this class (TP / (TP + FP)). `nil` when the model
    /// made no positive predictions for this class.
    public let precision: Double?

    /// Recall for this class (TP / (TP + FN)). `nil` when there are no
    /// actual samples of this class in the ground truth.
    public let recall: Double?

    /// F1 score for this class — the harmonic mean of precision and recall.
    /// `nil` when either precision or recall is undefined.
    public let f1Score: Double?

    /// The number of actual samples of this class in the ground truth.
    /// For aggregate rows, this is the total sample count.
    public let support: Int

    public init(label: Int, precision: Double?, recall: Double?, f1Score: Double?, support: Int) {
        self.label = label
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.support = support
    }
}

extension ClassMetrics: CustomStringConvertible {
    public var description: String {
        let p = precision.map { String(format: "%.2f", $0) } ?? "N/A"
        let r = recall.map { String(format: "%.2f", $0) } ?? "N/A"
        let f = f1Score.map { String(format: "%.2f", $0) } ?? "N/A"
        return "label \(label): precision \(p), recall \(r), f1 \(f), support \(support)"
    }
}

/// A frozen snapshot of a multi-class classification's evaluation metrics.
///
/// Returned by `[Int].classificationReport(actual:)`. Provides per-class
/// metrics keyed by label, overall accuracy, and macro and weighted
/// averages. The `description` property reproduces the formatted table
/// that earlier versions returned as a `String`, so existing
/// `print(predictions.classificationReport(actual: actual))` callers see
/// identical output.
public struct ClassificationReport: Equatable, Codable, Sendable {

    /// Per-class metrics keyed by class label.
    public let perClass: [Int: ClassMetrics]

    /// Class labels in sorted ascending order — the canonical iteration order.
    public let classOrder: [Int]

    /// Overall accuracy across all samples.
    public let accuracy: Double

    /// Unweighted mean of per-class metrics.
    public let macroAverage: ClassMetrics

    /// Mean of per-class metrics weighted by support.
    public let weightedAverage: ClassMetrics

    /// Total number of samples across all classes.
    public let totalSupport: Int

    public init(
        perClass: [Int: ClassMetrics],
        classOrder: [Int],
        accuracy: Double,
        macroAverage: ClassMetrics,
        weightedAverage: ClassMetrics,
        totalSupport: Int
    ) {
        self.perClass = perClass
        self.classOrder = classOrder
        self.accuracy = accuracy
        self.macroAverage = macroAverage
        self.weightedAverage = weightedAverage
        self.totalSupport = totalSupport
    }

    /// Renders the report as a Markdown table — one row per class, plus the aggregate rows.
    public func markdownTable() -> String {
        var lines = [
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |"
        ]
        for label in classOrder {
            guard let m = perClass[label] else { continue }
            lines.append("| \(label) | \(formatOptional(m.precision)) | \(formatOptional(m.recall)) | \(formatOptional(m.f1Score)) | \(m.support) |")
        }
        lines.append("| accuracy |  |  | \(formatOptional(accuracy)) | \(totalSupport) |")
        lines.append("| macro avg | \(formatOptional(macroAverage.precision)) | \(formatOptional(macroAverage.recall)) | \(formatOptional(macroAverage.f1Score)) | \(totalSupport) |")
        lines.append("| weighted avg | \(formatOptional(weightedAverage.precision)) | \(formatOptional(weightedAverage.recall)) | \(formatOptional(weightedAverage.f1Score)) | \(totalSupport) |")
        return lines.joined(separator: "\n")
    }

    /// Renders the report as CSV with one row per class plus the aggregate rows.
    public func csvRows() -> String {
        var lines = ["class,precision,recall,f1,support"]
        for label in classOrder {
            guard let m = perClass[label] else { continue }
            lines.append("\(label),\(csvOptional(m.precision)),\(csvOptional(m.recall)),\(csvOptional(m.f1Score)),\(m.support)")
        }
        lines.append("accuracy,,,\(accuracy),\(totalSupport)")
        lines.append("macro avg,\(csvOptional(macroAverage.precision)),\(csvOptional(macroAverage.recall)),\(csvOptional(macroAverage.f1Score)),\(totalSupport)")
        lines.append("weighted avg,\(csvOptional(weightedAverage.precision)),\(csvOptional(weightedAverage.recall)),\(csvOptional(weightedAverage.f1Score)),\(totalSupport)")
        return lines.joined(separator: "\n")
    }
}

extension ClassificationReport: CustomStringConvertible {
    public var description: String {
        // Reproduces the exact format the previous String-returning classificationReport(actual:) emitted.
        // Format helpers — match the previous behavior where nil renders as " 0.00".
        func fmt(_ val: Double?) -> String {
            guard let value = val else { return " 0.00" }
            return String(format: "%5.2f", value)
        }
        func fmtSupport(_ val: Int) -> String {
            return String(format: "%10d", val)
        }

        let labelStrings = classOrder.map { "\($0)" }
        let labelWidth = Swift.max(labelStrings.map { $0.count }.max() ?? 1, 12)

        let headerLabel = String(repeating: " ", count: labelWidth)
        let header = "\(headerLabel)  precision    recall  f1-score   support"
        var lines = [header, ""]

        for (i, label) in classOrder.enumerated() {
            guard let m = perClass[label] else { continue }
            let pad = String(repeating: " ", count: labelWidth - labelStrings[i].count)
            lines.append("\(pad)\(labelStrings[i])      \(fmt(m.precision))     \(fmt(m.recall))     \(fmt(m.f1Score))\(fmtSupport(m.support))")
        }

        lines.append("")

        let accPad = String(repeating: " ", count: labelWidth - 8)
        lines.append("\(accPad)accuracy                          \(fmt(accuracy))\(fmtSupport(totalSupport))")

        let macroPad = String(repeating: " ", count: labelWidth - 9)
        lines.append("\(macroPad)macro avg      \(fmt(macroAverage.precision))     \(fmt(macroAverage.recall))     \(fmt(macroAverage.f1Score))\(fmtSupport(totalSupport))")

        let weightPad = String(repeating: " ", count: labelWidth - 12)
        lines.append("\(weightPad)weighted avg      \(fmt(weightedAverage.precision))     \(fmt(weightedAverage.recall))     \(fmt(weightedAverage.f1Score))\(fmtSupport(totalSupport))")

        return lines.joined(separator: "\n")
    }
}

private func formatOptional(_ value: Double?) -> String {
    guard let v = value else { return "N/A" }
    return String(format: "%.2f", v)
}

private func csvOptional(_ value: Double?) -> String {
    guard let v = value else { return "" }
    return String(v)
}
