import Foundation

/// A frozen snapshot of per-column descriptive statistics for a `Panel`.
///
/// Returned by `Panel.summary()`. Each column's statistics live in a
/// `ColumnSummary` keyed by column name. The `description` property
/// reproduces the formatted table that `Panel.summary()` returned as a
/// `String` in earlier versions, so existing `print(panel.summary())`
/// callers see identical output.
public struct PanelSummary: Equatable, Codable, Sendable {

    /// Column names in their original declaration order. Reading
    /// `columns` directly returns a dictionary, which has no guaranteed
    /// order; `columnNames` is the canonical iteration order.
    public let columnNames: [String]

    /// Per-column descriptive statistics, keyed by column name.
    public let columns: [String: ColumnSummary]

    public init(columnNames: [String], columns: [String: ColumnSummary]) {
        self.columnNames = columnNames
        self.columns = columns
    }

    /// Renders the summary as a multi-column Markdown table — column names
    /// across the top, statistics down the side. Pastes cleanly into a PR
    /// comment, Substack draft, or stakeholder report.
    public func markdownTable() -> String {
        let stats: [(String, (ColumnSummary) -> String)] = [
            ("count", { "\($0.count)" }),
            ("mean", { formatNumber($0.mean) }),
            ("std", { formatNumber($0.std) }),
            ("min", { formatNumber($0.min) }),
            ("max", { formatNumber($0.max) })
        ]

        var headerRow = "| Statistic |"
        var separatorRow = "| --- |"
        for name in columnNames {
            headerRow += " \(name) |"
            separatorRow += " --- |"
        }

        var lines = [headerRow, separatorRow]
        for (statName, accessor) in stats {
            var row = "| \(statName) |"
            for name in columnNames {
                if let col = columns[name] {
                    row += " \(accessor(col)) |"
                } else {
                    row += "  |"
                }
            }
            lines.append(row)
        }
        return lines.joined(separator: "\n")
    }

    /// Renders the summary as CSV with one row per column and one column per statistic.
    public func csvRows() -> String {
        var lines = ["column,count,mean,std,min,max"]
        for name in columnNames {
            guard let col = columns[name] else { continue }
            lines.append("\(name),\(col.count),\(col.mean),\(col.std),\(col.min),\(col.max)")
        }
        return lines.joined(separator: "\n")
    }
}

extension PanelSummary: CustomStringConvertible {
    public var description: String {
        // Reproduces the exact format the previous String-returning Panel.summary() emitted.
        let headers = ["column", "count", "mean", "std", "min", "max"]
        var rows: [[String]] = []

        for name in columnNames {
            guard let col = columns[name] else { continue }
            rows.append([
                name,
                "\(col.count)",
                formatNumber(col.mean),
                formatNumber(col.std),
                formatNumber(col.min),
                formatNumber(col.max)
            ])
        }

        // Width per column based on header and data
        var widths = headers.map { $0.count }
        for row in rows {
            for (c, cellValue) in row.enumerated() {
                widths[c] = Swift.max(widths[c], cellValue.count)
            }
        }

        // Header: first column left-aligned, rest right-aligned
        let headerParts = headers.enumerated().map { c, h in
            c == 0
                ? h.padding(toLength: widths[c], withPad: " ", startingAt: 0)
                : String(repeating: " ", count: widths[c] - h.count) + h
        }
        var lines = [headerParts.joined(separator: "  ")]
        lines.append(String(repeating: "-", count: lines[0].count))

        // Data rows
        for row in rows {
            let parts = row.enumerated().map { c, cellValue in
                c == 0
                    ? cellValue.padding(toLength: widths[c], withPad: " ", startingAt: 0)
                    : String(repeating: " ", count: widths[c] - cellValue.count) + cellValue
            }
            lines.append(parts.joined(separator: "  "))
        }

        return lines.joined(separator: "\n")
    }
}

// Drops trailing ".0" for whole numbers, otherwise renders four decimal places.
// Same convention as ColumnSummary and the previous Panel.summary() formatter.
private func formatNumber(_ value: Double) -> String {
    if value == value.rounded(.towardZero) && !value.isNaN && !value.isInfinite {
        return "\(Int(value)).0"
    }
    return String(format: "%.4f", value)
}
