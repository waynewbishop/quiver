// Copyright 2026 Wayne W Bishop. All rights reserved.
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

/// A confidence interval for a single regression coefficient.
public struct ConfidenceInterval: Equatable, Codable, Sendable {

    /// The lower bound of the interval.
    public let lower: Double

    /// The upper bound of the interval.
    public let upper: Double

    public init(lower: Double, upper: Double) {
        self.lower = lower
        self.upper = upper
    }
}

/// A frozen snapshot of inference statistics for a fitted ``LinearRegression``.
///
/// Returned by `LinearRegression.summary(features:targets:)`. Carries everything
/// downstream callers need to interpret a regression fit — coefficients, standard
/// errors, t-statistics, two-tailed p-values, confidence intervals, R² and adjusted
/// R², plus sample size and degrees of freedom. Field names match the conventional
/// `statsmodels.OLS` output where possible so existing reference values translate
/// directly.
///
/// Standard errors come from `σ² · (X'X)⁻¹` where `σ²` is the residual variance.
/// The diagonal of that matrix gives one SE per coefficient. T-statistics are
/// `coefficient / SE`. P-values are computed via the Student's t-distribution
/// from `Distributions.t.cdf`. Confidence intervals use `coefficient ± t_critical · SE`.
///
/// **Singular `X'X`.** When the design matrix is singular or near-singular, the
/// underlying matrix inversion is unreliable and the standard errors would be
/// silently meaningless. ``LinearRegression/summary(features:targets:level:)``
/// throws `MatrixError.singular` in that case rather than returning a struct full
/// of corrupted numbers — matching the existing throwing contract on
/// ``LinearRegression/fit(features:targets:intercept:)``.
public struct RegressionSummary: Equatable, Codable, Sendable {

    /// The fitted coefficient vector. When the model has an intercept the first
    /// element is the intercept; the remainder are feature weights.
    public let coefficients: [Double]

    /// One standard error per coefficient. Same length and ordering as ``coefficients``.
    public let standardErrors: [Double]

    /// One t-statistic per coefficient. Equal to `coefficient / standardError`.
    public let tStatistics: [Double]

    /// One two-tailed p-value per coefficient.
    public let pValues: [Double]

    /// One confidence interval per coefficient at the level specified when the
    /// summary was built (default 95%).
    public let confidenceIntervals: [ConfidenceInterval]

    /// The coefficient of determination, `R² = 1 - SS_res / SS_tot`.
    public let rSquared: Double

    /// The adjusted coefficient of determination,
    /// `1 - (1 - R²) · (n - 1) / (n - p - 1)` where `p` is the number of fitted
    /// coefficients (intercept counts when present in this convention,
    /// matching `statsmodels`).
    public let adjustedRSquared: Double

    /// The sample size — number of training rows.
    public let n: Int

    /// The residual degrees of freedom, `n - p`.
    public let degreesOfFreedom: Int

    /// The residual standard error, `sqrt(SS_res / df)`.
    public let residualStandardError: Double

    /// The confidence level used to compute ``confidenceIntervals``,
    /// in `(0, 1)`. The default is `0.95`.
    public let confidenceLevel: Double

    public init(
        coefficients: [Double],
        standardErrors: [Double],
        tStatistics: [Double],
        pValues: [Double],
        confidenceIntervals: [ConfidenceInterval],
        rSquared: Double,
        adjustedRSquared: Double,
        n: Int,
        degreesOfFreedom: Int,
        residualStandardError: Double,
        confidenceLevel: Double
    ) {
        self.coefficients = coefficients
        self.standardErrors = standardErrors
        self.tStatistics = tStatistics
        self.pValues = pValues
        self.confidenceIntervals = confidenceIntervals
        self.rSquared = rSquared
        self.adjustedRSquared = adjustedRSquared
        self.n = n
        self.degreesOfFreedom = degreesOfFreedom
        self.residualStandardError = residualStandardError
        self.confidenceLevel = confidenceLevel
    }

    /// Renders the summary as a Markdown table — one row per coefficient, plus a
    /// header line carrying R² and the residual standard error.
    public func markdownTable() -> String {
        var lines: [String] = []
        lines.append("| Term | Coef | Std Err | t | P>\\|t\\| | CI Lower | CI Upper |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for i in 0..<coefficients.count {
            let label = _termLabel(index: i)
            lines.append(
                "| \(label) | \(_format(coefficients[i])) | \(_format(standardErrors[i])) | "
                + "\(_format(tStatistics[i])) | \(_formatP(pValues[i])) | "
                + "\(_format(confidenceIntervals[i].lower)) | "
                + "\(_format(confidenceIntervals[i].upper)) |"
            )
        }
        lines.append("")
        let level = Int((confidenceLevel * 100.0).rounded())
        lines.append("**n** = \(n)  |  **R²** = \(_format(rSquared))  |  "
            + "**Adj. R²** = \(_format(adjustedRSquared))  |  "
            + "**Resid. SE** = \(_format(residualStandardError))  |  "
            + "**df** = \(degreesOfFreedom)  |  **CI** = \(level)%")
        return lines.joined(separator: "\n")
    }

    /// Renders the summary as CSV — one row per coefficient.
    public func csvRows() -> String {
        var lines = ["term,coef,se,t,p,ci_lower,ci_upper"]
        for i in 0..<coefficients.count {
            let label = _termLabel(index: i)
            lines.append("\(label),\(coefficients[i]),\(standardErrors[i]),"
                + "\(tStatistics[i]),\(pValues[i]),"
                + "\(confidenceIntervals[i].lower),\(confidenceIntervals[i].upper)")
        }
        return lines.joined(separator: "\n")
    }

    private func _termLabel(index: Int) -> String {
        // First coefficient is the intercept when one is present; in this struct
        // we don't track that flag explicitly, but the convention from
        // `LinearRegression.summary` is that the intercept (when fitted) is
        // always coefficient[0]. The caller can rename rows downstream.
        if index == 0 {
            return "intercept_or_x1"
        }
        return "x\(index)"
    }
}

extension RegressionSummary: CustomStringConvertible {
    public var description: String {
        // statsmodels-style table — one row per coefficient.
        let level = Int((confidenceLevel * 100.0).rounded())
        var lines: [String] = []
        lines.append("Linear Regression Summary")
        lines.append("=========================")
        lines.append("n = \(n), df = \(degreesOfFreedom)")
        lines.append("R²    = \(_format(rSquared))")
        lines.append("Adj R² = \(_format(adjustedRSquared))")
        lines.append("Resid SE = \(_format(residualStandardError))")
        lines.append("")

        // Build columns; pad each cell to a fixed width per column.
        let headers = ["term", "coef", "std err", "t", "P>|t|",
                       "[\(level)% lo", "\(level)% hi]"]
        var rows: [[String]] = [headers]
        for i in 0..<coefficients.count {
            rows.append([
                "x\(i)",
                _format(coefficients[i]),
                _format(standardErrors[i]),
                _format(tStatistics[i]),
                _formatP(pValues[i]),
                _format(confidenceIntervals[i].lower),
                _format(confidenceIntervals[i].upper),
            ])
        }
        // Width per column = max length across all rows for that column.
        let columnCount = headers.count
        var widths = [Int](repeating: 0, count: columnCount)
        for row in rows {
            for c in 0..<columnCount {
                widths[c] = Swift.max(widths[c], row[c].count)
            }
        }
        // First column left-aligned, rest right-aligned.
        func line(_ row: [String]) -> String {
            var parts: [String] = []
            for c in 0..<columnCount {
                let cell = row[c]
                let pad = widths[c] - cell.count
                if c == 0 {
                    parts.append(cell + String(repeating: " ", count: pad))
                } else {
                    parts.append(String(repeating: " ", count: pad) + cell)
                }
            }
            return parts.joined(separator: "  ")
        }
        let headerLine = line(headers)
        lines.append(headerLine)
        lines.append(String(repeating: "-", count: headerLine.count))
        for r in 1..<rows.count {
            lines.append(line(rows[r]))
        }
        return lines.joined(separator: "\n")
    }
}

// Renders a number with four decimals, dropping trailing ".0000" for whole numbers.
// Matches the convention in the other typed-summary structs.
private func _format(_ value: Double) -> String {
    if value.isNaN { return "NaN" }
    if value.isInfinite { return value > 0 ? "Inf" : "-Inf" }
    if value == value.rounded(.towardZero) && abs(value) < 1e16 {
        return "\(Int(value)).0"
    }
    return String(format: "%.4f", value)
}

// P-values smaller than 1e-4 print in scientific notation; otherwise four decimals.
// Matches the convention used by R's lm summary and statsmodels.
private func _formatP(_ p: Double) -> String {
    if p.isNaN { return "NaN" }
    if !p.isFinite { return "Inf" }
    if p < 1e-4 && p > 0 {
        return String(format: "%.2e", p)
    }
    return String(format: "%.4f", p)
}
