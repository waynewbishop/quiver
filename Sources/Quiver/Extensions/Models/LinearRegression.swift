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

// MARK: - Linear Regression

/// A trained ordinary least squares regression model.
///
/// Linear regression finds the best-fit line (or hyperplane) through training data
/// by minimizing prediction error. This implementation uses the normal
/// equation θ = (X'X)⁻¹X'y, which gives an exact closed-form solution without
/// iterative optimization.
///
/// This is a value type — once created via one of the fit methods,
/// the model is immutable. There is no separate "unfitted" state, which eliminates
/// the common bug of calling predict before fit.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [
///     [1.0], [2.0], [3.0], [4.0], [5.0]
/// ]
/// let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
///
/// let model = try LinearRegression.fit(features: features, targets: targets)
/// let predictions = model.predict([[6.0], [7.0]])
/// // predictions ≈ [11.8, 13.8]
/// ```
public struct LinearRegression: Regressor, Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let featureLabel = featureCount == 1 ? "feature" : "features"
        var parts = "LinearRegression: \(featureCount) \(featureLabel)"

        if hasIntercept {
            parts += ", intercept: \(String(format: "%.2f", coefficients[0]))"
            let weights = Array(coefficients.dropFirst())
            if featureCount == 1, let slope = weights.first {
                parts += ", slope: \(String(format: "%.2f", slope))"
            } else {
                let formatted = weights.map { String(format: "%.2f", $0) }
                parts += ", weights: [\(formatted.joined(separator: ", "))]"
            }
        } else {
            if featureCount == 1, let slope = coefficients.first {
                parts += ", slope: \(String(format: "%.2f", slope))"
            } else {
                let formatted = coefficients.map { String(format: "%.2f", $0) }
                parts += ", weights: [\(formatted.joined(separator: ", "))]"
            }
        }

        return parts
    }

    /// The fitted coefficient vector.
    ///
    /// When `hasIntercept` is true, the first element is the bias (intercept) term
    /// and the remaining elements are the feature weights. When false, all elements
    /// are feature weights.
    public let coefficients: [Double]

    /// Number of features the model was trained on.
    public let featureCount: Int

    /// Whether this model includes an intercept (bias) term.
    public let hasIntercept: Bool

    /// Fits a linear regression model to the given training data.
    ///
    /// Solves the normal equation θ = (X'X)⁻¹X'y to find the coefficient vector
    /// that minimizes the sum of squared residuals. The returned model is ready
    /// for prediction immediately.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - targets: 1D array of target values, one per sample.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Complexity: O(*n*·*f*² + *f*³) where *n* is the number of samples and
    ///   *f* is the feature count. The *f*³ term comes from matrix inversion.
    ///   Performs well for feature counts up to a few hundred.
    /// - Returns: A trained ``LinearRegression`` model.
    /// - Throws: `MatrixError.singular` if the features are linearly dependent.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        intercept: Bool = true
    ) throws -> LinearRegression {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples")

        let featureCount = features[0].count
        let theta = try _Regression.fitNormalEquation(
            features: features,
            targets: targets,
            intercept: intercept
        )

        return LinearRegression(
            coefficients: theta,
            featureCount: featureCount,
            hasIntercept: intercept
        )
    }

    /// Fits a linear regression model from a single feature array.
    ///
    /// A convenience overload that accepts a flat `[Double]` instead of `[[Double]]`
    /// when training on a single feature. Each element is treated as one sample
    /// with one feature.
    ///
    /// ```swift
    /// let sqft   = [1200.0, 1500.0, 1800.0, 2100.0]
    /// let prices = [250_000.0, 320_000.0, 380_000.0, 440_000.0]
    ///
    /// let model = try LinearRegression.fit(features: sqft, targets: prices)
    /// let predicted = model.predict([2000.0])
    /// ```
    ///
    /// - Parameters:
    ///   - features: 1D array of feature values, one per sample.
    ///   - targets: 1D array of target values, one per sample.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Returns: A trained ``LinearRegression`` model with `featureCount` of 1.
    /// - Throws: `MatrixError.singular` if the features are constant.
    public static func fit(
        features: [Double],
        targets: [Double],
        intercept: Bool = true
    ) throws -> LinearRegression {
        return try fit(features: features.map { [$0] }, targets: targets, intercept: intercept)
    }

    /// Predicts target values for one or more samples.
    ///
    /// Computes ŷ = Xθ for each input sample using the fitted coefficients.
    ///
    /// - Parameter features: 2D array where each row is a sample to predict.
    /// - Returns: An array of predicted values, one per sample.
    public func predict(_ features: [[Double]]) -> [Double] {
        return _Regression.predict(
            features: features,
            coefficients: coefficients,
            intercept: hasIntercept
        )
    }

    /// Predicts target values for single-feature regression.
    ///
    /// A convenience overload that accepts a flat `[Double]` instead of `[[Double]]`
    /// when the model was trained on a single feature. Each element is treated as
    /// one sample with one feature.
    ///
    /// ```swift
    /// let model = try LinearRegression.fit(
    ///     features: [[1.0], [2.0], [3.0]],
    ///     targets: [2.0, 4.0, 6.0]
    /// )
    /// let trendX = [Double].linspace(start: 0.0, end: 4.0, count: 50)
    /// let trendY = model.predict(trendX)
    /// ```
    ///
    /// - Parameter values: 1D array of feature values for single-feature prediction.
    /// - Returns: An array of predicted values, one per input.
    public func predict(_ values: [Double]) -> [Double] {
        precondition(featureCount == 1,
            "Single-feature predict requires a model trained on 1 feature, got \(featureCount)")
        return predict(values.map { [$0] })
    }

    /// Builds a typed inference summary — coefficients, standard errors,
    /// t-statistics, p-values, confidence intervals, R², adjusted R² — for the
    /// model evaluated on the training data that produced it.
    ///
    /// The summary covers everything downstream callers need to interpret a
    /// regression fit. It mirrors the output of `statsmodels.OLS.fit().summary()`:
    /// per-coefficient inference plus the goodness-of-fit summary.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let model = try LinearRegression.fit(features: X, targets: y)
    /// let report = try model.summary(features: X, targets: y)
    ///
    /// print(report)                  // human-readable table
    /// report.coefficients[1]         // weight on the first feature
    /// report.pValues[1]              // two-tailed p-value for that weight
    /// report.confidenceIntervals[1]  // (lower, upper) at the chosen level
    /// ```
    ///
    /// **Singular `X'X` (Li Chen Flag 2).** When the design matrix is singular
    /// or near-singular, the inverse used to compute the variance–covariance
    /// matrix is unreliable and standard errors would be silently meaningless.
    /// `summary` throws `MatrixError.singular` in that case rather than returning
    /// a struct full of corrupted numbers — matching the existing throwing
    /// contract on the `fit` methods.
    ///
    /// - Parameters:
    ///   - features: The same `[[Double]]` design matrix passed to `fit`.
    ///   - targets: The same target vector passed to `fit`.
    ///   - level: Confidence level for ``RegressionSummary/confidenceIntervals``,
    ///     in `(0, 1)`. Defaults to `0.95`.
    /// - Returns: A typed ``RegressionSummary`` value.
    /// - Throws: `MatrixError.singular` when `X'X` cannot be inverted.
    public func summary(
        features: [[Double]],
        targets: [Double],
        level: Double = 0.95
    ) throws -> RegressionSummary {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples")
        precondition(level > 0.0 && level < 1.0,
            "Confidence level must be in (0, 1), got \(level)")

        // Build design matrix X (with leading ones column when intercept is enabled).
        let X: [[Double]]
        if hasIntercept {
            X = features.map { row in [1.0] + row }
        } else {
            X = features
        }
        precondition(X[0].count == coefficients.count,
            "Feature count in summary input does not match the fitted model")

        let n = X.count
        let p = coefficients.count
        let df = n - p
        precondition(df > 0,
            "Need more samples than coefficients to compute residual statistics; n=\(n), p=\(p)")

        // (X'X)^-1 — throws `MatrixError.singular` when the matrix is rank-deficient.
        let Xt = X.transposed()
        let XtX = Xt.multiplyMatrix(X)
        let XtX_inv = try XtX.inverted()

        // Residuals and residual variance.
        let predictions = _Regression.predict(
            features: features,
            coefficients: coefficients,
            intercept: hasIntercept
        )
        var rss = 0.0
        for i in 0..<n {
            let r = targets[i] - predictions[i]
            rss += r * r
        }
        let sigma2 = rss / Double(df)
        let residualSE = Foundation.sqrt(sigma2)

        // Standard errors from sqrt(σ² · diag(XtX^-1)).
        var standardErrors = [Double](repeating: 0.0, count: p)
        for i in 0..<p {
            let variance = sigma2 * XtX_inv[i][i]
            standardErrors[i] = variance >= 0 ? Foundation.sqrt(variance) : .nan
        }

        // t-statistics and two-tailed p-values via Distributions.t.cdf.
        var tStats = [Double](repeating: 0.0, count: p)
        var pValues = [Double](repeating: 0.0, count: p)
        for i in 0..<p {
            let se = standardErrors[i]
            if se > 0 {
                tStats[i] = coefficients[i] / se
            } else {
                tStats[i] = .nan
            }
            // Two-tailed p-value: 2 · (1 - tCDF(|t|, df)).
            // The absolute value is critical — symmetry of t makes this work.
            let absT = abs(tStats[i])
            if let upper = Distributions.t.cdf(x: absT, df: Double(df)) {
                let p = 2.0 * (1.0 - upper)
                pValues[i] = Swift.max(0.0, Swift.min(1.0, p))
            } else {
                pValues[i] = .nan
            }
        }

        // Confidence intervals at the requested level.
        let alpha = 1.0 - level
        guard let tCrit = Distributions.t.quantile(p: 1.0 - alpha / 2.0, df: Double(df)) else {
            // Should not happen for the default level — but guard anyway.
            throw MatrixError.singular
        }
        var cis = [ConfidenceInterval](repeating: ConfidenceInterval(lower: 0, upper: 0), count: p)
        for i in 0..<p {
            let half = tCrit * standardErrors[i]
            cis[i] = ConfidenceInterval(
                lower: coefficients[i] - half,
                upper: coefficients[i] + half
            )
        }

        // R² and adjusted R².
        var targetMean = 0.0
        for value in targets { targetMean += value }
        targetMean /= Double(n)
        var tss = 0.0
        for value in targets {
            let diff = value - targetMean
            tss += diff * diff
        }
        let r2 = tss > 0 ? (1.0 - rss / tss) : 0.0
        // Statsmodels convention: adj R² = 1 - (1 - R²) · (n - 1) / (n - p).
        // Here `p` is the total number of fitted parameters (intercept counted
        // when fitted), which is what `coefficients.count` already gives us.
        let adjR2: Double
        if df > 0 {
            adjR2 = 1.0 - (1.0 - r2) * Double(n - 1) / Double(df)
        } else {
            adjR2 = .nan
        }

        return RegressionSummary(
            coefficients: coefficients,
            standardErrors: standardErrors,
            tStatistics: tStats,
            pValues: pValues,
            confidenceIntervals: cis,
            rSquared: r2,
            adjustedRSquared: adjR2,
            n: n,
            degreesOfFreedom: df,
            residualStandardError: residualSE,
            confidenceLevel: level
        )
    }
}
