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
public struct LinearRegression: Codable, CustomStringConvertible, Equatable {

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
    /// let trendX = Array.linspace(start: 0.0, end: 4.0, count: 50)
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
}
