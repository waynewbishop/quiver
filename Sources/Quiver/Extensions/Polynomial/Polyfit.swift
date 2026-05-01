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

// MARK: - Polynomial fitting

public extension Array where Element == Double {

    /// Fits a polynomial of the given degree to `(x, y)` pairs by ordinary least squares.
    ///
    /// Builds a Vandermonde-style design matrix whose row `i` contains the powers
    /// `x[i]¹, x[i]², ..., x[i]ⁿ`, then defers to ``LinearRegression`` to solve
    /// the normal equation. The recovered intercept becomes the constant term
    /// of the polynomial and the recovered weights become the higher-power
    /// coefficients in ascending order. For `degree: 1`, the result matches
    /// `LinearRegression.fit(features: x, targets: y)` exactly.
    ///
    /// For the constant-fit case of `degree: 0` — fitting a horizontal
    /// line — the returned polynomial has the mean of `y` as its only coefficient.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let x = [1.0, 2.0, 3.0, 4.0, 5.0]
    /// let y = [6.0, 15.0, 28.0, 45.0, 66.0]   // 2x² + 3x + 1
    /// if let p = [Double].polyfit(x: x, y: y, degree: 2) {
    ///     p.coefficients   // ≈ [1.0, 3.0, 2.0]
    ///     p(6)              // ≈ 91.0
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - x: The independent-variable samples.
    ///   - y: The dependent-variable samples, paired with `x` by index.
    ///   - degree: The polynomial degree to fit. Must be at least `0`.
    /// - Returns: The fitted polynomial, or `nil` if the inputs are invalid
    ///   (mismatched lengths, fewer points than `degree + 1`, negative `degree`)
    ///   or if the underlying linear system is ill-conditioned.
    static func polyfit(x: [Double], y: [Double], degree: Int) -> Polynomial? {
        // Validate inputs.
        guard degree >= 0 else { return nil }
        guard x.count == y.count else { return nil }
        guard x.count > degree else { return nil }

        // Degree 0 is the constant fit — least squares reduces to the mean.
        if degree == 0 {
            let mean = y.reduce(0.0, +) / Double(y.count)
            return Polynomial([mean])
        }

        // Build the design matrix WITHOUT a bias column. LinearRegression
        // prepends its own ones column when `intercept: true`, and the
        // returned coefficient vector starts with the intercept — exactly
        // the layout we need for the polynomial constant term.
        // Row i, columns: [x[i]¹, x[i]², ..., x[i]^degree]
        let features: [[Double]] = x.map { xi in
            var row = [Double](repeating: 0, count: degree)
            var power = xi
            for j in 0..<degree {
                row[j] = power
                power *= xi
            }
            return row
        }

        guard let model = try? LinearRegression.fit(
            features: features,
            targets: y,
            intercept: true
        ) else {
            return nil
        }

        // model.coefficients is [intercept, slope_x¹, slope_x², ..., slope_xⁿ]
        // which is exactly the [a₀, a₁, ..., aₙ] order Polynomial expects.
        return Polynomial(model.coefficients)
    }
}
