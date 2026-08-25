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

// MARK: - Internal Regression Computation

/// Internal namespace for regression math.
///
/// Solves the normal equation X'X·θ = X'y using existing Quiver
/// matrix operations. Separated from the public API so that the fitting
/// logic stays testable without exposing implementation details.
internal enum _Regression {

    /// Solves the ordinary least squares normal equation.
    ///
    /// Prepends a column of ones to the feature matrix when `intercept` is true,
    /// then solves X'X·θ = X'y. Rather than forming the explicit inverse (X'X)⁻¹,
    /// the system is solved directly through an LU factorization: solving Ax = b
    /// by forward and back substitution is both faster and more numerically stable
    /// than inverting A and multiplying. Throws if X'X is singular.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a feature.
    ///   - targets: 1D array of target values, one per sample.
    ///   - intercept: Whether to prepend a ones column for the bias term.
    /// - Returns: Coefficient vector θ (intercept first when `intercept` is true).
    /// - Throws: `MatrixError.singular` if the feature matrix is rank-deficient.
    static func fitNormalEquation(
        features: [[Double]],
        targets: [Double],
        intercept: Bool
    ) throws -> [Double] {

        // Build the design matrix X, optionally prepending a ones column
        let X: [[Double]]
        if intercept {
            X = features.map { row in
                var newRow = [1.0]
                newRow.append(contentsOf: row)
                return newRow
            }
        } else {
            X = features
        }

        let Xt = X.transposed()              // X' — shape [p × n]
        let XtX = Xt.multiplyMatrix(X)        // X'X — shape [p × p]

        // X'y as a plain vector: multiply X' (p×n) by y (n×1), then flatten.
        let yColumn = targets.map { [$0] }              // n×1 column matrix
        let Xty = Xt.multiplyMatrix(yColumn).map { $0[0] }  // p-vector

        // Solve X'X·θ = X'y directly. luDecomposed() throws MatrixError.singular
        // on a rank-deficient X'X, preserving the same contract the previous
        // (X'X)⁻¹ path surfaced; the subsequent solve then cannot fail.
        let theta = try XtX.luDecomposed().solve(Xty)   // p-vector

        return theta
    }

    /// Computes predictions ŷ = Xθ for the given features and coefficients.
    ///
    /// - Parameters:
    ///   - features: 2D array of samples.
    ///   - coefficients: The fitted coefficient vector.
    ///   - intercept: Whether the first coefficient is a bias term.
    /// - Returns: Predicted values, one per sample.
    static func predict(
        features: [[Double]],
        coefficients: [Double],
        intercept: Bool
    ) -> [Double] {
        return features.map { sample in
            if intercept {
                // θ₀ + θ₁x₁ + θ₂x₂ + ...
                var sum = coefficients[0]
                for i in 0..<sample.count {
                    sum += coefficients[i + 1] * sample[i]
                }
                return sum
            } else {
                return sample.dot(coefficients)
            }
        }
    }

    /// Computes the coefficient of determination R².
    ///
    /// R² = 1 - SS_res / SS_tot where SS_res = Σ(yᵢ - ŷᵢ)² and
    /// SS_tot = Σ(yᵢ - ȳ)². Returns 0 when SS_tot is zero (constant target).
    static func rSquared(predicted: [Double], actual: [Double]) -> Double {
        let mean = actual.reduce(0.0, +) / Double(actual.count)
        var ssRes = 0.0
        var ssTot = 0.0
        for i in 0..<actual.count {
            let residual = actual[i] - predicted[i]
            ssRes += residual * residual
            let deviation = actual[i] - mean
            ssTot += deviation * deviation
        }
        guard ssTot > 0 else { return 0.0 }
        return 1.0 - ssRes / ssTot
    }

    /// Computes mean squared error: MSE = Σ(yᵢ - ŷᵢ)² / n.
    static func meanSquaredError(predicted: [Double], actual: [Double]) -> Double {
        var sum = 0.0
        for i in 0..<actual.count {
            let diff = actual[i] - predicted[i]
            sum += diff * diff
        }
        return sum / Double(actual.count)
    }
}
