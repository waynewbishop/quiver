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

// MARK: - Covariance Matrix

public extension Array where Element == [Double] {

    /// Computes the covariance matrix of a design matrix whose rows are samples
    /// and columns are features.
    ///
    /// The input follows the design-matrix orientation used by `StandardScaler`
    /// and the fitted models: each row is one observation, each column is one
    /// feature. The result is a p × p matrix for p feature columns, where cell
    /// `[j][k]` holds the covariance between feature `j` and feature `k`, and the
    /// diagonal holds each feature's variance.
    ///
    /// The computation is two-pass and centered: column means are computed first,
    /// the matrix is centered once, and every cell accumulates products of the
    /// same centered values. Only the upper triangle is computed; the lower
    /// triangle is mirrored, so the result is exactly symmetric.
    ///
    /// The eigenvalues of a covariance matrix are the squared singular values of
    /// the centered data divided by n − ddof. Forming the covariance matrix
    /// squares the condition number of the data, which concentrates rounding
    /// error in the smallest eigenvalues. For well-conditioned data this loses
    /// nothing measurable; when features are nearly collinear, the smallest
    /// eigenvalues lose precision while the largest remain accurate to machine
    /// precision. Check `conditionNumber` on the result when features may be
    /// redundant.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let samples = [
    ///     [1.0, 2.0],
    ///     [2.0, 4.0],
    ///     [3.0, 6.0]
    /// ]
    /// let covariance = samples.covarianceMatrix()
    /// // [[1.0, 2.0],
    /// //  [2.0, 4.0]]
    /// ```
    ///
    /// - Note: `correlationMatrix()` uses the opposite orientation — its rows are
    ///   series, not samples. Feeding a samples-as-rows design matrix to
    ///   `correlationMatrix()` produces an n × n matrix where p × p was intended.
    /// - Parameter ddof: Delta degrees of freedom. The divisor is
    ///   `count - ddof`, so the default of 1 gives the sample covariance and 0
    ///   gives the population covariance.
    /// - Returns: A p × p symmetric covariance matrix, or `nil` if the array is
    ///   empty, the rows have inconsistent lengths, `ddof` is negative, or there
    ///   are fewer than `ddof + 1` rows.
    /// - Complexity: O(*n*·*p*²) where *n* is the number of samples and *p* is
    ///   the number of features.
    func covarianceMatrix(ddof: Int = 1) -> [[Double]]? {
        guard ddof >= 0, count > ddof else { return nil }

        // meanVector() returns nil on ragged rows, validating the shape here.
        guard let means = meanVector() else { return nil }
        let featureCount = means.count

        // Center the matrix once so every cell sees bit-identical means.
        let negatedMeans = means.map { -$0 }
        let centered = broadcast(addingToEachRow: negatedMeans)

        let divisor = Double(count - ddof)
        var matrix = [[Double]](
            repeating: [Double](repeating: 0.0, count: featureCount),
            count: featureCount
        )

        // Accumulate the upper triangle, then mirror for exact symmetry.
        for j in 0..<featureCount {
            for k in j..<featureCount {
                var sum = 0.0
                for row in centered {
                    sum += row[j] * row[k]
                }
                let value = sum / divisor
                matrix[j][k] = value
                matrix[k][j] = value
            }
        }

        return matrix
    }
}
