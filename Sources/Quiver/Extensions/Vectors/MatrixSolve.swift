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

// MARK: - Linear system solver

public extension Array where Element == [Double] {

    /// Solves the linear system `A · x = b` for the unknown vector `x`.
    ///
    /// Treats `self` as a square coefficient matrix `A` and `b` as the
    /// right-hand-side vector. Computes `x = A⁻¹ · b` by inverting `A` once
    /// and applying the inverse to `b`. This is convenient for school-level
    /// systems where the matrix is small and the user has already arranged
    /// the equations in matrix form; for large systems prefer a dedicated
    /// LU or QR factorization.
    ///
    /// > Note: Singular matrices return `nil`. Near-singular matrices
    /// > (condition number above roughly `10¹⁰`) return a non-`nil` value
    /// > that may be numerically unreliable. For ill-conditioned systems
    /// > the result should be treated with care — see <doc:Determinants-Primer>.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // 2x +  y = 5
    /// //  x + 3y = 10
    /// let A = [[2.0, 1.0],
    ///          [1.0, 3.0]]
    /// let b = [5.0, 10.0]
    /// A.solve(b)  // [1.0, 3.0]
    /// ```
    ///
    /// - Parameter b: The right-hand-side vector. Must contain one element per
    ///   row of `self`.
    /// - Returns: The solution vector `x`, or `nil` if `self` is not square,
    ///   has rows of inconsistent length, has a row count that disagrees with
    ///   `b.count`, or is singular (no unique solution).
    func solve(_ b: [Double]) -> [Double]? {
        // Validate shape: non-empty, square, uniform row length, and matched RHS.
        guard !self.isEmpty else { return nil }
        let n = self.count
        for row in self {
            if row.count != n { return nil }
        }
        guard b.count == n else { return nil }

        // Invert; a singular matrix throws and we surface that as nil.
        guard let inverse = try? self.inverted() else { return nil }

        // x = A⁻¹ · b — multiply each inverse row against b.
        var x = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = 0.0
            for j in 0..<n {
                sum += inverse[i][j] * b[j]
            }
            x[i] = sum
        }
        return x
    }
}
