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

// MARK: - LU Decomposition Result Type

/// The result of an LU factorization with partial pivoting.
///
/// Expresses a square matrix `A` as `P · A = L · U`, where `L` is unit lower
/// triangular (1s on the diagonal), `U` is upper triangular, and `P` is the row
/// permutation chosen during pivoting. The factorization is computed once and
/// can then solve `A · x = b` for many right-hand sides cheaply: each solve is
/// an O(*n*²) pair of substitutions rather than another O(*n*³) factorization.
///
/// Example:
/// ```swift
/// let A = [[2.0, 1.0, 1.0],
///          [4.0, -6.0, 0.0],
///          [-2.0, 7.0, 2.0]]
///
/// let lu = try A.luDecomposed()
/// let x = lu.solve([5.0, -2.0, 9.0])  // [1.0, 1.0, 2.0]
/// ```
public struct LUDecomposition: Sendable {

    /// Unit lower triangular factor (1s on the diagonal, multipliers below).
    public let l: [[Double]]

    /// Upper triangular factor.
    public let u: [[Double]]

    /// Row permutation applied during pivoting. `p[i]` is the original index of
    /// the row now occupying position `i`, so `(P · b)[i] == b[p[i]]`.
    public let p: [Int]

    /// Sign contributed by the row swaps: `+1` for an even number of swaps,
    /// `-1` for an odd number. Used to recover the determinant.
    public let permutationSign: Double

    /// The determinant of the original matrix, recovered as the permutation sign
    /// times the product of the diagonal of `U`.
    public var determinant: Double {
        var result = permutationSign
        for i in 0..<u.count {
            result *= u[i][i]
        }
        return result
    }
}

// MARK: - LU Factorization

public extension Array where Element == [Double] {

    /// Factors a square matrix into `P · A = L · U` using partial pivoting.
    ///
    /// Partial pivoting selects the largest available pivot in each column to
    /// limit the growth of rounding error. The returned ``LUDecomposition`` can
    /// solve `A · x = b` for many different `b` vectors without refactoring.
    ///
    /// Example:
    /// ```swift
    /// let A = [[2.0, 1.0],
    ///          [1.0, 3.0]]
    /// let lu = try A.luDecomposed()
    /// lu.solve([5.0, 10.0])  // [1.0, 3.0]
    /// ```
    ///
    /// - Complexity: O(*n*³) where *n* is the matrix dimension. Each subsequent
    ///   ``LUDecomposition/solve(_:)`` is O(*n*²).
    /// - Returns: The `L`, `U`, and permutation factors of the matrix.
    /// - Throws: `MatrixError.notSquare` if the matrix is not square or has rows
    ///   of inconsistent length; `MatrixError.singular` if the matrix is singular.
    func luDecomposed() throws -> LUDecomposition {
        guard !self.isEmpty, self.count == self[0].count else {
            throw MatrixError.notSquare
        }
        let n = self.count
        for row in self where row.count != n {
            throw MatrixError.notSquare
        }

        // Work in place: below-diagonal entries accumulate L's multipliers,
        // on/above-diagonal entries become U.
        var a = self
        var p = [Int](0..<n)
        var sign = 1.0
        let epsilon = Double.ulpOfOne * 1000

        for i in 0..<n {
            // Partial pivot: find the largest magnitude entry in this column.
            var maxRow = i
            for k in (i + 1)..<n where abs(a[k][i]) > abs(a[maxRow][i]) {
                maxRow = k
            }

            if abs(a[maxRow][i]) < epsilon {
                throw MatrixError.singular
            }

            if maxRow != i {
                a.swapAt(i, maxRow)
                p.swapAt(i, maxRow)
                sign = -sign
            }

            // Eliminate the column below the pivot, storing each multiplier
            // in the (now-vacated) lower-triangular slot.
            for k in (i + 1)..<n {
                let factor = a[k][i] / a[i][i]
                a[k][i] = factor
                for j in (i + 1)..<n {
                    a[k][j] -= factor * a[i][j]
                }
            }
        }

        // Split the packed working matrix into separate L and U.
        var l = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var u = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            l[i][i] = 1
            for j in 0..<n {
                if j < i {
                    l[i][j] = a[i][j]
                } else {
                    u[i][j] = a[i][j]
                }
            }
        }

        return LUDecomposition(l: l, u: u, p: p, permutationSign: sign)
    }
}

// MARK: - Solving with a Factorization

public extension LUDecomposition {

    /// Solves `A · x = b` using the stored factorization.
    ///
    /// Applies the permutation to `b`, solves `L · y = P · b` by forward
    /// substitution, then solves `U · x = y` by back substitution. Because the
    /// factorization already guaranteed a non-singular `U`, this step always
    /// succeeds for a correctly sized `b`.
    ///
    /// Example:
    /// ```swift
    /// let lu = try A.luDecomposed()
    /// let monday  = lu.solve(b1)   // reuse the same factorization
    /// let tuesday = lu.solve(b2)
    /// ```
    ///
    /// - Parameter b: The right-hand-side vector, one element per matrix row.
    /// - Returns: The solution vector `x`.
    /// - Complexity: O(*n*²).
    func solve(_ b: [Double]) -> [Double] {
        let n = l.count
        precondition(b.count == n, "Right-hand side must match the matrix dimension")

        // Apply the row permutation: (P · b)[i] = b[p[i]].
        var pb = [Double](repeating: 0, count: n)
        for i in 0..<n {
            pb[i] = b[p[i]]
        }

        // Forward substitution: L · y = P · b  (L has a unit diagonal).
        var y = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var sum = pb[i]
            for j in 0..<i {
                sum -= l[i][j] * y[j]
            }
            y[i] = sum
        }

        // Back substitution: U · x = y.
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = y[i]
            for j in (i + 1)..<n {
                sum -= u[i][j] * x[j]
            }
            x[i] = sum / u[i][i]
        }

        return x
    }
}
