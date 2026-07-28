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

// MARK: - Eigendecomposition Result Type

/// The result of a symmetric eigendecomposition.
///
/// Expresses a symmetric matrix `A` as `V · Λ · Vᵀ`, where the columns of `V`
/// are orthonormal eigenvectors and `Λ` is the diagonal matrix of eigenvalues.
/// Here the decomposition is stored one eigenvector per row: `eigenvectors[k]`
/// pairs with `eigenvalues[k]`, so with a 3 × 3 input, `eigenvectors` has three
/// rows of three values and `eigenvectors[0]` is the direction of largest
/// variance. Eigenvalues are sorted descending.
///
/// The same input always produces the same decomposition — bit for bit, on
/// every input size — because a single deterministic algorithm handles every
/// shape.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let matrix = [[2.0, 1.0],
///               [1.0, 2.0]]
///
/// let eigen = try matrix.eigenDecomposed()
/// eigen.eigenvalues     // [3.0, 1.0]
/// eigen.eigenvectors[0] // [0.7071..., 0.7071...]
/// ```
public struct EigenDecomposition: Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let count = eigenvalues.count
        let valueNoun = count == 1 ? "eigenvalue" : "eigenvalues"
        let sweepNoun = sweepsUsed == 1 ? "sweep" : "sweeps"
        if converged {
            return "EigenDecomposition: \(count) \(valueNoun), converged in \(sweepsUsed) \(sweepNoun)"
        }
        return "EigenDecomposition: \(count) \(valueNoun), not converged after \(sweepsUsed) \(sweepNoun)"
    }

    /// Eigenvalues sorted in descending order.
    public let eigenvalues: [Double]

    /// Orthonormal eigenvectors, one per row. Row `k` pairs with `eigenvalues[k]`.
    /// Each eigenvector is sign-fixed so its largest-magnitude element is positive.
    public let eigenvectors: [[Double]]

    /// The number of full Jacobi sweeps the solver performed.
    public let sweepsUsed: Int

    /// Whether the off-diagonal norm reached the convergence tolerance within
    /// the sweep limit. An unconverged result is still a valid decomposition —
    /// the iterates remain exactly orthogonal-similar to the input — but its
    /// off-diagonal residual is larger than the tolerance guarantees.
    public let converged: Bool
}

// MARK: - Symmetric Eigendecomposition

public extension Array where Element == [Double] {

    /// Computes the eigenvalues and eigenvectors of a symmetric matrix.
    ///
    /// Uses the cyclic Jacobi rotation method: repeated sweeps of plane
    /// rotations shrink the off-diagonal mass until the matrix is diagonal to
    /// within a relative tolerance of `1e-12` against the Frobenius norm of
    /// the input, with a cap of 50 sweeps. Well-conditioned matrices converge
    /// in about 5 sweeps. If the cap is reached, the result is returned with
    /// ``EigenDecomposition/converged`` set to `false` rather than thrown
    /// away, because Jacobi iterates are always exactly orthogonal-similar to
    /// the input.
    ///
    /// Symmetry is checked with a relative tolerance:
    /// `|A[i][j] − A[j][i]| ≤ 1e-12 · max(|A[i][j]|, |A[j][i]|, 1.0)`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let covariance = [[4.0, 2.0],
    ///                   [2.0, 3.0]]
    ///
    /// let eigen = try covariance.eigenDecomposed()
    /// eigen.eigenvalues  // [5.56..., 1.43...], descending
    /// ```
    ///
    /// - Returns: The eigenvalues (descending) and matching eigenvectors.
    /// - Throws: `MatrixError.notSquare` if the matrix is empty, non-square,
    ///   or has rows of inconsistent length; `MatrixError.notSymmetric` if any
    ///   off-diagonal pair differs beyond the relative tolerance.
    /// - Complexity: O(*n*³) per sweep where *n* is the matrix dimension.
    ///   Performs well for matrices up to a few hundred rows.
    func eigenDecomposed() throws -> EigenDecomposition {
        guard !self.isEmpty, self.count == self[0].count else {
            throw MatrixError.notSquare
        }
        let n = self.count
        for row in self where row.count != n {
            throw MatrixError.notSquare
        }

        // Symmetry check with a relative tolerance, so matrices measured in
        // millions are not rejected for a last-digit mismatch.
        for i in 0..<n {
            for j in (i + 1)..<n {
                let difference = abs(self[i][j] - self[j][i])
                let scale = Swift.max(abs(self[i][j]), abs(self[j][i]), 1.0)
                guard difference <= 1e-12 * scale else {
                    throw MatrixError.notSymmetric
                }
            }
        }

        return _Eigen.decompose(self)
    }
}
