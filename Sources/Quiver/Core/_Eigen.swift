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

// MARK: - Jacobi Eigensolver

/// Namespace for the symmetric eigendecomposition kernel.
internal enum _Eigen {

    /// Maximum number of cyclic sweeps before the solver stops. Measured sweep
    /// counts are 5 for well-conditioned 6×6 input and at most 12 at n = 80,
    /// so the cap carries roughly 4× headroom rather than acting as a tuning knob.
    static let sweepLimit = 50

    /// Relative convergence tolerance applied against the Frobenius norm of
    /// the original matrix.
    static let relativeTolerance = 1e-12

    /// Rotations skip a pair only when its off-diagonal entry is a true zero
    /// at this threshold; termination is owned by the relative convergence test.
    static let skipThreshold = 1e-300

    /// Diagonalizes a symmetric matrix using cyclic Jacobi rotations.
    ///
    /// Performs full sweeps over every off-diagonal pair (p, q) with p < q in
    /// cyclic order, rotating each pair through the smaller angle so that
    /// convergence is monotone. After each sweep the off-diagonal Frobenius
    /// norm `off(A)` is tested against `1e-12 · ‖A‖_F`, where `‖A‖_F` is
    /// computed once from the original matrix and held fixed — the Frobenius
    /// norm is invariant under orthogonal similarity, so the fixed value is
    /// exact. The solver stops at 50 sweeps if the test never passes; the
    /// iterates remain exactly orthogonal-similar to the input, so an
    /// unconverged result is still a valid decomposition and is returned with
    /// `converged` set to `false` rather than thrown away.
    ///
    /// Eigenvalues are returned sorted descending with eigenvectors permuted
    /// to match, and each eigenvector is sign-fixed so its largest-magnitude
    /// element is positive (ties on magnitude resolve to the lowest index).
    ///
    /// The caller is responsible for validating shape: the input must be a
    /// non-empty square symmetric matrix. `eigenDecomposed()` performs that
    /// validation for public callers.
    ///
    /// - Parameter matrix: A non-empty square symmetric matrix.
    /// - Returns: The eigenvalues, eigenvectors, sweep count, and convergence flag.
    /// - Complexity: O(*n*³) per sweep for an *n* × *n* matrix.
    static func decompose(_ matrix: [[Double]]) -> EigenDecomposition {
        let n = matrix.count

        // A 1×1 matrix is already diagonal.
        if n == 1 {
            return EigenDecomposition(
                eigenvalues: [matrix[0][0]],
                eigenvectors: [[1.0]],
                sweepsUsed: 0,
                converged: true
            )
        }

        var a = matrix

        // Eigenvector accumulator, initialized to the identity. Column k of v
        // is eigenvector k until extraction transposes into rows.
        var v = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        for i in 0..<n {
            v[i][i] = 1.0
        }

        // Frobenius norm of the original matrix, computed once and held fixed.
        var squaredSum = 0.0
        for i in 0..<n {
            for j in 0..<n {
                squaredSum += a[i][j] * a[i][j]
            }
        }
        let frobeniusNorm = Foundation.sqrt(squaredSum)

        // An all-zeros matrix has zero eigenvalues and the identity as its basis.
        if frobeniusNorm == 0.0 {
            return extract(diagonalOf: a, vectors: v, sweepsUsed: 0, converged: true)
        }

        let threshold = relativeTolerance * frobeniusNorm
        var sweepsUsed = 0
        var converged = false

        while sweepsUsed < sweepLimit {
            // One cyclic sweep over every pair (p, q), p < q.
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = a[p][q]
                    if abs(apq) <= skipThreshold {
                        continue
                    }

                    // Stable rotation: computes the smaller root as a quotient
                    // of well-separated quantities, avoiding the catastrophic
                    // cancellation of the naive t = -theta ± sqrt(theta² + 1)
                    // form as theta grows. sign(0) resolves to +1.
                    let theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                    let t: Double
                    if theta >= 0.0 {
                        t = 1.0 / (theta + Foundation.sqrt(theta * theta + 1.0))
                    } else {
                        t = -1.0 / (-theta + Foundation.sqrt(theta * theta + 1.0))
                    }
                    let c = 1.0 / Foundation.sqrt(t * t + 1.0)
                    let s = t * c

                    // Update the (p, q) block: the rotation zeroes a[p][q] exactly.
                    let app = a[p][p]
                    let aqq = a[q][q]
                    a[p][p] = app - t * apq
                    a[q][q] = aqq + t * apq
                    a[p][q] = 0.0
                    a[q][p] = 0.0

                    // Update only rows and columns p and q — O(n) per pair.
                    for i in 0..<n where i != p && i != q {
                        let aip = a[i][p]
                        let aiq = a[i][q]
                        a[i][p] = c * aip - s * aiq
                        a[p][i] = a[i][p]
                        a[i][q] = s * aip + c * aiq
                        a[q][i] = a[i][q]
                    }

                    // Accumulate the rotation into the eigenvector columns.
                    for i in 0..<n {
                        let vip = v[i][p]
                        let viq = v[i][q]
                        v[i][p] = c * vip - s * viq
                        v[i][q] = s * vip + c * viq
                    }
                }
            }

            sweepsUsed += 1

            if offDiagonalNorm(of: a) <= threshold {
                converged = true
                break
            }
        }

        return extract(diagonalOf: a, vectors: v, sweepsUsed: sweepsUsed, converged: converged)
    }

    /// Computes the Frobenius norm of the off-diagonal part of a square matrix.
    ///
    /// - Parameter matrix: The square matrix to measure.
    /// - Returns: `sqrt(Σᵢ Σⱼ≠ᵢ matrix[i][j]²)`.
    private static func offDiagonalNorm(of matrix: [[Double]]) -> Double {
        var squaredSum = 0.0
        let n = matrix.count
        for i in 0..<n {
            for j in 0..<n where j != i {
                squaredSum += matrix[i][j] * matrix[i][j]
            }
        }
        return Foundation.sqrt(squaredSum)
    }

    /// Extracts eigenvalues from the diagonal, sorts them descending with
    /// eigenvectors permuted to match, and applies the sign convention.
    ///
    /// The sign convention negates any eigenvector whose largest-magnitude
    /// element is negative, with ties on magnitude resolving to the lowest
    /// index. Eigenvector k arrives as column k of `vectors` and leaves as
    /// row k of the result.
    ///
    /// - Parameters:
    ///   - matrix: The diagonalized working matrix; its diagonal holds the eigenvalues.
    ///   - vectors: The accumulated rotation product; column k is eigenvector k.
    ///   - sweepsUsed: The number of full sweeps the solver performed.
    ///   - converged: Whether the convergence test passed within the sweep limit.
    /// - Returns: The assembled decomposition with descending eigenvalues.
    private static func extract(
        diagonalOf matrix: [[Double]],
        vectors: [[Double]],
        sweepsUsed: Int,
        converged: Bool
    ) -> EigenDecomposition {
        let n = matrix.count

        var diagonal = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            diagonal[i] = matrix[i][i]
        }

        // sortedIndices() is ascending; reverse for descending eigenvalues.
        let order = Array(diagonal.sortedIndices().reversed())

        var eigenvalues = [Double]()
        eigenvalues.reserveCapacity(n)
        var eigenvectors = [[Double]]()
        eigenvectors.reserveCapacity(n)

        for k in order {
            eigenvalues.append(diagonal[k])

            var vector = [Double](repeating: 0.0, count: n)
            var largestMagnitude = 0.0
            var largestIndex = 0
            for i in 0..<n {
                vector[i] = vectors[i][k]
                if abs(vector[i]) > largestMagnitude {
                    largestMagnitude = abs(vector[i])
                    largestIndex = i
                }
            }
            if vector[largestIndex] < 0.0 {
                for i in 0..<n {
                    vector[i] = -vector[i]
                }
            }
            eigenvectors.append(vector)
        }

        return EigenDecomposition(
            eigenvalues: eigenvalues,
            eigenvectors: eigenvectors,
            sweepsUsed: sweepsUsed,
            converged: converged
        )
    }
}
