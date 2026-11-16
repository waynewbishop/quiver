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

// MARK: - Internal Singularity Tolerance

/// Internal namespace for the singular-matrix test shared by every elimination routine.
///
/// A pivot is treated as zero when its magnitude is at or below a cutoff scaled
/// to the largest entry in the matrix. A fixed cutoff fails in both directions:
/// with large entries (sums of squared house prices, say) the rounding left in a
/// pivot that should be zero sits far above any fixed value, so a rank-deficient
/// matrix slips through; with tiny entries a perfectly healthy matrix trips it.
/// Scaling the cutoff means multiplying the whole matrix by a constant leaves the
/// verdict unchanged, within representable limits. Rescaling individual columns,
/// such as expressing one feature in different units, changes the matrix's
/// conditioning and can change the verdict. When the largest entry is 1 the cutoff equals the previous fixed value of
/// `1000 · ulpOfOne`; it grows and shrinks in proportion to the matrix's scale.
internal enum _MatrixTolerance {

    /// Returns the pivot magnitude at or below which the matrix is treated as singular.
    static func pivotCutoff<T: FloatingPoint>(for matrix: [[T]]) -> T {
        var largest = T.zero
        for row in matrix {
            for value in row where abs(value) > largest {
                largest = abs(value)
            }
        }
        return T.ulpOfOne * T(1000) * largest
    }

    /// Returns the two elimination pivots of a 2×2 matrix, or `nil` when either fails the cutoff.
    ///
    /// The pivots are computed exactly as partial-pivoting elimination computes
    /// them: pivot on the larger first-column entry, then subtract a multiple of
    /// that row whose magnitude is at most 1. Working with the pivots rather than
    /// the raw products `a·d` and `b·c` keeps the test free of overflow and
    /// underflow at extreme scales. `sign` is the determinant's sign, including
    /// the flip from a row swap.
    static func pivots2x2<T: FloatingPoint>(_ matrix: [[T]]) -> (first: T, second: T, sign: T)? {
        let cutoff = pivotCutoff(for: matrix)
        let swapped = abs(matrix[1][0]) > abs(matrix[0][0])
        let top = swapped ? matrix[1] : matrix[0]
        let bottom = swapped ? matrix[0] : matrix[1]

        guard abs(top[0]) > cutoff else { return nil }
        let factor = bottom[0] / top[0]
        let second = bottom[1] - factor * top[1]
        guard abs(second) > cutoff else { return nil }

        var sign: T = swapped ? -1 : 1
        if top[0] < 0 { sign = -sign }
        if second < 0 { sign = -sign }
        return (top[0], second, sign)
    }

    /// Returns whether a 2×2 matrix fails the same pivot test partial-pivoting elimination applies.
    ///
    /// Sharing the pivot test keeps the closed-form 2×2 determinant in agreement
    /// with the LU path on which matrices are singular. A nonsingular matrix can
    /// still report a `0.0` determinant when the product of its pivots underflows,
    /// so a zero determinant alone does not establish singularity.
    static func isSingular2x2<T: FloatingPoint>(_ matrix: [[T]]) -> Bool {
        pivots2x2(matrix) == nil
    }
}
