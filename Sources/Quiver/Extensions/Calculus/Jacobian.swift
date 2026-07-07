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

// MARK: - Jacobian

/// The Jacobian of a vector-valued function, approximated by central differences.
///
/// For a function that maps an n-vector to an m-vector, evaluated at `x`, returns
/// the m×n matrix J of first-order partial derivatives, where J[i][j] is the rate
/// at which output i changes as input j changes. Near the evaluation point,
/// multiplying a small input change by J predicts the resulting output change —
/// the same idea as a matrix transforming a vector.
///
/// The derivative in each input direction is estimated by perturbing that
/// coordinate a small step in both directions and comparing the outputs. Central
/// differences give second-order accuracy for one extra evaluation per input. The
/// default step size lands near the cube root of machine epsilon, which balances
/// the two competing errors: too large a step adds truncation error, too small a
/// step magnifies floating-point round-off.
///
/// ```swift
/// import Quiver
///
/// // f(length, width) = [area, perimeter]
/// func plotMetrics(_ v: [Double]) -> [Double] {
///     let length = v[0]
///     let width = v[1]
///     let area = length * width
///     let perimeter = 2 * (length + width)
///     return [area, perimeter]
/// }
///
/// let point = [3.0, 2.0]
/// let j = jacobian(of: plotMetrics, at: point)
/// // [[2.0, 3.0],   ∂area/∂length = width = 2,  ∂area/∂width = length = 3
/// //  [2.0, 2.0]]   ∂perimeter/∂length = 2,     ∂perimeter/∂width = 2
/// ```
///
/// - Parameters:
///   - f: The function to differentiate, mapping an n-vector to an m-vector.
///   - x: The point at which to evaluate the Jacobian.
///   - h: The finite-difference step size. The default (6e-6) is a reasonable
///        choice for Double inputs of order one; scale it up for larger inputs.
/// - Returns: The m×n Jacobian matrix.
public func jacobian(of f: ([Double]) -> [Double],
                     at x: [Double],
                     step h: Double = 6e-6) -> [[Double]] {
    // Each column is one input direction perturbed ±h, so the collected columns
    // form an n×m matrix; transposing lands the conventional m×n shape.
    let columns = (0..<x.count).map { j -> [Double] in
        var forward = x, backward = x
        forward[j]  += h
        backward[j] -= h
        // (f(x + h·eⱼ) − f(x − h·eⱼ)) / 2h  →  the j-th column of J
        return f(forward).subtract(f(backward)).broadcast(dividingBy: 2 * h)
    }
    return columns.transposed()
}
