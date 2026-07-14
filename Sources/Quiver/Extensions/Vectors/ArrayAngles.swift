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

// MARK: - Angle Calculations for Double

public extension Array where Element == Double {
    /// Returns the angle between two vectors in radians.
    ///
    /// Computes the arc cosine of the cosine similarity, producing
    /// a value from 0 (parallel) to π (opposite directions).
    ///
    /// Example:
    /// ```swift
    /// let v = [4.0, 3.0]
    /// let w = [5.0, 12.0]
    /// let theta = v.angle(with: w)
    /// // 0.533 radians (dot = 56, |v| = 5, |w| = 13, cos = 56/65 ≈ 0.862)
    /// ```
    ///
    /// Returns π/2 (90°) if either vector has zero magnitude. A zero vector
    /// has no direction, so ``cosineOfAngle(with:)`` returns 0.0 and `acos(0.0)`
    /// yields π/2 by convention — not a measured angle.
    ///
    /// For the raw similarity score itself, see ``cosineOfAngle(with:)``. For
    /// the angle in degrees, see ``angleInDegrees(with:)->Double``.
    ///
    /// - Parameter other: The vector to measure the angle against (must have the same number of elements)
    /// - Returns: The angle in radians, in the range [0, π]
    func angle(with other: [Double]) -> Double {
        let cosine = self.cosineOfAngle(with: other)
        return acos(Swift.min(1.0, Swift.max(-1.0, cosine)))
    }

    /// Returns the angle between two vectors in degrees.
    ///
    /// Convenience wrapper around ``angle(with:)->Double`` that converts the result
    /// from radians to degrees.
    ///
    /// Example:
    /// ```swift
    /// let v = [4.0, 3.0]
    /// let w = [5.0, 12.0]
    /// let theta = v.angleInDegrees(with: w)
    /// // 30.5 degrees (dot = 56, |v| = 5, |w| = 13, cos = 56/65 ≈ 0.862)
    /// ```
    ///
    /// Returns 90° if either vector has zero magnitude. A zero vector has no
    /// direction, so the angle is π/2 (90°) by convention — not a measured angle.
    ///
    /// For the raw similarity score itself, see ``cosineOfAngle(with:)``. For
    /// the angle in radians, see ``angle(with:)->Double``.
    ///
    /// - Parameter other: The vector to measure the angle against (must have the same number of elements)
    /// - Returns: The angle in degrees, in the range [0, 180]
    func angleInDegrees(with other: [Double]) -> Double {
        return angle(with: other) * 180 / .pi
    }
}

// MARK: - Angle Calculations for Float

public extension Array where Element == Float {
    /// Returns the angle between two vectors in radians.
    ///
    /// Computes the arc cosine of the cosine similarity, producing
    /// a value from 0 (parallel) to π (opposite directions).
    ///
    /// Example:
    /// ```swift
    /// let v: [Float] = [4.0, 3.0]
    /// let w: [Float] = [5.0, 12.0]
    /// let theta = v.angle(with: w)
    /// // 0.533 radians (dot = 56, |v| = 5, |w| = 13, cos = 56/65 ≈ 0.862)
    /// ```
    ///
    /// Returns π/2 (90°) if either vector has zero magnitude. A zero vector
    /// has no direction, so ``cosineOfAngle(with:)`` returns 0.0 and `acos(0.0)`
    /// yields π/2 by convention — not a measured angle.
    ///
    /// For the raw similarity score itself, see ``cosineOfAngle(with:)``. For
    /// the angle in degrees, see ``angleInDegrees(with:)->Float``.
    ///
    /// - Parameter other: The vector to measure the angle against (must have the same number of elements)
    /// - Returns: The angle in radians, in the range [0, π]
    func angle(with other: [Float]) -> Float {
        let cosine = self.cosineOfAngle(with: other)
        return acos(Swift.min(1.0, Swift.max(-1.0, cosine)))
    }

    /// Returns the angle between two vectors in degrees.
    ///
    /// Convenience wrapper around ``angle(with:)->Float`` that converts the result
    /// from radians to degrees.
    ///
    /// Example:
    /// ```swift
    /// let v: [Float] = [4.0, 3.0]
    /// let w: [Float] = [5.0, 12.0]
    /// let theta = v.angleInDegrees(with: w)
    /// // 30.5 degrees (dot = 56, |v| = 5, |w| = 13, cos = 56/65 ≈ 0.862)
    /// ```
    ///
    /// Returns 90° if either vector has zero magnitude. A zero vector has no
    /// direction, so the angle is π/2 (90°) by convention — not a measured angle.
    ///
    /// For the raw similarity score itself, see ``cosineOfAngle(with:)``. For
    /// the angle in radians, see ``angle(with:)->Float``.
    ///
    /// - Parameter other: The vector to measure the angle against (must have the same number of elements)
    /// - Returns: The angle in degrees, in the range [0, 180]
    func angleInDegrees(with other: [Float]) -> Float {
        return angle(with: other) * 180 / .pi
    }
}
