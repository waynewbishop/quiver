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

// MARK: - Log Determinant Result Type

/// Represents the sign and natural logarithm of a matrix determinant.
///
/// This type enables numerically stable determinant computations for large matrices
/// where the raw determinant value would overflow or underflow floating-point range.
///
/// The determinant can be reconstructed via the `value` property, which computes
/// `sign * exp(logAbsValue)`.
public struct LogDeterminant: Sendable {

    /// The sign of the determinant: -1, 0, or 1
    public let sign: Double

    /// The natural logarithm of the absolute determinant value
    public let logAbsValue: Double

    /// Reconstructs the determinant value (sign times exp of log absolute value)
    public var value: Double {
        sign * Foundation.exp(logAbsValue)
    }
}
