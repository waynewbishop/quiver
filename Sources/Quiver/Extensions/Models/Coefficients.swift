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

// MARK: - Coefficients

/// A model whose fit is summarized by a vector of coefficients.
///
/// The regressors that report a fit as `coefficients` — ``LinearRegression``,
/// ``Ridge``, ``GradientDescent`` — conform (intercept first, then one weight
/// per feature). Distance- and tree-based models do not, so this capability is
/// its own protocol rather than a requirement on every regressor.
///
/// ``LogisticRegression`` also stores coefficients, but it is a ``Classifier``,
/// not a ``Regressor``, so it is not wrapped by `ResidualModel` and is left out
/// of this protocol. The exclusion rests on meaning, not on a structural
/// barrier: a residual `observed − predicted` is defined for a continuous
/// target, not for a probability or a 0/1 label.
public protocol Coefficients {
    /// The fitted coefficients: intercept first, then one weight per feature.
    var coefficients: [Double] { get }
}

extension Coefficients {

    /// Renders the fit as a readable equation, such as `y = 38000.00 + 110.00x₁`.
    ///
    /// The formula is built from ``coefficients`` directly: `coefficients[0]` is
    /// the intercept and each remaining weight pairs with a subscripted variable
    /// `x₁`, `x₂`, … in input order. Terms with a zero weight are dropped, and a
    /// weight of exactly one renders as the bare variable (`x₁`, not `1.00x₁`).
    /// The weights are rendered in the units the model was fit in — when the
    /// features were standardized first, each is a per-standard-deviation slope.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let model = try LinearRegression.fit(features: x, targets: y)
    /// print(model.equation())  // y = 38000.00 + 110.00x₁
    /// ```
    ///
    /// - Returns: The fitted formula as a string. An empty coefficient vector
    ///   renders as `y = 0`.
    public func equation() -> String {
        let c = coefficients
        guard !c.isEmpty else { return "y = 0" }

        // coefficients[0] is the intercept; the rest pair with x₁, x₂, …
        var pieces = [String(format: "%.2f", c[0])]

        for (offset, weight) in c.dropFirst().enumerated() where weight != 0 {
            let variable = "x" + Self.subscriptDigits(offset + 1)
            let magnitude = Swift.abs(weight)
            // A weight of exactly 1 renders as the bare variable (x₁, not 1.00x₁).
            let coefficientText = magnitude == 1 ? "" : String(format: "%.2f", magnitude)
            pieces.append((weight < 0 ? "- " : "+ ") + coefficientText + variable)
        }

        return "y = " + pieces.joined(separator: " ")
    }

    /// Converts a positive integer into Unicode subscript digits (10 → "₁₀").
    private static func subscriptDigits(_ value: Int) -> String {
        let table: [Character: Character] = [
            "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
            "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        ]
        return String(String(value).map { table[$0] ?? $0 })
    }
}

extension LinearRegression: Coefficients {}
extension Ridge: Coefficients {}
extension GradientDescent: Coefficients {}
