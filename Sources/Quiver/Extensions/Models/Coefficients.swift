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

/// A model you can read the numbers out of.
///
/// Some models make a prediction by multiplying each input by a learned number
/// and adding the results up. Those learned numbers are the model's
/// **coefficients** — they are the model. A house-price model might learn that
/// each square foot is worth about $110 and that a sale starts from a base of
/// $38,000 before any size is counted. Predict a price and you are really just
/// doing `38,000 + 110 × size`.
///
/// `Coefficients` is the protocol for any model whose learning boils down to a
/// list of those numbers. Conforming to it means the model can hand back its
/// learned values, and — because reading them is the same act for every such
/// model — print itself as a plain formula.
///
/// This matters for two everyday reasons. First, it lets you **inspect** what a
/// model actually decided: a big number means that input mattered a lot, a small
/// one means it barely moved the prediction. Second, it lets you **explain** the
/// model. A model that prints `y = 38000.00 + 110.00x` is not a black box; a
/// teammate, a reviewer, or a student can read it and check it by hand.
///
/// ```swift
/// import Quiver
///
/// // Learn a price from square footage.
/// let model = try LinearRegression.fit(features: x, targets: y)
///
/// model.coefficients   // [38000.0, 110.0] — the base value, then the price per square foot
/// model.equation()     // "y = 38000.00 + 110.00x" — the same model, written out
/// ```
///
/// ## How the numbers are arranged
///
/// `coefficients` is one `[Double]`, and the order is always the same: the base
/// value (the **intercept**) comes first, then one number per input, in the
/// order the inputs were given. So `coefficients[0]` is the starting value
/// before any input is counted, and the rest are how much each input pushes the
/// prediction up or down.
///
/// ## Which models conform
///
/// The models whose fit *is* a list of numbers conform: ``LinearRegression``,
/// ``Ridge``, and ``GradientDescent``. Models that predict a different way — by
/// comparing a new point to stored examples, like nearest-neighbors, or by
/// splitting on thresholds, like a tree — have no such list, so they do not
/// conform. That is why this is its own protocol and not something every model
/// is forced to provide.
///
/// To go deeper — what a number really tells you, and when it can mislead you —
/// see <doc:Model-Interpretation-Primer>. For more on printing models and
/// matrices as readable math, see <doc:Rendering-Math-Primer>.
public protocol Coefficients {
    /// The model's learned numbers: the base value (intercept) first, then one
    /// weight per input, in the order the inputs were given.
    var coefficients: [Double] { get }
}

extension Coefficients {

    /// Renders the fit as a readable equation, such as `y = 38000.00 + 110.00x`.
    ///
    /// The formula is built from ``coefficients`` directly: `coefficients[0]` is
    /// the intercept and each remaining weight pairs with a variable in input
    /// order. A single-feature model uses a bare `x`; a multi-feature model uses
    /// subscripts (`x₁`, `x₂`, …) so the terms stay distinct. Terms with a zero
    /// weight are dropped, and a weight of exactly one renders as the bare
    /// variable (`x`, not `1.00x`). The weights are rendered in the units the
    /// model was fit in — when the features were standardized first, each is a
    /// per-standard-deviation slope.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let model = try LinearRegression.fit(features: x, targets: y)
    /// print(model.equation())  // y = 38000.00 + 110.00x
    /// ```
    ///
    /// - Returns: The fitted formula as a string. An empty coefficient vector
    ///   renders as `y = 0`.
    public func equation() -> String {
        // A lone feature uses a bare `x`; two or more take subscripts so the
        // terms never collide.
        let weightCount = Swift.max(coefficients.count - 1, 0)
        var variables: [String] = []
        for index in 0..<weightCount {
            variables.append(weightCount == 1 ? "x" : "x" + Self.subscriptDigits(index + 1))
        }
        return Self.render(coefficients, variables: variables, response: "y", joiner: "")
    }

    /// Renders the fit as a readable equation with each variable named, such as
    /// `price = 38000.00 + 110.00·sqft`.
    ///
    /// This is ``equation()`` with names in place of `x₁`, `x₂`, …: each name
    /// pairs with a weight in input order, joined by a middle dot, and the
    /// left-hand side reads `response`. The formatting rules are the same —
    /// two decimal places, zero weights dropped, and a weight of exactly one
    /// rendered as the bare name.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let model = try LinearRegression.fit(features: x, targets: y)
    /// print(model.equation(variables: ["sqft"], response: "price"))
    /// // price = 38000.00 + 110.00·sqft
    /// ```
    ///
    /// - Parameters:
    ///   - variables: One name per weight, in the order the inputs were given.
    ///   - response: The name of the predicted quantity. Defaults to `y`.
    /// - Returns: The fitted formula as a string. An empty coefficient vector
    ///   renders as `response = 0`.
    /// - Precondition: `variables` holds exactly one name per weight, which is
    ///   `coefficients.count - 1`.
    public func equation(variables: [String], response: String = "y") -> String {
        let c = coefficients
        guard !c.isEmpty else { return response + " = 0" }
        precondition(variables.count == c.count - 1,
                     "equation(variables:) needs one name per weight: expected \(c.count - 1), got \(variables.count)")
        return Self.render(c, variables: variables, response: response, joiner: "·")
    }

    /// Joins the intercept and each named, signed weight into `response = …`.
    private static func render(_ c: [Double], variables: [String], response: String,
                               joiner: String) -> String {
        guard !c.isEmpty else { return response + " = 0" }

        // coefficients[0] is the intercept; the rest are the feature weights.
        var pieces = [String(format: "%.2f", c[0])]

        for (offset, weight) in c.dropFirst().enumerated() where weight != 0 {
            let magnitude = Swift.abs(weight)
            // A weight of exactly 1 renders as the bare variable (x, not 1.00x).
            let term = magnitude == 1
                ? variables[offset]
                : String(format: "%.2f", magnitude) + joiner + variables[offset]
            pieces.append((weight < 0 ? "- " : "+ ") + term)
        }

        return response + " = " + pieces.joined(separator: " ")
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
