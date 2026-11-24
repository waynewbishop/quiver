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
/// A model fit with `intercept: false` has no base value, so every element is a
/// weight and `coefficients[0]` is the first input's weight. ``hasIntercept``
/// reports which layout a model uses, and ``equation()`` reads it so the first
/// weight is never printed as an intercept.
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
/// ``LogisticRegression`` also learns a list of weights, but they predict the
/// log-odds of a class rather than `y`, so it prints its own formula with
/// ``LogisticRegression/equation(form:)`` instead of conforming.
///
/// To go deeper — what a number really tells you, and when it can mislead you —
/// see <doc:Model-Interpretation-Primer>. For more on printing models and
/// matrices as readable math, see <doc:Rendering-Math-Primer>.
public protocol Coefficients {
    /// The model's learned numbers: the base value (intercept) first, then one
    /// weight per input, in the order the inputs were given.
    var coefficients: [Double] { get }

    /// Whether ``coefficients`` begins with an intercept. When `false`, every
    /// element is a feature weight. Defaults to `true`.
    var hasIntercept: Bool { get }
}

extension Coefficients {

    /// Models conform with an intercept-first layout unless they say otherwise.
    public var hasIntercept: Bool { true }

    /// Renders the fit as a readable equation, such as `y = 38000.00 + 110.00x`.
    ///
    /// The formula is built from ``coefficients`` directly: `coefficients[0]` is
    /// the intercept and each remaining weight pairs with a variable in input
    /// order. When ``hasIntercept`` is `false` there is no intercept term, and
    /// every element pairs with a variable, as in `y = 2.00x`. A single-feature
    /// model uses a bare `x`; a multi-feature model uses subscripts (`x₁`, `x₂`,
    /// …) so the terms stay distinct. Terms with a zero weight are dropped, and a weight of exactly one renders as the bare
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
        let variables = _LinearEquation.subscriptVariables(
            count: _LinearEquation.weightCount(coefficients, hasIntercept: hasIntercept))
        return "y = " + _LinearEquation.render(coefficients, hasIntercept: hasIntercept,
                                               variables: variables, joiner: "")
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
    ///   `coefficients.count - 1` with an intercept and `coefficients.count`
    ///   without one.
    public func equation(variables: [String], response: String = "y") -> String {
        let c = coefficients
        guard !c.isEmpty else { return response + " = 0" }
        let expected = _LinearEquation.weightCount(c, hasIntercept: hasIntercept)
        precondition(variables.count == expected,
                     "equation(variables:) needs one name per weight: expected \(expected), got \(variables.count)")
        return response + " = " + _LinearEquation.render(c, hasIntercept: hasIntercept,
                                                         variables: variables, joiner: "·")
    }
}

// MARK: - Shared rendering

/// Renders a linear combination of weights and variables for every model that
/// prints its fit, so the coefficient-based models and ``LogisticRegression``
/// format their formulas identically.
enum _LinearEquation {

    /// Returns the number of feature weights in a coefficient vector.
    static func weightCount(_ c: [Double], hasIntercept: Bool) -> Int {
        hasIntercept ? Swift.max(c.count - 1, 0) : c.count
    }

    /// Returns `x` for a single weight, otherwise `x₁`, `x₂`, … so terms never collide.
    static func subscriptVariables(count: Int) -> [String] {
        var variables: [String] = []
        for index in 0..<count {
            variables.append(count == 1 ? "x" : "x" + subscriptDigits(index + 1))
        }
        return variables
    }

    /// Joins the optional intercept and each signed, named weight into the
    /// right-hand side of a formula, such as `-0.20 + 3.10·tion - 2.50·ment`.
    static func render(_ c: [Double], hasIntercept: Bool, variables: [String],
                       joiner: String) -> String {
        guard !c.isEmpty else { return "0" }

        var pieces: [String] = []
        // With an intercept, coefficients[0] is the base value and always prints.
        if hasIntercept {
            pieces.append(fixed(c[0]))
        }
        let weights = hasIntercept ? Array(c.dropFirst()) : c

        for (offset, weight) in weights.enumerated() where weight != 0 {
            let magnitude = Swift.abs(weight)
            // A weight of exactly 1 renders as the bare variable (x, not 1.00x).
            let term = magnitude == 1
                ? variables[offset]
                : fixed(magnitude) + joiner + variables[offset]
            if pieces.isEmpty {
                // The leading term carries its own sign: "-2.50x", never "- 2.50x".
                pieces.append((weight < 0 ? "-" : "") + term)
            } else {
                pieces.append((weight < 0 ? "- " : "+ ") + term)
            }
        }

        // No intercept and every weight zero leaves nothing to print.
        return pieces.isEmpty ? "0" : pieces.joined(separator: " ")
    }

    /// Formats a value to two decimals, printing a value that rounds to zero as
    /// `0.00` rather than `-0.00`.
    private static func fixed(_ value: Double) -> String {
        let text = String(format: "%.2f", value)
        return text == "-0.00" ? "0.00" : text
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
