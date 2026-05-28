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

// MARK: - Polynomial

/// A polynomial in a single real variable, stored as ordered coefficients.
///
/// `Polynomial` represents an expression of the form `a₀ + a₁x + a₂x² + ... + aₙxⁿ`,
/// keeping the coefficients in a `[Double]` indexed by power of `x`. The constant
/// term lives at index `0`, the linear term at index `1`, and so on. This is the
/// same convention used by school algebra and by most numerical computing
/// libraries that work with polynomials.
///
/// The type is a value type with no hidden state. Two polynomials with the same
/// coefficients are equal, encode to the same JSON, and are safe to share across
/// concurrent contexts. Evaluation uses Horner's method for numerical stability,
/// avoiding the precision loss of a naive `pow(x, k)` accumulation.
///
/// Example:
/// ```swift
/// import Quiver
///
/// // 2x² + 3x + 1
/// let p = Polynomial([1, 3, 2])
/// p(2)              // 15.0  — evaluate at a single point
/// p([0, 1, 2, 3])   // [1.0, 6.0, 15.0, 28.0] — vectorized for plotting
/// p.derivative()    // 4x + 3
/// String(describing: p)  // "2x² + 3x + 1"
/// ```
public struct Polynomial: Equatable, Codable, Sendable, CustomStringConvertible {

    /// The polynomial coefficients in ascending order of power.
    ///
    /// Element `i` is the coefficient of `xⁱ`. The array is stored as supplied
    /// to the initializer; trailing zeros are kept (so `Polynomial([1, 0])` is
    /// the constant `1` viewed as a degree-1 polynomial with a zero linear
    /// coefficient). Use ``degree`` for the actual highest non-zero index.
    public let coefficients: [Double]

    /// Creates a polynomial from coefficients in ascending order of power.
    ///
    /// The first element is the constant term, the second is the linear term,
    /// and so on. An empty array is treated as the zero polynomial — equivalent
    /// to `Polynomial([0])`. The coefficient array is stored as supplied; this
    /// initializer does not trim trailing zeros, since callers may want to
    /// preserve a specific degree representation.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let constant = Polynomial([5])           // 5
    /// let linear = Polynomial([2, 3])          // 3x + 2
    /// let quadratic = Polynomial([1, 3, 2])    // 2x² + 3x + 1
    /// ```
    ///
    /// - Parameter coefficients: Coefficients indexed by power, with `coefficients[0]`
    ///   being the constant term. An empty array becomes the zero polynomial.
    public init(_ coefficients: [Double]) {
        if coefficients.isEmpty {
            self.coefficients = [0]
        } else {
            self.coefficients = coefficients
        }
    }

    /// The degree of the polynomial — the highest power with a non-zero coefficient.
    ///
    /// Returns `0` for any constant polynomial, including the zero polynomial.
    /// Trailing zeros in ``coefficients`` are ignored when computing the degree,
    /// so `Polynomial([1, 2, 0])` reports a degree of `1`, not `2`.
    public var degree: Int {
        for i in stride(from: coefficients.count - 1, through: 0, by: -1) {
            if coefficients[i] != 0 {
                return i
            }
        }
        return 0
    }

    /// Evaluates the polynomial at a single point using Horner's method.
    ///
    /// Horner's method evaluates `a₀ + a₁x + a₂x² + ... + aₙxⁿ` by rewriting
    /// the expression as `a₀ + x·(a₁ + x·(a₂ + ... + x·aₙ))`. This costs `n`
    /// multiplications and `n` additions, and avoids the precision loss of
    /// repeatedly computing `pow(x, k)`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 3, 2])  // 2x² + 3x + 1
    /// p(2)   // 15.0
    /// p(-1)  // 0.0
    /// ```
    ///
    /// - Parameter x: The point at which to evaluate.
    /// - Returns: The polynomial value at `x`.
    public func callAsFunction(_ x: Double) -> Double {
        // Horner's method: evaluate from the highest-degree coefficient down.
        var result = 0.0
        for i in stride(from: coefficients.count - 1, through: 0, by: -1) {
            result = result * x + coefficients[i]
        }
        return result
    }

    /// Evaluates the polynomial at each point in an array.
    ///
    /// Maps the scalar evaluation across the input. Useful for plotting — pass
    /// a dense grid of `x` values and chart the result against them.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 3, 2])  // 2x² + 3x + 1
    /// p([0, 1, 2, 3])  // [1.0, 6.0, 15.0, 28.0]
    /// ```
    ///
    /// - Parameter xs: The points at which to evaluate.
    /// - Returns: An array of polynomial values, one per input.
    public func callAsFunction(_ xs: [Double]) -> [Double] {
        return xs.map { self($0) }
    }

    /// Returns the derivative polynomial.
    ///
    /// Differentiation follows the **power rule** applied term by term: the
    /// derivative of `aᵢ · xⁱ` is `i · aᵢ · xⁱ⁻¹`. The constant term `a₀`
    /// vanishes because its derivative is zero, and every higher-degree term
    /// is multiplied by its exponent and dropped one power. For a polynomial
    /// stored as `a₀ + a₁x + a₂x² + ... + aₙxⁿ`, the result is
    /// `a₁ + 2a₂x + 3a₃x² + ... + n·aₙxⁿ⁻¹` — the storage shifts down by one
    /// index and each coefficient is scaled by its original index.
    ///
    /// A constant polynomial differentiates to the zero polynomial. The
    /// returned value is canonical: trailing zeros that would arise from a
    /// `0` leading coefficient on the original polynomial are trimmed, so
    /// the degree of the derivative is exactly one less than the degree of
    /// the input whenever the input has degree ≥ 1.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // 2x² + 3x + 1 → derivative is 4x + 3
    /// let p = Polynomial([1, 3, 2])
    /// p.derivative().asExpression()    // "4x + 3"
    ///
    /// // A constant polynomial has a zero derivative.
    /// Polynomial([-7]).derivative()    // 0
    ///
    /// // Sparse polynomials work the same way — interior zeros stay zero.
    /// // -2x³ + 5 → derivative is -6x²
    /// Polynomial([5, 0, 0, -2]).derivative().asExpression()   // "-6x²"
    ///
    /// // x⁴ - 3x² + 1 → derivative is 4x³ - 6x
    /// Polynomial([1, 0, -3, 0, 1]).derivative().asExpression()  // "4x³ - 6x"
    /// ```
    ///
    /// The chain rule for composing derivatives is not provided directly; for
    /// a `Polynomial` composed with another function, take the derivative
    /// here and combine with the outer derivative manually. See
    /// <doc:Calculus-Primer> for the broader treatment of differentiation in
    /// Quiver, and <doc:Polynomials> for the type's evaluation,
    /// least-squares fitting, and arithmetic surface.
    ///
    /// - Returns: A new `Polynomial` whose coefficients are the
    ///   power-rule-scaled coefficients of `self`, with the constant term
    ///   dropped. The zero polynomial is returned for constants.
    /// - Complexity: O(*n*) where *n* is `coefficients.count`.
    public func derivative() -> Polynomial {
        guard coefficients.count > 1 else {
            return Polynomial([0])
        }
        var derived = [Double]()
        derived.reserveCapacity(coefficients.count - 1)
        for i in 1..<coefficients.count {
            derived.append(Double(i) * coefficients[i])
        }
        return Polynomial(derived)
    }

    /// Returns a polynomial with trailing zero coefficients removed.
    ///
    /// Two polynomials that represent the same function may have different
    /// coefficient arrays — the initializer accepts the array as given, so
    /// `Polynomial([1, 2])` and `Polynomial([1, 2, 0])` evaluate identically
    /// but are not equal under the synthesized `==`. Calling `trimmed()` on
    /// both yields polynomials that compare equal. The zero polynomial
    /// canonicalizes to `Polynomial([0])`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 2, 0, 0])
    /// p.trimmed()                       // Polynomial([1, 2])
    /// p.trimmed() == Polynomial([1, 2]) // true
    /// ```
    ///
    /// - Returns: A polynomial whose final coefficient is non-zero (or `[0]`
    ///   for the zero polynomial).
    public func trimmed() -> Polynomial {
        var coeffs = coefficients
        while coeffs.count > 1 && coeffs.last == 0 {
            coeffs.removeLast()
        }
        return Polynomial(coeffs)
    }

    /// Adds two polynomials term by term.
    ///
    /// Pads the shorter coefficient array with zeros to match the longer one,
    /// then adds element-wise. Trailing zeros in the result are trimmed so the
    /// reported ``degree`` reflects the true highest-power term.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 3, 2])     // 2x² + 3x + 1
    /// let q = Polynomial([4, -3, -2])   // -2x² - 3x + 4
    /// p + q                              // 5
    /// ```
    public static func + (lhs: Polynomial, rhs: Polynomial) -> Polynomial {
        let n = max(lhs.coefficients.count, rhs.coefficients.count)
        var sum = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let l = i < lhs.coefficients.count ? lhs.coefficients[i] : 0
            let r = i < rhs.coefficients.count ? rhs.coefficients[i] : 0
            sum[i] = l + r
        }
        return Polynomial(_trimmingTrailingZeros(sum))
    }

    /// Multiplies two polynomials by convolution.
    ///
    /// The product of polynomials of degree `m` and `n` has degree `m + n`.
    /// The coefficient at index `k` of the result is `Σ lhs[i] · rhs[k - i]`
    /// over all valid `i` — the discrete convolution of the two coefficient
    /// arrays.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 1])   // x + 1
    /// let q = Polynomial([-1, 1])  // x - 1
    /// p * q                         // x² - 1
    /// ```
    public static func * (lhs: Polynomial, rhs: Polynomial) -> Polynomial {
        let m = lhs.coefficients.count
        let n = rhs.coefficients.count
        var product = [Double](repeating: 0, count: m + n - 1)
        for i in 0..<m {
            for j in 0..<n {
                product[i + j] += lhs.coefficients[i] * rhs.coefficients[j]
            }
        }
        return Polynomial(_trimmingTrailingZeros(product))
    }

    /// Multiplies every coefficient by a scalar.
    ///
    /// Multiplying by `0` produces the zero polynomial. Multiplying by any
    /// non-zero scalar preserves the degree.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 3, 2])  // 2x² + 3x + 1
    /// 3.0 * p                         // 6x² + 9x + 3
    /// ```
    public static func * (lhs: Double, rhs: Polynomial) -> Polynomial {
        let scaled = rhs.coefficients.map { lhs * $0 }
        return Polynomial(_trimmingTrailingZeros(scaled))
    }

    /// Returns the polynomial as a string in the conventional
    /// highest-power-first form, using Unicode superscript digits for powers
    /// of two or more.
    ///
    /// Sibling to `Double.asFraction(maxDenominator:)`. Both methods convert
    /// a value into a display form: `asFraction` returns rational structure as
    /// a `Fraction`; `asExpression` returns mathematical structure as a
    /// `String`. The returned string is identical to ``description`` — use
    /// this method in code where the rendering is the point, so the call
    /// itself signals intent.
    ///
    /// Coefficients of `0` are omitted, coefficients of `±1` are rendered
    /// without the leading digit (writing `x²` instead of `1x²`), and negative
    /// terms are joined with ` - ` rather than ` + -`. The zero polynomial
    /// renders as `"0"`.
    ///
    /// Fitted polynomials commonly carry numerical-noise coefficients near
    /// machine zero — a degree-3 fit to quadratic data produces a leading
    /// `x³` coefficient of order `1e-17`. The `relativeZeroTolerance`
    /// parameter suppresses such terms by dropping any coefficient whose
    /// magnitude is below `tolerance · max(|aᵢ|)`. The default of `1e-12`
    /// matches the conditioning floor of Quiver's least-squares solver.
    /// Pass `0` to disable suppression and see every coefficient as
    /// computed.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// Polynomial([1, 3, 2]).asExpression()   // "2x² + 3x + 1"
    /// Polynomial([0, -1]).asExpression()     // "-x"
    /// Polynomial([0]).asExpression()         // "0"
    /// ```
    ///
    /// See <doc:Rendering-Math-Primer> for the full family.
    ///
    /// - Parameter relativeZeroTolerance: Coefficients with `|aᵢ| <
    ///   tolerance · max(|aⱼ|)` are treated as zero. Defaults to `1e-12`.
    ///   Pass `0` to render every coefficient.
    /// - Returns: The polynomial as a Unicode mathematical expression.
    public func asExpression(relativeZeroTolerance: Double = 1e-12) -> String {
        // Find the largest-magnitude coefficient. The relative tolerance is
        // scaled against this so that a polynomial whose coefficients all
        // live near `1e-13` doesn't suppress every term.
        let maxMagnitude = coefficients.map { abs($0) }.max() ?? 0
        let threshold = maxMagnitude * relativeZeroTolerance

        // The zero polynomial — every coefficient is below threshold (or
        // there are none) — renders as "0".
        let highest = degree
        if maxMagnitude == 0 || (highest == 0 && abs(coefficients[0]) <= threshold) {
            return "0"
        }

        var pieces: [String] = []
        for i in stride(from: highest, through: 0, by: -1) {
            let c = coefficients[i]
            if abs(c) <= threshold { continue }

            let isFirstPiece = pieces.isEmpty
            let sign: String
            let magnitude = abs(c)
            if isFirstPiece {
                sign = c < 0 ? "-" : ""
            } else {
                sign = c < 0 ? " - " : " + "
            }

            let coefficientText: String
            if i >= 1 && magnitude == 1 {
                // Coefficients of ±1 on x or higher powers: drop the "1".
                coefficientText = ""
            } else {
                coefficientText = UnicodeMath.formatCell(magnitude)
            }

            let powerText: String
            switch i {
            case 0:
                powerText = ""
            case 1:
                powerText = "x"
            default:
                powerText = "x" + UnicodeMath.superscript(i)
            }

            pieces.append(sign + coefficientText + powerText)
        }

        // If the threshold suppressed every term, fall back to "0".
        return pieces.isEmpty ? "0" : pieces.joined()
    }

    /// A human-readable rendering of the polynomial in the conventional
    /// highest-power-first form.
    ///
    /// Identical to ``asExpression(relativeZeroTolerance:)``. Use that method
    /// directly in code where the rendering is the point; this property exists so
    /// `String(describing:)`, `print(_:)`, and string interpolation produce
    /// the same output without an explicit method call.
    public var description: String {
        asExpression()
    }

    // MARK: - Internal helpers

    /// Removes trailing zeros from a coefficient array, preserving at least
    /// one element so the result is always a valid polynomial representation.
    internal static func _trimmingTrailingZeros(_ values: [Double]) -> [Double] {
        var trimmed = values
        while trimmed.count > 1, trimmed.last == 0 {
            trimmed.removeLast()
        }
        return trimmed
    }

}
