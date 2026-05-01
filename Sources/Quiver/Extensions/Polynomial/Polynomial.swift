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
    /// For `a₀ + a₁x + a₂x² + ... + aₙxⁿ`, the derivative is
    /// `a₁ + 2a₂x + 3a₃x² + ... + n·aₙxⁿ⁻¹`. A constant polynomial differentiates
    /// to the zero polynomial.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let p = Polynomial([1, 3, 2])  // 2x² + 3x + 1
    /// p.derivative()                  // 4x + 3
    /// ```
    ///
    /// - Returns: The derivative polynomial.
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

    /// A human-readable rendering of the polynomial in the conventional
    /// highest-power-first form.
    ///
    /// Coefficients of `0` are omitted, coefficients of `1` and `-1` are
    /// rendered without the leading digit (writing `x²` instead of `1x²`),
    /// and negative terms are joined with ` - ` rather than ` + -`. Powers
    /// of two or more use Unicode superscript digits. The zero polynomial
    /// renders as `"0"`.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// String(describing: Polynomial([1, 3, 2]))   // "2x² + 3x + 1"
    /// String(describing: Polynomial([0, -1]))     // "-x"
    /// String(describing: Polynomial([0]))         // "0"
    /// ```
    public var description: String {
        // Find the highest-degree non-zero coefficient. If none exists, we are
        // the zero polynomial.
        let highest = degree
        if highest == 0 && coefficients[0] == 0 {
            return "0"
        }

        var pieces: [String] = []
        for i in stride(from: highest, through: 0, by: -1) {
            let c = coefficients[i]
            if c == 0 { continue }

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
                coefficientText = Polynomial._formatCoefficient(magnitude)
            }

            let powerText: String
            switch i {
            case 0:
                powerText = ""
            case 1:
                powerText = "x"
            default:
                powerText = "x" + Polynomial._superscript(i)
            }

            pieces.append(sign + coefficientText + powerText)
        }

        return pieces.joined()
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

    /// Renders a non-negative coefficient as a string, omitting unnecessary
    /// decimal noise on whole values (so `2.0` becomes `"2"`).
    internal static func _formatCoefficient(_ value: Double) -> String {
        if value == value.rounded() && value.isFinite {
            return String(Int(value))
        }
        return String(value)
    }

    /// Returns the Unicode superscript representation of a non-negative
    /// integer power. Multi-digit powers concatenate the per-digit superscript
    /// characters — `10` becomes `"¹⁰"`.
    internal static func _superscript(_ power: Int) -> String {
        let map: [Character: Character] = [
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
        ]
        let digits = String(power)
        var result = ""
        for ch in digits {
            if let s = map[ch] {
                result.append(s)
            }
        }
        return result
    }
}
