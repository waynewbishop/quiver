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

// MARK: - Fraction Type

/// A rational number representation for educational display of numeric results.
///
/// `Fraction` converts floating-point values into their simplest rational form,
/// making mathematical relationships visible. For example, a matrix inverse with
/// determinant 13 produces elements like `5/13` and `-1/13` — the shared denominator
/// immediately connects back to the determinant.
///
/// This type is presentation-only. All Quiver operations continue to return standard
/// numeric types. Use `.asFraction()` on any `Double` or `.asFractions()` on any
/// `[Double]` or `[[Double]]` to see the rational representation.
///
/// ```swift
/// let inverse = try matrix.inverted()
/// let display = inverse.asFractions()
/// print(display[0][0])        // "5/13"
/// print(display[0][0].value)  // 0.384615...
/// ```
public struct Fraction: CustomStringConvertible, Equatable {

    /// The numerator of the fraction
    public let numerator: Int

    /// The denominator of the fraction (always positive)
    public let denominator: Int

    /// The floating-point value of this fraction
    public var value: Double {
        Double(numerator) / Double(denominator)
    }

    /// Displays the fraction as "a/b", or just "a" for whole numbers
    public var description: String {
        if denominator == 1 {
            return "\(numerator)"
        }
        return "\(numerator)/\(denominator)"
    }

    /// Creates a fraction from an explicit numerator and denominator
    public init(numerator: Int, denominator: Int) {
        precondition(denominator != 0, "Denominator cannot be zero")
        let sign = (denominator < 0) ? -1 : 1
        let g = Fraction.gcd(abs(numerator), abs(denominator))
        self.numerator = sign * numerator / g
        self.denominator = sign * denominator / g
    }

    /// Converts a Double to its simplest fractional representation
    /// using continued fraction expansion.
    ///
    /// - Parameters:
    ///   - value: The floating-point value to convert
    ///   - maxDenominator: The largest denominator to consider (default: 1000)
    ///   - tolerance: How close the fraction must be to the original value (default: 1e-9)
    public init(_ value: Double, maxDenominator: Int = 1000, tolerance: Double = 1e-9) {
        // Handle zero
        guard value != 0.0 else {
            self.numerator = 0
            self.denominator = 1
            return
        }

        // Handle negative values
        let sign = value < 0 ? -1 : 1
        let absValue = abs(value)

        // Handle whole numbers
        let rounded = absValue.rounded()
        if abs(absValue - rounded) < tolerance {
            self.numerator = sign * Int(rounded)
            self.denominator = 1
            return
        }

        // Continued fraction / Stern-Brocot convergence
        var bestNumerator = Int(rounded)
        var bestDenominator = 1
        var bestError = abs(absValue - rounded)

        // Mediant-based search through the Stern-Brocot tree
        var lowerN = Int(absValue)     // floor
        var lowerD = 1
        var upperN = lowerN + 1        // ceil
        var upperD = 1

        while lowerD + upperD <= maxDenominator {
            let (medN, nOverflow) = lowerN.addingReportingOverflow(upperN)
            let (medD, dOverflow) = lowerD.addingReportingOverflow(upperD)
            if nOverflow || dOverflow { break }
            let medValue = Double(medN) / Double(medD)
            let error = abs(absValue - medValue)

            if error < bestError {
                bestNumerator = medN
                bestDenominator = medD
                bestError = error
            }

            if bestError < tolerance {
                break
            }

            if medValue < absValue {
                lowerN = medN
                lowerD = medD
            } else {
                upperN = medN
                upperD = medD
            }
        }

        self.numerator = sign * bestNumerator
        self.denominator = bestDenominator
    }

    /// Computes the greatest common divisor of two integers
    private static func gcd(_ a: Int, _ b: Int) -> Int {
        b == 0 ? a : gcd(b, a % b)
    }
}

// MARK: - Double Conversion

public extension Double {

    /// Converts this value to its simplest fractional representation.
    ///
    /// ```swift
    /// let x = 0.384615384615
    /// x.asFraction()              // 5/13
    /// x.asFraction().numerator    // 5
    /// x.asFraction().denominator  // 13
    /// ```
    ///
    /// - Parameter maxDenominator: The largest denominator to consider (default: 1000)
    /// - Returns: A `Fraction` representing this value
    func asFraction(maxDenominator: Int = 1000) -> Fraction {
        Fraction(self, maxDenominator: maxDenominator)
    }
}

// MARK: - Vector Conversion

public extension Array where Element == Double {

    /// Converts each element to its simplest fractional representation.
    ///
    /// ```swift
    /// let normalized = [0.6, 0.8]
    /// normalized.asFractions()  // [3/5, 4/5]
    /// ```
    ///
    /// - Parameter maxDenominator: The largest denominator to consider (default: 1000)
    /// - Returns: An array of `Fraction` values
    func asFractions(maxDenominator: Int = 1000) -> [Fraction] {
        map { Fraction($0, maxDenominator: maxDenominator) }
    }
}

// MARK: - Matrix Conversion

public extension Array where Element == [Double] {

    /// Converts each element to its simplest fractional representation.
    ///
    /// ```swift
    /// let inverse = try matrix.inverted()
    /// let display = inverse.asFractions()
    /// // [[5/13, -1/13], [-2/13, 3/13]]
    /// ```
    ///
    /// - Parameter maxDenominator: The largest denominator to consider (default: 1000)
    /// - Returns: A 2D array of `Fraction` values
    func asFractions(maxDenominator: Int = 1000) -> [[Fraction]] {
        map { $0.asFractions(maxDenominator: maxDenominator) }
    }
}
