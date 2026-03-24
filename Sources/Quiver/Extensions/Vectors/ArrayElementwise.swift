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

// MARK: - Element-wise Operations for Double Arrays

public extension Array where Element == Double {
    /// Raises each element to the specified power.
    ///
    /// - Parameter exponent: The power to raise each element to (supports fractional and negative values)
    /// - Returns: A new array where each element is raised to the given exponent
    func power(_ exponent: Double) -> [Double] {
        return self.map { Foundation.pow($0, exponent) }
    }

    /// Returns the sine of each element in the array.
    ///
    /// - Returns: A new array of sine values, with inputs interpreted as radians
    func sin() -> [Double] {
        return self.map { Foundation.sin($0) }
    }

    /// Returns the cosine of each element in the array.
    ///
    /// - Returns: A new array of cosine values, with inputs interpreted as radians
    func cos() -> [Double] {
        return self.map { Foundation.cos($0) }
    }

    /// Returns the tangent of each element in the array.
    ///
    /// - Returns: A new array of tangent values, with inputs interpreted as radians
    func tan() -> [Double] {
        return self.map { Foundation.tan($0) }
    }

    /// Returns the floor of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded down to the nearest integer
    func floor() -> [Double] {
        return self.map { Foundation.floor($0) }
    }

    /// Returns the ceiling of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded up to the nearest integer
    func ceil() -> [Double] {
        return self.map { Foundation.ceil($0) }
    }

    /// Returns the rounded value of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded to the nearest integer
    func round() -> [Double] {
        return self.map { Foundation.round($0) }
    }

    /// Returns the natural logarithm of each element in the array.
    ///
    /// - Returns: A new array of natural log values (base e)
    func log() -> [Double] {
        return self.map { Foundation.log($0) }
    }

    /// Returns the base-10 logarithm of each element in the array.
    ///
    /// - Returns: A new array of base-10 log values
    func log10() -> [Double] {
        return self.map { Foundation.log10($0) }
    }

    /// Returns e raised to the power of each element in the array.
    ///
    /// - Returns: A new array where each element is the exponential e^x
    func exp() -> [Double] {
        return self.map { Foundation.exp($0) }
    }

    /// Returns the square root of each element in the array.
    ///
    /// - Returns: A new array of square root values
    func sqrt() -> [Double] {
        return self.map { Foundation.sqrt($0) }
    }

    /// Returns the square of each element in the array.
    ///
    /// - Returns: A new array where each element is multiplied by itself
    func square() -> [Double] {
        return self.map { $0 * $0 }
    }

    /// Converts raw scores into a probability distribution that sums to 1.0.
    ///
    /// Softmax applies exp(xᵢ) / Σexp(xⱼ) to each element, turning arbitrary
    /// real-valued scores into values between 0 and 1 that sum to 1.0. This is
    /// the standard output layer for multi-class classification and the core of
    /// attention mechanisms in transformers.
    ///
    /// The name "softmax" is a historical misnomer — it was introduced by John
    /// Bridle in 1990 as a differentiable approximation of argmax, not max. The
    /// function is more accurately described as a "normalized exponential," but
    /// "softmax" is universal across ML literature, frameworks, and textbooks.
    ///
    /// Uses the numerically stable variant: subtracts the maximum value before
    /// exponentiation to prevent overflow when scores are large.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let logits = [2.0, 1.0, 0.1]
    /// let probs = logits.softMax()
    /// // [0.659, 0.242, 0.099] — sums to 1.0
    /// ```
    ///
    /// - Returns: A probability distribution with the same length as the input.
    func softMax() -> [Double] {
        guard let maxVal = self.max() else { return [] }
        let shifted = self.map { Foundation.exp($0 - maxVal) }
        let total = shifted.reduce(0.0, +)
        return shifted.map { $0 / total }
    }

    /// Applies the sigmoid function to each element: σ(x) = 1 / (1 + e⁻ˣ).
    ///
    /// Sigmoid squashes each value into the range (0, 1), making it the standard
    /// activation function for binary classification. Large positive values map
    /// near 1.0, large negative values map near 0.0, and 0 maps to exactly 0.5.
    ///
    /// While ``softMax()->[Double]`` converts a vector of scores into a multi-class
    /// probability distribution, sigmoid operates element-wise — each output depends
    /// only on its own input. This makes sigmoid the right choice for binary
    /// classification (one output) and multi-label classification (independent outputs),
    /// while softmax is for multi-class (mutually exclusive outputs).
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let logits = [-2.0, 0.0, 2.0, 5.0]
    /// let probs = logits.sigmoid()
    /// // [0.119, 0.5, 0.881, 0.993]
    /// ```
    ///
    /// - Returns: An array of values in (0, 1), one per input element.
    func sigmoid() -> [Double] {
        return self.map { 1.0 / (1.0 + Foundation.exp(-$0)) }
    }
}

// MARK: - Element-wise Operations for Float Arrays

public extension Array where Element == Float {
    /// Raises each element to the specified power.
    ///
    /// - Parameter exponent: The power to raise each element to (supports fractional and negative values)
    /// - Returns: A new array where each element is raised to the given exponent
    func power(_ exponent: Float) -> [Float] {
        return self.map { Foundation.powf($0, exponent) }
    }

    /// Returns the sine of each element in the array.
    ///
    /// - Returns: A new array of sine values, with inputs interpreted as radians
    func sin() -> [Float] {
        return self.map { Foundation.sinf($0) }
    }

    /// Returns the cosine of each element in the array.
    ///
    /// - Returns: A new array of cosine values, with inputs interpreted as radians
    func cos() -> [Float] {
        return self.map { Foundation.cosf($0) }
    }

    /// Returns the tangent of each element in the array.
    ///
    /// - Returns: A new array of tangent values, with inputs interpreted as radians
    func tan() -> [Float] {
        return self.map { Foundation.tanf($0) }
    }

    /// Returns the floor of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded down to the nearest integer
    func floor() -> [Float] {
        return self.map { Foundation.floorf($0) }
    }

    /// Returns the ceiling of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded up to the nearest integer
    func ceil() -> [Float] {
        return self.map { Foundation.ceilf($0) }
    }

    /// Returns the rounded value of each element in the array.
    ///
    /// - Returns: A new array where each element is rounded to the nearest integer
    func round() -> [Float] {
        return self.map { Foundation.roundf($0) }
    }

    /// Returns the natural logarithm of each element in the array.
    ///
    /// - Returns: A new array of natural log values (base e)
    func log() -> [Float] {
        return self.map { Foundation.logf($0) }
    }

    /// Returns the base-10 logarithm of each element in the array.
    ///
    /// - Returns: A new array of base-10 log values
    func log10() -> [Float] {
        return self.map { Foundation.log10f($0) }
    }

    /// Returns e raised to the power of each element in the array.
    ///
    /// - Returns: A new array where each element is the exponential e^x
    func exp() -> [Float] {
        return self.map { Foundation.expf($0) }
    }

    /// Returns the square root of each element in the array.
    ///
    /// - Returns: A new array of square root values
    func sqrt() -> [Float] {
        return self.map { Foundation.sqrtf($0) }
    }

    /// Returns the square of each element in the array.
    ///
    /// - Returns: A new array where each element is multiplied by itself
    func square() -> [Float] {
        return self.map { $0 * $0 }
    }

    /// Converts raw scores into a probability distribution that sums to 1.0.
    ///
    /// Float version of ``Swift/Array/softMax()->[Double]``. See the Double version
    /// for full documentation and usage examples.
    ///
    /// - Returns: A probability distribution with the same length as the input.
    func softMax() -> [Float] {
        guard let maxVal = self.max() else { return [] }
        let shifted = self.map { Foundation.expf($0 - maxVal) }
        let total = shifted.reduce(0.0, +)
        return shifted.map { $0 / total }
    }

    /// Applies the sigmoid function to each element: σ(x) = 1 / (1 + e⁻ˣ).
    ///
    /// Float version of ``Swift/Array/sigmoid()->[Double]``. See the Double version
    /// for full documentation and usage examples.
    ///
    /// - Returns: An array of values in (0, 1), one per input element.
    func sigmoid() -> [Float] {
        return self.map { 1.0 / (1.0 + Foundation.expf(-$0)) }
    }
}
