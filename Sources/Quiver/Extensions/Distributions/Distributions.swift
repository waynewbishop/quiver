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

// MARK: - Distributions

/// Probability distributions exposed as a stateless namespace.
///
/// `Distributions` groups the probability density, cumulative density, and quantile
/// functions for named distributions. Each nested type is a case-less enum acting
/// as its own namespace — for example, `Distributions.normal`. Callers reach the
/// functions by their distribution name, keeping every call site self-documenting.
///
/// The functions are stateless. A typical call passes the distribution parameters
/// directly: `Distributions.normal.cdf(x: 1.96, mean: 0, std: 1)`. There is no
/// "fitted distribution" object to construct, no shared state to mis-configure,
/// and every call site is self-documenting.
public enum Distributions: Sendable {

    /// Normal distribution functions.
    ///
    /// Provides probability density (``pdf(x:mean:std:)``), log-density
    /// (``logPDF(x:mean:std:)``), cumulative density (``cdf(x:mean:std:)``),
    /// and quantile (``quantile(p:mean:std:)``) for the normal distribution
    /// with the given mean and standard deviation.
    ///
    /// All functions return `Double?` and produce `nil` when `std <= 0`, when
    /// the inputs are otherwise out of domain, or when the computation would
    /// produce a non-finite result (`NaN` or `±infinity` from underflow at very
    /// small `std`, etc.). Following Quiver's existing pattern (``Swift/Array/mean()``,
    /// etc.), out-of-domain input maps to `nil` rather than a runtime trap or
    /// silently propagating `NaN`.
    public enum normal: Sendable {

        /// Returns the probability density of the normal distribution at `x`.
        ///
        /// The probability density function (PDF) gives the relative likelihood
        /// of observing the value `x` under a normal distribution with mean `mean`
        /// and standard deviation `std`. The PDF is positive everywhere on the real
        /// line, peaks at the mean, and integrates to 1.0 over its full support.
        ///
        /// For numerically demanding work — multiplying many densities together,
        /// or evaluating a density far in the tail — prefer ``logPDF(x:mean:std:)``,
        /// which returns the natural log of the same quantity and avoids underflow.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // Density at the mean of a standard normal
        /// let peak = Distributions.normal.pdf(x: 0, mean: 0, std: 1)  // ≈ 0.3989
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the density.
        ///   - mean: The distribution mean (μ).
        ///   - std: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The probability density at `x`, or `nil` if `std <= 0`.
        public static func pdf(x: Double, mean: Double, std: Double) -> Double? {
            guard let logValue = logPDF(x: x, mean: mean, std: std) else { return nil }
            let value = Foundation.exp(logValue)
            return value.isFinite ? value : nil
        }

        /// Returns the natural log of the normal probability density at `x`.
        ///
        /// Working in log-space is the standard tactic for numerical work that
        /// combines many density values — the products become sums, and densities
        /// far in the tail (which round to zero in linear space) stay representable.
        /// `GaussianNaiveBayes` uses this internally; it is exposed publicly so
        /// other downstream callers can use the same well-tested implementation.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // log-density 4σ from the mean
        /// let lp = Distributions.normal.logPDF(x: 4, mean: 0, std: 1)  // ≈ -8.919
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the log-density.
        ///   - mean: The distribution mean (μ).
        ///   - std: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The natural log of the density at `x`, or `nil` if `std <= 0`.
        public static func logPDF(x: Double, mean: Double, std: Double) -> Double? {
            guard std > 0 else { return nil }
            let variance = std * std
            let diff = x - mean
            let value = -0.5 * (Foundation.log(2.0 * .pi) + Foundation.log(variance) + (diff * diff) / variance)
            return value.isFinite ? value : nil
        }

        /// Returns the cumulative probability `P(X <= x)` under the normal distribution.
        ///
        /// The cumulative distribution function (CDF) gives the probability that a
        /// normally distributed random variable falls at or below `x`. It rises
        /// monotonically from 0 at negative infinity to 1 at positive infinity,
        /// and equals 0.5 at the mean. Computed via the error function `erf`.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // P(Z <= 1.96) for a standard normal
        /// let p = Distributions.normal.cdf(x: 1.96, mean: 0, std: 1)  // ≈ 0.975
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the CDF.
        ///   - mean: The distribution mean (μ).
        ///   - std: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The cumulative probability at `x`, or `nil` if `std <= 0`.
        public static func cdf(x: Double, mean: Double, std: Double) -> Double? {
            guard std > 0 else { return nil }
            let z = (x - mean) / std
            let value = 0.5 * (1.0 + Foundation.erf(z / Foundation.sqrt(2.0)))
            return value.isFinite ? value : nil
        }

        /// Returns the quantile (inverse CDF) of the normal distribution at probability `p`.
        ///
        /// The quantile function answers the question "what value of `x` puts probability
        /// `p` below it?" It is the inverse of ``cdf(x:mean:std:)``. For example, the
        /// quantile at `p = 0.975` of a standard normal is approximately `1.96` — the
        /// critical value used to build a 95% confidence interval.
        ///
        /// Implemented via the Beasley-Springer-Moro rational approximation, which
        /// uses a central polynomial in `0.08 <= p <= 0.92` and a tail formula
        /// elsewhere. Accuracy is roughly 7 decimals in the body of the distribution
        /// and 4 decimals deep in the tails — the documented limit of BSM.
        ///
        /// Out-of-domain input maps to `nil`: `p <= 0`, `p >= 1`, or `std <= 0`.
        /// This includes `p = 1e-15` and `p = 1 - 1e-15` — values past the BSM
        /// representable range.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // 95% critical value for a standard normal
        /// let z = Distributions.normal.quantile(p: 0.975, mean: 0, std: 1)  // ≈ 1.96
        /// ```
        ///
        /// - Parameters:
        ///   - p: The cumulative probability, in `(0, 1)`.
        ///   - mean: The distribution mean (μ).
        ///   - std: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The quantile at `p`, or `nil` if `p` is outside `(0, 1)`,
        ///   `p <= 1e-15` or `p >= 1 - 1e-15`, or `std <= 0`.
        public static func quantile(p: Double, mean: Double, std: Double) -> Double? {
            guard std > 0 else { return nil }
            guard p > 0.0 && p < 1.0 else { return nil }
            // BSM is not numerically stable at the extreme tails; treat values
            // closer than 1e-15 to 0 or 1 as out-of-domain.
            guard p > 1e-15 && p < 1.0 - 1e-15 else { return nil }
            guard let z = _bsmStandardQuantile(p: p) else { return nil }
            let value = mean + std * z
            return value.isFinite ? value : nil
        }

        // MARK: - Internal helpers

        /// Beasley-Springer-Moro standard-normal quantile.
        ///
        /// Returns the inverse standard-normal CDF for `p` in `(0, 1)`. Uses a
        /// rational approximation in the central region (`0.08 <= p <= 0.92`)
        /// and a tail formula elsewhere. The constants are the original BSM
        /// (1991) coefficients, accurate to roughly 7 decimals in the body
        /// and 4 decimals in the tails.
        internal static func _bsmStandardQuantile(p: Double) -> Double? {
            // Central-region coefficients (Beasley-Springer 1977).
            let a: [Double] = [
                -3.969683028665376e+01,
                 2.209460984245205e+02,
                -2.759285104469687e+02,
                 1.383577518672690e+02,
                -3.066479806614716e+01,
                 2.506628277459239e+00
            ]
            let b: [Double] = [
                -5.447609879822406e+01,
                 1.615858368580409e+02,
                -1.556989798598866e+02,
                 6.680131188771972e+01,
                -1.328068155288572e+01
            ]
            // Tail-region coefficients (Moro 1995).
            let c: [Double] = [
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e+00,
                -2.549732539343734e+00,
                 4.374664141464968e+00,
                 2.938163982698783e+00
            ]
            let d: [Double] = [
                 7.784695709041462e-03,
                 3.224671290700398e-01,
                 2.445134137142996e+00,
                 3.754408661907416e+00
            ]

            let pLow = 0.02425
            let pHigh = 1.0 - pLow

            if p < pLow {
                // Lower tail.
                let q = Foundation.sqrt(-2.0 * Foundation.log(p))
                let num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
                let den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
                return num / den
            } else if p <= pHigh {
                // Central region — rational approximation.
                let q = p - 0.5
                let r = q * q
                let num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
                let den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
                return num / den
            } else {
                // Upper tail.
                let q = Foundation.sqrt(-2.0 * Foundation.log(1.0 - p))
                let num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
                let den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
                return -num / den
            }
        }
    }

}
