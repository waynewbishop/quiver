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
/// directly: `Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)`. There is no
/// "fitted distribution" object to construct, no shared state to mis-configure,
/// and every call site is self-documenting.
public enum Distributions: Sendable {

    /// Normal distribution functions.
    ///
    /// Provides probability density (``pdf(x:mean:standardDeviation:)``), log-density
    /// (``logPDF(x:mean:standardDeviation:)``), cumulative density (``cdf(x:mean:standardDeviation:)``),
    /// and quantile (``quantile(p:mean:standardDeviation:)``) for the normal distribution
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
        /// or evaluating a density far in the tail — prefer ``logPDF(x:mean:standardDeviation:)``,
        /// which returns the natural log of the same quantity and avoids underflow.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // Density at the mean of a standard normal
        /// let peak = Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)  // ≈ 0.3989
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the density.
        ///   - mean: The distribution mean (μ).
        ///   - standardDeviation: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The probability density at `x`, or `nil` if `standardDeviation <= 0`.
        public static func pdf(x: Double, mean: Double, standardDeviation: Double) -> Double? {
            guard let logValue = logPDF(x: x, mean: mean, standardDeviation: standardDeviation) else { return nil }
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
        /// let lp = Distributions.normal.logPDF(x: 4, mean: 0, standardDeviation: 1)  // ≈ -8.919
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the log-density.
        ///   - mean: The distribution mean (μ).
        ///   - standardDeviation: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The natural log of the density at `x`, or `nil` if `standardDeviation <= 0`.
        public static func logPDF(x: Double, mean: Double, standardDeviation: Double) -> Double? {
            guard standardDeviation > 0 else { return nil }
            let variance = standardDeviation * standardDeviation
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
        /// let p = Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)  // ≈ 0.975
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the CDF.
        ///   - mean: The distribution mean (μ).
        ///   - standardDeviation: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The cumulative probability at `x`, or `nil` if `standardDeviation <= 0`.
        public static func cdf(x: Double, mean: Double, standardDeviation: Double) -> Double? {
            guard standardDeviation > 0 else { return nil }
            let z = (x - mean) / standardDeviation
            let value = 0.5 * (1.0 + Foundation.erf(z / Foundation.sqrt(2.0)))
            return value.isFinite ? value : nil
        }

        /// Returns the quantile (inverse CDF) of the normal distribution at probability `p`.
        ///
        /// The quantile function answers the question "what value of `x` puts probability
        /// `p` below it?" It is the inverse of ``cdf(x:mean:standardDeviation:)``. For example, the
        /// quantile at `p = 0.975` of a standard normal is approximately `1.96` — the
        /// critical value used to build a 95% confidence interval.
        ///
        /// Implemented via the Beasley-Springer-Moro rational approximation, which
        /// uses a central polynomial in `0.08 <= p <= 0.92` and a tail formula
        /// elsewhere. Accuracy is roughly 7 decimals in the body of the distribution
        /// and 4 decimals deep in the tails — the documented limit of BSM.
        ///
        /// Out-of-domain input maps to `nil`: `p <= 0`, `p >= 1`, or `standardDeviation <= 0`.
        /// This includes `p = 1e-15` and `p = 1 - 1e-15` — values past the BSM
        /// representable range.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // 95% critical value for a standard normal
        /// let z = Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1)  // ≈ 1.96
        /// ```
        ///
        /// - Parameters:
        ///   - p: The cumulative probability, in `(0, 1)`.
        ///   - mean: The distribution mean (μ).
        ///   - standardDeviation: The distribution standard deviation (σ). Must be positive.
        /// - Returns: The quantile at `p`, or `nil` if `p` is outside `(0, 1)`,
        ///   `p <= 1e-15` or `p >= 1 - 1e-15`, or `standardDeviation <= 0`.
        public static func quantile(p: Double, mean: Double, standardDeviation: Double) -> Double? {
            guard standardDeviation > 0 else { return nil }
            guard p > 0.0 && p < 1.0 else { return nil }
            // BSM is not numerically stable at the extreme tails; treat values
            // closer than 1e-15 to 0 or 1 as out-of-domain.
            guard p > 1e-15 && p < 1.0 - 1e-15 else { return nil }
            guard let z = _bsmStandardQuantile(p: p) else { return nil }
            let value = mean + standardDeviation * z
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

    /// Student's t-distribution functions.
    ///
    /// Provides cumulative density (``cdf(x:df:)``) and quantile
    /// (``quantile(p:df:)``) for the t-distribution with `df` degrees of
    /// freedom. The mean is zero; the spread depends on `df`. As `df → ∞`
    /// the t-distribution converges to the standard normal, and that
    /// limit is verified directly in the test suite.
    ///
    /// All functions return `Double?` and produce `nil` when the inputs
    /// are out of domain (`df <= 0`, `p` outside `(0, 1)`) or when the
    /// computation would otherwise produce a non-finite result. Following
    /// the convention established by ``Distributions/normal``, out-of-domain
    /// input maps to `nil` rather than a runtime trap.
    public enum t: Sendable {

        /// Returns the cumulative probability `P(T <= x)` under the t-distribution with `df` degrees of freedom.
        ///
        /// The cumulative distribution function (CDF) gives the probability that a
        /// t-distributed random variable falls at or below `x`. Computed via the
        /// regularized incomplete beta function `I_x(a, b)` — series form for the
        /// rapidly-converging branch and Lentz's continued fraction otherwise — with
        /// the standard transition at `x = (a + 1) / (a + b + 2)`. The route gives
        /// the symmetry invariant `tCDF(-x, df) + tCDF(x, df) = 1` to machine precision.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // P(T <= 1.96) for a t with df = 30 — close to but not equal to the normal value
        /// let p = Distributions.t.cdf(x: 1.96, df: 30)  // ≈ 0.9703
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the CDF.
        ///   - df: The degrees of freedom. Must be positive.
        /// - Returns: The cumulative probability at `x`, or `nil` if `df <= 0`
        ///   or the computation produces a non-finite result.
        public static func cdf(x: Double, df: Double) -> Double? {
            guard df > 0 else { return nil }
            // The two textbook identities — pick the branch that keeps `xBeta` away from 1.
            // Closed form for the special points avoids dividing by zero in the limit.
            if x == 0 {
                return 0.5
            }
            let dfPlusXSquared = df + x * x
            guard dfPlusXSquared.isFinite else {
                // x is so large that df + x*x overflowed — the answer is the asymptotic limit.
                return x > 0 ? 1.0 : 0.0
            }
            let xBeta = df / dfPlusXSquared
            guard let ibeta = _regularizedIncompleteBeta(x: xBeta, a: 0.5 * df, b: 0.5) else {
                return nil
            }
            let value: Double
            if x > 0 {
                value = 1.0 - 0.5 * ibeta
            } else {
                value = 0.5 * ibeta
            }
            // Clamp to [0, 1] in case of tiny floating-point drift.
            let clamped = Swift.max(0.0, Swift.min(1.0, value))
            return clamped.isFinite ? clamped : nil
        }

        /// Returns the quantile (inverse CDF) of the t-distribution at probability `p`.
        ///
        /// The quantile function answers "what value of `x` puts probability `p` below
        /// it under a t-distribution with `df` degrees of freedom?" For example, the
        /// quantile at `p = 0.975` with `df = 30` is approximately `2.042` — the
        /// critical value used to build a 95% confidence interval from a sample of size 31.
        ///
        /// Implemented via bisection on ``cdf(x:df:)`` with an initial bracket guided
        /// by the normal approximation. Bisection is unconditionally stable on the
        /// monotonic CDF; convergence to roughly 1e-10 takes at most ~50 iterations.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // 95% one-tailed critical value with df = 10
        /// let t10 = Distributions.t.quantile(p: 0.95, df: 10)  // ≈ 1.812
        /// ```
        ///
        /// - Parameters:
        ///   - p: The cumulative probability, in `(0, 1)`.
        ///   - df: The degrees of freedom. Must be positive.
        /// - Returns: The quantile at `p`, or `nil` if `p` is outside `(0, 1)`,
        ///   `df <= 0`, or the computation produces a non-finite result.
        public static func quantile(p: Double, df: Double) -> Double? {
            guard df > 0 else { return nil }
            guard p > 0.0 && p < 1.0 else { return nil }
            if p == 0.5 { return 0.0 }

            // Symmetric — solve for upper tail and negate when p < 0.5.
            let upper = p > 0.5
            let target = upper ? p : 1.0 - p

            // Initial bracket — start from the normal quantile then expand if needed.
            // For small df the t-quantile can be much larger than the normal quantile,
            // so the upper bound widens until cdf(hi, df) > target.
            var lo = 0.0
            var hi: Double
            if let z = Distributions.normal._bsmStandardQuantile(p: target) {
                hi = Swift.max(z, 1.0)
            } else {
                hi = 1.0
            }

            // Expand `hi` until cdf(hi) > target. Cap iterations to prevent runaway.
            var expandIterations = 0
            while expandIterations < 64 {
                guard let cdfHi = cdf(x: hi, df: df) else { return nil }
                if cdfHi >= target { break }
                hi *= 2.0
                expandIterations += 1
            }
            if expandIterations >= 64 { return nil }

            // Bisection — at most ~52 iterations to bring 1.0 down to 2.2e-16.
            let tolerance = 1e-12
            for _ in 0..<200 {
                let mid = 0.5 * (lo + hi)
                if hi - lo < tolerance * Swift.max(1.0, abs(mid)) { break }
                guard let cdfMid = cdf(x: mid, df: df) else { return nil }
                if cdfMid < target {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            let value = 0.5 * (lo + hi)
            let signed = upper ? value : -value
            return signed.isFinite ? signed : nil
        }
    }

    /// Chi-squared distribution functions.
    ///
    /// Provides cumulative density (``cdf(x:df:)``) for the chi-squared
    /// distribution with `df` degrees of freedom. Computed via the regularized
    /// lower incomplete gamma function `P(a, x)`.
    ///
    /// All functions return `Double?` and produce `nil` when `df <= 0` or
    /// when the computation would produce a non-finite result. Following the
    /// convention established by ``Distributions/normal``, out-of-domain
    /// input maps to `nil` rather than a runtime trap.
    public enum chiSquared: Sendable {

        /// Returns the cumulative probability `P(X² <= x)` under the chi-squared distribution with `df` degrees of freedom.
        ///
        /// The chi-squared CDF gives the probability that a chi-squared random variable
        /// with `df` degrees of freedom falls at or below `x`. Negative `x` returns 0
        /// (the distribution has support on `[0, ∞)`). Computed via the regularized
        /// lower incomplete gamma function `P(df / 2, x / 2)` — series expansion when
        /// `x < a + 1` and Lentz's continued fraction otherwise (Numerical Recipes 6.2).
        ///
        /// At `df = 2` the chi-squared CDF has the closed form `1 - exp(-x / 2)`,
        /// and the test suite verifies this anchor to machine precision.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // 0.95 critical value with df = 5 is ≈ 11.07
        /// let p = Distributions.chiSquared.cdf(x: 11.07, df: 5)  // ≈ 0.95
        /// ```
        ///
        /// - Parameters:
        ///   - x: The point at which to evaluate the CDF.
        ///   - df: The degrees of freedom. Must be positive.
        /// - Returns: The cumulative probability at `x`, or `nil` if `df <= 0`
        ///   or the computation produces a non-finite result.
        public static func cdf(x: Double, df: Double) -> Double? {
            guard df > 0 else { return nil }
            if x <= 0 { return 0.0 }
            guard let p = _regularizedIncompleteGammaP(a: 0.5 * df, x: 0.5 * x) else {
                return nil
            }
            let clamped = Swift.max(0.0, Swift.min(1.0, p))
            return clamped.isFinite ? clamped : nil
        }
    }

    /// Poisson distribution functions.
    ///
    /// Provides probability mass (``pmf(k:lambda:)``), log probability mass
    /// (``logPMF(k:lambda:)``), cumulative density (``cdf(k:lambda:)``), and
    /// quantile (``quantile(p:lambda:)``) for the Poisson distribution with
    /// rate parameter `λ`. The Poisson distribution is discrete with support
    /// on the non-negative integers, so the density function is a probability
    /// mass function (`pmf`) rather than a probability density function
    /// (`pdf`) — each value carries an actual probability rather than a
    /// density.
    ///
    /// The mean and variance both equal `λ`. The function ``mean(lambda:)`` and
    /// ``variance(lambda:)`` make this contract explicit at the call site.
    ///
    /// All functions return `Double?` (or `Int?` for `quantile`) and produce
    /// `nil` when `lambda <= 0`, when the input is otherwise out of domain,
    /// or when the computation would produce a non-finite result. Following
    /// the convention established by ``Distributions/normal``, out-of-domain
    /// input maps to `nil` rather than a runtime trap.
    public enum poisson: Sendable {

        /// Returns the probability mass `P(K = k)` under a Poisson distribution with rate `lambda`.
        ///
        /// The Poisson distribution describes the count of independent events
        /// that occur in a fixed window when the events arrive at an average
        /// rate of `λ` per window. The mass function is `P(K = k) = e^{-λ} · λ^k / k!`.
        /// Computed in log space (see ``logPMF(k:lambda:)``) and exponentiated,
        /// which keeps the calculation finite for moderate `λ` where the naive
        /// `λ^k / k!` form overflows.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // Probability of seeing exactly 2 calls when 3.5 are expected
        /// let p = Distributions.poisson.pmf(k: 2, lambda: 3.5)  // ≈ 0.1850
        /// ```
        ///
        /// - Parameters:
        ///   - k: The count to evaluate. Must be non-negative.
        ///   - lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The probability mass at `k`, or `nil` if `k < 0` or `lambda <= 0`.
        public static func pmf(k: Int, lambda: Double) -> Double? {
            guard let logValue = logPMF(k: k, lambda: lambda) else { return nil }
            let value = Foundation.exp(logValue)
            return value.isFinite ? value : nil
        }

        /// Returns the natural log of the Poisson probability mass at `k`.
        ///
        /// Working in log space is the standard tactic for combining many
        /// probabilities — the products become sums, and tail probabilities
        /// that would underflow in linear space stay representable. The
        /// formula `logP(K = k) = -λ + k·log(λ) - log(k!)` uses `lgamma(k+1)`
        /// for the factorial term, which is finite for any reachable `k`.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // log probability of 2 events with rate 3.5
        /// let lp = Distributions.poisson.logPMF(k: 2, lambda: 3.5)  // ≈ -1.687
        /// ```
        ///
        /// - Parameters:
        ///   - k: The count to evaluate. Must be non-negative.
        ///   - lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The log probability mass at `k`, or `nil` if `k < 0` or `lambda <= 0`.
        public static func logPMF(k: Int, lambda: Double) -> Double? {
            guard lambda > 0, k >= 0 else { return nil }
            let kd = Double(k)
            let value = -lambda + kd * Foundation.log(lambda) - Foundation.lgamma(kd + 1.0)
            return value.isFinite ? value : nil
        }

        /// Returns the cumulative probability `P(K <= k)` under a Poisson distribution with rate `lambda`.
        ///
        /// The Poisson CDF gives the probability that the event count falls at
        /// or below `k`. Computed via the regularized upper incomplete gamma
        /// function `Q(k+1, λ) = 1 - P(k+1, λ)`, which is numerically stable
        /// across the full range of `λ` and avoids the precision loss of
        /// naive partial-sum approaches.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // P(K ≤ 5) when 3.5 calls are expected
        /// let p = Distributions.poisson.cdf(k: 5, lambda: 3.5)  // ≈ 0.8576
        /// ```
        ///
        /// - Parameters:
        ///   - k: The upper count bound. Returns 0 for `k < 0`.
        ///   - lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The cumulative probability at `k`, or `nil` if `lambda <= 0`.
        public static func cdf(k: Int, lambda: Double) -> Double? {
            guard lambda > 0 else { return nil }
            if k < 0 { return 0.0 }
            guard let lowerP = _regularizedIncompleteGammaP(a: Double(k + 1), x: lambda) else {
                return nil
            }
            let upperQ = 1.0 - lowerP
            let clamped = Swift.max(0.0, Swift.min(1.0, upperQ))
            return clamped.isFinite ? clamped : nil
        }

        /// Returns the smallest integer `k` such that `P(K <= k) >= p`.
        ///
        /// The Poisson quantile (inverse CDF) returns the count at which the
        /// cumulative probability first crosses `p`. The Poisson distribution
        /// is discrete, so the result is an integer rather than a continuous
        /// value. Seeded by a normal approximation `λ + z·√λ` and refined by
        /// stepping along the CDF until the threshold is crossed.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // The 95th percentile of a Poisson(3.5)
        /// let k = Distributions.poisson.quantile(p: 0.95, lambda: 3.5)  // 7
        /// ```
        ///
        /// - Parameters:
        ///   - p: The cumulative probability. Must be in `[0, 1]`.
        ///   - lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The smallest `k` for which `cdf(k:lambda:) >= p`,
        ///   or `nil` if `p` is outside `[0, 1]` or `lambda <= 0`.
        public static func quantile(p: Double, lambda: Double) -> Int? {
            guard lambda > 0, p >= 0, p <= 1 else { return nil }
            if p == 0 { return 0 }
            if p == 1 { return Int.max }

            // Cornish-Fisher style normal approximation as a seed
            guard let z = normal.quantile(p: p, mean: 0, standardDeviation: 1) else {
                return nil
            }
            let seed = lambda + z * Foundation.sqrt(lambda)
            var k = Swift.max(0, Int(seed.rounded(.down)))

            // Walk downward then upward until the smallest k with cdf >= p
            guard let current = cdf(k: k, lambda: lambda) else { return nil }
            if current >= p {
                while k > 0 {
                    guard let prev = cdf(k: k - 1, lambda: lambda) else { break }
                    if prev < p { break }
                    k -= 1
                }
                return k
            }
            while true {
                k += 1
                guard let next = cdf(k: k, lambda: lambda) else { return nil }
                if next >= p { return k }
            }
        }

        /// Returns the mean of a Poisson distribution. Equal to the rate `lambda`.
        ///
        /// - Parameter lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The mean, or `nil` if `lambda <= 0`.
        public static func mean(lambda: Double) -> Double? {
            guard lambda > 0 else { return nil }
            return lambda
        }

        /// Returns the variance of a Poisson distribution. Equal to the rate `lambda`.
        ///
        /// The mean-equals-variance property is a defining feature of the
        /// Poisson distribution and a quick diagnostic for whether observed
        /// count data is plausibly Poisson.
        ///
        /// - Parameter lambda: The distribution rate parameter (λ). Must be positive.
        /// - Returns: The variance, or `nil` if `lambda <= 0`.
        public static func variance(lambda: Double) -> Double? {
            guard lambda > 0 else { return nil }
            return lambda
        }
    }

    /// Binomial distribution functions.
    ///
    /// Provides probability mass (``pmf(k:n:p:)``), log probability mass
    /// (``logPMF(k:n:p:)``), cumulative density (``cdf(k:n:p:)``), and
    /// quantile (``quantile(p:n:probability:)``) for the binomial distribution
    /// with `n` independent trials each succeeding with probability `p`. The
    /// binomial distribution is discrete with support on `{0, 1, ..., n}`, so
    /// the density function is a probability mass function (`pmf`) rather
    /// than a probability density function (`pdf`).
    ///
    /// The mean is `n·p` and the variance is `n·p·(1-p)`. Functions
    /// ``mean(n:p:)`` and ``variance(n:p:)`` make these contracts explicit.
    ///
    /// All functions return `Double?` (or `Int?` for `quantile`) and produce
    /// `nil` when `n < 0`, `p` is outside `[0, 1]`, or the input is otherwise
    /// out of domain. Following the convention established by
    /// ``Distributions/normal``, out-of-domain input maps to `nil`.
    public enum binomial: Sendable {

        /// Returns the probability mass `P(K = k)` of seeing `k` successes in `n` independent trials.
        ///
        /// The binomial distribution describes the count of successes in a
        /// fixed number of independent yes-or-no trials, each succeeding with
        /// the same probability `p`. The mass function is
        /// `P(K = k) = C(n, k) · p^k · (1-p)^(n-k)`, where `C(n, k)` is the
        /// binomial coefficient. Computed in log space (see
        /// ``logPMF(k:n:p:)``) and exponentiated, which keeps the calculation
        /// finite for the large `n` values that overflow the naive factorial form.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // Probability of exactly 3 successes in 10 trials at p = 0.4
        /// let pr = Distributions.binomial.pmf(k: 3, n: 10, p: 0.4)  // ≈ 0.2150
        /// ```
        ///
        /// - Parameters:
        ///   - k: The number of successes. Must be in `[0, n]`.
        ///   - n: The number of trials. Must be non-negative.
        ///   - p: The per-trial success probability. Must be in `[0, 1]`.
        /// - Returns: The probability mass at `k`, or `nil` if any input is out of domain.
        public static func pmf(k: Int, n: Int, p: Double) -> Double? {
            guard let logValue = logPMF(k: k, n: n, p: p) else { return nil }
            let value = Foundation.exp(logValue)
            return value.isFinite ? value : nil
        }

        /// Returns the natural log of the binomial probability mass at `k`.
        ///
        /// Computed as `log C(n, k) + k·log(p) + (n-k)·log(1-p)`, with the
        /// binomial coefficient evaluated via `lgamma` for numerical stability
        /// at large `n`. Working in log space lets the function return a
        /// finite value even when the underlying probability rounds to zero
        /// in linear space.
        ///
        /// Boundary cases follow probability convention: `0^0 = 1`, so
        /// `pmf(k: 0, n: n, p: 0)` returns `1.0` and `pmf(k: n, n: n, p: 1)`
        /// returns `1.0`.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // log probability of 3 successes in 10 trials at p = 0.4
        /// let lp = Distributions.binomial.logPMF(k: 3, n: 10, p: 0.4)  // ≈ -1.537
        /// ```
        ///
        /// - Parameters:
        ///   - k: The number of successes. Must be in `[0, n]`.
        ///   - n: The number of trials. Must be non-negative.
        ///   - p: The per-trial success probability. Must be in `[0, 1]`.
        /// - Returns: The log probability mass at `k`, or `nil` if any input is out of domain.
        public static func logPMF(k: Int, n: Int, p: Double) -> Double? {
            guard n >= 0, k >= 0, k <= n, p >= 0, p <= 1 else { return nil }

            // Boundary probabilities: handle p = 0 and p = 1 explicitly so
            // log(0) does not appear in the formula.
            if p == 0 {
                return k == 0 ? 0.0 : -.infinity
            }
            if p == 1 {
                return k == n ? 0.0 : -.infinity
            }

            let nd = Double(n)
            let kd = Double(k)
            let logChoose = Foundation.lgamma(nd + 1.0) - Foundation.lgamma(kd + 1.0) - Foundation.lgamma(nd - kd + 1.0)
            let value = logChoose + kd * Foundation.log(p) + (nd - kd) * Foundation.log(1.0 - p)
            return value.isFinite ? value : nil
        }

        /// Returns the cumulative probability `P(K <= k)` of at most `k` successes in `n` trials.
        ///
        /// The binomial CDF gives the probability that the success count falls
        /// at or below `k`. Computed via the regularized incomplete beta
        /// function `I_{1-p}(n-k, k+1)`, which is the standard route for
        /// stable evaluation across a wide range of `n` and `p`. Direct
        /// partial summation of the PMF accumulates rounding error and
        /// underflows for large `n`; the beta route avoids both.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // P(K ≤ 3) in 10 trials at p = 0.4
        /// let pr = Distributions.binomial.cdf(k: 3, n: 10, p: 0.4)  // ≈ 0.3823
        /// ```
        ///
        /// - Parameters:
        ///   - k: The upper success count. Returns 0 for `k < 0` and 1 for `k >= n`.
        ///   - n: The number of trials. Must be non-negative.
        ///   - p: The per-trial success probability. Must be in `[0, 1]`.
        /// - Returns: The cumulative probability at `k`, or `nil` if any input is out of domain.
        public static func cdf(k: Int, n: Int, p: Double) -> Double? {
            guard n >= 0, p >= 0, p <= 1 else { return nil }
            if k < 0 { return 0.0 }
            if k >= n { return 1.0 }

            // Boundary probabilities sidestep the beta evaluation entirely
            if p == 0 { return 1.0 }
            if p == 1 { return 0.0 }

            // P(K <= k) = I_{1-p}(n-k, k+1)
            guard let beta = _regularizedIncompleteBeta(x: 1.0 - p, a: Double(n - k), b: Double(k + 1)) else {
                return nil
            }
            let clamped = Swift.max(0.0, Swift.min(1.0, beta))
            return clamped.isFinite ? clamped : nil
        }

        /// Returns the smallest integer `k` such that `P(K <= k) >= p`.
        ///
        /// The binomial quantile (inverse CDF) returns the success count at
        /// which the cumulative probability first crosses `p`. The
        /// distribution is discrete, so the result is an integer. Seeded by
        /// a normal approximation `n·probability + z·√(n·probability·(1-probability))`
        /// and refined by stepping along the CDF until the threshold is crossed.
        ///
        /// Example:
        /// ```swift
        /// import Quiver
        ///
        /// // 95th percentile of Binomial(n: 10, p: 0.4)
        /// let k = Distributions.binomial.quantile(p: 0.95, n: 10, probability: 0.4)  // 7
        /// ```
        ///
        /// - Parameters:
        ///   - p: The cumulative probability. Must be in `[0, 1]`.
        ///   - n: The number of trials. Must be non-negative.
        ///   - probability: The per-trial success probability. Must be in `[0, 1]`.
        ///     Named to avoid colliding with the cumulative `p` parameter.
        /// - Returns: The smallest `k` for which `cdf(k:n:p:) >= p`, or `nil`
        ///   if any input is out of domain.
        public static func quantile(p: Double, n: Int, probability: Double) -> Int? {
            guard n >= 0, probability >= 0, probability <= 1, p >= 0, p <= 1 else { return nil }
            if p == 0 { return 0 }
            if p == 1 { return n }

            // Normal approximation as a seed
            let mean = Double(n) * probability
            let std = Foundation.sqrt(Double(n) * probability * (1.0 - probability))
            guard let z = normal.quantile(p: p, mean: 0, standardDeviation: 1) else {
                return nil
            }
            let seed = mean + z * std
            var k = Swift.max(0, Swift.min(n, Int(seed.rounded(.down))))

            guard let current = cdf(k: k, n: n, p: probability) else { return nil }
            if current >= p {
                while k > 0 {
                    guard let prev = cdf(k: k - 1, n: n, p: probability) else { break }
                    if prev < p { break }
                    k -= 1
                }
                return k
            }
            while k < n {
                k += 1
                guard let next = cdf(k: k, n: n, p: probability) else { return nil }
                if next >= p { return k }
            }
            return n
        }

        /// Returns the mean of a binomial distribution. Equal to `n·p`.
        ///
        /// - Parameters:
        ///   - n: The number of trials. Must be non-negative.
        ///   - p: The per-trial success probability. Must be in `[0, 1]`.
        /// - Returns: The mean, or `nil` if any input is out of domain.
        public static func mean(n: Int, p: Double) -> Double? {
            guard n >= 0, p >= 0, p <= 1 else { return nil }
            return Double(n) * p
        }

        /// Returns the variance of a binomial distribution. Equal to `n·p·(1-p)`.
        ///
        /// The variance vanishes at the boundaries `p = 0` and `p = 1` because
        /// every trial produces the same outcome with certainty. The maximum
        /// variance for a given `n` occurs at `p = 0.5`.
        ///
        /// - Parameters:
        ///   - n: The number of trials. Must be non-negative.
        ///   - p: The per-trial success probability. Must be in `[0, 1]`.
        /// - Returns: The variance, or `nil` if any input is out of domain.
        public static func variance(n: Int, p: Double) -> Double? {
            guard n >= 0, p >= 0, p <= 1 else { return nil }
            return Double(n) * p * (1.0 - p)
        }
    }

    // MARK: - Internal special-function helpers

    /// Regularized incomplete beta function `I_x(a, b)`.
    ///
    /// Returns `B(x; a, b) / B(a, b)` for `x ∈ [0, 1]`, `a > 0`, `b > 0`.
    /// Combines the series form (rapid convergence near 0) with Lentz's
    /// continued fraction (NR 5.2 / 6.4), switching at `x = (a + 1) / (a + b + 2)`
    /// — the boundary where the series form starts to lose accuracy.
    /// Returns `nil` for out-of-domain input or non-finite results.
    internal static func _regularizedIncompleteBeta(x: Double, a: Double, b: Double) -> Double? {
        guard a > 0, b > 0 else { return nil }
        if x <= 0 { return 0.0 }
        if x >= 1 { return 1.0 }

        // ln of B(x;a,b)/B(a,b) prefactor. Split into named subexpressions
        // so the Swift 6.3+ type checker can resolve the arithmetic in
        // reasonable time.
        let lgammaSum: Double = Foundation.lgamma(a + b)
        let lgammaA: Double = Foundation.lgamma(a)
        let lgammaB: Double = Foundation.lgamma(b)
        let logX: Double = Foundation.log(x)
        let log1mX: Double = Foundation.log(1.0 - x)
        let lnPrefactor: Double = lgammaSum - lgammaA - lgammaB + a * logX + b * log1mX
        let prefactor = Foundation.exp(lnPrefactor)
        guard prefactor.isFinite else { return nil }

        let threshold = (a + 1.0) / (a + b + 2.0)
        if x < threshold {
            // Use the continued fraction for I_x(a, b) directly.
            guard let cf = _betaContinuedFraction(x: x, a: a, b: b) else { return nil }
            return prefactor * cf / a
        } else {
            // Symmetry: I_x(a, b) = 1 - I_{1-x}(b, a).
            guard let cf = _betaContinuedFraction(x: 1.0 - x, a: b, b: a) else { return nil }
            return 1.0 - prefactor * cf / b
        }
    }

    /// Continued-fraction evaluation for the regularized incomplete beta.
    ///
    /// Implements the recurrence from Numerical Recipes 6.4 with Lentz's
    /// modification (NR 5.2). The caller multiplies by the prefactor and
    /// divides by `a` (or `b` for the symmetric branch) to recover `I_x(a, b)`.
    /// Returns `nil` if the iteration fails to converge.
    private static func _betaContinuedFraction(x: Double, a: Double, b: Double) -> Double? {
        let maxIterations = 200
        let epsilon = 3.0e-16
        let fpMin = 1.0e-300

        let qab = a + b
        let qap = a + 1.0
        let qam = a - 1.0
        var c = 1.0
        var d = 1.0 - qab * x / qap
        if abs(d) < fpMin { d = fpMin }
        d = 1.0 / d
        var h = d

        for m in 1...maxIterations {
            let mDouble = Double(m)
            let m2 = 2.0 * mDouble

            // Even step.
            var aa = mDouble * (b - mDouble) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpMin { d = fpMin }
            c = 1.0 + aa / c
            if abs(c) < fpMin { c = fpMin }
            d = 1.0 / d
            h *= d * c

            // Odd step.
            aa = -(a + mDouble) * (qab + mDouble) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpMin { d = fpMin }
            c = 1.0 + aa / c
            if abs(c) < fpMin { c = fpMin }
            d = 1.0 / d
            let del = d * c
            h *= del

            if abs(del - 1.0) < epsilon {
                return h
            }
        }
        return nil
    }

    /// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`.
    ///
    /// Uses the series expansion when `x < a + 1` and the upper-incomplete
    /// continued fraction (NR 6.2) otherwise. The two routes converge in
    /// complementary regimes; switching at `x = a + 1` keeps both fast.
    /// Returns `nil` for out-of-domain input or non-finite results.
    internal static func _regularizedIncompleteGammaP(a: Double, x: Double) -> Double? {
        guard a > 0 else { return nil }
        if x <= 0 { return 0.0 }
        if x < a + 1.0 {
            return _gammaSeries(a: a, x: x)
        } else {
            guard let q = _gammaContinuedFraction(a: a, x: x) else { return nil }
            return 1.0 - q
        }
    }

    /// Series form for the regularized lower incomplete gamma `P(a, x)`.
    ///
    /// Converges rapidly when `x < a + 1`. Returns `nil` if the iteration
    /// fails to converge or produces a non-finite result.
    private static func _gammaSeries(a: Double, x: Double) -> Double? {
        let maxIterations = 200
        let epsilon = 3.0e-16

        let logX: Double = Foundation.log(x)
        let lgammaA: Double = Foundation.lgamma(a)
        let lnPrefactor: Double = -x + a * logX - lgammaA
        guard lnPrefactor.isFinite else { return nil }
        let prefactor = Foundation.exp(lnPrefactor)

        var ap = a
        var sum = 1.0 / a
        var del = sum
        for _ in 0..<maxIterations {
            ap += 1.0
            del *= x / ap
            sum += del
            if abs(del) < abs(sum) * epsilon {
                let value = sum * prefactor
                return value.isFinite ? value : nil
            }
        }
        return nil
    }

    /// Lentz's continued fraction for the regularized upper incomplete gamma `Q(a, x)`.
    ///
    /// Caller computes `P = 1 - Q`. Converges rapidly when `x >= a + 1`.
    /// Returns `nil` if the iteration fails to converge.
    private static func _gammaContinuedFraction(a: Double, x: Double) -> Double? {
        let maxIterations = 200
        let epsilon = 3.0e-16
        let fpMin = 1.0e-300

        let logX: Double = Foundation.log(x)
        let lgammaA: Double = Foundation.lgamma(a)
        let lnPrefactor: Double = -x + a * logX - lgammaA
        guard lnPrefactor.isFinite else { return nil }
        let prefactor = Foundation.exp(lnPrefactor)

        var b = x + 1.0 - a
        var c = 1.0 / fpMin
        var d = 1.0 / b
        var h = d
        for i in 1...maxIterations {
            let iDouble = Double(i)
            let an = -iDouble * (iDouble - a)
            b += 2.0
            d = an * d + b
            if abs(d) < fpMin { d = fpMin }
            c = b + an / c
            if abs(c) < fpMin { c = fpMin }
            d = 1.0 / d
            let del = d * c
            h *= del
            if abs(del - 1.0) < epsilon {
                let value = prefactor * h
                return value.isFinite ? value : nil
            }
        }
        return nil
    }

}
