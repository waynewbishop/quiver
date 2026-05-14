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

// MARK: - Random Array Generation for Double

public extension Array where Element == Double {
    /// Creates a 1D array of random values uniformly distributed between 0 and 1.
    ///
    /// - Parameter count: The number of random values to generate (must be non-negative)
    /// - Returns: An array of uniformly distributed random doubles in [0, 1]
    static func random(_ count: Int) -> [Double] {
        var rng = SystemRandomNumberGenerator()
        return random(count, using: &rng)
    }

    /// Creates a 1D array of random values uniformly distributed between 0 and 1, drawing from the given generator.
    ///
    /// Pass a `SeededRandomNumberGenerator` for reproducible output, or any other type
    /// conforming to `RandomNumberGenerator`. Mirrors Swift stdlib's `Array.shuffled(using:)`
    /// pattern so the same generator can thread through Quiver and stdlib calls.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - generator: The random number generator to draw from
    /// - Returns: An array of uniformly distributed random doubles in [0, 1]
    static func random<G: RandomNumberGenerator>(_ count: Int, using generator: inout G) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        var result = [Double]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(Double.random(in: 0...1, using: &generator))
        }
        return result
    }

    /// Creates a 2D array of random values uniformly distributed between 0 and 1.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    /// - Returns: A matrix of uniformly distributed random doubles in [0, 1]
    static func random(_ rows: Int, _ columns: Int) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        return random(rows, columns, using: &rng)
    }

    /// Creates a 2D array of random values uniformly distributed between 0 and 1, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - generator: The random number generator to draw from
    /// - Returns: A matrix of uniformly distributed random doubles in [0, 1]
    static func random<G: RandomNumberGenerator>(_ rows: Int, _ columns: Int, using generator: inout G) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in random(columns, using: &generator) }
    }
}

// MARK: - Custom Range Random for Double

public extension Array where Element == Double {
    /// Creates a 1D array of random values uniformly distributed in the given range.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - range: The closed range to sample from
    /// - Returns: An array of uniformly distributed random doubles within the specified range
    static func random(_ count: Int, in range: ClosedRange<Double>) -> [Double] {
        var rng = SystemRandomNumberGenerator()
        return random(count, in: range, using: &rng)
    }

    /// Creates a 1D array of random values uniformly distributed in the given range, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - range: The closed range to sample from
    ///   - generator: The random number generator to draw from
    /// - Returns: An array of uniformly distributed random doubles within the specified range
    static func random<G: RandomNumberGenerator>(_ count: Int, in range: ClosedRange<Double>, using generator: inout G) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        var result = [Double]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(Double.random(in: range, using: &generator))
        }
        return result
    }

    /// Creates a 2D array of random values uniformly distributed in the given range.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - range: The closed range to sample from
    /// - Returns: A matrix of uniformly distributed random doubles within the specified range
    static func random(_ rows: Int, _ columns: Int, in range: ClosedRange<Double>) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        return random(rows, columns, in: range, using: &rng)
    }

    /// Creates a 2D array of random values uniformly distributed in the given range, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - range: The closed range to sample from
    ///   - generator: The random number generator to draw from
    /// - Returns: A matrix of uniformly distributed random doubles within the specified range
    static func random<G: RandomNumberGenerator>(_ rows: Int, _ columns: Int, in range: ClosedRange<Double>, using generator: inout G) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in random(columns, in: range, using: &generator) }
    }
}

// MARK: - Normal Distribution Random for Double

public extension Array where Element == Double {
    /// Generates a pair of standard normal values using the Box-Muller transform.
    private static func _boxMullerPair<G: RandomNumberGenerator>(using generator: inout G) -> (Double, Double) {
        let u1 = Double.random(in: Double.leastNonzeroMagnitude...1.0, using: &generator)
        let u2 = Double.random(in: 0.0...1.0, using: &generator)
        let r = (-2.0 * Foundation.log(u1)).squareRoot()
        let theta = 2.0 * .pi * u2
        return (r * Foundation.cos(theta), r * Foundation.sin(theta))
    }

    /// Creates a 1D array of random values from a normal (Gaussian) distribution.
    ///
    /// Uses the Box-Muller transform to generate normally distributed values.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: An array of normally distributed random doubles
    static func randomNormal(_ count: Int, mean: Double = 0.0, standardDeviation: Double = 1.0) -> [Double] {
        var rng = SystemRandomNumberGenerator()
        return randomNormal(count, mean: mean, standardDeviation: standardDeviation, using: &rng)
    }

    /// Creates a 1D array of random values from a normal (Gaussian) distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    ///   - generator: The random number generator to draw from
    /// - Returns: An array of normally distributed random doubles
    static func randomNormal<G: RandomNumberGenerator>(_ count: Int, mean: Double = 0.0, standardDeviation: Double = 1.0, using generator: inout G) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(standardDeviation >= 0, "Standard deviation must be non-negative")
        var result = [Double]()
        result.reserveCapacity(count)
        while result.count < count {
            let (z1, z2) = _boxMullerPair(using: &generator)
            result.append(mean + z1 * standardDeviation)
            if result.count < count {
                result.append(mean + z2 * standardDeviation)
            }
        }
        return result
    }

    /// Creates a 2D array of random values from a normal (Gaussian) distribution.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: A matrix of normally distributed random doubles
    static func randomNormal(_ rows: Int, _ columns: Int, mean: Double = 0.0, standardDeviation: Double = 1.0) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        return randomNormal(rows, columns, mean: mean, standardDeviation: standardDeviation, using: &rng)
    }

    /// Creates a 2D array of random values from a normal (Gaussian) distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    ///   - generator: The random number generator to draw from
    /// - Returns: A matrix of normally distributed random doubles
    static func randomNormal<G: RandomNumberGenerator>(_ rows: Int, _ columns: Int, mean: Double = 0.0, standardDeviation: Double = 1.0, using generator: inout G) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(standardDeviation >= 0, "Standard deviation must be non-negative")
        return (0..<rows).map { _ in randomNormal(columns, mean: mean, standardDeviation: standardDeviation, using: &generator) }
    }
}

// MARK: - Exponential Distribution Random for Double

public extension Array where Element == Double {
    /// Creates a 1D array of random values from an exponential distribution.
    ///
    /// Models the time between independent events occurring at a constant average rate —
    /// call durations, transaction intervals, equipment lifetimes. Generated by inverse-
    /// transform sampling against the cumulative distribution: `-log(1 - u) / rate`.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - rate: The rate parameter λ controlling decay (default 1.0, must be positive). The mean of the distribution is `1 / rate`.
    /// - Returns: An array of exponentially distributed random doubles, all non-negative
    static func randomExponential(_ count: Int, rate: Double = 1.0) -> [Double] {
        var rng = SystemRandomNumberGenerator()
        return randomExponential(count, rate: rate, using: &rng)
    }

    /// Creates a 1D array of random values from an exponential distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - rate: The rate parameter λ controlling decay (default 1.0, must be positive)
    ///   - generator: The random number generator to draw from
    /// - Returns: An array of exponentially distributed random doubles
    static func randomExponential<G: RandomNumberGenerator>(_ count: Int, rate: Double = 1.0, using generator: inout G) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(rate > 0, "Rate must be positive")
        var result = [Double]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            // Draw u from (0, 1] so that 1 - u is in [0, 1) and log is finite
            let u = Double.random(in: Double.leastNonzeroMagnitude...1.0, using: &generator)
            result.append(-Foundation.log(1.0 - u) / rate)
        }
        return result
    }

    /// Creates a 2D array of random values from an exponential distribution.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - rate: The rate parameter λ controlling decay (default 1.0, must be positive)
    /// - Returns: A matrix of exponentially distributed random doubles
    static func randomExponential(_ rows: Int, _ columns: Int, rate: Double = 1.0) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        return randomExponential(rows, columns, rate: rate, using: &rng)
    }

    /// Creates a 2D array of random values from an exponential distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - rate: The rate parameter λ controlling decay (default 1.0, must be positive)
    ///   - generator: The random number generator to draw from
    /// - Returns: A matrix of exponentially distributed random doubles
    static func randomExponential<G: RandomNumberGenerator>(_ rows: Int, _ columns: Int, rate: Double = 1.0, using generator: inout G) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(rate > 0, "Rate must be positive")
        return (0..<rows).map { _ in randomExponential(columns, rate: rate, using: &generator) }
    }
}

// MARK: - Binomial Distribution Random for Double

public extension Array where Element == Double {
    /// Creates a 1D array of random values from a binomial distribution.
    ///
    /// Each value is the count of successes in `n` independent Bernoulli trials with
    /// success probability `p`. Models conversion counts, defect counts, and vote tallies —
    /// anywhere a fixed number of independent yes/no trials are aggregated.
    ///
    /// Returns `[Double]` with whole-number values (`0.0`, `1.0`, ..., `Double(n)`) so
    /// results compose with `mean()`, `std()`, and `histogram(bins:)` without conversion.
    /// The simple-and-clear loop runs each trial individually (O(count × n)) — adequate
    /// for teaching examples and any `n` up to a few hundred.
    ///
    /// - Parameters:
    ///   - count: The number of binomial samples to generate (must be non-negative)
    ///   - n: The number of Bernoulli trials per sample (must be non-negative; `0` returns zeros)
    ///   - p: The probability of success on each trial (must be in `[0, 1]`)
    /// - Returns: An array of binomial counts as whole-number doubles
    static func randomBinomial(_ count: Int, n: Int, p: Double) -> [Double] {
        var rng = SystemRandomNumberGenerator()
        return randomBinomial(count, n: n, p: p, using: &rng)
    }

    /// Creates a 1D array of random values from a binomial distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - count: The number of binomial samples to generate (must be non-negative)
    ///   - n: The number of Bernoulli trials per sample (must be non-negative)
    ///   - p: The probability of success on each trial (must be in `[0, 1]`)
    ///   - generator: The random number generator to draw from
    /// - Returns: An array of binomial counts as whole-number doubles
    static func randomBinomial<G: RandomNumberGenerator>(_ count: Int, n: Int, p: Double, using generator: inout G) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(n >= 0, "n must be non-negative")
        precondition(p >= 0 && p <= 1, "p must be in [0, 1]")
        var result = [Double]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            var successes = 0
            for _ in 0..<n {
                if Double.random(in: 0..<1, using: &generator) < p { successes += 1 }
            }
            result.append(Double(successes))
        }
        return result
    }

    /// Creates a 2D array of random values from a binomial distribution.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - n: The number of Bernoulli trials per sample (must be non-negative)
    ///   - p: The probability of success on each trial (must be in `[0, 1]`)
    /// - Returns: A matrix of binomial counts as whole-number doubles
    static func randomBinomial(_ rows: Int, _ columns: Int, n: Int, p: Double) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        return randomBinomial(rows, columns, n: n, p: p, using: &rng)
    }

    /// Creates a 2D array of random values from a binomial distribution, drawing from the given generator.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - n: The number of Bernoulli trials per sample (must be non-negative)
    ///   - p: The probability of success on each trial (must be in `[0, 1]`)
    ///   - generator: The random number generator to draw from
    /// - Returns: A matrix of binomial counts as whole-number doubles
    static func randomBinomial<G: RandomNumberGenerator>(_ rows: Int, _ columns: Int, n: Int, p: Double, using generator: inout G) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(n >= 0, "n must be non-negative")
        precondition(p >= 0 && p <= 1, "p must be in [0, 1]")
        return (0..<rows).map { _ in randomBinomial(columns, n: n, p: p, using: &generator) }
    }
}

// MARK: - Random Array Generation for Float

public extension Array where Element == Float {
    /// Creates a 1D array of random values uniformly distributed between 0 and 1.
    ///
    /// - Parameter count: The number of random values to generate (must be non-negative)
    /// - Returns: An array of uniformly distributed random floats in [0, 1]
    static func random(_ count: Int) -> [Float] {
        precondition(count >= 0, "Count must be non-negative")
        return (0..<count).map { _ in Float.random(in: 0...1) }
    }

    /// Creates a 2D array of random values uniformly distributed between 0 and 1.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    /// - Returns: A matrix of uniformly distributed random floats in [0, 1]
    static func random(_ rows: Int, _ columns: Int) -> [[Float]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in (0..<columns).map { _ in Float.random(in: 0...1) } }
    }
}

// MARK: - Custom Range Random for Float

public extension Array where Element == Float {
    /// Creates a 1D array of random values uniformly distributed in the given range.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - range: The closed range to sample from
    /// - Returns: An array of uniformly distributed random floats within the specified range
    static func random(_ count: Int, in range: ClosedRange<Float>) -> [Float] {
        precondition(count >= 0, "Count must be non-negative")
        return (0..<count).map { _ in Float.random(in: range) }
    }

    /// Creates a 2D array of random values uniformly distributed in the given range.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - range: The closed range to sample from
    /// - Returns: A matrix of uniformly distributed random floats within the specified range
    static func random(_ rows: Int, _ columns: Int, in range: ClosedRange<Float>) -> [[Float]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in (0..<columns).map { _ in Float.random(in: range) } }
    }
}

// MARK: - Normal Distribution Random for Float

public extension Array where Element == Float {
    /// Creates a 1D array of random values from a normal (Gaussian) distribution.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: An array of normally distributed random floats
    static func randomNormal(_ count: Int, mean: Float = 0.0, standardDeviation: Float = 1.0) -> [Float] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(standardDeviation >= 0, "Standard deviation must be non-negative")
        return [Double].randomNormal(count, mean: Double(mean), standardDeviation: Double(standardDeviation)).map { Float($0) }
    }

    /// Creates a 2D array of random values from a normal (Gaussian) distribution.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - standardDeviation: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: A matrix of normally distributed random floats
    static func randomNormal(_ rows: Int, _ columns: Int, mean: Float = 0.0, standardDeviation: Float = 1.0) -> [[Float]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(standardDeviation >= 0, "Standard deviation must be non-negative")
        return [Double].randomNormal(rows, columns, mean: Double(mean), standardDeviation: Double(standardDeviation)).map { $0.map { Float($0) } }
    }
}

// MARK: - Random Integer Generation

public extension Array where Element == Int {
    /// Creates a 1D array of random integers in the given half-open range.
    ///
    /// - Parameters:
    ///   - count: The number of random values to generate (must be non-negative)
    ///   - range: The half-open range to sample from (must not be empty)
    /// - Returns: An array of random integers within the specified range
    static func random(_ count: Int, in range: Range<Int>) -> [Int] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(!range.isEmpty, "Range must not be empty")
        return (0..<count).map { _ in Int.random(in: range) }
    }

    /// Creates a 2D array of random integers in the given half-open range.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - range: The half-open range to sample from (must not be empty)
    /// - Returns: A matrix of random integers within the specified range
    static func random(_ rows: Int, _ columns: Int, in range: Range<Int>) -> [[Int]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(!range.isEmpty, "Range must not be empty")
        return (0..<rows).map { _ in (0..<columns).map { _ in Int.random(in: range) } }
    }
}
