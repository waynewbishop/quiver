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
        precondition(count >= 0, "Count must be non-negative")
        return (0..<count).map { _ in Double.random(in: 0...1) }
    }

    /// Creates a 2D array of random values uniformly distributed between 0 and 1.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    /// - Returns: A matrix of uniformly distributed random doubles in [0, 1]
    static func random(_ rows: Int, _ columns: Int) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in (0..<columns).map { _ in Double.random(in: 0...1) } }
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
        precondition(count >= 0, "Count must be non-negative")
        return (0..<count).map { _ in Double.random(in: range) }
    }

    /// Creates a 2D array of random values uniformly distributed in the given range.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - range: The closed range to sample from
    /// - Returns: A matrix of uniformly distributed random doubles within the specified range
    static func random(_ rows: Int, _ columns: Int, in range: ClosedRange<Double>) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        return (0..<rows).map { _ in (0..<columns).map { _ in Double.random(in: range) } }
    }
}

// MARK: - Normal Distribution Random for Double

public extension Array where Element == Double {
    /// Generates a pair of standard normal values using the Box-Muller transform
    private static func _boxMullerPair() -> (Double, Double) {
        let u1 = Double.random(in: Double.leastNonzeroMagnitude...1.0)
        let u2 = Double.random(in: 0.0...1.0)
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
    ///   - std: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: An array of normally distributed random doubles
    static func randomNormal(_ count: Int, mean: Double = 0.0, std: Double = 1.0) -> [Double] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(std >= 0, "Standard deviation must be non-negative")
        var result = [Double]()
        result.reserveCapacity(count)
        while result.count < count {
            let (z1, z2) = _boxMullerPair()
            result.append(mean + z1 * std)
            if result.count < count {
                result.append(mean + z2 * std)
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
    ///   - std: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: A matrix of normally distributed random doubles
    static func randomNormal(_ rows: Int, _ columns: Int, mean: Double = 0.0, std: Double = 1.0) -> [[Double]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(std >= 0, "Standard deviation must be non-negative")
        return (0..<rows).map { _ in randomNormal(columns, mean: mean, std: std) }
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
    ///   - std: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: An array of normally distributed random floats
    static func randomNormal(_ count: Int, mean: Float = 0.0, std: Float = 1.0) -> [Float] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(std >= 0, "Standard deviation must be non-negative")
        return [Double].randomNormal(count, mean: Double(mean), std: Double(std)).map { Float($0) }
    }

    /// Creates a 2D array of random values from a normal (Gaussian) distribution.
    ///
    /// - Parameters:
    ///   - rows: The number of rows (must be positive)
    ///   - columns: The number of columns (must be positive)
    ///   - mean: The center of the distribution (default 0.0)
    ///   - std: The standard deviation controlling spread (default 1.0, must be non-negative)
    /// - Returns: A matrix of normally distributed random floats
    static func randomNormal(_ rows: Int, _ columns: Int, mean: Float = 0.0, std: Float = 1.0) -> [[Float]] {
        precondition(rows > 0 && columns > 0, "Dimensions must be positive")
        precondition(std >= 0, "Standard deviation must be non-negative")
        return [Double].randomNormal(rows, columns, mean: Double(mean), std: Double(std)).map { $0.map { Float($0) } }
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
