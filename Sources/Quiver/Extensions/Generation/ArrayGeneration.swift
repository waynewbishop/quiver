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

// MARK: - Array Generation for Numeric Types

public extension Array where Element: Numeric {
    /// Creates a 1D array filled with zeros.
    ///
    /// - Parameter count: The number of elements to generate
    /// - Returns: An array of the specified length with all elements set to zero
    static func zeros(_ count: Int) -> [Element] {
        return _Vector<Element>.zeros(count).elements
    }

    /// Creates a 1D array filled with ones.
    ///
    /// - Parameter count: The number of elements to generate
    /// - Returns: An array of the specified length with all elements set to one
    static func ones(_ count: Int) -> [Element] {
        return _Vector<Element>.ones(count).elements
    }

    /// Creates a 1D array filled with a specific value.
    ///
    /// - Parameters:
    ///   - count: The number of elements to generate
    ///   - value: The value to fill every element with
    /// - Returns: An array of the specified length with all elements set to the given value
    static func full(_ count: Int, value: Element) -> [Element] {
        return _Vector<Element>.full(count, value: value).elements
    }

    /// Creates a 2D array filled with zeros.
    ///
    /// - Parameters:
    ///   - rows: The number of rows
    ///   - columns: The number of columns
    /// - Returns: A matrix of the specified dimensions with all elements set to zero
    static func zeros(_ rows: Int, _ columns: Int) -> [[Element]] {
        return _Vector<Element>.zeros2D(rows, columns)
    }

    /// Creates a 2D array filled with ones.
    ///
    /// - Parameters:
    ///   - rows: The number of rows
    ///   - columns: The number of columns
    /// - Returns: A matrix of the specified dimensions with all elements set to one
    static func ones(_ rows: Int, _ columns: Int) -> [[Element]] {
        return _Vector<Element>.ones2D(rows, columns)
    }

    /// Creates a 2D array filled with a specific value.
    ///
    /// - Parameters:
    ///   - rows: The number of rows
    ///   - columns: The number of columns
    ///   - value: The value to fill every element with
    /// - Returns: A matrix of the specified dimensions with all elements set to the given value
    static func full(_ rows: Int, _ columns: Int, value: Element) -> [[Element]] {
        return _Vector<Element>.full2D(rows, columns, value: value)
    }
    
    /// Creates an identity matrix of size n × n.
    ///
    /// The identity matrix is the neutral element for matrix multiplication, analogous to multiplying
    /// by 1 in scalar arithmetic. It has ones along the main diagonal and zeros elsewhere.
    ///
    /// Mathematical form for n=3:
    /// ```
    /// [[1, 0, 0],
    ///  [0, 1, 0],
    ///  [0, 0, 1]]
    /// ```
    ///
    /// Key properties:
    /// - **Neutral transformation**: Leaves vectors unchanged when applied
    /// - **Multiplication identity**: `I × M = M × I = M` for any matrix M
    /// - **Basis vectors**: Each column represents a standard basis vector (i-hat, j-hat, k-hat, ...)
    ///
    /// Common uses:
    /// - Starting point for transformation composition
    /// - Verifying matrix inverses: `M × M⁻¹ = I`
    /// - Representing "no transformation" in graphics pipelines
    ///
    /// Example:
    /// ```swift
    /// // Create 2D identity matrix
    /// let identity = [Double].identity(2)
    /// // [[1.0, 0.0],
    /// //  [0.0, 1.0]]
    ///
    /// // Verify it leaves vectors unchanged
    /// let vector = [3.0, 4.0]
    /// let result = vector.transformedBy(identity)
    /// // [3.0, 4.0] - unchanged
    ///
    /// // Math: [3, 4] = 3×[1, 0] + 4×[0, 1] = [3, 4]
    ///
    /// // Verify multiplication identity property
    /// let rotation = [[0.0, -1.0], [1.0, 0.0]]
    /// let unchanged = identity.multiplyMatrix(rotation)
    /// // [[0.0, -1.0], [1.0, 0.0]] - same as rotation
    /// ```
    ///
    /// - Parameter n: The size of the square matrix (must be positive)
    /// - Returns: An n × n identity matrix
    static func identity(_ n: Int) -> [[Element]] {
        return _Vector<Element>.identity(n)
    }
    
    /// Creates a diagonal matrix from a vector of diagonal values.
    ///
    /// A diagonal matrix has non-zero values only along its main diagonal (top-left to bottom-right),
    /// with all other elements set to zero. This is fundamental for scaling transformations and
    /// representing various linear algebra operations.
    ///
    /// Mathematical form for vector `[a, b, c]`:
    /// ```
    /// [[a, 0, 0],
    ///  [0, b, 0],
    ///  [0, 0, c]]
    /// ```
    ///
    /// Common use cases:
    /// - **Scaling transformations**: Apply different scale factors per axis
    /// - **Identity matrix**: `diag([1, 1, 1])` creates the identity matrix
    /// - **Reflection**: `diag([-1, 1])` reflects across y-axis
    /// - **Eigenvalue matrices**: Store eigenvalues along the diagonal
    ///
    /// Example:
    /// ```swift
    /// // Create 2D scaling transformation (3× horizontal, 2× vertical)
    /// let scale = [Double].diag([3.0, 2.0])
    /// // [[3.0, 0.0],
    /// //  [0.0, 2.0]]
    ///
    /// // Apply to vector [4, 5]
    /// let vector = [4.0, 5.0]
    /// let scaled = vector.transformedBy(scale)
    /// // [12.0, 10.0]
    ///
    /// // Math: 4×[3, 0] + 5×[0, 2] = [12, 0] + [0, 10] = [12, 10]
    /// ```
    ///
    /// - Parameter diagonal: Vector of values to place on the matrix diagonal
    /// - Returns: Square matrix with diagonal values from input vector, zeros elsewhere
    static func diag(_ diagonal: [Element]) -> [[Element]] {
        return _Vector<Element>.diag(_Vector(elements: diagonal))
    }
    
    /// Creates a sequence of evenly spaced values within a half-open interval.
    ///
    /// Values are generated from `start` up to (but not including) `stop`,
    /// incrementing by `step`. Supports both positive and negative step values.
    ///
    /// - Parameters:
    ///   - start: The starting value of the sequence
    ///   - stop: The exclusive upper bound of the sequence
    ///   - step: The spacing between consecutive values (must not be zero)
    /// - Returns: An array of evenly spaced values from start up to stop
    static func arange(_ start: Element, _ stop: Element, step: Element) -> [Element] where Element: Comparable {
        var result = [Element]()
        
        if step > .zero {
            var current = start
            while current < stop {
                result.append(current)
                current = current + step
            }
        } else if step < .zero {
            var current = start
            while current > stop {
                result.append(current)
                current = current + step
            }
        } else {
            preconditionFailure("Step size cannot be zero")
        }
        
        return result
    }
}

// MARK: - Array Generation for FloatingPoint Types

public extension Array where Element: FloatingPoint {
    /// Creates evenly spaced numbers over a specified closed interval.
    ///
    /// Unlike `arange`, this method specifies the number of points rather than the step size,
    /// and includes both endpoints.
    ///
    /// - Parameters:
    ///   - start: The starting value of the sequence
    ///   - end: The ending value of the sequence (included)
    ///   - count: The number of evenly spaced values to generate
    /// - Returns: An array of `count` evenly spaced values from start to end (inclusive)
    static func linspace(start: Element, end: Element, count: Int) -> [Element] {
        return _Vector<Element>.linspace(start: start, end: end, count: count).elements
    }
    
    /// Creates a sequence of evenly spaced values within a half-open interval.
    ///
    /// - Parameters:
    ///   - start: The starting value of the sequence
    ///   - stop: The exclusive upper bound of the sequence
    ///   - step: The spacing between consecutive values (default 1)
    /// - Returns: An array of evenly spaced values from start up to stop
    static func arange(_ start: Element, _ stop: Element, step: Element = 1) -> [Element] {
        return _Vector<Element>.arange(start, stop, step: step).elements
    }
    
}
