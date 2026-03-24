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

// MARK: - Basic Arithmetic Operations

public extension Array where Element: Numeric {
    /// Element-wise addition of two vectors.
    ///
    /// ```swift
    /// let velocity = [3.0, 0.0]
    /// let current = [0.0, 2.0]
    /// velocity.add(current)  // [3.0, 2.0]
    /// ```
    ///
    /// - Parameter other: The second vector (must have the same length as `self`)
    /// - Returns: A new array where each element is the sum of corresponding elements
    func add(_ other: [Element]) -> [Element] {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: other)
        return _Vector.add(v1, v2).elements
    }

    /// Element-wise subtraction of two vectors.
    ///
    /// Subtraction produces the displacement between two points. The ``Swift/Array/distance(to:)``
    /// method uses this internally: `self.subtract(other).magnitude`.
    ///
    /// ```swift
    /// let player = [100.0, 200.0]
    /// let enemy = [130.0, 170.0]
    /// player.subtract(enemy)  // [-30.0, 30.0]
    /// ```
    ///
    /// - Parameter other: The second vector (must have the same length as `self`)
    /// - Returns: A new array where each element is the difference of corresponding elements
    func subtract(_ other: [Element]) -> [Element] {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: other)
        return _Vector.subtract(v1, v2).elements
    }

    /// Element-wise multiplication of two vectors (Hadamard product).
    ///
    /// This is not matrix multiplication — each element is multiplied by its
    /// corresponding element. For matrix multiplication, use `multiplyMatrix(_:)`.
    ///
    /// ```swift
    /// let features = [2.0, 5.0, 3.0]
    /// let weights = [0.5, 0.3, 0.2]
    /// features.multiply(weights)  // [1.0, 1.5, 0.6]
    /// ```
    ///
    /// - Parameter other: The second vector (must have the same length as `self`)
    /// - Returns: A new array where each element is the product of corresponding elements
    func multiply(_ other: [Element]) -> [Element] {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: other)
        return _Vector.multiply(v1, v2).elements
    }
}

// Division is only available for floating point types
public extension Array where Element: FloatingPoint {
    /// Element-wise division of two vectors.
    ///
    /// ```swift
    /// let values = [10.0, 20.0, 30.0]
    /// let scales = [2.0, 5.0, 10.0]
    /// values.divide(scales)  // [5.0, 4.0, 3.0]
    /// ```
    ///
    /// - Parameter other: The divisor vector (must have the same length as `self`, no element may be zero)
    /// - Returns: A new array where each element is the quotient of corresponding elements
    func divide(_ other: [Element]) -> [Element] {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: other)
        return _Vector<Element>.divide(v1, v2).elements
    }
}

// MARK: - Matrix Arithmetic Operations (Double)

public extension Array where Element == [Double] {
    /// Element-wise addition of two matrices.
    ///
    /// ```swift
    /// let m1 = [[1.0, 2.0], [3.0, 4.0]]
    /// let m2 = [[5.0, 6.0], [7.0, 8.0]]
    /// m1.add(m2)  // [[6.0, 8.0], [10.0, 12.0]]
    /// ```
    ///
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the sum of corresponding elements
    func add(_ other: [[Double]]) -> [[Double]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot add empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.add(row2)
        }
    }

    /// Element-wise subtraction of two matrices.
    ///
    /// ```swift
    /// let actual = [[90.0, 85.0], [78.0, 92.0]]
    /// let predicted = [[88.0, 87.0], [80.0, 90.0]]
    /// actual.subtract(predicted)  // [[2.0, -2.0], [-2.0, 2.0]]
    /// ```
    ///
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the difference of corresponding elements
    func subtract(_ other: [[Double]]) -> [[Double]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot subtract empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.subtract(row2)
        }
    }

    /// Element-wise multiplication of two matrices (Hadamard product).
    ///
    /// ```swift
    /// let data = [[1.0, 2.0], [3.0, 4.0]]
    /// let mask = [[1.0, 0.0], [0.0, 1.0]]
    /// data.multiply(mask)  // [[1.0, 0.0], [0.0, 4.0]]
    /// ```
    ///
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the product of corresponding elements
    func multiply(_ other: [[Double]]) -> [[Double]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot multiply empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.multiply(row2)
        }
    }

    /// Element-wise division of two matrices.
    ///
    /// ```swift
    /// let totals = [[100.0, 200.0], [300.0, 400.0]]
    /// let counts = [[4.0, 5.0], [6.0, 8.0]]
    /// totals.divide(counts)  // [[25.0, 40.0], [50.0, 50.0]]
    /// ```
    ///
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the quotient of corresponding elements
    func divide(_ other: [[Double]]) -> [[Double]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot divide empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.divide(row2)
        }
    }
}

// MARK: - Matrix Arithmetic Operations (Float)

public extension Array where Element == [Float] {
    /// Element-wise addition of two matrices
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the sum of corresponding elements
    func add(_ other: [[Float]]) -> [[Float]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot add empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.add(row2)
        }
    }

    /// Element-wise subtraction of two matrices
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the difference of corresponding elements
    func subtract(_ other: [[Float]]) -> [[Float]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot subtract empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.subtract(row2)
        }
    }

    /// Element-wise multiplication of two matrices (Hadamard product)
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the product of corresponding elements
    func multiply(_ other: [[Float]]) -> [[Float]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot multiply empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.multiply(row2)
        }
    }

    /// Element-wise division of two matrices
    /// - Parameter other: The second matrix
    /// - Returns: A new matrix where each element is the quotient of corresponding elements
    func divide(_ other: [[Float]]) -> [[Float]] {
        precondition(self.count == other.count, "Matrices must have same number of rows")
        precondition(!self.isEmpty, "Cannot divide empty matrices")

        return zip(self, other).map { row1, row2 in
            row1.divide(row2)
        }
    }
}
