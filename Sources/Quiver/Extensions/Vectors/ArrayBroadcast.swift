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

// MARK: - Broadcasting for scalar operations on 1D arrays

public extension Array where Element: Numeric {
    /// Broadcasts a scalar addition operation to each element
    /// - Parameter value: The scalar value to add to each element
    /// - Returns: A new array with the scalar added to each element
    func broadcast(adding value: Element) -> [Element] {
        return self.map { $0 + value }
    }

    /// Broadcasts a scalar multiplication operation to each element
    /// - Parameter value: The scalar value to multiply each element by
    /// - Returns: A new array with each element multiplied by the scalar
    func broadcast(multiplyingBy value: Element) -> [Element] {
        return self.map { $0 * value }
    }

    /// Broadcasts a scalar subtraction operation to each element
    /// - Parameter value: The scalar value to subtract from each element
    /// - Returns: A new array with the scalar subtracted from each element
    func broadcast(subtracting value: Element) -> [Element] {
        return self.map { $0 - value }
    }
}

// Add division only for floating point types
public extension Array where Element: FloatingPoint {
    /// Broadcasts a scalar division operation to each element
    /// - Parameter value: The scalar value to divide each element by
    /// - Returns: A new array with each element divided by the scalar
    func broadcast(dividingBy value: Element) -> [Element] {
        precondition(value != 0, "Cannot divide by zero")
        return self.map { $0 / value }
    }
}

// MARK: - Broadcasting for array operations on 2D arrays

public extension Array where Element: Collection, Element.Element: Numeric {
    /// Broadcasts a row vector across each row of the matrix
    /// - Parameter vector: The vector to add to each row
    /// - Returns: A new matrix with the vector added to each row
    func broadcast(addingToEachRow vector: [Element.Element]) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard let firstRow = matrix.first, firstRow.count == vector.count else {
            preconditionFailure("Row vector length must match matrix column count")
        }

        // Add the vector to each row
        return matrix.map { row in
            zip(row, vector).map { $0 + $1 }
        }
    }

    /// Broadcasts a column vector across each column of the matrix
    /// - Parameter vector: The vector to add to each column
    /// - Returns: A new matrix with the vector added to each column
    func broadcast(addingToEachColumn vector: [Element.Element]) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard matrix.count == vector.count else {
            preconditionFailure("Column vector length must match matrix row count")
        }

        // Add the corresponding vector element to each element in a row
        return matrix.enumerated().map { rowIndex, row in
            return row.map { $0 + vector[rowIndex] }
        }
    }

    /// Broadcasts a row vector multiplication across each row of the matrix
    /// - Parameter vector: The vector to multiply each row by
    /// - Returns: A new matrix with each row multiplied by the vector
    func broadcast(multiplyingEachRowBy vector: [Element.Element]) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard let firstRow = matrix.first, firstRow.count == vector.count else {
            preconditionFailure("Row vector length must match matrix column count")
        }

        // Multiply each row by the vector
        return matrix.map { row in
            zip(row, vector).map { $0 * $1 }
        }
    }

    /// Broadcasts a column vector multiplication across each column of the matrix
    /// - Parameter vector: The vector to multiply each column by
    /// - Returns: A new matrix with each column multiplied by the vector
    func broadcast(multiplyingEachColumnBy vector: [Element.Element]) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard matrix.count == vector.count else {
            preconditionFailure("Column vector length must match matrix row count")
        }

        // Multiply each element in a row by the corresponding vector element
        return matrix.enumerated().map { rowIndex, row in
            return row.map { $0 * vector[rowIndex] }
        }
    }
}

// MARK: - General broadcasting with closure

public extension Array where Element: Numeric {
    /// Broadcasts a custom operation with a scalar value to each element
    /// - Parameters:
    ///   - value: The scalar value to use in the operation
    ///   - operation: A closure that defines the operation to perform
    /// - Returns: A new array with the operation applied to each element
    func broadcast(with value: Element, operation: (Element, Element) -> Element) -> [Element] {
        return self.map { operation($0, value) }
    }
}

public extension Array where Element: Collection, Element.Element: Numeric {
    /// Broadcasts a custom row operation across the matrix
    /// - Parameters:
    ///   - vector: The vector to use in the operation
    ///   - operation: A closure that defines the operation to perform
    /// - Returns: A new matrix with the operation applied to each row
    func broadcast(withRowVector vector: [Element.Element],
                  operation: (Element.Element, Element.Element) -> Element.Element) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard let firstRow = matrix.first, firstRow.count == vector.count else {
            preconditionFailure("Row vector length must match matrix column count")
        }

        // Apply operation between each row and the vector
        return matrix.map { row in
            zip(row, vector).map { operation($0, $1) }
        }
    }

    /// Broadcasts a custom column operation across the matrix
    /// - Parameters:
    ///   - vector: The vector to use in the operation
    ///   - operation: A closure that defines the operation to perform
    /// - Returns: A new matrix with the operation applied to each column
    func broadcast(withColumnVector vector: [Element.Element],
                  operation: (Element.Element, Element.Element) -> Element.Element) -> [[Element.Element]] {
        // Convert self to [[Element.Element]] for easier manipulation
        let matrix = self.map { Swift.Array($0) }

        // Verify dimensions
        guard matrix.count == vector.count else {
            preconditionFailure("Column vector length must match matrix row count")
        }

        // Apply operation between each element and the corresponding vector element
        return matrix.enumerated().map { rowIndex, row in
            return row.map { operation($0, vector[rowIndex]) }
        }
    }
}

// MARK: - Scalar Broadcasting Operators for Vectors

public extension Array where Element: Numeric {
    /// Broadcasts scalar addition to each element of the vector
    /// - Parameters:
    ///   - lhs: The vector
    ///   - rhs: The scalar value to add
    /// - Returns: A new vector with the scalar added to each element
    static func + (lhs: [Element], rhs: Element) -> [Element] {
        return lhs.broadcast(adding: rhs)
    }

    /// Broadcasts scalar addition to each element of the vector (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to add
    ///   - rhs: The vector
    /// - Returns: A new vector with the scalar added to each element
    static func + (lhs: Element, rhs: [Element]) -> [Element] {
        return rhs + lhs
    }

    /// Broadcasts scalar subtraction from each element of the vector
    /// - Parameters:
    ///   - lhs: The vector
    ///   - rhs: The scalar value to subtract
    /// - Returns: A new vector with the scalar subtracted from each element
    static func - (lhs: [Element], rhs: Element) -> [Element] {
        return lhs.broadcast(subtracting: rhs)
    }

    /// Broadcasts scalar subtraction with reversed operands (scalar - vector)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The vector
    /// - Returns: A new vector with each element subtracted from the scalar
    static func - (lhs: Element, rhs: [Element]) -> [Element] {
        return rhs.map { lhs - $0 }
    }

    /// Broadcasts scalar multiplication to each element of the vector
    /// - Parameters:
    ///   - lhs: The vector
    ///   - rhs: The scalar value to multiply
    /// - Returns: A new vector with each element multiplied by the scalar
    static func * (lhs: [Element], rhs: Element) -> [Element] {
        return lhs.broadcast(multiplyingBy: rhs)
    }

    /// Broadcasts scalar multiplication to each element of the vector (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to multiply
    ///   - rhs: The vector
    /// - Returns: A new vector with each element multiplied by the scalar
    static func * (lhs: Element, rhs: [Element]) -> [Element] {
        return rhs * lhs
    }
}

public extension Array where Element: FloatingPoint {
    /// Broadcasts scalar division to each element of the vector
    /// - Parameters:
    ///   - lhs: The vector
    ///   - rhs: The scalar value to divide by
    /// - Returns: A new vector with each element divided by the scalar
    static func / (lhs: [Element], rhs: Element) -> [Element] {
        return lhs.broadcast(dividingBy: rhs)
    }

    /// Broadcasts scalar division with reversed operands (scalar / vector)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The vector
    /// - Returns: A new vector with the scalar divided by each element
    static func / (lhs: Element, rhs: [Element]) -> [Element] {
        precondition(!rhs.contains(0), "Cannot divide by zero")
        return rhs.map { lhs / $0 }
    }
}

// MARK: - Scalar Broadcasting Operators for Matrices (Double)

public extension Array where Element == [Double] {
    /// Broadcasts scalar addition to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to add
    /// - Returns: A new matrix with the scalar added to each element
    static func + (lhs: [[Double]], rhs: Double) -> [[Double]] {
        return lhs.map { $0 + rhs }  // Uses vector + scalar operator
    }

    /// Broadcasts scalar addition to each element of the matrix (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to add
    ///   - rhs: The matrix
    /// - Returns: A new matrix with the scalar added to each element
    static func + (lhs: Double, rhs: [[Double]]) -> [[Double]] {
        return rhs + lhs
    }

    /// Broadcasts scalar subtraction from each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to subtract
    /// - Returns: A new matrix with the scalar subtracted from each element
    static func - (lhs: [[Double]], rhs: Double) -> [[Double]] {
        return lhs.map { $0 - rhs }  // Uses vector - scalar operator
    }

    /// Broadcasts scalar subtraction with reversed operands (scalar - matrix)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The matrix
    /// - Returns: A new matrix with each element subtracted from the scalar
    static func - (lhs: Double, rhs: [[Double]]) -> [[Double]] {
        return rhs.map { row in row.map { lhs - $0 } }
    }

    /// Broadcasts scalar multiplication to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to multiply
    /// - Returns: A new matrix with each element multiplied by the scalar
    static func * (lhs: [[Double]], rhs: Double) -> [[Double]] {
        return lhs.map { $0 * rhs }  // Uses vector * scalar operator
    }

    /// Broadcasts scalar multiplication to each element of the matrix (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to multiply
    ///   - rhs: The matrix
    /// - Returns: A new matrix with each element multiplied by the scalar
    static func * (lhs: Double, rhs: [[Double]]) -> [[Double]] {
        return rhs * lhs
    }

    /// Broadcasts scalar division to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to divide by
    /// - Returns: A new matrix with each element divided by the scalar
    static func / (lhs: [[Double]], rhs: Double) -> [[Double]] {
        return lhs.map { $0 / rhs }  // Uses vector / scalar operator
    }

    /// Broadcasts scalar division with reversed operands (scalar / matrix)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The matrix
    /// - Returns: A new matrix with the scalar divided by each element
    static func / (lhs: Double, rhs: [[Double]]) -> [[Double]] {
        return rhs.map { row in row.map { lhs / $0 } }
    }
}

// MARK: - Scalar Broadcasting Operators for Matrices (Float)

public extension Array where Element == [Float] {
    /// Broadcasts scalar addition to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to add
    /// - Returns: A new matrix with the scalar added to each element
    static func + (lhs: [[Float]], rhs: Float) -> [[Float]] {
        return lhs.map { $0 + rhs }  // Uses vector + scalar operator
    }

    /// Broadcasts scalar addition to each element of the matrix (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to add
    ///   - rhs: The matrix
    /// - Returns: A new matrix with the scalar added to each element
    static func + (lhs: Float, rhs: [[Float]]) -> [[Float]] {
        return rhs + lhs
    }

    /// Broadcasts scalar subtraction from each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to subtract
    /// - Returns: A new matrix with the scalar subtracted from each element
    static func - (lhs: [[Float]], rhs: Float) -> [[Float]] {
        return lhs.map { $0 - rhs }  // Uses vector - scalar operator
    }

    /// Broadcasts scalar subtraction with reversed operands (scalar - matrix)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The matrix
    /// - Returns: A new matrix with each element subtracted from the scalar
    static func - (lhs: Float, rhs: [[Float]]) -> [[Float]] {
        return rhs.map { row in row.map { lhs - $0 } }
    }

    /// Broadcasts scalar multiplication to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to multiply
    /// - Returns: A new matrix with each element multiplied by the scalar
    static func * (lhs: [[Float]], rhs: Float) -> [[Float]] {
        return lhs.map { $0 * rhs }  // Uses vector * scalar operator
    }

    /// Broadcasts scalar multiplication to each element of the matrix (commutative)
    /// - Parameters:
    ///   - lhs: The scalar value to multiply
    ///   - rhs: The matrix
    /// - Returns: A new matrix with each element multiplied by the scalar
    static func * (lhs: Float, rhs: [[Float]]) -> [[Float]] {
        return rhs * lhs
    }

    /// Broadcasts scalar division to each element of the matrix
    /// - Parameters:
    ///   - lhs: The matrix
    ///   - rhs: The scalar value to divide by
    /// - Returns: A new matrix with each element divided by the scalar
    static func / (lhs: [[Float]], rhs: Float) -> [[Float]] {
        return lhs.map { $0 / rhs }  // Uses vector / scalar operator
    }

    /// Broadcasts scalar division with reversed operands (scalar / matrix)
    /// - Parameters:
    ///   - lhs: The scalar value
    ///   - rhs: The matrix
    /// - Returns: A new matrix with the scalar divided by each element
    static func / (lhs: Float, rhs: [[Float]]) -> [[Float]] {
        return rhs.map { row in row.map { lhs / $0 } }
    }
}
