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

// MARK: - Basic Element-wise Operations

extension _Vector where Element: Numeric {
    
    /// Element-wise addition of two vectors
    static func add(_ lhs: _Vector<Element>, _ rhs: _Vector<Element>) -> _Vector<Element> {
        precondition(lhs.elements.count == rhs.elements.count, "Vectors must have the same dimension")
        
        var result = [Element]()
        for i in 0..<lhs.elements.count {
            result.append(lhs.elements[i] + rhs.elements[i])
        }
        return _Vector(elements: result)
    }
    
    /// Element-wise subtraction of two vectors
    static func subtract(_ lhs: _Vector<Element>, _ rhs: _Vector<Element>) -> _Vector<Element> {
        precondition(lhs.elements.count == rhs.elements.count, "Vectors must have the same dimension")
        
        var result = [Element]()
        for i in 0..<lhs.elements.count {
            result.append(lhs.elements[i] - rhs.elements[i])
        }
        return _Vector(elements: result)
    }
    
    /// Element-wise multiplication of two vectors
    static func multiply(_ lhs: _Vector<Element>, _ rhs: _Vector<Element>) -> _Vector<Element> {
        precondition(lhs.elements.count == rhs.elements.count, "Vectors must have the same dimension")
        
        var result = [Element]()
        for i in 0..<lhs.elements.count {
            result.append(lhs.elements[i] * rhs.elements[i])
        }
        return _Vector(elements: result)
    }
    
    
    /// Calculates the dot product of two vectors
    static func dot(_ lhs: _Vector<Element>, _ rhs: _Vector<Element>) -> Element {
        precondition(lhs.elements.count == rhs.elements.count, "Vectors must have the same dimension")
        
        var result = Element.zero
        for i in 0..<lhs.elements.count {
            result += lhs.elements[i] * rhs.elements[i]
        }
        return result
    }
    
    /// Matrix-vector multiplication
    static func matrixVectorTransform(_ matrix: [[Element]], _ vector: _Vector<Element>) -> _Vector<Element> {
        precondition(matrix.first?.count == vector.elements.count, "Matrix columns must match vector length")
        
        var resultElements = [Element]()
        for row in matrix {
            let rowVector = _Vector(elements: row)
            resultElements.append(_Vector.dot(rowVector, vector))
        }
        
        return _Vector(elements: resultElements)
    }
    
    /// Returns the transpose of a matrix
    static func transpose(_ matrix: [[Element]]) -> [[Element]] {
        guard !matrix.isEmpty, !matrix[0].isEmpty else { return [] }

        let rows = matrix.count
        let cols = matrix[0].count

        // Validate rectangular structure
        for row in matrix {
            precondition(row.count == cols,
                "Matrix has inconsistent row lengths — expected \(cols) columns, found \(row.count)")
        }

        // Create an empty result matrix with swapped dimensions
        var result = [[Element]]()
        for _ in 0..<cols {
            result.append([Element](repeating: .zero, count: rows))
        }

        // Fill the transposed matrix
        for i in 0..<rows {
            for j in 0..<cols {
                result[j][i] = matrix[i][j]
            }
        }

        return result
    }

    /// Matrix-matrix multiplication (A × B)
    /// Multiplies two matrices following standard matrix multiplication rules
    /// Dimensions: (n×k) × (k×m) → (n×m)
    static func matrixMatrixMultiply(_ lhs: [[Element]], _ rhs: [[Element]]) -> [[Element]] {
        // Validate matrices are not empty
        guard !lhs.isEmpty, !rhs.isEmpty else {
            preconditionFailure("Cannot multiply empty matrices")
        }

        guard let lhsFirstRow = lhs.first, !lhsFirstRow.isEmpty else {
            preconditionFailure("Left matrix is empty")
        }

        guard let rhsFirstRow = rhs.first, !rhsFirstRow.isEmpty else {
            preconditionFailure("Right matrix is empty")
        }

        let n = lhs.count           // Rows in A
        let k = lhsFirstRow.count   // Columns in A / Rows in B
        let m = rhsFirstRow.count   // Columns in B

        // Verify all rows have consistent dimensions
        for row in lhs {
            guard row.count == k else {
                preconditionFailure("Left matrix has inconsistent row lengths")
            }
        }

        for row in rhs {
            guard row.count == m else {
                preconditionFailure("Right matrix has inconsistent row lengths")
            }
        }

        // Check dimension compatibility: columns of A must equal rows of B
        guard k == rhs.count else {
            preconditionFailure("Matrix dimensions incompatible: (\(n)×\(k)) × (\(rhs.count)×\(m)). Columns of first matrix (\(k)) must equal rows of second matrix (\(rhs.count)).")
        }

        // Perform matrix multiplication
        // Result[i][j] = sum of (A[i][k] * B[k][j]) for all k
        var result = [[Element]]()

        for i in 0..<n {
            var row = [Element]()
            for j in 0..<m {
                var sum = Element.zero
                for kIdx in 0..<k {
                    sum += lhs[i][kIdx] * rhs[kIdx][j]
                }
                row.append(sum)
            }
            result.append(row)
        }

        return result
    }


}

// MARK: - Floating Point Operations

extension _Vector where Element: FloatingPoint {
    /// Element-wise division of two vectors
    static func divide(_ lhs: _Vector<Element>, _ rhs: _Vector<Element>) -> _Vector<Element> {
        precondition(lhs.elements.count == rhs.elements.count, "Vectors must have the same dimension")
        
        var result = [Element]()
        for i in 0..<lhs.elements.count {
            precondition(rhs.elements[i] != 0, "Division by zero")
            result.append(lhs.elements[i] / rhs.elements[i])
        }
        return _Vector(elements: result)
    }
    
    /// Calculates the magnitude (length) of a vector
    func magnitude() -> Element {
        var sumOfSquares = Element.zero
        for element in elements {
            sumOfSquares += element * element
        }
        return sumOfSquares.squareRoot()
    }
    
    /// Returns a normalized version of the vector (unit vector)
    /// Returns a zero vector if the input has zero magnitude
    func normalized() -> _Vector<Element> {
        let mag = magnitude()
        guard mag > 0 else {
            return _Vector(elements: Array(repeating: Element.zero, count: elements.count))
        }

        var normalizedElements = [Element]()
        for element in elements {
            normalizedElements.append(element / mag)
        }
        return _Vector(elements: normalizedElements)
    }

    /// Calculates the scalar projection of this vector onto another vector
    /// Returns 0 if the target has zero magnitude
    func scalarProjection(onto other: _Vector<Element>) -> Element {
        // A zero vector has no direction, so there is no axis to cast a
        // shadow along and the projected length is zero by convention —
        // matching normalized() and cosineOfAngle(with:).
        let magnitude = other.magnitude()
        guard magnitude > 0 else { return Element.zero }

        return _Vector.dot(self, other) / magnitude
    }
    
    /// Calculates the vector projection of this vector onto another vector
    /// Returns a zero vector if the target has zero magnitude
    func vectorProjection(onto other: _Vector<Element>) -> _Vector<Element> {
        // Guards on magnitude, matching scalarProjection, so both methods
        // draw the zero boundary at the same place. Guarding on
        // dot(other, other) here and magnitude there meant two conditions
        // to reason about for one question.
        let magnitude = other.magnitude()
        guard magnitude > 0 else {
            return _Vector(elements: Array(repeating: Element.zero, count: elements.count))
        }

        // The scalar multiple, as the scalar projection over the magnitude
        // a second time — equivalently dot(self, other) / dot(other, other),
        // reusing the guarded magnitude rather than re-forming that sum.
        let scalar = self.scalarProjection(onto: other) / magnitude

        // Create a new vector by scaling each component
        var projectedElements = [Element]()
        for element in other.elements {
            projectedElements.append(element * scalar)
        }
        
        return _Vector(elements: projectedElements)
    }
}
