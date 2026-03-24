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

// MARK: - Matrix Error Type

/// Errors thrown by matrix operations that can fail at runtime.
public enum MatrixError: Error, Equatable, CustomStringConvertible {
    /// The operation requires a square matrix but received a non-square one.
    case notSquare
    /// The matrix is singular (determinant = 0) and cannot be inverted.
    case singular

    public var description: String {
        switch self {
        case .notSquare:
            return "Matrix operation requires a square matrix"
        case .singular:
            return "Matrix is singular and cannot be inverted (determinant = 0)"
        }
    }
}

// MARK: - Log Determinant Result Type

/// Represents the sign and natural logarithm of a matrix determinant.
///
/// This type enables numerically stable determinant computations for large matrices
/// where the raw determinant value would overflow or underflow floating-point range.
///
/// The determinant can be reconstructed via the `value` property, which computes
/// `sign * exp(logAbsValue)`.
public struct LogDeterminant {

    /// The sign of the determinant: -1, 0, or 1
    public let sign: Double

    /// The natural logarithm of the absolute determinant value
    public let logAbsValue: Double

    /// Reconstructs the determinant value (sign times exp of log absolute value)
    public var value: Double {
        sign * Foundation.exp(logAbsValue)
    }
}

// MARK: - Standard Numeric Vector Operations

public extension Array where Element: Numeric {

    /// Calculates the dot product of two vectors.
    ///
    /// The dot product multiplies corresponding elements of two vectors and sums the results.
    /// It measures how much two vectors point in the same direction and is foundational to
    /// operations like cosine similarity, projections, and matrix transformations.
    ///
    /// Mathematical operation:
    /// ```
    /// dot(a, b) = a[0]×b[0] + a[1]×b[1] + ... + a[n]×b[n]
    /// ```
    ///
    /// Key properties:
    /// - **Positive result**: Vectors point in similar directions
    /// - **Zero result**: Vectors are perpendicular (orthogonal)
    /// - **Negative result**: Vectors point in opposing directions
    ///
    /// Example:
    /// ```swift
    /// let ratings = [5.0, 3.0, 4.0, 1.0]
    /// let weights = [0.4, 0.3, 0.2, 0.1]
    /// let score = ratings.dot(weights)
    /// // 3.8 (weighted average of ratings)
    ///
    /// let a = [1.0, 0.0]
    /// let b = [0.0, 1.0]
    /// let perpendicular = a.dot(b)
    /// // 0.0 (vectors are orthogonal)
    /// ```
    ///
    /// - Parameter other: The vector to compute the dot product with (must have the same number of elements)
    /// - Returns: The scalar dot product of the two vectors
    func dot(_ other: [Element]) -> Element {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: other)
        return _Vector.dot(v1, v2)
    }
    
    /// Transforms this vector using a matrix (matrix-vector multiplication).
    ///
    /// This method applies a linear transformation to the vector by multiplying it with a matrix.
    /// The transformation represents how the basis vectors of the coordinate system are modified,
    /// and the vector's coordinates are recalculated in the new coordinate system.
    ///
    /// Mathematical operation:
    /// ```
    /// For matrix M and vector v:
    /// result[i] = M[i][0]×v[0] + M[i][1]×v[1] + ... + M[i][n]×v[n]
    /// ```
    ///
    /// Conceptually, a vector `[x, y]` represents:
    /// ```
    /// x×i-hat + y×j-hat
    /// ```
    /// When the matrix transforms the basis vectors, the result is:
    /// ```
    /// x×(transformed i-hat) + y×(transformed j-hat)
    /// ```
    ///
    /// Common transformations:
    /// - **Rotation**: Spin the vector around the origin
    /// - **Scaling**: Stretch or compress along axes
    /// - **Reflection**: Mirror across an axis
    /// - **Shear**: Slant the coordinate system
    ///
    /// Example:
    /// ```swift
    /// // Rotate vector 90° counterclockwise
    /// let rotation = [
    ///     [0.0, -1.0],
    ///     [1.0,  0.0]
    /// ]
    ///
    /// let vector = [3.0, 4.0]
    /// let rotated = vector.transformedBy(rotation)
    /// // [-4.0, 3.0]
    ///
    /// // Math breakdown:
    /// // [3, 4] = 3×[1, 0] + 4×[0, 1]  (original basis)
    /// // After rotation:
    /// // = 3×[0, 1] + 4×[-1, 0]        (transformed basis)
    /// // = [0, 3] + [-4, 0]
    /// // = [-4, 3]
    /// ```
    ///
    /// - Parameter matrix: The transformation matrix (must have same number of columns as vector elements)
    /// - Returns: The transformed vector
    func transformedBy(_ matrix: [[Element]]) -> [Element] {
        // Check if the matrix dimensions are compatible with this vector
        guard !matrix.isEmpty,
              let firstRow = matrix.first,
              firstRow.count == self.count else {
            preconditionFailure("Invalid matrix dimensions or vector length")
        }
        
        // Ensure all rows have the same length
        for row in matrix {
            guard row.count == firstRow.count else {
                preconditionFailure("All matrix rows must have the same length")
            }
        }
        
        // Convert to internal _Vector and use the internal implementation
        let vectorObj = _Vector(elements: self)
        
        let result = _Vector.matrixVectorTransform(matrix, vectorObj)
        return result.elements
    }
}

public extension Array where Element: Collection, Element.Element: Numeric {
    /// Returns the transpose of a matrix
    ///
    /// This method is only available on 2D arrays (arrays of collections) where the inner elements
    /// are numeric types. For example, it works with [[1, 2], [3, 4]] or [[1.0, 2.0], [3.0, 4.0]],
    /// but not with [1, 2, 3, 4] or [["a", "b"], ["c", "d"]].
    ///
    /// The transpose operation converts rows into columns and columns into rows:
    /// - For a matrix with dimensions m×n, the result will have dimensions n×m
    /// - Each element at position (i,j) in the original matrix will be at position (j,i) in the transposed matrix
    ///
    /// - Complexity: O(*n*·*m*) where *n* is the number of rows and *m* is the
    ///   number of columns.
    /// - Returns: A new matrix where rows become columns and columns become rows
    func transpose() -> [[Element.Element]] {
        guard !self.isEmpty, !self[0].isEmpty else { return [] }

        // Convert to array of arrays for internal implementation
        let matrixArray = self.map { row -> [Element.Element] in
            return row.map { $0 }
        }

        // Call the internal implementation from _Vector
        return _Vector.transpose(matrixArray)
    }

    /// Returns the transpose of a matrix (convenience method matching Swift naming conventions)
    ///
    /// This is an alias for `transpose()` that follows Swift's convention of using past participle
    /// forms for methods that return transformed copies.
    ///
    /// - Returns: A new matrix where rows become columns and columns become rows
    func transposed() -> [[Element.Element]] {
        return self.transpose()
    }

    /// Extracts a column from a matrix at the specified index
    ///
    /// This method provides an intuitive way to extract vertical slices from matrices,
    /// which is otherwise awkward in Swift. For example:
    ///
    /// ```swift
    /// let matrix = [[1, 2, 3],
    ///               [4, 5, 6],
    ///               [7, 8, 9]]
    /// let secondColumn = matrix.column(at: 1)  // [2, 5, 8]
    /// ```
    ///
    /// - Parameter index: The column index to extract
    /// - Returns: An array containing all elements from the specified column
    func column(at index: Element.Index) -> [Element.Element] {
        return self.map { $0[index] }
    }

    /// Transforms a vector by this matrix (matrix-vector multiplication)
    ///
    /// This method provides a more intuitive API for matrix-vector multiplication where
    /// the matrix acts on the vector, which matches mathematical notation: **Mv = w**
    ///
    /// For example, to rotate a 2D vector 90 degrees counterclockwise:
    /// ```swift
    /// let rotationMatrix = [[0.0, -1.0],
    ///                       [1.0,  0.0]]
    /// let vector = [1.0, 0.0]
    /// let rotated = rotationMatrix.transform(vector)  // [0.0, 1.0]
    /// ```
    ///
    /// - Parameter vector: The vector to transform
    /// - Returns: The transformed vector
    func transform(_ vector: [Element.Element]) -> [Element.Element] {
        // Convert self (which is [Element] where Element: Collection) to [[Element.Element]]
        let matrixArray = self.map { row -> [Element.Element] in
            return row.map { $0 }
        }
        return vector.transformedBy(matrixArray)
    }

    /// Multiplies this matrix by another matrix to compose transformations.
    ///
    /// Matrix multiplication combines two transformations into a single operation. The result
    /// represents applying the second transformation first, then applying the first transformation.
    /// This operation is **non-commutative**: `A × B ≠ B × A` in general.
    ///
    /// Mathematical operation for matrices A (n×k) and B (k×m):
    /// ```
    /// C[i][j] = A[i][0]×B[0][j] + A[i][1]×B[1][j] + ... + A[i][k]×B[k][j]
    /// ```
    ///
    /// Transformation composition means:
    /// ```
    /// v.transformedBy(A.multiplyMatrix(B))
    /// ```
    /// is equivalent to:
    /// ```
    /// v.transformedBy(B).transformedBy(A)  // B applied first, then A
    /// ```
    ///
    /// Order matters:
    /// - **Right-to-left application**: `A.multiplyMatrix(B)` means "apply B first, then A"
    /// - **Scale then rotate ≠ Rotate then scale**: Different orders produce different results
    ///
    /// Common patterns:
    /// - **Transformation pipelines**: Object → World → Camera → Screen
    /// - **Animation systems**: Interpolate between transformation matrices
    /// - **Caching compositions**: Compose once, apply to many vectors efficiently
    ///
    /// Example:
    /// ```swift
    /// // Rotate 90° counterclockwise
    /// let rotation = [
    ///     [0.0, -1.0],
    ///     [1.0,  0.0]
    /// ]
    ///
    /// // Scale 2× horizontally, 3× vertically
    /// let scaling = [Double].diag([2.0, 3.0])
    ///
    /// // Compose: rotate first, then scale
    /// let combined = scaling.multiplyMatrix(rotation)
    ///
    /// // Apply to vector [1, 0]
    /// let point = [1.0, 0.0]
    /// let result = point.transformedBy(combined)
    /// // [0.0, 3.0]
    ///
    /// // Math breakdown:
    /// // Rotation: [1, 0] → [0, 1]
    /// // Scaling:  [0, 1] → [0, 3]
    ///
    /// // Order matters:
    /// let reversed = rotation.multiplyMatrix(scaling)
    /// let different = point.transformedBy(reversed)
    /// // [0.0, 2.0] - different result!
    ///
    /// // Scaling:  [1, 0] → [2, 0]
    /// // Rotation: [2, 0] → [0, 2]
    /// ```
    ///
    /// - Complexity: O(*n*·*m*·*p*) for an (*n*×*m*) times (*m*×*p*) multiplication.
    ///   Performs well for matrices up to a few hundred rows and columns.
    /// - Parameter other: The matrix to multiply with (must have compatible dimensions: this.columns == other.rows)
    /// - Returns: The resulting composed transformation matrix
    func multiplyMatrix(_ other: [[Element.Element]]) -> [[Element.Element]] {
        // Convert self to [[Element.Element]]
        let lhsMatrix = self.map { row -> [Element.Element] in
            return row.map { $0 }
        }

        return _Vector<Element.Element>.matrixMatrixMultiply(lhsMatrix, other)
    }
}

// MARK: - Matrix Operations (FloatingPoint)

public extension Array where Element: Collection, Element.Element: FloatingPoint {

    /// Returns the determinant of a square matrix.
    ///
    /// The determinant is a scalar value that provides information about the matrix:
    /// - det = 0: Matrix is singular (not invertible)
    /// - det ≠ 0: Matrix is invertible
    ///
    /// Example:
    /// ```swift
    /// let matrix = [[4.0, 3.0],
    ///               [6.0, 3.0]]
    /// let det = matrix.determinant  // -6.0
    /// ```
    ///
    /// - Complexity: O(*n*³) where *n* is the matrix dimension. Performs well
    ///   for matrices up to a few hundred rows.
    /// - Returns: The determinant value
    var determinant: Element.Element {
        let matrix = self.map { $0.map { $0 } }
        precondition(!matrix.isEmpty && matrix.count == matrix[0].count,
                     "Determinant requires a square matrix")

        let n = matrix.count

        // Base cases
        if n == 1 {
            return matrix[0][0]
        }
        if n == 2 {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        }

        // LU decomposition for larger matrices
        var A = matrix
        var det = Element.Element(1)

        for i in 0..<n {
            // Find pivot
            var maxRow = i
            for k in (i+1)..<n {
                if abs(A[k][i]) > abs(A[maxRow][i]) {
                    maxRow = k
                }
            }

            // Check for singular matrix (nearly zero pivot)
            let epsilon = Element.Element.ulpOfOne * 1000
            if abs(A[maxRow][i]) < epsilon {
                return 0
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                det = -det
            }

            det *= A[i][i]

            // Eliminate column
            for k in (i+1)..<n {
                let factor = A[k][i] / A[i][i]
                for j in (i+1)..<n {
                    A[k][j] -= factor * A[i][j]
                }
            }
        }

        return det
    }

    /// Returns the inverse of a square matrix.
    ///
    /// The inverse matrix A⁻¹ satisfies: A × A⁻¹ = I (identity matrix)
    /// Only non-singular matrices (determinant ≠ 0) have an inverse.
    ///
    /// Example:
    /// ```swift
    /// let matrix = [[4.0, 7.0],
    ///               [2.0, 6.0]]
    /// let inverse = try matrix.inverted()
    /// // [[0.6, -0.7], [-0.2, 0.4]]
    /// ```
    ///
    /// - Complexity: O(*n*³) where *n* is the matrix dimension. Performs well
    ///   for matrices up to a few hundred rows. Check ``conditionNumber``
    ///   first if numerical stability is a concern.
    /// - Returns: The inverted matrix
    /// - Throws: `MatrixError.notSquare` if the matrix is not square,
    ///           `MatrixError.singular` if the matrix is singular.
    func inverted() throws -> [[Element.Element]] {
        let matrix = self.map { $0.map { $0 } }
        guard !matrix.isEmpty, matrix.count == matrix[0].count else {
            throw MatrixError.notSquare
        }

        let n = matrix.count
        var A = matrix
        var inv = [[Element.Element]](repeating: [Element.Element](repeating: 0, count: n), count: n)

        // Initialize inv as identity matrix
        for i in 0..<n {
            inv[i][i] = 1
        }

        // Gaussian elimination with partial pivoting
        for i in 0..<n {
            // Find pivot
            var maxRow = i
            for k in (i+1)..<n {
                if abs(A[k][i]) > abs(A[maxRow][i]) {
                    maxRow = k
                }
            }

            // Check for singular matrix (nearly zero pivot)
            let epsilon = Element.Element.ulpOfOne * 1000
            if abs(A[maxRow][i]) < epsilon {
                throw MatrixError.singular
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                inv.swapAt(i, maxRow)
            }

            // Scale pivot row
            let pivot = A[i][i]
            for j in 0..<n {
                A[i][j] /= pivot
                inv[i][j] /= pivot
            }

            // Eliminate column
            for k in 0..<n where k != i {
                let factor = A[k][i]
                for j in 0..<n {
                    A[k][j] -= factor * A[i][j]
                    inv[k][j] -= factor * inv[i][j]
                }
            }
        }

        return inv
    }

}

// MARK: - Matrix Diagnostics (Double)

public extension Array where Element == [Double] {

    /// Returns the sign and natural logarithm of the absolute determinant.
    ///
    /// For large matrices, the determinant can overflow or underflow Double's range.
    /// This method works in log-space to avoid that problem, returning the sign
    /// separately from the logarithm of the absolute value.
    ///
    /// The determinant can be reconstructed as `sign * exp(logAbsValue)`, but for
    /// many applications (comparing determinants, checking singularity thresholds)
    /// the log form is more useful directly.
    ///
    /// Example:
    /// ```swift
    /// let matrix = [[4.0, 3.0],
    ///               [6.0, 3.0]]
    /// let ld = matrix.logDeterminant
    /// // ld.sign == -1.0
    /// // ld.logAbsValue == log(6.0) ≈ 1.7918
    /// // ld.value == -6.0 (reconstructed)
    /// ```
    ///
    /// - Complexity: O(*n*³) where *n* is the matrix dimension. Preferred
    ///   over ``determinant`` for large matrices where the raw value may
    ///   overflow or underflow.
    /// - Returns: A `LogDeterminant` containing the sign (-1, 0, or 1) and log of the absolute determinant
    var logDeterminant: LogDeterminant {
        precondition(!self.isEmpty && self.count == self[0].count,
                     "Log determinant requires a square matrix")

        let n = self.count

        // Base cases
        if n == 1 {
            let val = self[0][0]
            if val == 0 {
                return LogDeterminant(sign: 0, logAbsValue: -.infinity)
            }
            return LogDeterminant(sign: val < 0 ? -1.0 : 1.0, logAbsValue: Foundation.log(abs(val)))
        }
        if n == 2 {
            let val = self[0][0] * self[1][1] - self[0][1] * self[1][0]
            if val == 0 {
                return LogDeterminant(sign: 0, logAbsValue: -.infinity)
            }
            return LogDeterminant(sign: val < 0 ? -1.0 : 1.0, logAbsValue: Foundation.log(abs(val)))
        }

        // LU decomposition accumulating in log-space
        var A = self
        var sign: Double = 1
        var logAbsDet: Double = 0

        for i in 0..<n {
            // Find pivot
            var maxRow = i
            for k in (i+1)..<n {
                if abs(A[k][i]) > abs(A[maxRow][i]) {
                    maxRow = k
                }
            }

            // Check for singular matrix
            let epsilon = Double.ulpOfOne * 1000
            if abs(A[maxRow][i]) < epsilon {
                return LogDeterminant(sign: 0, logAbsValue: -.infinity)
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                sign = -sign
            }

            let pivot = A[i][i]
            if pivot < 0 {
                sign = -sign
            }
            logAbsDet += Foundation.log(abs(pivot))

            // Eliminate column
            for k in (i+1)..<n {
                let factor = A[k][i] / A[i][i]
                for j in (i+1)..<n {
                    A[k][j] -= factor * A[i][j]
                }
            }
        }

        return LogDeterminant(sign: sign, logAbsValue: logAbsDet)
    }

    /// Returns the condition number of the matrix using the 1-norm.
    ///
    /// The condition number measures how sensitive the matrix inverse is to
    /// numerical perturbations. A large condition number indicates an
    /// ill-conditioned matrix where small input changes produce large output changes.
    ///
    /// Interpretation:
    /// - Near 1.0: Well-conditioned, safe to invert
    /// - 10³–10⁶: Moderate conditioning, results may lose precision
    /// - Above 10⁶: Ill-conditioned, inversion results are unreliable
    /// - Infinity: Singular matrix, no inverse exists
    ///
    /// Example:
    /// ```swift
    /// let identity = [[1.0, 0.0],
    ///                 [0.0, 1.0]]
    /// identity.conditionNumber  // 1.0 (perfectly conditioned)
    ///
    /// let illConditioned = [[1.0, 1.0],
    ///                       [1.0, 1.0000001]]
    /// illConditioned.conditionNumber  // Very large (near-singular)
    /// ```
    ///
    /// - Complexity: O(*n*³) where *n* is the matrix dimension. Computes the
    ///   matrix inverse internally. Performs well for matrices up to a few
    ///   hundred rows.
    /// - Returns: The 1-norm condition number, or `.infinity` for singular matrices
    var conditionNumber: Double {
        precondition(!self.isEmpty && self.count == self[0].count,
                     "Condition number requires a square matrix")

        let n = self.count

        // Compute 1-norm of the original matrix (max absolute column sum)
        let norm1A = _matrixNorm1(self, size: n)

        // Attempt inversion without fatalError
        guard let inv = _tryInvert(self, size: n) else {
            return .infinity
        }

        // Compute 1-norm of the inverse
        let norm1Inv = _matrixNorm1(inv, size: n)

        return norm1A * norm1Inv
    }

    // Computes the 1-norm of a square matrix (max absolute column sum)
    private func _matrixNorm1(_ matrix: [[Double]], size n: Int) -> Double {
        var maxColSum: Double = 0
        for j in 0..<n {
            var colSum: Double = 0
            for i in 0..<n {
                colSum += abs(matrix[i][j])
            }
            if colSum > maxColSum {
                maxColSum = colSum
            }
        }
        return maxColSum
    }

    // Non-fatal matrix inversion (returns nil for singular matrices)
    private func _tryInvert(_ matrix: [[Double]], size n: Int) -> [[Double]]? {
        var A = matrix
        var inv = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)

        for i in 0..<n {
            inv[i][i] = 1
        }

        for i in 0..<n {
            var maxRow = i
            for k in (i+1)..<n {
                if abs(A[k][i]) > abs(A[maxRow][i]) {
                    maxRow = k
                }
            }

            let epsilon = Double.ulpOfOne * 1000
            if abs(A[maxRow][i]) < epsilon {
                return nil
            }

            if maxRow != i {
                A.swapAt(i, maxRow)
                inv.swapAt(i, maxRow)
            }

            let pivot = A[i][i]
            for j in 0..<n {
                A[i][j] /= pivot
                inv[i][j] /= pivot
            }

            for k in 0..<n where k != i {
                let factor = A[k][i]
                for j in 0..<n {
                    A[k][j] -= factor * A[i][j]
                    inv[k][j] -= factor * inv[i][j]
                }
            }
        }

        return inv
    }
}


// MARK: - FloatingPoint Vector Operations

public extension Array where Element: FloatingPoint {
    /// Calculates the magnitude (length) of the vector.
    ///
    /// The magnitude represents the Euclidean length of the vector, computed as the
    /// square root of the sum of squared elements. It quantifies the "size" of a vector
    /// independent of its direction.
    ///
    /// Mathematical operation:
    /// ```
    /// magnitude = √(x² + y² + z² + ...)
    /// ```
    ///
    /// Common uses:
    /// - **Normalization**: Divide by magnitude to get a unit vector
    /// - **Distance**: The magnitude of `a - b` gives the Euclidean distance between two points
    /// - **Similarity**: Used in the denominator of cosine similarity
    ///
    /// Example:
    /// ```swift
    /// let velocity = [3.0, 4.0]
    /// let speed = velocity.magnitude
    /// // 5.0 (classic 3-4-5 right triangle)
    ///
    /// let features = [1.0, 2.0, 2.0]
    /// let length = features.magnitude
    /// // 3.0
    /// ```
    var magnitude: Element {
        let v = _Vector(elements: self)
        return v.magnitude()
    }
    
    /// Returns a normalized version of the vector (unit vector).
    ///
    /// A unit vector preserves direction but has a magnitude of 1.0. Normalization
    /// is essential for cosine similarity, direction comparisons, and any operation
    /// where only orientation matters.
    ///
    /// ```swift
    /// let v = [3.0, 4.0]
    /// v.normalized           // [0.6, 0.8]
    /// v.normalized.magnitude // 1.0
    /// ```
    ///
    /// - Returns: A new array with the same direction and magnitude 1.0, or a zero vector if the original has zero magnitude
    var normalized: [Element] {
        let v = _Vector(elements: self)
        return v.normalized().elements
    }
    
    /// Calculates the Euclidean distance between two points or vectors.
    ///
    /// The distance is computed as the magnitude of the difference vector,
    /// equivalent to `(self - other).magnitude`. This operation powers
    /// nearest-neighbor search and cluster assignment in Quiver's ML models.
    ///
    /// ```swift
    /// let a = [1.0, 2.0]
    /// let b = [4.0, 6.0]
    /// a.distance(to: b)  // 5.0
    /// ```
    ///
    /// - Parameter other: The vector to measure distance to (must have the same number of elements)
    /// - Returns: The Euclidean distance between the two vectors
    func distance(to other: [Element]) -> Element {
        return self.subtract(other).magnitude
    }
        
    /// Returns the cosine of the angle between two vectors.
    ///
    /// Cosine similarity measures how closely two vectors align in direction, regardless of
    /// their magnitude. Values range from -1 (opposite directions) through 0 (perpendicular)
    /// to 1 (same direction). This makes it ideal for comparing items represented as vectors,
    /// such as documents, user preferences, or feature embeddings.
    ///
    /// Mathematical operation:
    /// ```
    /// cosine(a, b) = dot(a, b) / (|a| × |b|)
    /// ```
    ///
    /// Interpretation:
    /// - **1.0**: Vectors point in the same direction (identical orientation)
    /// - **0.0**: Vectors are perpendicular (no similarity)
    /// - **-1.0**: Vectors point in opposite directions
    ///
    /// Example:
    /// ```swift
    /// let userA = [5.0, 1.0, 4.0]  // action fan
    /// let userB = [4.0, 2.0, 5.0]  // similar taste
    /// let userC = [1.0, 5.0, 1.0]  // drama fan
    ///
    /// let similarTaste = userA.cosineOfAngle(with: userB)
    /// // 0.965 (very similar preferences)
    ///
    /// let differentTaste = userA.cosineOfAngle(with: userC)
    /// // 0.497 (low similarity)
    /// ```
    ///
    /// Returns 0.0 if either vector has zero magnitude.
    ///
    /// - Parameter other: The vector to compare against
    /// - Returns: The cosine of the angle between the two vectors, in the range [-1, 1]
    func cosineOfAngle(with other: [Element]) -> Element {
        let dotProduct = self.dot(other)
        let magnitudeProduct = self.magnitude * other.magnitude
        guard magnitudeProduct > 0 else { return Element.zero }
        return dotProduct / magnitudeProduct
    }

    /// Calculates the scalar projection of this vector onto another vector.
    ///
    /// The scalar projection represents the signed length of the shadow cast by this vector
    /// onto the direction of the other vector. A positive value means the vectors point
    /// in the same general direction; a negative value means they point in opposite directions.
    ///
    /// ```swift
    /// let v = [3.0, 4.0]
    /// let axis = [1.0, 0.0]
    /// v.scalarProjection(onto: axis)  // 3.0
    /// ```
    ///
    /// - Parameter vector: The vector to project onto (must not be a zero vector)
    /// - Returns: The scalar length of the projection along the target vector's direction
    func scalarProjection(onto vector: [Element]) -> Element {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: vector)
        return v1.scalarProjection(onto: v2)
    }
    
    /// Calculates the vector projection of this vector onto another vector.
    ///
    /// The vector projection is the component of this vector that lies along
    /// the direction of the target vector. Together with `orthogonalComponent(to:)`,
    /// it decomposes this vector into parallel and perpendicular parts.
    ///
    /// ```swift
    /// let v = [3.0, 4.0]
    /// let axis = [1.0, 0.0]
    /// v.vectorProjection(onto: axis)  // [3.0, 0.0]
    /// ```
    ///
    /// - Parameter vector: The vector to project onto (must not be a zero vector)
    /// - Returns: A new vector pointing in the direction of the target with the projected magnitude
    func vectorProjection(onto vector: [Element]) -> [Element] {
        let v1 = _Vector(elements: self)
        let v2 = _Vector(elements: vector)
        return v1.vectorProjection(onto: v2).elements
    }
    
    /// Returns the component of this vector that is perpendicular to another vector.
    ///
    /// The orthogonal component is computed as `self - vectorProjection(onto: vector)`.
    /// Together with `vectorProjection(onto:)`, it decomposes this vector into
    /// parallel and perpendicular parts relative to the reference vector.
    ///
    /// ```swift
    /// let v = [3.0, 4.0]
    /// let axis = [1.0, 0.0]
    /// v.orthogonalComponent(to: axis)  // [0.0, 4.0]
    /// ```
    ///
    /// - Parameter vector: The reference vector to measure perpendicularity against (must not be a zero vector)
    /// - Returns: A new vector perpendicular to the reference vector
    func orthogonalComponent(to vector: [Element]) -> [Element] {
        let projection = self.vectorProjection(onto: vector)
        return self.subtract(projection)
    }
}

// MARK: - Double Vector Operations

public extension Array where Element == [Double] {

    /// Returns whether all vectors in the collection have the same dimension count.
    /// Used to validate that a collection of vectors can be used together in mathematical operations.
    func areValidVectorDimensions() -> Bool {
        guard let firstCount = self.first?.count else {
            return false
        }
        
        return self.allSatisfy { $0.count == firstCount }
    }
    
    /// Calculates the element-wise average of a collection of vectors.
    ///
    /// Each element in the result is the mean of the corresponding elements across all vectors.
    /// This is commonly used to compute centroids for clustering, average word embeddings
    /// into document vectors, and aggregate feature vectors.
    ///
    /// ```swift
    /// let wordVectors = [
    ///     [0.8, 0.2, 0.1],  // "running"
    ///     [0.7, 0.3, 0.2],  // "athletic"
    ///     [0.6, 0.1, 0.3]   // "shoes"
    /// ]
    /// wordVectors.averaged()  // [0.7, 0.2, 0.2]
    /// ```
    ///
    /// - Returns: A vector where each element is the mean across all input vectors, or `nil` if the array is empty or vectors have inconsistent dimensions
    func averaged() -> [Double]? {
        // Return nil if no vectors to average
        guard !self.isEmpty else { return nil }
        
        // Ensure all vectors have consistent dimensions
        guard self.areValidVectorDimensions() else { return nil }
        
        // Initialize sum vector with matching dimensions
        let dimensions = self[0].count
        var sum = [Double].zeros(dimensions)
        
        // Sum all vectors element-wise
        for vector in self {
            sum = sum.add(vector)
        }
        
        // Divide by count to get average
        return sum.broadcast(dividingBy: Double(self.count))
    }
    
    /// Calculates cosine similarity between each vector in the array and a target vector.
    ///
    /// Each score measures directional alignment: 1.0 means identical orientation,
    /// 0.0 means perpendicular, and -1.0 means opposite directions.
    ///
    /// - Parameter target: The reference vector to compare each vector against
    /// - Returns: An array of similarity scores in the range [-1, 1], one per vector
    func cosineSimilarities(to target: [Double]) -> [Double] {
        return self.map { $0.cosineOfAngle(with: target) }
    }

    /// Find pairs of vectors with similarity above the specified threshold.
    ///
    /// Computes pairwise cosine similarities and returns all pairs that meet or exceed
    /// the threshold. Useful for duplicate detection, near-neighbor search, and identifying
    /// similar items in datasets.
    ///
    /// Example:
    /// ```swift
    /// let documents = [
    ///     [0.8, 0.6, 0.9],
    ///     [0.8, 0.6, 0.9],  // Duplicate of first
    ///     [0.1, 0.2, 0.1]
    /// ]
    /// let duplicates = documents.findDuplicates(threshold: 0.95)
    /// // Returns: [(index1: 0, index2: 1, similarity: 1.0)]
    /// ```
    ///
    /// - Complexity: O(*n*²) where *n* is the number of vectors. Performs well
    ///   for collections up to low thousands. For larger datasets, consider
    ///   partitioning by category first, then comparing within each partition.
    /// - Parameter threshold: Minimum cosine similarity (0.0 to 1.0). Default is 0.95.
    /// - Returns: Array of tuples containing pair indices and similarity scores, sorted by similarity (highest first)
    func findDuplicates(threshold: Double = 0.95) -> [(index1: Int, index2: Int, similarity: Double)] {
        let n = self.count
        guard n > 1 else { return [] }

        // Performance: This inlines the cosine similarity formula rather than calling
        // cosineOfAngle(with:) in the inner loop. With n² pair comparisons, the _Vector
        // wrapper allocation overhead per call becomes the dominant cost. Precomputing
        // magnitudes avoids n² redundant magnitude calculations. The math is identical
        // to cosineOfAngle(with:): dot(a,b) / (|a| * |b|).
        let magnitudes = self.map { vec -> Double in
            var sum = 0.0
            for v in vec { sum += v * v }
            return Foundation.sqrt(sum)
        }

        var results = [(index1: Int, index2: Int, similarity: Double)]()
        for i in 0..<n {
            let magI = magnitudes[i]
            guard magI > 0 else { continue }
            let vecI = self[i]
            for j in (i + 1)..<n {
                let magJ = magnitudes[j]
                guard magJ > 0 else { continue }
                var dot = 0.0
                for d in 0..<vecI.count { dot += vecI[d] * self[j][d] }
                let similarity = dot / (magI * magJ)
                if similarity >= threshold {
                    results.append((i, j, similarity))
                }
            }
        }
        return results.sorted { $0.similarity > $1.similarity }
    }

    /// Calculate average pairwise similarity as a measure of cluster cohesion.
    ///
    /// Computes the mean cosine similarity between all pairs of vectors in the collection.
    /// Higher values indicate tighter, more homogeneous clusters. This metric is useful
    /// for validating clustering quality and measuring group coherence.
    ///
    /// Example:
    /// ```swift
    /// let cluster = [
    ///     [0.8, 0.7, 0.9],
    ///     [0.7, 0.8, 0.8],
    ///     [0.9, 0.6, 0.9]
    /// ]
    /// let cohesion = cluster.clusterCohesion()
    /// // Returns value between 0.0 (unrelated) and 1.0 (identical)
    /// ```
    ///
    /// - Complexity: O(*n*²) where *n* is the number of vectors. Performs well
    ///   for clusters up to low thousands. Use on individual cluster groups
    ///   rather than the full dataset.
    /// - Returns: Average pairwise similarity (0.0 to 1.0), or 0.0 if fewer than 2 vectors
    func clusterCohesion() -> Double {
        let n = self.count
        guard n > 1 else { return 0.0 }

        // Performance: Inlines cosine similarity and accumulates directly rather than
        // building an intermediate array of all pairwise results. Same math as
        // cosineOfAngle(with:) — see findDuplicates comment for rationale.
        let magnitudes = self.map { vec -> Double in
            var sum = 0.0
            for v in vec { sum += v * v }
            return Foundation.sqrt(sum)
        }

        var totalSimilarity = 0.0
        var pairCount = 0
        for i in 0..<n {
            let magI = magnitudes[i]
            guard magI > 0 else { continue }
            let vecI = self[i]
            for j in (i + 1)..<n {
                let magJ = magnitudes[j]
                guard magJ > 0 else { continue }
                var dot = 0.0
                for d in 0..<vecI.count { dot += vecI[d] * self[j][d] }
                totalSimilarity += dot / (magI * magJ)
                pairCount += 1
            }
        }

        return pairCount > 0 ? totalSimilarity / Double(pairCount) : 0.0
    }
}

// MARK: - Array Ranking Operations

public extension Array where Element == Double {

    /// Returns the indices and values of the top K highest elements.
    ///
    /// This method sorts the array in descending order and returns the indices and values
    /// of the top K elements. Commonly used in similarity search, recommendation systems,
    /// and ranking operations.
    ///
    /// Example:
    /// ```swift
    /// let scores = [0.3, 0.9, 0.1, 0.7, 0.5]
    /// let top3 = scores.topIndices(k: 3)
    /// // Returns: [(index: 1, score: 0.9), (index: 3, score: 0.7), (index: 4, score: 0.5)]
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    /// - Parameter k: Number of top elements to return
    /// - Returns: Array of tuples containing index and score, sorted by score (highest first)
    func topIndices(k: Int) -> [(index: Int, score: Double)] {
        return self.enumerated()
            .map { (index: $0.offset, score: $0.element) }
            .sorted { $0.score > $1.score }
            .prefix(k)
            .map { $0 }
    }

    /// Returns the top K highest scores with corresponding labels
    ///
    /// This convenience method combines top-K selection with label mapping, eliminating the need
    /// to manually map indices back to labels. Particularly useful for word prediction, recommendation
    /// systems, and any scenario where you need both the score and associated label.
    ///
    /// Example:
    /// ```swift
    /// let scores = [0.3, 0.9, 0.1, 0.7]
    /// let words = ["the", "cat", "dog", "sat"]
    ///
    /// let predictions = scores.topIndices(k: 2, labels: words)
    /// // Returns: [(label: "cat", score: 0.9), (label: "sat", score: 0.7)]
    /// ```
    ///
    /// - Parameters:
    ///   - k: Number of top elements to return
    ///   - labels: Array of labels corresponding to each score
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    /// - Returns: Array of tuples containing label and score, sorted by score (highest first)
    func topIndices<T>(k: Int, labels: [T]) -> [(label: T, score: Double)] {
        precondition(self.count == labels.count, "Scores and labels must have the same count")

        return self.topIndices(k: k).map { result in
            (label: labels[result.index], score: result.score)
        }
    }

    /// Returns the indices that would sort the array in ascending order.
    ///
    /// This method provides functionality equivalent to NumPy's `argsort()`,
    /// returning the indices of elements in sorted order rather than the sorted elements themselves.
    /// This is useful when you need to maintain a mapping between sorted and original positions.
    ///
    /// Example:
    /// ```swift
    /// let values = [40.0, 10.0, 30.0, 20.0]
    /// let indices = values.sortedIndices()
    /// // [1, 3, 2, 0] - indices that would sort the array
    ///
    /// // Use indices to access elements in sorted order
    /// let sorted = indices.map { values[$0] }
    /// // [10.0, 20.0, 30.0, 40.0]
    /// ```
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the number of elements.
    /// - Returns: Array of indices representing the sorted order
    func sortedIndices() -> [Int] {
        return self.enumerated()
            .sorted { $0.element < $1.element }
            .map { $0.offset }
    }

}

// MARK: - Float Vector Operations

public extension Array where Element == [Float] {
    /// Calculates cosine similarity between each vector in the array and a target vector.
    ///
    /// Each score measures directional alignment: 1.0 means identical orientation,
    /// 0.0 means perpendicular, and -1.0 means opposite directions.
    ///
    /// - Parameter target: The reference vector to compare each vector against
    /// - Returns: An array of similarity scores in the range [-1, 1], one per vector
    func cosineSimilarities(to target: [Float]) -> [Float] {
        return self.map { $0.cosineOfAngle(with: target) }
    }
}

