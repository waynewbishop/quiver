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

// MARK: - 1D Reshape (Vector → Matrix)

public extension Array where Element: Numeric {

    /// Reshapes a one-dimensional array into a two-dimensional matrix with the specified
    /// dimensions using row-major order.
    ///
    /// Elements fill the matrix row by row from left to right. The total number of elements
    /// must equal `rows × columns`. This is fundamental to data preparation workflows where
    /// flat data streams need to be organized into structured matrices for analysis.
    ///
    /// Row-major order means elements are read sequentially and placed left-to-right across
    /// each row before advancing to the next row:
    /// ```
    /// [1, 2, 3, 4, 5, 6]  →  [[1, 2, 3],
    ///                          [4, 5, 6]]
    /// ```
    ///
    /// Example:
    /// ```swift
    /// // Generate a sequence and organize into a matrix
    /// let values = [Double].arange(1, 7)  // [1, 2, 3, 4, 5, 6]
    /// let matrix = values.reshaped(rows: 2, columns: 3)
    /// // [[1.0, 2.0, 3.0],
    /// //  [4.0, 5.0, 6.0]]
    ///
    /// // Create column and row vectors from a flat array
    /// let data = [10.0, 20.0, 30.0]
    /// let columnVector = data.reshaped(rows: 3, columns: 1)  // [[10], [20], [30]]
    /// let rowVector = data.reshaped(rows: 1, columns: 3)     // [[10, 20, 30]]
    /// ```
    ///
    /// - Parameters:
    ///   - rows: The number of rows in the resulting matrix (must be positive).
    ///   - columns: The number of columns in the resulting matrix (must be positive).
    /// - Returns: A two-dimensional array with the specified shape.
    func reshaped(rows: Int, columns: Int) -> [[Element]] {
        return _Vector<Element>.reshape(self, rows: rows, columns: columns)
    }
}

// MARK: - 2D Reshape and Flatten (Matrix → Vector, Matrix → Matrix)

public extension Array where Element: Collection, Element.Element: Numeric {

    /// Returns the shape of a two-dimensional array as a `(rows, columns)` tuple.
    ///
    /// The row count is the number of inner arrays, and the column count is taken
    /// from the first row. This property provides a quick way to inspect the
    /// dimensions of a matrix without iterating over every element.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let matrix: [[Double]] = [
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ]
    /// let (rows, columns) = matrix.shape
    /// // rows == 2, columns == 3
    /// ```
    ///
    /// - Returns: A tuple of `(rows: Int, columns: Int)`. Returns `(0, 0)` for
    ///   an empty matrix.
    var shape: (rows: Int, columns: Int) {
        return _Vector<Element.Element>.shape(of: self)
    }

    /// Returns the total number of elements across all dimensions.
    ///
    /// For a matrix with shape `(rows, columns)`, this equals the sum of every
    /// row's element count. Unlike `count`, which returns only the number of rows,
    /// `size` reflects the full element population of the matrix.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let matrix: [[Double]] = [
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ]
    /// matrix.size   // 6
    /// matrix.count  // 2 (number of rows only)
    /// ```
    ///
    /// - Returns: The total element count. Returns `0` for an empty matrix.
    var size: Int {
        return _Vector<Element.Element>.size(of: self)
    }

    /// Flattens a two-dimensional matrix into a one-dimensional array by concatenating
    /// rows in row-major order.
    ///
    /// All elements from the first row appear first, followed by elements from the second
    /// row, and so on. This is the inverse of `reshaped(rows:columns:)` — flattening a
    /// matrix and reshaping back to the original dimensions returns the same matrix.
    ///
    /// Flattening is useful when an algorithm expects a single vector of features rather
    /// than a structured matrix, or when serializing matrix data for storage or transmission.
    ///
    /// ```
    /// [[1, 2, 3],     →  [1, 2, 3, 4, 5, 6]
    ///  [4, 5, 6]]
    /// ```
    ///
    /// Example:
    /// ```swift
    /// let scores = [
    ///     [95.0, 88.0, 92.0],  // Player A
    ///     [87.0, 90.0, 89.0]   // Player B
    /// ]
    /// let allScores = scores.flattened()
    /// // [95.0, 88.0, 92.0, 87.0, 90.0, 89.0]
    ///
    /// // Round-trip: flatten and reshape restores the original
    /// let restored = allScores.reshaped(rows: 2, columns: 3)
    /// // [[95.0, 88.0, 92.0], [87.0, 90.0, 89.0]]
    /// ```
    ///
    /// - Returns: A one-dimensional array containing all matrix elements in row-major order.
    ///   Returns an empty array if the matrix is empty.
    func flattened() -> [Element.Element] {
        let matrix = self.map { row -> [Element.Element] in row.map { $0 } }
        return _Vector<Element.Element>.flatten(matrix)
    }

    /// Reshapes a matrix to new dimensions, preserving all elements in row-major order.
    ///
    /// The total number of elements must remain the same — a 2×6 matrix can become 3×4,
    /// 4×3, 6×2, 1×12, or 12×1. Internally, the matrix is flattened row by row, then
    /// filled into the new shape.
    ///
    /// This operation is useful when restructuring data between different representations.
    /// For example, converting a wide feature matrix into a tall format for different
    /// algorithms, or reorganizing batch data into new sample groupings.
    ///
    /// ```
    /// [[1, 2, 3],     →  [[1, 2],
    ///  [4, 5, 6]]         [3, 4],
    ///                      [5, 6]]
    /// ```
    ///
    /// Example:
    /// ```swift
    /// // Convert a 2×3 matrix to 3×2
    /// let wide: [[Double]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    /// let tall: [[Double]] = wide.reshaped(rows: 3, columns: 2)
    /// // [[1.0, 2.0],
    /// //  [3.0, 4.0],
    /// //  [5.0, 6.0]]
    ///
    /// // Reshape to a single row
    /// let row: [[Double]] = wide.reshaped(rows: 1, columns: 6)
    /// // [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    /// ```
    ///
    /// - Parameters:
    ///   - rows: The number of rows in the resulting matrix (must be positive).
    ///   - columns: The number of columns in the resulting matrix (must be positive).
    /// - Returns: A two-dimensional array with the specified shape.
    func reshaped(rows: Int, columns: Int) -> [[Element.Element]] {
        let flat: [Element.Element] = self.flattened()
        return _Vector<Element.Element>.reshape(flat, rows: rows, columns: columns)
    }
}
