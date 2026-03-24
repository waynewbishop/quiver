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

// MARK: - Array Information for Numeric Types

public extension Array where Element: Numeric {

    /// Returns a summary of the array including count, type, and a preview of elements.
    ///
    /// For basic numeric arrays (integers and other non-floating-point types), the summary
    /// includes the element count, the Swift type, and the first five elements.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let data = [1, 2, 3, 4, 5]
    /// print(data.info())
    /// // Array Information:
    /// // Count: 5
    /// // Type: Int.Type
    /// //
    /// // First 5 items:
    /// // [0]: 1
    /// // [1]: 2
    /// // [2]: 3
    /// // [3]: 4
    /// // [4]: 5
    /// ```
    ///
    /// - Returns: A formatted string summarizing the array
    func info() -> String {
        var result = "Array Information:\n"
        result += "Count: \(self.count)\n"
        result += "Type: \(type(of: Element.self))\n"

        if !self.isEmpty {
            let previewCount = Swift.min(5, self.count)
            result += "\nFirst \(previewCount) items:\n"
            for i in 0..<previewCount {
                result += "[\(i)]: \(self[i])\n"
            }
        }

        return result
    }
}

// MARK: - Array Information for Floating Point Types

public extension Array where Element: FloatingPoint {

    /// Returns a statistical summary of the array including count, type, mean, standard
    /// deviation, min, and max.
    ///
    /// This overload provides richer output for floating-point arrays by including key
    /// statistical measures alongside the element preview. The standard deviation uses
    /// population statistics (ddof: 0).
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let scores = [85.0, 92.0, 78.0, 95.0, 88.0]
    /// print(scores.info())
    /// // Array Information:
    /// // Count: 5
    /// // Type: Double.Type
    /// // Mean: 87.6
    /// // Std: 5.85
    /// // Min: 78.0
    /// // Max: 95.0
    /// //
    /// // First 5 items:
    /// // [0]: 85.0
    /// // [1]: 92.0
    /// // [2]: 78.0
    /// // [3]: 95.0
    /// // [4]: 88.0
    /// ```
    ///
    /// - Returns: A formatted string summarizing the array with statistics
    func info() -> String {
        var result = "Array Information:\n"
        result += "Count: \(self.count)\n"
        result += "Type: \(type(of: Element.self))\n"

        if !self.isEmpty {
            if let mean = self.mean() {
                result += "Mean: \(mean)\n"
            }
            if let std = self.std() {
                result += "Std: \(std)\n"
            }
            if let min = self.min() {
                result += "Min: \(min)\n"
            }
            if let max = self.max() {
                result += "Max: \(max)\n"
            }

            let previewCount = Swift.min(5, self.count)
            result += "\nFirst \(previewCount) items:\n"
            for i in 0..<previewCount {
                result += "[\(i)]: \(self[i])\n"
            }
        }

        return result
    }
}

// MARK: - Array Information for Matrices

public extension Array where Element == [Double] {

    /// Returns a statistical summary of a 2D matrix including shape, size, type,
    /// and aggregate statistics computed across all elements.
    ///
    /// The summary flattens the matrix to compute mean, standard deviation, min, and max
    /// across every element. Shape reports `(rows, columns)` and size reports the total
    /// element count. The preview shows the first three rows.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let matrix: [[Double]] = [
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ]
    /// print(matrix.info())
    /// // Matrix Information:
    /// // Shape: (2, 3)
    /// // Size: 6
    /// // Type: Double.Type
    /// // Mean: 3.5
    /// // Std: 1.707825127659933
    /// // Min: 1.0
    /// // Max: 6.0
    /// //
    /// // First 2 rows:
    /// // [0]: [1.0, 2.0, 3.0]
    /// // [1]: [4.0, 5.0, 6.0]
    /// ```
    ///
    /// - Returns: A formatted string summarizing the matrix with shape and statistics
    func info() -> String {
        let dims = self.shape
        var result = "Matrix Information:\n"
        result += "Shape: (\(dims.rows), \(dims.columns))\n"
        result += "Size: \(self.size)\n"
        result += "Type: \(type(of: Double.self))\n"

        if !self.isEmpty {
            let flat = self.flattened()

            if let mean = flat.mean() {
                result += "Mean: \(mean)\n"
            }
            if let std = flat.std() {
                result += "Std: \(std)\n"
            }
            if let min = flat.min() {
                result += "Min: \(min)\n"
            }
            if let max = flat.max() {
                result += "Max: \(max)\n"
            }

            let previewCount = Swift.min(3, self.count)
            result += "\nFirst \(previewCount) rows:\n"
            for i in 0..<previewCount {
                result += "[\(i)]: \(self[i])\n"
            }
        }

        return result
    }
}
