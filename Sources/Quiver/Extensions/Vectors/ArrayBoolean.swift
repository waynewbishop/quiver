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

// MARK: - Boolean Comparison Operations

public extension Array where Element: Comparable {
    /// Checks if this array is equal to another array element-wise.
    ///
    /// ```swift
    /// let predicted = [3, 5, 2, 5]
    /// let actual = [3, 4, 2, 5]
    /// predicted.isEqual(to: actual)  // [true, false, true, true]
    /// ```
    ///
    /// - Parameter other: The array to compare with
    /// - Returns: An array of boolean values indicating equality at each position
    func isEqual(to other: [Element]) -> [Bool] {
        let v1 = _VectorComparable(elements: self)
        let v2 = _VectorComparable(elements: other)
        return v1.isEqual(to: v2)
    }
    
    /// Checks if each element in this array is greater than the specified value.
    ///
    /// ```swift
    /// let scores = [85.0, 45.0, 92.0, 38.0]
    /// scores.isGreaterThan(80.0)  // [true, false, true, false]
    /// ```
    ///
    /// - Parameter value: The value to compare against
    /// - Returns: An array of boolean values
    func isGreaterThan(_ value: Element) -> [Bool] {
        let vector = _VectorComparable(elements: self)
        return vector.isGreaterThan(value)
    }
    
    /// Checks if each element in this array is less than the specified value.
    ///
    /// ```swift
    /// let readings = [23.5, 150.0, 22.9, -10.0]
    /// readings.isLessThan(0.0)  // [false, false, false, true]
    /// ```
    ///
    /// - Parameter value: The value to compare against
    /// - Returns: An array of boolean values
    func isLessThan(_ value: Element) -> [Bool] {
        let vector = _VectorComparable(elements: self)
        return vector.isLessThan(value)
    }
    
    /// Checks if each element in this array is greater than or equal to the specified value.
    ///
    /// ```swift
    /// let confidence = [0.92, 0.45, 0.87, 0.31, 0.78]
    /// confidence.isGreaterThanOrEqual(0.75)  // [true, false, true, false, true]
    /// ```
    ///
    /// - Parameter value: The value to compare against
    /// - Returns: An array of boolean values
    func isGreaterThanOrEqual(_ value: Element) -> [Bool] {
        let vector = _VectorComparable(elements: self)
        return vector.isGreaterThanOrEqual(value)
    }
    
    /// Checks if each element in this array is less than or equal to the specified value.
    ///
    /// ```swift
    /// let temperatures = [15.0, 22.0, 35.0, 18.0]
    /// temperatures.isLessThanOrEqual(20.0)  // [true, false, false, true]
    /// ```
    ///
    /// - Parameter value: The value to compare against
    /// - Returns: An array of boolean values
    func isLessThanOrEqual(_ value: Element) -> [Bool] {
        let vector = _VectorComparable(elements: self)
        return vector.isLessThanOrEqual(value)
    }
}

// Helper extension for Boolean arrays
public extension Array where Element == Bool {
    /// Returns the indices of all `true` elements in the boolean array.
    ///
    /// ```swift
    /// let mask = [true, false, true, false, true]
    /// mask.trueIndices  // [0, 2, 4]
    /// ```
    ///
    /// - Returns: An array of integer indices where the value is `true`
    var trueIndices: [Int] {
        return self.enumerated()
                  .filter { $0.element }
                  .map { $0.offset }
    }

    /// Performs element-wise logical AND with another boolean array.
    ///
    /// ```swift
    /// let ageOk = [true, true, false, true]
    /// let hasLicense = [true, false, false, true]
    /// ageOk.and(hasLicense)  // [true, false, false, true]
    /// ```
    ///
    /// - Parameter other: The boolean array to combine with (must have the same length)
    /// - Returns: A new boolean array where each element is `true` only if both corresponding elements are `true`
    func and(_ other: [Bool]) -> [Bool] {
        precondition(self.count == other.count, "Arrays must have the same length")
        return zip(self, other).map { $0 && $1 }
    }
    
    /// Performs element-wise logical OR with another boolean array.
    ///
    /// ```swift
    /// let tempHigh = [false, false, true, false]
    /// let humidHigh = [false, true, false, false]
    /// tempHigh.or(humidHigh)  // [false, true, true, false]
    /// ```
    ///
    /// - Parameter other: The boolean array to combine with (must have the same length)
    /// - Returns: A new boolean array where each element is `true` if either corresponding element is `true`
    func or(_ other: [Bool]) -> [Bool] {
        precondition(self.count == other.count, "Arrays must have the same length")
        return zip(self, other).map { $0 || $1 }
    }
    
    /// Returns the element-wise logical negation of the boolean array.
    ///
    /// ```swift
    /// let valid = [true, true, false, true, false]
    /// valid.not  // [false, false, true, false, true]
    /// ```
    ///
    /// - Returns: A new boolean array where each element is inverted
    var not: [Bool] {
        return self.map { !$0 }
    }
}

// MARK: - Boolean Indexing
public extension Array {
    /// Returns elements where the mask is true (like NumPy's `array[mask]`).
    ///
    /// ```swift
    /// let scores = [85.0, 45.0, 92.0, 38.0, 76.0]
    /// let passing = scores.isGreaterThanOrEqual(50.0)
    /// scores.masked(by: passing)  // [85.0, 92.0, 76.0]
    /// ```
    ///
    /// - Parameter mask: The boolean mask to apply
    /// - Returns: Array containing only elements where the mask is true
    func masked(by mask: [Bool]) -> [Element] {
        precondition(self.count == mask.count, "Array and mask must have the same length")
        return zip(self, mask)
            .filter { $0.1 }
            .map { $0.0 }
    }
    
    /// Returns a new array with elements conditionally chosen from this array or another.
    ///
    /// ```swift
    /// let readings = [23.5, 150.0, 22.9, -10.0]
    /// let valid = readings.isGreaterThanOrEqual(0.0)
    ///     .and(readings.isLessThanOrEqual(50.0))
    /// let defaults = [Double](repeating: 25.0, count: readings.count)
    /// readings.choose(where: valid, otherwise: defaults)
    /// // [23.5, 25.0, 22.9, 25.0]
    /// ```
    ///
    /// - Parameters:
    ///   - condition: Boolean mask determining which array to choose from
    ///   - other: The alternative array to choose elements from when condition is false
    /// - Returns: Array with elements chosen conditionally from this array or other
    func choose(where condition: [Bool], otherwise other: [Element]) -> [Element] {
        precondition(self.count == condition.count && self.count == other.count,
                     "All arrays must have the same length")
        return zip(self, zip(condition, other))
            .map { $0.1.0 ? $0.0 : $0.1.1 }
    }
}
