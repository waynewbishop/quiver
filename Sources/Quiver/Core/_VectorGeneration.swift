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

// MARK: - Vector Generation Functions (Internal Implementation)

extension _Vector where Element: Numeric {
    /// Creates a vector filled with zeros
    static func zeros(_ count: Int) -> _Vector<Element> {
        precondition(count >= 0, "Count must be non-negative")
        return _Vector(elements: [Element](repeating: .zero, count: count))
    }
    
    /// Creates a vector filled with ones
    static func ones(_ count: Int) -> _Vector<Element> {
        precondition(count >= 0, "Count must be non-negative")
        
        guard let one = Element(exactly: 1) else {
            preconditionFailure("Element type does not support exact conversion from 1")
        }
        return _Vector(elements: [Element](repeating: one, count: count))
    }
    
    /// Creates a vector filled with a specific value
    static func full(_ count: Int, value: Element) -> _Vector<Element> {
        precondition(count >= 0, "Count must be non-negative")
        return _Vector(elements: [Element](repeating: value, count: count))
    }
    
    /// Creates a 2D array filled with zeros
    static func zeros2D(_ rows: Int, _ columns: Int) -> [[Element]] {
        precondition(rows >= 0 && columns >= 0, "Dimensions must be non-negative")
        let row = [Element](repeating: .zero, count: columns)
        return [[Element]](repeating: row, count: rows)
    }
    
    /// Creates a 2D array filled with ones
    static func ones2D(_ rows: Int, _ columns: Int) -> [[Element]] {
        precondition(rows >= 0 && columns >= 0, "Dimensions must be non-negative")
        guard let one = Element(exactly: 1) else {
            preconditionFailure("Element type does not support exact conversion from 1")
        }
        let row = [Element](repeating: one, count: columns)
        return [[Element]](repeating: row, count: rows)
    }
    
    /// Creates a 2D array filled with a specific value
    static func full2D(_ rows: Int, _ columns: Int, value: Element) -> [[Element]] {
        precondition(rows >= 0 && columns >= 0, "Dimensions must be non-negative")
        let row = [Element](repeating: value, count: columns)
        return [[Element]](repeating: row, count: rows)
    }
    
    /// Creates an identity matrix of size n x n
    static func identity(_ n: Int) -> [[Element]] {
        precondition(n > 0, "Matrix dimension must be positive")

        // Create matrix of zeros
        var result = zeros2D(n, n)

        // Set diagonal elements to one
        for i in 0..<n {
            if let one = Element(exactly: 1) {
                result[i][i] = one
            } else {
                // Handle types that can't convert from Int
                // This could be more sophisticated for complex numeric types
                preconditionFailure("Element type does not support exact conversion from 1")
            }
        }

        return result
    }
    
    /// Creates a diagonal matrix from a vector
    static func diag(_ vector: _Vector<Element>) -> [[Element]] {
        let n = vector.elements.count
        precondition(n > 0, "Vector must not be empty")
        
        var result = zeros2D(n, n)
        for i in 0..<n {
            result[i][i] = vector.elements[i]
        }
        
        return result
    }
}

extension _Vector where Element: FloatingPoint {
    /// Creates evenly spaced numbers over a specified interval
    static func linspace(start: Element, end: Element, count: Int) -> _Vector<Element> {
        precondition(count > 0, "Number of samples must be positive")

        if count == 1 {
            return _Vector(elements: [start])
        }

        var result = [Element]()
        let step = (end - start) / Element(count - 1)

        for i in 0..<count {
            result.append(start + step * Element(i))
        }

        return _Vector(elements: result)
    }
    
    /// Creates a sequence of values with a specified step size
    static func arange(_ start: Element, _ stop: Element, step: Element) -> _Vector<Element> {
        precondition(step != 0, "Step size cannot be zero")
        
        var result = [Element]()
        
        if step > 0 {
            var current = start
            while current < stop {
                result.append(current)
                current += step
            }
        } else {
            var current = start
            while current > stop {
                result.append(current)
                current += step
            }
        }
        
        return _Vector(elements: result)
    }
}
