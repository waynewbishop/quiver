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

// MARK: - Statistical Extensions

extension _Vector where Element: Numeric {
    /// Calculates the sum of all elements in the vector
    func sum() -> Element {
        return elements.reduce(Element.zero, +)
    }
    
    /// Calculates the product of all elements in the vector
    func product() -> Element {
        guard !elements.isEmpty else { return Element.zero }
        
        var result = elements[0]
        for i in 1..<elements.count {
            result = result * elements[i]
        }
        return result
    }
    
    /// Calculates the cumulative sum of elements in the vector
    func cumulativeSum() -> _Vector<Element> {
        guard !elements.isEmpty else { return _Vector(elements: []) }
        
        var result = [Element](repeating: Element.zero, count: elements.count)
        result[0] = elements[0]
        
        for i in 1..<elements.count {
            result[i] = result[i-1] + elements[i]
        }
        
        return _Vector(elements: result)
    }
    
    /// Calculates the cumulative product of elements in the vector
    func cumulativeProduct() -> _Vector<Element> {
        guard !elements.isEmpty else { return _Vector(elements: []) }
        
        var result = [Element](repeating: Element.zero, count: elements.count)
        result[0] = elements[0]
        
        for i in 1..<elements.count {
            result[i] = result[i-1] * elements[i]
        }
        
        return _Vector(elements: result)
    }
}

// MARK: - Operations requiring comparison

extension _Vector where Element: Numeric & Comparable {
    /// Finds the minimum element and its index in the vector
    func minWithIndex() -> (value: Element, index: Int)? {
        guard !elements.isEmpty else { return nil }
        
        var minIndex = 0
        var minValue = elements[0]
        
        for i in 1..<elements.count {
            if elements[i] < minValue {
                minValue = elements[i]
                minIndex = i
            }
        }
        
        return (minValue, minIndex)
    }
    
    /// Finds the maximum element and its index in the vector
    func maxWithIndex() -> (value: Element, index: Int)? {
        guard !elements.isEmpty else { return nil }
        
        var maxIndex = 0
        var maxValue = elements[0]
        
        for i in 1..<elements.count {
            if elements[i] > maxValue {
                maxValue = elements[i]
                maxIndex = i
            }
        }
        
        return (maxValue, maxIndex)
    }
}

extension _Vector where Element: FloatingPoint {
    /// Calculates the mean (average) of all elements in the vector
    func mean() -> Element? {
        guard !elements.isEmpty else { return nil }
        return sum() / Element(elements.count)
    }
    
    /// Calculates the median value of the vector
    func median() -> Element? {
        guard !elements.isEmpty else { return nil }
        
        let sorted = elements.sorted()
        let count = sorted.count
        
        if count % 2 == 0 {
            // Even number of elements, average the middle two
            return (sorted[count/2 - 1] + sorted[count/2]) / 2
        } else {
            // Odd number of elements, return the middle one
            return sorted[count/2]
        }
    }
    
    /// Calculates the variance of all elements in the vector
    func variance(ddof: Int = 1) -> Element? {
        guard elements.count > ddof else { return nil }

        guard let mean = self.mean() else { return nil }

        let squaredDifferences = elements.map { ($0 - mean) * ($0 - mean) }
        let sum = squaredDifferences.reduce(Element.zero, +)

        return sum / Element(elements.count - ddof)
    }

    /// Calculates the standard deviation of all elements in the vector
    func standardDeviation(ddof: Int = 1) -> Element? {
        guard let variance = self.variance(ddof: ddof) else { return nil }
        return variance.squareRoot()
    }

    /// Calculates the standard error of the mean for the elements in the vector
    func standardError(ddof: Int = 1) -> Element? {
        guard let std = self.standardDeviation(ddof: ddof) else { return nil }
        return std / Element(elements.count).squareRoot()
    }

}
