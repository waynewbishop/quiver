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

public extension Array where Element == Double {

    /// Returns the area under the curve traced by the samples using the trapezoid rule.
    ///
    /// The trapezoid rule treats each adjacent pair of samples as the two parallel sides
    /// of a trapezoid: average the two heights, multiply by the time between them, sum
    /// every slice across the signal. The result approximates the total accumulated by
    /// a rate over the sampled interval — for example, total work done from a power
    /// curve sampled once per second, or total distance from a velocity series.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// // Cycling power, sampled once per second over a 10-second interval.
    /// let powerWatts = [250.0, 255.0, 260.0, 258.0, 262.0,
    ///                   265.0, 263.0, 261.0, 259.0, 257.0]
    ///
    /// // Total work in joules: area under the power-vs-time curve.
    /// let totalJ = powerWatts.trapezoidalIntegral(dt: 1.0) ?? 0
    /// ```
    ///
    /// Newton described the trapezoid rule in the *Principia*; it is the standard
    /// first-order numerical integrator for discrete samples of a continuous rate.
    /// For more sophisticated quadrature (Simpson's rule, Gaussian quadrature), use
    /// the building blocks elsewhere in Quiver and the existing element-wise methods.
    ///
    /// - Parameter dt: The time between samples in seconds. Must be positive.
    /// - Returns: The accumulated area under the samples, or `nil` if the array has
    ///   fewer than two elements.
    /// - Complexity: O(*n*) where *n* is the number of samples.
    func trapezoidalIntegral(dt: Double) -> Double? {
        guard self.count >= 2 else { return nil }
        var area = 0.0
        for i in 1..<self.count {
            area += 0.5 * (self[i - 1] + self[i]) * dt
        }
        return area
    }

    /// Returns the running area under the samples — total area accumulated at each
    /// time step, plotted as a curve the same length as the input.
    ///
    /// Where ``trapezoidalIntegral(dt:)`` returns a single total, this method returns
    /// the entire growth curve. The first element is always `0` (no area accumulated
    /// before any time has passed); each later element is the running sum of trapezoid
    /// contributions up to that sample. Pass the result directly to Swift Charts to
    /// visualise total work, total distance, or any cumulative quantity over time.
    ///
    /// Example:
    /// ```swift
    /// import Quiver
    ///
    /// let powerWatts = [250.0, 255.0, 260.0, 258.0, 262.0,
    ///                   265.0, 263.0, 261.0, 259.0, 257.0]
    ///
    /// let workCurve = powerWatts.cumulativeTrapezoidal(dt: 1.0)
    /// // workCurve[0] == 0.0
    /// // workCurve[9] is total joules done by the end of the interval
    /// ```
    ///
    /// - Parameter dt: The time between samples in seconds. Must be positive.
    /// - Returns: A new array with the same length as the input; the first element is
    ///   zero and each later element is the running trapezoidal total. Returns an
    ///   empty array when the input is empty.
    /// - Complexity: O(*n*) where *n* is the number of samples.
    func cumulativeTrapezoidal(dt: Double) -> [Double] {
        guard !self.isEmpty else { return [] }
        var result = [Double](repeating: 0.0, count: self.count)
        for i in 1..<self.count {
            result[i] = result[i - 1] + 0.5 * (self[i - 1] + self[i]) * dt
        }
        return result
    }
}
