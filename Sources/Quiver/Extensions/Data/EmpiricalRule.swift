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

/// Observed and expected fractions for the empirical (68-95-99.7) rule.
///
/// Returned by `[Double].empiricalRule()`. Holds the observed fraction of values within
/// one, two, and three sample standard deviations of the mean, alongside the theoretical
/// fractions for a Gaussian distribution. The expected fractions are exposed as instance
/// properties so that observed and expected values read from the same struct at the call site.
public struct EmpiricalRule: Equatable, Sendable {

    /// The number of elements summarized.
    public let count: Int

    /// The observed fraction of values within one standard deviation of the mean.
    public let within1Sigma: Double

    /// The observed fraction of values within two standard deviations of the mean.
    public let within2Sigma: Double

    /// The observed fraction of values within three standard deviations of the mean.
    public let within3Sigma: Double

    /// The fraction expected within one standard deviation for a Gaussian distribution (0.6827).
    public var expected1Sigma: Double { 0.6827 }

    /// The fraction expected within two standard deviations for a Gaussian distribution (0.9545).
    public var expected2Sigma: Double { 0.9545 }

    /// The fraction expected within three standard deviations for a Gaussian distribution (0.9973).
    public var expected3Sigma: Double { 0.9973 }

    public init(count: Int, within1Sigma: Double, within2Sigma: Double, within3Sigma: Double) {
        self.count = count
        self.within1Sigma = within1Sigma
        self.within2Sigma = within2Sigma
        self.within3Sigma = within3Sigma
    }
}

extension EmpiricalRule: CustomStringConvertible {
    public var description: String {
        let header = "Empirical rule check (n = \(count))"
        let columns = "              actual    expected    diff"
        let row1 = format(label: "within 1\u{03C3}:", actual: within1Sigma, expected: expected1Sigma)
        let row2 = format(label: "within 2\u{03C3}:", actual: within2Sigma, expected: expected2Sigma)
        let row3 = format(label: "within 3\u{03C3}:", actual: within3Sigma, expected: expected3Sigma)
        return [header, columns, row1, row2, row3].joined(separator: "\n")
    }

    private func format(label: String, actual: Double, expected: Double) -> String {
        let actualStr = String(format: "%.3f", actual)
        let expectedStr = String(format: "%.3f", expected)
        let diff = actual - expected
        let sign = diff >= 0 ? "+" : "-"
        let diffStr = sign + String(format: "%.3f", abs(diff))
        return "  \(label)  \(actualStr)     \(expectedStr)       \(diffStr)"
    }
}

extension EmpiricalRule: Codable {
    private enum CodingKeys: String, CodingKey {
        case count, within1Sigma, within2Sigma, within3Sigma
    }
}
