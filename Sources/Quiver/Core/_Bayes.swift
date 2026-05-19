// Copyright 2026 Wayne W Bishop. All rights reserved.
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

/// Internal Bayes computation primitives.
internal enum _Bayes {

    /// Returns log(sum(exp(values))) computed in a numerically stable way.
    ///
    /// Subtracts the maximum before exponentiating so the largest term becomes
    /// `exp(0) = 1` and the remaining terms stay representable. Without this
    /// trick, summing exponentials of moderately negative log-probabilities
    /// underflows to zero in `Double`.
    static func logSumExp(_ values: [Double]) -> Double {
        guard let maxValue = values.max() else { return -.infinity }
        if maxValue == -.infinity { return -.infinity }
        var accumulator = 0.0
        for value in values {
            accumulator += Foundation.exp(value - maxValue)
        }
        return maxValue + Foundation.log(accumulator)
    }
}
