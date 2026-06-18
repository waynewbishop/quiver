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

// MARK: - Aggregation Method

/// Specifies how values within a group or window are combined into a single result.
///
/// Used by `groupBy(_:using:)`, `groupedData(by:using:)`, and `downsample(factor:using:)`
/// to control how multiple values are aggregated.
public enum AggregationMethod: Sendable {
    /// Sum all values in the group
    case sum
    /// Calculate the arithmetic mean of the group
    case mean
    /// Count the number of values in the group
    case count
    /// Select the minimum value in the group
    case min
    /// Select the maximum value in the group
    case max
    /// Sum all values per group, then normalize so all groups sum to 100
    case percentage
}
