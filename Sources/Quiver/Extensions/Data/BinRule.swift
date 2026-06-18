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

// MARK: - Bin Rule

/// Names a rule for choosing the number of bins in a histogram.
///
/// Passed to `histogram(rule:)`. Each rule picks a bin count from the data
/// itself, freeing the caller from guessing a number. The three rules cover
/// the common design space: a quick exploration rule that ignores the data's
/// shape, a classical rule derived under a normal assumption, and a modern
/// robust rule that uses the interquartile range. See the "Choosing a bin
/// rule" section of <doc:Identifying-A-Distribution> for the history of each
/// rule and the trade-offs that motivate the choice.
public enum BinRule: Sendable {
    /// `k = ⌈√n⌉`. Depends only on sample size. Quick first look.
    case squareRoot
    /// `k = ⌈log₂(n) + 1⌉`. Sturges, 1926; assumes roughly normal data.
    case sturges
    /// `width = 2·IQR / n^(1/3)`, then `k = ⌈(max − min) / width⌉`. Freedman-Diaconis, 1981; robust to outliers.
    case freedmanDiaconis
}
