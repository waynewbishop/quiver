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

// MARK: - Skewness Agreement

/// How an outlier-sensitive and an outlier-resistant skewness measure relate.
///
/// `skewnessReport()` computes skewness two independent ways — the moment coefficient,
/// which weighs every value (and so is pulled hard by extremes), and the Bowley quartile
/// coefficient, which is built only from the quartiles and so ignores the tails. When the
/// two measures point the same way the shape is corroborated; when they conflict, a few
/// extreme values are likely distorting the moment number.
public enum SkewnessAgreement: Equatable, Sendable, Codable {

    /// Both measures land in their symmetric band, or both lean the same direction.
    /// The skew, whatever its size, is corroborated by a measure that ignores outliers.
    case agree

    /// One measure reads as roughly symmetric while the other reads as decisively skewed.
    /// The decisive measure is not corroborated — a sign that extremes may be at work.
    case mixed

    /// The two measures lean in opposite directions. The strongest signal that the moment
    /// number is driven by a few extreme values rather than the underlying shape.
    case direction
}
