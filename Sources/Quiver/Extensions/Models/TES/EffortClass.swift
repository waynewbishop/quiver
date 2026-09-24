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

/// The four effort bands. The first three are named as training zones; the top band is named
/// for the load itself, because it holds heavy muscular work as well as cardiovascular effort.
/// The `threshold` band anchors the score, so one hour held at threshold reads 100 on the
/// raw score, following the convention of power-based training stress scores.
///
/// - `easy`: recovery and aerobic base.
/// - `tempo`: sustained sub-threshold effort, roughly marathon-to-half pace.
/// - `threshold`: at lactate threshold, the score anchor.
/// - `hard`: above threshold, or a heavy muscular load that heart rate does not show, such as a
///   steep eccentric descent or a power-hike.
///
/// The tempo and threshold bands share a soft boundary: the classifier separates them across a
/// run but blurs them per moment, since the real distinction is blood lactate rather than a
/// kinematic signal. The bands are weight-adjacent, so a confusion between them is the cheapest
/// misclassification in the model.
public enum EffortClass: String, Codable, Equatable, Hashable, CaseIterable, Sendable {
    case easy
    case tempo
    case threshold
    case hard

    /// A display name for a watch face or summary, distinct from the raw case name used for
    /// storage and logic.
    public var label: String {
        switch self {
        case .easy:      return "Easy"
        case .tempo:     return "Tempo"
        case .threshold: return "Threshold"
        case .hard:      return "Hard"
        }
    }

    /// The band's load weight. Public because the weights are published math — the raw score is
    /// `100 × Σ(weight·Δt) / 2700` — and a reader has to be able to reach them.
    public var weight: Double {
        switch self {
        case .easy:      return 0.25
        case .tempo:     return 0.50
        case .threshold: return 0.75
        case .hard:      return 1.00
        }
    }

    /// The integer the classifier trains and predicts on, ordered by ascending weight. Internal;
    /// the public surface stays the domain-typed `EffortClass`.
    var ordinal: Int {
        switch self {
        case .easy:      return 0
        case .tempo:     return 1
        case .threshold: return 2
        case .hard:      return 3
        }
    }

    /// Maps a classifier prediction back to a band, clamping out-of-range values.
    init(clampingOrdinal value: Int) {
        switch Swift.min(3, Swift.max(0, value)) {
        case 0:  self = .easy
        case 1:  self = .tempo
        case 2:  self = .threshold
        default: self = .hard
        }
    }
}
