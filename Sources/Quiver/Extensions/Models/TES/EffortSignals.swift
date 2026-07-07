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

/// The standardized signals for one moment — each value is the z-score of that signal against the
/// classifier's training distribution, so a reading near 0 is typical and a large magnitude is an
/// outlier. This is how a UI reads *why* a moment is unusual: on a steep downhill, `grade` is a
/// large negative outlier while `heartRate` stays near 0, which is the calm-heart, hard-legs
/// signature the raw numbers hide.
///
/// The z-scores are the same standardization the classifier applies, surfaced rather than
/// discarded — no new math, just the intermediate made visible.
public struct EffortSignals: Equatable, Codable, Sendable {

    /// Heart rate, in standard deviations from the training mean.
    public let heartRate: Double

    /// Pace, in standard deviations from the training mean.
    public let pace: Double

    /// Cadence, in standard deviations from the training mean.
    public let cadence: Double

    /// Grade, in standard deviations from the training mean. A large negative value is a steep
    /// downhill.
    public let grade: Double

    /// Vertical oscillation, in standard deviations from the training mean.
    public let verticalOscillation: Double

    public init(
        heartRate: Double,
        pace: Double,
        cadence: Double,
        grade: Double,
        verticalOscillation: Double
    ) {
        self.heartRate = heartRate
        self.pace = pace
        self.cadence = cadence
        self.grade = grade
        self.verticalOscillation = verticalOscillation
    }

    /// The z-scores in the classifier's feature order:
    /// `[heartRate, pace, cadence, grade, verticalOscillation]`.
    public var asArray: [Double] {
        [heartRate, pace, cadence, grade, verticalOscillation]
    }
}

extension EffortSignals: CustomStringConvertible {
    public var description: String {
        func z(_ value: Double) -> String { String(format: "%+.1fσ", value) }
        return "EffortSignals(hr: \(z(heartRate)), pace: \(z(pace)), "
            + "cadence: \(z(cadence)), grade: \(z(grade)), vertOsc: \(z(verticalOscillation)))"
    }
}
