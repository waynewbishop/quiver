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

/// A completed run, folded into history and the unit the baseline is fit on. Each moment is stored
/// as raw signal values rather than a wrapper type, so a caller reads a workout back the same way
/// they recorded it. Heart rate is stored as an outcome and used only as the regression target.
public struct Workout: Codable, Equatable, Sendable {

    /// One recorded moment, as raw sensor values. Internal: callers record via
    /// `TrueEffortScore.record(...)` and read a workout back through the arrays below.
    struct Moment: Codable, Equatable, Sendable {
        let heartRate: Double
        let pace: Double
        let cadence: Double
        let grade: Double               // signed; negative is downhill
        let verticalOscillation: Double
        let altitude: Double
        let hrTrust: Double              // optical reliability, 0...1
        let deltaTime: TimeInterval      // seconds since the previous moment; the first is 0

        /// What the baseline sees: every workload signal except heart rate, which is the target.
        var regressionFeatures: [Double] { [pace, cadence, grade, verticalOscillation, altitude] }

        /// What the classifier sees: the kinematic signature, heart rate included.
        var classifierFeatures: [Double] { [heartRate, pace, cadence, grade, verticalOscillation] }
    }

    var moments: [Moment]

    /// When the run began.
    public let startDate: Date

    /// The recorded heart rates, in order.
    public var heartRates: [Double] { moments.map(\.heartRate) }

    /// The per-moment durations, in seconds; the first is zero.
    public var deltaTimes: [TimeInterval] { moments.map(\.deltaTime) }

    /// The number of recorded moments.
    public var count: Int { moments.count }

    init(moments: [Moment], startDate: Date) {
        self.moments = moments
        self.startDate = startDate
    }
}
