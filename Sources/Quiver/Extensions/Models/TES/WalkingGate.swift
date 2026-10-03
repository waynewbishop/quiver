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

// The walking gate. The anchor set holds running efforts only, so a walking moment's nearest
// neighbors are the steep power-hike rows and a flat walk read Hard. Walking moments are labeled
// here instead, by grade, vertical speed and heart rate, and never reach the classifier.
//
// Provisional: thresholds from the exercise-science review on 2026-10-01, calibrated on one
// runner's recordings. Heart-rate bands are absolute beats per minute, the same scale the anchor
// rows use, so they share the anchors' one-runner-profile limit. Vertical oscillation is not
// read, because watches do not report it while walking.

extension TrueEffortScore {

    /// Cadence below which a moment can be walking, in steps per minute. Walking sits near 95–120
    /// and running near 150 or more, so 130 falls in the gap.
    static let walkingCadenceCeiling = 130.0

    /// Pace above which a moment can be walking, in minutes per kilometer.
    static let walkingPaceFloor = 8.0

    /// Grade at or above which a walking moment is a climb, in percent.
    static let walkingClimbGrade = 3.0

    /// Grade at or below which a walking moment is a steep descent, in percent.
    static let walkingDescentGrade = -10.0

    /// Vertical speed, in meters per hour, at which a climb reaches Tempo, Threshold and Hard.
    static let climbVerticalSpeedBands = (tempo: 300.0, threshold: 500.0, hard: 700.0)

    /// Vertical speed, in meters per hour, at which a steep descent reaches Tempo.
    static let descentVerticalSpeedTempo = 500.0

    /// Heart rate at which a climb reaches Threshold and Hard, in beats per minute.
    static let climbHeartRateBands = (threshold: 150.0, hard: 165.0)

    /// Heart-rate trust below which the climb's heart-rate band is ignored.
    static let climbHeartRateMinimumTrust = 0.5

    /// Labels a walking moment, or returns nil for a running moment the classifier should label.
    /// A climb takes the higher of its vertical-speed band and its heart-rate band, so a fit
    /// runner's lower heart rate cannot pull a fast climb down, and a dropped or doubted heart rate
    /// leaves the climb on vertical speed alone. Heart rate is read raw, not capped at the
    /// baseline's expectation, because a flat-trained baseline would otherwise cap a hard climb.
    /// A steep descent is Easy, or Tempo when fast, and never Hard: walking-speed braking loads
    /// the legs far less than a running descent. Everything else is Recovery.
    static func walkingEffort(for moment: Workout.Moment) -> EffortClass? {
        guard moment.cadence < walkingCadenceCeiling, moment.pace > walkingPaceFloor else {
            return nil
        }
        // |grade| / 100 × (1000 / (60 × pace)) m/s × 3600 s/h
        let verticalSpeed = abs(moment.grade) * 600 / moment.pace

        if moment.grade >= walkingClimbGrade {
            let bands = climbVerticalSpeedBands
            let byVerticalSpeed: EffortClass =
                verticalSpeed >= bands.hard ? .hard
                : verticalSpeed >= bands.threshold ? .threshold
                : verticalSpeed >= bands.tempo ? .tempo
                : .easy
            guard moment.hrTrust >= climbHeartRateMinimumTrust else { return byVerticalSpeed }
            let byHeartRate: EffortClass =
                moment.heartRate >= climbHeartRateBands.hard ? .hard
                : moment.heartRate >= climbHeartRateBands.threshold ? .threshold
                : .easy
            return byHeartRate.ordinal > byVerticalSpeed.ordinal ? byHeartRate : byVerticalSpeed
        }

        if moment.grade <= walkingDescentGrade {
            return verticalSpeed >= descentVerticalSpeedTempo ? .tempo : .easy
        }

        return .recovery
    }
}
