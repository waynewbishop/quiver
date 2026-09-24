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

// The population anchor set the classifier is seeded from at every construction. Rows are the
// classifier feature vectors [heartRate, pace, cadence, grade, verticalOscillation], parallel to
// anchorLabels. Public so a reader can inspect what the classifier learned from.
//
// FREYA: this is a PLACEHOLDER anchor set pending your signed, fixture-pinned values. It was
// ported from the shipped watchOS demo's history, remapping the demo's four-label scheme
// (0 easy / 1 steady / 2 tempo / 3 hard) onto the EffortClass zones; the per-row remap reason is
// carried inline. Replace before release.

extension TrueEffortScore {

    /// Classifier feature rows for the anchor set: [heartRate, pace, cadence, grade, vertOsc].
    public static let anchorSamples: [[Double]] = [
        [130, 6.5, 165,  0.0, 8.5],   // easy — flat, slow
        [135, 6.2, 168,  0.0, 8.3],   // easy — flat
        [128, 6.8, 162, -1.0, 9.0],   // easy — gentle downhill
        [131, 6.6, 164,  0.0, 8.7],   // easy — settling in
        [127, 7.0, 160,  0.0, 9.1],   // easy — relaxed cruise
        [133, 6.4, 166,  0.5, 8.6],   // easy — gentle rise
        [129, 6.7, 163, -0.5, 8.9],   // easy — slight dip
        [132, 5.0, 158, -6.0, 11.0],  // hard — steep downhill, eccentric load
        [130, 5.2, 156, -5.5, 10.8],  // hard — steep downhill, eccentric load
        [165, 9.5, 145,  9.0, 6.5],   // hard — power hike, steep climb
        [168, 9.8, 142, 10.0, 6.3],   // hard — power hike, steep climb
        [172, 5.8, 178,  5.0, 7.4],   // hard — running uphill, still striding
        [176, 6.0, 176,  6.0, 7.2],   // hard — running uphill, still striding
        [150, 5.5, 172,  1.0, 8.0],   // tempo — steady, slight climb
        [148, 5.8, 170,  0.5, 8.1],   // tempo — steady, rolling
        [155, 5.3, 174,  1.5, 7.9],   // threshold — climbing
        [158, 5.0, 176,  1.0, 7.6],   // threshold — fast
        [170, 4.8, 180,  0.0, 7.0],   // threshold — fast, flat (race pace)
        [178, 4.6, 181,  0.0, 6.9],   // hard — sustained above threshold
        [175, 4.5, 182,  0.5, 6.8],   // hard — fast, slight climb
        [172, 4.6, 178,  0.0, 7.1],   // threshold — fast, flat
    ]

    /// The effort labels parallel to `anchorSamples`. The old-hard rows split between `.threshold`
    /// (sustained race pace) and `.hard` (climb, descent, and above-threshold surges) by their
    /// kinematics; that split is the main judgment call awaiting Freya's sign-off.
    public static let anchorLabels: [EffortClass] = [
        .easy, .easy, .easy, .easy, .easy, .easy, .easy,   // demo label 0 → easy
        .hard, .hard,                                      // steep downhill eccentric → hard
        .hard, .hard,                                      // power hike steep climb → hard
        .hard, .hard,                                      // uphill running → hard
        .tempo, .tempo,                                    // demo label 1 (steady) → tempo
        .threshold, .threshold,                            // demo label 2 (old tempo) → threshold
        .threshold,                                        // fast flat race pace → threshold
        .hard, .hard,                                      // sustained above threshold → hard
        .threshold,                                        // fast flat → threshold
    ]
}
