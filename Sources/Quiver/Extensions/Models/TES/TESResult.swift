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

/// The breakdown of one finalized run. `adjusted` is the headline, but it never stands alone:
/// `meanResidual` and the three session terms ride along as siblings, so a score driven by genuine
/// hard work stays distinguishable from one driven by heat or altitude decoupling. There is
/// deliberately no `score()` accessor that hides them.
public struct TESResult: Codable, Equatable, CustomStringConvertible, Sendable {

    public let adjusted: Double   // headline; fixed-anchor, about 100 for one hour at threshold
    public let raw: Double        // pre-adjustment score, 100 × Σ(weight·Δt) / (0.75 × 3600)

    /// Observed minus expected heart rate, the session mean and the core effort signal. Trust- and
    /// time-weighted, so a doubted optical stretch cannot swamp a few-bpm signal. It detects
    /// cardiac decoupling but does not diagnose its cause.
    public let meanResidual: Double

    public let effortDistribution: [EffortClass: Double]  // share of time per band

    // The three session terms are kept separate, never pre-multiplied into the headline.
    public let varianceMultiplier: Double
    public let durationFactor: Double
    public let transitionLoad: Double

    public let loadCurve: [Double]          // cumulative weighted load trace

    public let timerTime: TimeInterval      // active time, what the score is built on
    public let elapsedTime: TimeInterval    // wall clock, including pauses

    public let baselineExpression: String   // fitted baseline as math, carried onto the result

    /// Shows the breakdown, not just the headline.
    public var description: String {
        return """
        TESResult:
          adjusted:        \(String(format: "%.1f", adjusted))
          raw:             \(String(format: "%.1f", raw))
          meanResidual:    \(String(format: "%.1f", meanResidual)) bpm
          variance ×:      \(String(format: "%.3f", varianceMultiplier))
          duration ×:      \(String(format: "%.3f", durationFactor))
          transitionLoad:  \(String(format: "%.2f", transitionLoad))
          timer/elapsed:   \(Int(timerTime.rounded()))s / \(Int(elapsedTime.rounded()))s
          baseline:        \(baselineExpression)
        """
    }
}
