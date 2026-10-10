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

    public let adjusted: Double   // headline; the raw anchor is 100 for one hour at threshold
    public let raw: Double        // pre-adjustment score, 100 × Σ(weight·Δt) / (0.75 × 3600)

    /// The recorded heart rate averaged over active time, in beats per minute. Weighted by time
    /// alone, not by trust, so it matches the average a watch shows for the run. Present on every
    /// run, including a first run.
    public let averageHeartRate: Double

    /// Observed minus expected heart rate, the session mean and the core effort signal. Trust- and
    /// time-weighted, so a doubted optical stretch cannot swamp a few-bpm signal. It detects
    /// cardiac decoupling but does not diagnose its cause. nil on a first run, when no baseline
    /// exists to compare against.
    public let meanResidual: Double?

    /// The heart rate the baseline expected for this run's workload, the session mean in beats per
    /// minute. Weighted like `meanResidual`, so the two add up to the trust-weighted average heart
    /// rate, which equals `averageHeartRate` when no reading was doubted. nil on a first run.
    public let expectedHeartRate: Double?

    public let effortDistribution: [EffortClass: Double]  // share of time per category

    // The three session terms are kept separate, never pre-multiplied into the headline.
    public let varianceMultiplier: Double
    public let durationFactor: Double
    public let transitionLoad: Double

    public let loadCurve: [Double]          // cumulative weighted load trace

    public let timerTime: TimeInterval      // active time, what the score is built on
    public let elapsedTime: TimeInterval    // wall clock, including pauses

    /// The baseline that scored this run, nil on a first run before any baseline exists. It is
    /// captured before `finalize()` refits, so it can differ from `tes.baseline`, which has
    /// already learned from this run. Read it with `baseline?.equation()`.
    public let baseline: TESBaseline?

    /// Shows the breakdown, not just the headline.
    public var description: String {
        return """
        TESResult:
          adjusted:        \(String(format: "%.1f", adjusted))
          raw:             \(String(format: "%.1f", raw))
          average HR:      \(String(format: "%.1f bpm", averageHeartRate))
          expected HR:     \(expectedHeartRate.map { String(format: "%.1f bpm", $0) } ?? "n/a")
          meanResidual:    \(meanResidual.map { String(format: "%.1f bpm", $0) } ?? "n/a")
          variance ×:      \(String(format: "%.3f", varianceMultiplier))
          duration ×:      \(String(format: "%.3f", durationFactor))
          transitionLoad:  \(String(format: "%.2f", transitionLoad))
          timer/elapsed:   \(Int(timerTime.rounded()))s / \(Int(elapsedTime.rounded()))s
          baseline:        \(baseline?.equation() ?? "uncalibrated")
        """
    }
}

extension TESResult {

    /// Returns the active time spent in a category, in seconds, or 0 if the run never entered it.
    /// The times across all categories sum to `timerTime`.
    public func time(in category: EffortClass) -> TimeInterval {
        (effortDistribution[category] ?? 0) * timerTime
    }
}
