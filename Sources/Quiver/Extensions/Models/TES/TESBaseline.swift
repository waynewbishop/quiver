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

/// The personal trained sub-models plus readable diagnostics, reachable as `tes.baseline`. It is
/// produced by `TrueEffortScore`'s fit and replaced by each refit, never constructed directly.
/// The classifier is not here — it is anchor-seeded and always present, so it lives on
/// `TrueEffortScore.classifier`. This holds the personal half only, so `baseline == nil` means
/// exactly that the expected-heart-rate model is uncalibrated.
///
/// Each field is a wrapped Quiver model: a `StandardScaler`, a `Ridge`, and a `ResidualModel`
/// over that Ridge. The inspection surface below forwards their own methods, which is the whole
/// anti-black-box story.
public struct TESBaseline: Codable, Equatable, CustomStringConvertible, Sendable {

    public let lambda: Double

    public let scaler: StandardScaler
    public let expectedHeartRate: Ridge               // expected heart rate from workload
    public let residualModel: ResidualModel<Ridge>    // observed minus expected

    /// The 1-norm condition number of XᵀX at fit time, with `.infinity` mapped to nil. Cached so
    /// the diagnostic survives a Codable round-trip without re-deriving it.
    let conditionNumberValue: Double?

    /// The regression feature names, in coefficient order after the intercept.
    public static let featureNames = ["pace", "cadence", "grade", "verticalOscillation", "altitude"]

    /// The fitted baseline as readable math with anonymous subscripts, forwarded from
    /// `Ridge.equation()`. The weights are on standardized features, so each slope is the
    /// heart-rate change per standard deviation of its signal. See `labeledExpression` to name
    /// each signal.
    public var expression: String {
        expectedHeartRate.equation()
    }

    /// The fitted baseline with each slope named, such as
    /// "expected HR = 146.3 − 3.05·pace + 3.09·cadence". The slopes are per standard deviation, so
    /// they are directly comparable in size.
    public var labeledExpression: String {
        let weights = coefficients
        guard weights.count == Self.featureNames.count + 1 else { return expression }
        var text = "expected HR = " + Self.formatCoefficient(weights[0], leading: true)
        for (name, weight) in zip(Self.featureNames, weights.dropFirst()) {
            text += " " + Self.formatCoefficient(weight, leading: false) + "·" + name
        }
        return text
    }

    private static func formatCoefficient(_ value: Double, leading: Bool) -> String {
        let magnitude = String(format: "%.4g", abs(value))
        if leading {
            return value < 0 ? "-\(magnitude)" : magnitude
        }
        return (value < 0 ? "- " : "+ ") + magnitude
    }

    // There is no cached fit-quality metric by design: Quiver keeps quality off the fitted model
    // so it never returns a self-flattering training-set score. Compute it on held-out data via
    // the array extensions. This struct exposes ingredients and lets the caller judge.

    /// Conditioning of the standardized features — how redundant the signals are. A runner's pace,
    /// cadence, grade, and vertical oscillation move together, so this often reads in the tens,
    /// which is information rather than failure: the ridge penalty keeps the fit stable. nil when
    /// not meaningfully measurable. It is a 1-norm condition number of XᵀX, not an SVD condition
    /// number and not κ(X).
    public var conditioning: Double? {
        conditionNumberValue
    }

    /// Per-feature weights, intercept at index 0, forwarded from the fitted Ridge.
    public var coefficients: [Double] {
        expectedHeartRate.coefficients
    }

    public var description: String {
        let cond = conditioning.map { String(format: "%.1f", $0) } ?? "n/a"
        return "TESBaseline: λ=\(lambda), conditioning=\(cond), \(expression)"
    }
}
