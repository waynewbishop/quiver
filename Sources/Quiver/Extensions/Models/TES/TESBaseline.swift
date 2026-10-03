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

    /// The fitted baseline as a readable equation with each signal named, such as
    /// `expected HR = 142.69 - 4.62·pace + 2.87·cadence + …`. It forwards to the wrapped Ridge's
    /// ``Coefficients/equation(variables:response:)``, so it reads the same way as every other
    /// linear model's equation. The weights are on standardized features, so each slope is the
    /// heart-rate change per standard deviation of its signal and the sizes compare directly.
    public func equation() -> String {
        expectedHeartRate.equation(variables: Self.featureNames, response: "expected HR")
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
        return "TESBaseline: λ=\(lambda), conditioning=\(cond), \(equation())"
    }
}
