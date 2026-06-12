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

// MARK: - Residual Model

/// Pairs a fitted regression model with the residuals it leaves behind.
///
/// A `ResidualModel` wraps any fitted ``Regressor`` and exposes the part of the
/// signal that model could not explain: the residual `observed − predicted`.
/// Where the regressor answers "what value does the workload predict?", the
/// residual answers "how far did the real value drift from that prediction?" —
/// the information a single prediction throws away.
///
/// This is the standard move behind residual analysis and de-confounding: model
/// out the part of an outcome that a set of inputs explains, then study what is
/// left. A heart rate inflated by heat reads the same as one inflated by effort
/// until a baseline predicts the heart rate the workload alone should produce;
/// the gap between that prediction and the observed value isolates the heat.
///
/// You fit the baseline yourself, then hand it in — the wrapper holds a model it
/// is given rather than training one of its own, the same way ``Pipeline``
/// composes a model it is handed. Fit the baseline on one period and residualize
/// a later sample against it: residualizing the same data the baseline was fit on
/// gives optimistically small residuals that understate what the model misses on
/// data it has not seen.
///
/// This is a value type. Once created via `init(model:)`, the wrapper is
/// immutable and can be reused across many residual computations.
///
/// - Note: Unlike a feature scaler, computing a residual needs both the features
///   and the observed targets — there is no `observed − predicted` without the
///   observed value. So ``residuals(features:targets:)`` takes both, rather than
///   the unary `transform(_:)` a scaler exposes.
///
/// Example:
/// ```swift
/// import Quiver
///
/// // A baseline that predicts expected heart rate from workload signals,
/// // fit on an earlier block of the runner's recorded history.
/// let workload: [[Double]] = [[6.5, 165], [5.5, 172], [4.8, 180]]
/// let heartRates = [132.0, 150, 170]
/// let baseline = try Ridge.fit(features: workload, targets: heartRates, lambda: 1.0)
///
/// // Wrap it, then read what the workload could not explain on a later sample.
/// let residualModel = ResidualModel(model: baseline)
/// let drift = residualModel.residuals(features: laterWorkload, targets: laterHeartRates)
/// // each value is observed − predicted: near zero where the fit holds,
/// // large where something off-model (heat, drift) pushed the reading
/// ```
public struct ResidualModel<Model: Regressor & Codable & Equatable & Sendable>:
    Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        "ResidualModel: wrapping \(Model.self)"
    }

    /// The fitted regression model whose predictions define the expectation.
    public let model: Model

    /// Wraps an already-fitted regressor.
    ///
    /// The model is fitted by the caller and passed in, so a `ResidualModel`
    /// composes with any regressor without needing to know its training
    /// signature. The wrapper holds no state of its own beyond the model.
    ///
    /// Two `ResidualModel` values are equal when their wrapped models are equal,
    /// which is coefficient-equality: a closed-form fit on the same data is
    /// bit-identical, while two iteratively fit models match only when they
    /// converged to the same numbers.
    ///
    /// - Parameter model: A fitted ``Regressor``.
    public init(model: Model) {
        self.model = model
    }

    /// The expected values the wrapped model predicts for the given features.
    ///
    /// A thin pass-through to the model's own `predict`, named for the role the
    /// prediction plays here: the expectation the residual is measured against.
    ///
    /// - Parameter features: 2D array where each row is a sample.
    /// - Returns: One predicted value per sample.
    public func expected(_ features: [[Double]]) -> [Double] {
        model.predict(features)
    }

    /// The residuals `observed − predicted`, one per sample.
    ///
    /// This is the signal the wrapper exists to surface: the part of each
    /// observed target the model's prediction did not account for. A residual
    /// near zero means the model explained the value; a large residual means
    /// something outside the model's inputs moved it.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample.
    ///   - targets: The observed target values, one per sample.
    /// - Returns: One residual per sample, `targets[i] − predicted[i]`.
    public func residuals(features: [[Double]], targets: [Double]) -> [Double] {
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples " +
            "(\(features.count) vs \(targets.count))")

        let predicted = model.predict(features)
        var result: [Double] = []
        result.reserveCapacity(targets.count)
        for i in 0..<targets.count {
            result.append(targets[i] - predicted[i])
        }
        return result
    }

    /// The residual for a single observed value at one sample.
    ///
    /// A scalar convenience: one feature vector and its observed target in, one
    /// residual out, with no array to wrap or unwrap.
    ///
    /// - Parameters:
    ///   - features: One sample's feature vector.
    ///   - observed: The observed target value for that sample.
    /// - Returns: `observed − predicted` for the single sample.
    public func residual(features: [Double], observed: Double) -> Double {
        observed - model.predict([features])[0]
    }
}

// MARK: - Coefficient access

/// A model whose fit is summarized by a vector of coefficients.
///
/// The regressors that report a fit as `coefficients` — ``LinearRegression``,
/// ``Ridge``, ``GradientDescent`` — conform (intercept first, then one weight
/// per feature). Distance- and tree-based models do not, so this capability is
/// its own protocol rather than a requirement on every regressor.
///
/// ``LogisticRegression`` also stores coefficients, but it is a ``Classifier``,
/// not a ``Regressor``, so it is not wrapped by `ResidualModel` and is left out
/// of this protocol. The exclusion rests on meaning, not on a structural
/// barrier: a residual `observed − predicted` is defined for a continuous
/// target, not for a probability or a 0/1 label.
public protocol Coefficients {
    /// The fitted coefficients: intercept first, then one weight per feature.
    var coefficients: [Double] { get }
}

extension LinearRegression: Coefficients {}
extension Ridge: Coefficients {}
extension GradientDescent: Coefficients {}

/// When the wrapped model reports coefficients, the wrapper forwards them — so
/// reading `residualModel.coefficients` is consistent with reading them off any
/// other linear model. The conditional conformance means this convenience
/// exists only for coefficient-bearing models and is a compile error for the
/// rest, rather than a runtime surprise.
extension ResidualModel: Coefficients where Model: Coefficients {
    public var coefficients: [Double] { model.coefficients }
}
