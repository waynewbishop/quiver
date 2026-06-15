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

// MARK: - Cross-Entropy Strategy

/// The binary cross-entropy objective L(θ) = −(1/n)Σ[yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)]
/// where pᵢ = σ(xᵢ·θ), the gradient ∇L(θ) = (1/n)Xᵀ(σ(Xθ) − y).
///
/// Plugged into the shared ``_GradientDescent/descend(features:targets:learningRate:maxIterations:tolerance:strategy:)``
/// loop, this powers ``LogisticRegression``. It differs from
/// ``_MeanSquaredErrorStrategy`` at exactly the two points the math demands: the
/// prediction passes through the sigmoid before the residual is taken, and the
/// loss is cross-entropy rather than squared error. Everything else — the θ = 0
/// start, the step, the signed-relative convergence test, the divergence guards,
/// the trajectory — is the one shared loop, so logistic regression converges and
/// fails by the same rules as the squared-error models, and reuses the same
/// ``_GradientDescent/Outcome`` and ``_GradientDescent/DivergenceCause`` types.
///
/// The cross-entropy surface for a sigmoid hypothesis is convex with a unique
/// minimum on non-separable data, so a correct descent converges to the
/// maximum-likelihood coefficients. That correspondence is the validation oracle.
internal struct _CrossEntropyStrategy: _GradientStrategy {

    func gradient(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> [Double] {
        let n = features.count
        let p = parameters.count
        let oneOverN = 1.0 / Double(n)

        // Residual r = σ(Xθ) − y, accumulated into the gradient Xᵀr in one pass.
        // The single place this diverges from least squares: the prediction
        // passes through the sigmoid before the residual is taken. Note the
        // (1/n) factor versus least squares' (2/n) — the cross-entropy gradient
        // carries no factor of 2.
        let predictions = Self.predictProbabilities(features: features, parameters: parameters)
        var gradient = [Double](repeating: 0.0, count: p)
        for i in 0..<n {
            let r_i = predictions[i] - targets[i]
            let row = features[i]
            for j in 0..<p {
                gradient[j] += row[j] * r_i
            }
        }
        for j in 0..<p {
            gradient[j] *= oneOverN
        }
        return gradient
    }

    func loss(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> Double {
        Self.crossEntropyLoss(features: features, parameters: parameters, targets: targets)
    }

    // MARK: - Objective helpers

    /// Binary cross-entropy loss: L(θ) = −(1/n)Σ[yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)],
    /// where pᵢ = σ(xᵢ·θ).
    ///
    /// The predicted probability is clamped into `[ε, 1−ε]` before the logarithm.
    /// A perfectly confident, perfectly correct prediction drives σ to exactly 0
    /// or 1, at which point `log(0)` would produce `−∞` and spuriously trip the
    /// non-finite divergence guard even though the optimizer is behaving. The
    /// clamp bounds the loss without altering the gradient, which uses the raw
    /// `(p − y)` residual and is always finite.
    static func crossEntropyLoss(
        features: [[Double]],
        parameters: [Double],
        targets: [Double]
    ) -> Double {
        let n = features.count
        let clampEpsilon = 1.0e-15
        var sum = 0.0
        for i in 0..<n {
            let row = features[i]
            var z = 0.0
            for j in 0..<parameters.count {
                z += row[j] * parameters[j]
            }
            let raw = 1.0 / (1.0 + Foundation.exp(-z))
            // Clamp away from the open-interval boundaries so the logarithm stays finite.
            let p = Swift.min(Swift.max(raw, clampEpsilon), 1.0 - clampEpsilon)
            let y = targets[i]
            sum += y * Foundation.log(p) + (1.0 - y) * Foundation.log(1.0 - p)
        }
        return -sum / Double(n)
    }

    /// Computes predicted probabilities σ(Xθ) for the given features and parameters.
    ///
    /// Used inside the gradient computation to build the residual. The design
    /// matrix arrives already prepared — the caller prepends the ones column when
    /// an intercept is part of θ, matching the squared-error strategy's contract.
    static func predictProbabilities(
        features: [[Double]],
        parameters: [Double]
    ) -> [Double] {
        var result = [Double](repeating: 0.0, count: features.count)
        for i in 0..<features.count {
            let row = features[i]
            var z = 0.0
            for j in 0..<parameters.count {
                z += row[j] * parameters[j]
            }
            result[i] = 1.0 / (1.0 + Foundation.exp(-z))
        }
        return result
    }
}
