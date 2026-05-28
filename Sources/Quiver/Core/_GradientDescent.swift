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

// MARK: - Internal Gradient Descent Computation

/// Internal namespace for batch gradient descent on mean squared error loss.
///
/// Minimizes L(θ) = (1/n)‖Xθ − y‖² by iteratively stepping the parameters
/// opposite the gradient ∇L(θ) = (2/n)Xᵀ(Xθ − y). Separated from the public API
/// so the descent loop stays testable without exposing implementation details.
///
/// The loss surface here is convex with a unique global minimum, so any correct
/// descent must converge to the same θ that ``_Regression/fitNormalEquation(features:targets:intercept:)``
/// produces in closed form. That correspondence is the validation oracle for the
/// optimizer mechanics.
internal enum _GradientDescent {

    /// The outcome of a descent run, separated so the caller can distinguish
    /// "converged within tolerance" from "hit the iteration cap while still
    /// improving" without inspecting numerical magnitudes.
    enum Outcome {
        /// The loss-delta convergence test fired within `maxIterations`.
        case converged
        /// The optimizer used every iteration and never satisfied the convergence
        /// test. The returned parameters are the best-so-far, not corrupted.
        case maxIterationsReached
    }

    /// Runs batch gradient descent on a linear hypothesis Xθ.
    ///
    /// Computes ∇ = (2/n)Xᵀ(Xθ − y) each iteration as a matrix-vector product
    /// against the residual, reusing existing Quiver matrix primitives. The
    /// convergence test is signed and relative: the loop stops only when
    /// `(previousLoss − currentLoss) / max(previousLoss, ε)` falls into
    /// `[0, tolerance)`. A negative delta — loss strictly increasing — is
    /// treated as a divergence signal, not a convergence signal.
    ///
    /// Numerical safety: after every iteration the loss is checked for
    /// `.isNaN` and `.isInfinite`. Either condition throws `divergedNonFinite`
    /// rather than returning corrupted numbers. A strict loss increase of more
    /// than one tolerance unit also throws `divergedIncreasing`. These two
    /// states are distinct from `.maxIterationsReached`, which carries a valid
    /// best-so-far estimate.
    ///
    /// The defaults assumed by the public surface — `learningRate: 0.01`,
    /// `maxIterations: 1000`, `tolerance: 1e-6` — are valid on standardized
    /// features. On raw-scale features the curvature of the loss surface scales
    /// with the squared feature magnitude and the constant step size diverges
    /// immediately. The caller is responsible for scaling.
    ///
    /// - Parameters:
    ///   - features: Design matrix X. Each row is a sample, each column is a
    ///     feature. The caller prepends a leading ones column when an intercept
    ///     is desired.
    ///   - targets: Target vector y, one element per row of `features`.
    ///   - learningRate: Step size η. Must be positive.
    ///   - maxIterations: Hard cap on iterations. Must be positive.
    ///   - tolerance: Relative loss-delta threshold for convergence. Must be
    ///     positive.
    /// - Returns: A tuple of the converged parameter vector θ, the loss
    ///   trajectory (one entry per completed iteration, beginning with the
    ///   loss at the initial all-zeros θ), the iteration count when the loop
    ///   stopped, and an ``Outcome`` distinguishing convergence from cap-reached.
    /// - Throws: ``_GradientDescent/DivergenceCause`` when the iterate produces
    ///   non-finite loss or the loss strictly increases between iterations.
    static func descend(
        features: [[Double]],
        targets: [Double],
        learningRate: Double,
        maxIterations: Int,
        tolerance: Double
    ) throws -> (parameters: [Double], lossHistory: [Double], iterations: Int, outcome: Outcome) {

        let n = features.count
        let p = features[0].count

        // Initial parameters: θ⁰ = 0 — the canonical starting point and the
        // value at which the loss landscape is purely the target variance.
        var theta = [Double](repeating: 0.0, count: p)
        var lossHistory: [Double] = []
        lossHistory.reserveCapacity(maxIterations + 1)

        // Seed the trajectory with the initial loss so a reader watching
        // `lossHistory` sees descent from the starting point.
        var previousLoss = meanSquaredLoss(features: features, parameters: theta, targets: targets)
        lossHistory.append(previousLoss)

        // Safety against pathological inputs that produce non-finite loss
        // before a single step has been taken.
        if !previousLoss.isFinite {
            throw DivergenceCause.nonFinite(iteration: 0, loss: previousLoss)
        }

        let twoOverN = 2.0 / Double(n)
        // Small epsilon for the relative-tolerance denominator. Stops the
        // convergence test from dividing by zero when the loss reaches an exact
        // minimum (rare, but possible on synthetic data).
        let epsilon = 1.0e-12

        for iteration in 1...maxIterations {

            // Residual r = Xθ − y, one element per sample.
            let predictions = predict(features: features, parameters: theta)
            var residual = [Double](repeating: 0.0, count: n)
            for i in 0..<n {
                residual[i] = predictions[i] - targets[i]
            }

            // Gradient ∇ = (2/n) Xᵀ r — accumulate Xᵀ · r in one pass over the
            // sample axis. Equivalent to lifting r to an n×1 column and using
            // `multiplyMatrix`, but the explicit loop avoids the temporary
            // matrix allocation each iteration without changing the math.
            var gradient = [Double](repeating: 0.0, count: p)
            for i in 0..<n {
                let r_i = residual[i]
                let row = features[i]
                for j in 0..<p {
                    gradient[j] += row[j] * r_i
                }
            }
            for j in 0..<p {
                gradient[j] *= twoOverN
            }

            // Step opposite the gradient.
            for j in 0..<p {
                theta[j] -= learningRate * gradient[j]
            }

            // Evaluate the new loss for the convergence and divergence tests.
            let currentLoss = meanSquaredLoss(features: features, parameters: theta, targets: targets)
            lossHistory.append(currentLoss)

            // Divergence guard — the single most important guard. If the
            // iterate has blown up to ±∞ or NaN, every downstream prediction
            // would be meaningless. Throw rather than return corrupted numbers,
            // matching the contract on `LinearRegression.fit` for singular X'X.
            if !currentLoss.isFinite {
                throw DivergenceCause.nonFinite(iteration: iteration, loss: currentLoss)
            }

            // Signed relative loss-delta. The same quantity drives both the
            // divergence and convergence branches so they share units — a
            // mistake in earlier drafts compared the divergence side against
            // an absolute `tolerance` while the convergence side was relative,
            // which would silently accept a 1e-6 increase on a loss of 1e10
            // and silently flag a 1e-6 increase on a loss of 1e-3 differently.
            // Both must be relative to be honest.
            let delta = previousLoss - currentLoss
            let relativeDelta = delta / Swift.max(previousLoss, epsilon)

            // Divergence: loss increased by more than one tolerance unit
            // *relative to its current scale*. Tolerance is the convergence
            // band; anything outside that band on the negative side is the
            // step having overshot the minimum.
            if relativeDelta < -tolerance {
                throw DivergenceCause.increasing(
                    iteration: iteration,
                    previousLoss: previousLoss,
                    currentLoss: currentLoss
                )
            }

            // Relative convergence: Δ / max(previousLoss, ε) < tolerance, with
            // Δ already known to be ≥ −tolerance from the branch above.
            if relativeDelta >= 0 && relativeDelta < tolerance {
                return (theta, lossHistory, iteration, .converged)
            }

            previousLoss = currentLoss
        }

        // The loop ran the full cap without satisfying the convergence test
        // and without diverging. Return the best-so-far estimate observably —
        // the caller can distinguish this from convergence via the outcome.
        return (theta, lossHistory, maxIterations, .maxIterationsReached)
    }

    /// Mean squared error loss: L(θ) = (1/n)Σ(xᵢ·θ − yᵢ)².
    ///
    /// Kept paired with ``gradient(features:parameters:targets:)`` — the loss
    /// measured by the convergence test must be the same loss whose gradient
    /// is being followed. Dropping the (1/n) factor would make the safe
    /// learning rate scale inversely with sample count.
    static func meanSquaredLoss(
        features: [[Double]],
        parameters: [Double],
        targets: [Double]
    ) -> Double {
        let n = features.count
        var sum = 0.0
        for i in 0..<n {
            let row = features[i]
            var prediction = 0.0
            for j in 0..<parameters.count {
                prediction += row[j] * parameters[j]
            }
            let residual = prediction - targets[i]
            sum += residual * residual
        }
        return sum / Double(n)
    }

    /// Computes predictions ŷ = Xθ for the given features and parameters.
    ///
    /// Used both inside the descent loop and by the public predict surface.
    /// Distinct from ``_Regression/predict(features:coefficients:intercept:)``
    /// because the design matrix arrives already prepared — the caller
    /// prepends the ones column when an intercept is part of θ.
    static func predict(
        features: [[Double]],
        parameters: [Double]
    ) -> [Double] {
        var result = [Double](repeating: 0.0, count: features.count)
        for i in 0..<features.count {
            let row = features[i]
            var sum = 0.0
            for j in 0..<parameters.count {
                sum += row[j] * parameters[j]
            }
            result[i] = sum
        }
        return result
    }

    /// Internal divergence cause, mapped to the public ``GradientDescentError``
    /// by the public `fit` surface. Kept internal so the public error type can
    /// evolve without breaking the descent loop's contract.
    enum DivergenceCause: Error {
        /// The iterate produced a non-finite loss (`NaN` or `±∞`) at the given
        /// iteration. The associated value is the offending loss.
        case nonFinite(iteration: Int, loss: Double)
        /// The loss strictly increased between two consecutive iterations by
        /// more than one tolerance unit. The step overshot the minimum.
        case increasing(iteration: Int, previousLoss: Double, currentLoss: Double)
    }
}
