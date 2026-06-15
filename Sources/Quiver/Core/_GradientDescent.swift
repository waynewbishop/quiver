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

// MARK: - Gradient Strategy

/// The varying part of batch gradient descent: how the gradient and loss are
/// computed for one objective. Everything invariant across objectives — the
/// θ = 0 start, the parameter step, the signed-relative convergence test, the
/// divergence guards, and the loss trajectory — lives in
/// ``_GradientDescent/descend(features:targets:learningRate:maxIterations:tolerance:strategy:)``.
///
/// A strategy is a pure function of the current iterate. `gradient` and `loss`
/// are kept as separate calls — rather than a fused pair — so the shared loop
/// computes each exactly where the per-objective loops historically did, leaving
/// the numerics of every existing model bit-for-bit unchanged. Squared error
/// (``_MeanSquaredErrorStrategy``) and cross-entropy (``_CrossEntropyStrategy``)
/// are the two objectives; both share the one loop.
internal protocol _GradientStrategy: Sendable {

    /// The gradient ∇L(θ) at the given parameters.
    ///
    /// The returned vector already carries the per-objective scaling — the
    /// `(2/n)` of squared error, the `(1/n)` of cross-entropy, plus any penalty
    /// term — so the shared loop applies only the learning rate.
    ///
    /// - Parameters:
    ///   - features: Design matrix, intercept column prepended when fitting a bias.
    ///   - targets: Target vector — continuous for regression, binary `0`/`1` for
    ///     classification.
    ///   - parameters: The current coefficient vector θ.
    /// - Returns: The gradient at θ, one element per parameter.
    func gradient(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> [Double]

    /// The scalar loss L(θ) at the given parameters.
    ///
    /// Must be the exact objective whose gradient ``gradient(features:targets:parameters:)``
    /// returns, so the convergence test measures the surface being descended.
    ///
    /// - Parameters:
    ///   - features: Design matrix, matching the gradient call.
    ///   - targets: Target vector, matching the gradient call.
    ///   - parameters: The current coefficient vector θ.
    /// - Returns: The loss at θ.
    func loss(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> Double
}

// MARK: - Internal Gradient Descent Computation

/// Internal namespace for batch gradient descent, parameterized by objective.
///
/// Runs one shared descent loop over a ``_GradientStrategy``: starts from θ = 0
/// and steps opposite the gradient until the signed relative loss-delta falls
/// within tolerance or the iteration cap is reached.
///
/// Every iterative model in Quiver routes through this one loop — the dependents
/// are:
///
/// - ``GradientDescent`` — via ``_MeanSquaredErrorStrategy`` with `lambda == 0`.
/// - ``Ridge`` — via ``_MeanSquaredErrorStrategy`` with `lambda > 0`.
/// - ``LogisticRegression`` — via ``_CrossEntropyStrategy``.
///
/// (``LinearRegression`` is the closed-form sibling — it solves the normal
/// equation directly and does not use this loop.) Changing the loop's convergence
/// or divergence behavior changes all three dependents at once, so the equivalence
/// tests in `GradientStrategyTests` pin the squared-error path against its prior
/// numerics.
///
/// For squared error the loss surface is convex with a unique global minimum, so
/// any correct descent must converge to the same θ that
/// ``_Regression/fitNormalEquation(features:targets:intercept:)`` produces in
/// closed form. That correspondence is the validation oracle for the optimizer
/// mechanics.
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
    /// **L2 penalty.** Passing `lambda > 0` adds a ridge (Tikhonov) penalty to
    /// the objective: L(θ) = (1/n)‖Xθ − y‖² + λ‖θ‖², minimized by the gradient
    /// ∇ = (2/n)Xᵀ(Xθ − y) + 2λθ. The penalty contributes from `penalizeFromIndex`
    /// onward, so a caller that prepended a ones column passes
    /// `penalizeFromIndex: 1` to leave the intercept unpenalized — the standard
    /// statistical convention. The default `lambda: 0.0` makes every penalty
    /// term identically zero, so the unregularized path is reproduced bit-for-bit
    /// and the convergence/divergence test sees the same loss it always has.
    ///
    /// The monitored loss is the *penalized* objective, not the bare MSE: the
    /// loss the convergence test compares must be the same loss whose gradient
    /// is being followed, or the relative-delta test measures a different
    /// function than the one descending.
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
    ///   - lambda: L2 penalty strength. Must be non-negative. `0.0` (the default)
    ///     disables regularization and reproduces the plain-MSE path exactly.
    ///   - penalizeFromIndex: First parameter index the penalty applies to.
    ///     `0` penalizes every coefficient; `1` skips a leading intercept term.
    ///     Defaults to `0`. Ignored when `lambda == 0`.
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
        tolerance: Double,
        lambda: Double = 0.0,
        penalizeFromIndex: Int = 0
    ) throws -> (parameters: [Double], lossHistory: [Double], iterations: Int, outcome: Outcome) {
        // The squared-error path delegates to the shared strategy-driven loop.
        // `_MeanSquaredErrorStrategy` reproduces the exact gradient `(2/n)Xᵀr`,
        // the optional `2λθ` penalty term, and the penalized loss this method
        // computed inline before unification — so `LinearRegression`, `Ridge`,
        // and `GradientDescent` see identical numerics.
        return try descend(
            features: features,
            targets: targets,
            learningRate: learningRate,
            maxIterations: maxIterations,
            tolerance: tolerance,
            strategy: _MeanSquaredErrorStrategy(
                lambda: lambda, penalizeFromIndex: penalizeFromIndex
            )
        )
    }

    /// Runs batch gradient descent for the objective defined by `strategy`.
    ///
    /// Starts from θ = 0 and steps opposite the gradient each iteration until the
    /// signed relative loss-delta `(previousLoss − currentLoss) / max(previousLoss, ε)`
    /// falls into `[0, tolerance)`, or the iteration cap is reached. The strategy
    /// supplies the gradient and loss; this loop owns the step, the trajectory,
    /// and the convergence and divergence tests — so every objective converges
    /// and fails by identical rules.
    ///
    /// Numerical safety: after every iteration the loss is checked for `.isNaN`
    /// and `.isInfinite`, throwing ``DivergenceCause/nonFinite(iteration:loss:)``
    /// rather than returning corrupted numbers. A strict relative loss increase
    /// beyond one tolerance unit throws ``DivergenceCause/increasing(iteration:previousLoss:currentLoss:)``.
    /// Both are distinct from `.maxIterationsReached`, which carries a valid
    /// best-so-far estimate.
    ///
    /// - Parameters:
    ///   - features: Design matrix X, intercept column prepended by the caller
    ///     when a bias term is fit.
    ///   - targets: Target vector y, one element per row.
    ///   - learningRate: Step size η. Must be positive.
    ///   - maxIterations: Hard cap on iterations. Must be positive.
    ///   - tolerance: Relative loss-delta threshold for convergence. Must be positive.
    ///   - strategy: The objective — squared error or cross-entropy — supplying
    ///     the per-iteration gradient and loss.
    /// - Returns: The converged parameter vector θ, the loss trajectory (seeded
    ///   with the loss at θ = 0), the stopping iteration count, and an ``Outcome``.
    /// - Throws: ``DivergenceCause`` on non-finite or strictly increasing loss.
    static func descend(
        features: [[Double]],
        targets: [Double],
        learningRate: Double,
        maxIterations: Int,
        tolerance: Double,
        strategy: _GradientStrategy
    ) throws -> (parameters: [Double], lossHistory: [Double], iterations: Int, outcome: Outcome) {

        let p = features[0].count

        // Initial parameters: θ⁰ = 0 — the canonical starting point.
        var theta = [Double](repeating: 0.0, count: p)
        var lossHistory: [Double] = []
        lossHistory.reserveCapacity(maxIterations + 1)

        // Seed the trajectory with the loss at θ = 0 so a reader watching
        // `lossHistory` sees descent from the starting point.
        var previousLoss = strategy.loss(features: features, targets: targets, parameters: theta)
        lossHistory.append(previousLoss)

        // Safety against pathological inputs that produce non-finite loss
        // before a single step has been taken.
        if !previousLoss.isFinite {
            throw DivergenceCause.nonFinite(iteration: 0, loss: previousLoss)
        }

        // Small epsilon for the relative-tolerance denominator. Stops the
        // convergence test from dividing by zero when the loss reaches an exact
        // minimum (rare, but possible on synthetic data).
        let epsilon = 1.0e-12

        for iteration in 1...maxIterations {

            // Step opposite the gradient at the current iterate.
            let gradient = strategy.gradient(features: features, targets: targets, parameters: theta)
            for j in 0..<p {
                theta[j] -= learningRate * gradient[j]
            }

            // Evaluate the new loss for the convergence and divergence tests —
            // the same objective whose gradient was just followed.
            let currentLoss = strategy.loss(features: features, targets: targets, parameters: theta)
            lossHistory.append(currentLoss)

            // Divergence guard — the single most important guard. If the
            // iterate has blown up to ±∞ or NaN, every downstream prediction
            // would be meaningless. Throw rather than return corrupted numbers,
            // matching the contract on `LinearRegression.fit` for singular X'X.
            if !currentLoss.isFinite {
                throw DivergenceCause.nonFinite(iteration: iteration, loss: currentLoss)
            }

            // Signed relative loss-delta. The same quantity drives both the
            // divergence and convergence branches so they share units — both
            // must be relative to be honest across loss scales.
            let delta = previousLoss - currentLoss
            let relativeDelta = delta / Swift.max(previousLoss, epsilon)

            // Divergence: loss increased by more than one tolerance unit
            // *relative to its current scale*.
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
        // and without diverging. Return the best-so-far estimate observably.
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

    /// Penalized loss: L(θ) = (1/n)Σ(xᵢ·θ − yᵢ)² + λΣθⱼ² for j ≥ `penalizeFromIndex`.
    ///
    /// Composes the ridge (Tikhonov) penalty on top of ``meanSquaredLoss`` so
    /// the two share one residual definition. The penalty sums squared
    /// coefficients from `penalizeFromIndex` onward, leaving a leading intercept
    /// out of the norm when the caller passes `1`. With `lambda == 0` the second
    /// term is skipped entirely and the result equals the bare mean squared loss
    /// — the property that makes the unregularized descent path bit-for-bit
    /// unchanged.
    static func penalizedLoss(
        features: [[Double]],
        parameters: [Double],
        targets: [Double],
        lambda: Double,
        penalizeFromIndex: Int
    ) -> Double {
        let mse = meanSquaredLoss(features: features, parameters: parameters, targets: targets)
        guard lambda > 0 else { return mse }
        var penalty = 0.0
        for j in penalizeFromIndex..<parameters.count {
            penalty += parameters[j] * parameters[j]
        }
        return mse + lambda * penalty
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

// MARK: - Squared-Error Strategy

/// The squared-error objective L(θ) = (1/n)‖Xθ − y‖², optionally with an L2
/// ridge penalty λ‖θ‖² applied from `penalizeFromIndex` onward.
///
/// With `lambda == 0` this is ordinary least squares — the gradient `(2/n)Xᵀr`
/// and the bare mean-squared loss, reproducing the unregularized descent path
/// bit-for-bit. With `lambda > 0` and `penalizeFromIndex == 1` it is ridge
/// regression with an unpenalized intercept. Powers ``LinearRegression``,
/// ``Ridge``, and ``GradientDescent`` through ``_GradientDescent``.
internal struct _MeanSquaredErrorStrategy: _GradientStrategy {

    /// L2 penalty strength λ. `0.0` disables regularization (ordinary least squares).
    let lambda: Double

    /// First parameter index the penalty applies to. `1` leaves a leading
    /// intercept unpenalized — the standard statistical convention. Ignored when
    /// `lambda == 0`.
    let penalizeFromIndex: Int

    func gradient(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> [Double] {
        let n = features.count
        let p = parameters.count
        let twoOverN = 2.0 / Double(n)

        // Residual r = Xθ − y, accumulated into the gradient Xᵀr in one pass.
        let predictions = _GradientDescent.predict(features: features, parameters: parameters)
        var gradient = [Double](repeating: 0.0, count: p)
        for i in 0..<n {
            let r_i = predictions[i] - targets[i]
            let row = features[i]
            for j in 0..<p {
                gradient[j] += row[j] * r_i
            }
        }
        for j in 0..<p {
            gradient[j] *= twoOverN
        }

        // L2 penalty contribution: ∂/∂θⱼ (λ‖θ‖²) = 2λθⱼ. When lambda == 0 this
        // block is skipped and the plain-MSE gradient is left untouched.
        if lambda > 0 {
            for j in penalizeFromIndex..<p {
                gradient[j] += 2.0 * lambda * parameters[j]
            }
        }

        return gradient
    }

    func loss(
        features: [[Double]],
        targets: [Double],
        parameters: [Double]
    ) -> Double {
        _GradientDescent.penalizedLoss(
            features: features, parameters: parameters, targets: targets,
            lambda: lambda, penalizeFromIndex: penalizeFromIndex
        )
    }
}
