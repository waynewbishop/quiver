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

// MARK: - Ridge Regression

/// A regression model trained with an L2 (ridge) penalty on its coefficients.
///
/// Where ``LinearRegression`` minimizes the squared error alone, ``Ridge`` adds a
/// penalty proportional to the squared magnitude of the weights:
/// L(θ) = (1/n)‖Xθ − y‖² + λ‖θ‖². The penalty shrinks coefficients toward zero,
/// trading a little training-set accuracy for steadier predictions on unseen
/// data. The strength is controlled by `lambda` — at `lambda` of zero the penalty
/// vanishes and the fit is ordinary least squares; as `lambda` rises the weights
/// shrink. This is the quadratic, or Tikhonov, instance of regularization, and
/// the same λ that defines it is the term that stabilizes a near-singular XᵀX —
/// the situation ``Array/conditionNumber`` flags.
///
/// The model is fit by gradient descent on the penalized objective, sharing the
/// optimizer behind ``GradientDescent``. It conforms to ``Regressor``, so it is
/// interchangeable with the other regression models in downstream code and drops
/// into ``Pipeline`` without extra plumbing.
///
/// This is a value type — once created via one of the `fit` methods, the model is
/// immutable. There is no separate "unfitted" state.
///
/// **Standardize features before fitting.** A ridge penalty compares coefficient
/// magnitudes across features, so the features must share a scale or the penalty
/// falls unevenly — a large-range feature is penalized far less per unit than a
/// small-range one. Standardize with ``StandardScaler`` first. The intercept is
/// never penalized.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let scaler = StandardScaler.fit(features: rawFeatures)
/// let scaled = scaler.transform(rawFeatures)
///
/// let model = try Ridge.fit(features: scaled, targets: prices, lambda: 0.1)
/// let predictions = model.predict(newScaled)
/// ```
public struct Ridge: Regressor, Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let featureLabel = featureCount == 1 ? "feature" : "features"
        let lambdaFormatted = String(format: "%g", lambda)
        let lossFormatted = String(format: "%.4f", finalLoss)
        let convergence: String
        switch outcome {
        case .converged:
            convergence = "converged in \(iterations) iterations"
        case .maxIterationsReached:
            convergence = "reached max \(iterations) iterations"
        }
        return "Ridge: \(featureCount) \(featureLabel), λ=\(lambdaFormatted), \(convergence) (loss: \(lossFormatted))"
    }

    /// The outcome of the descent run that produced this model.
    ///
    /// Mirrors ``GradientDescent/Outcome`` — the penalized fit uses the same
    /// optimizer and so shares its convergence vocabulary.
    public enum Outcome: String, Codable, Equatable, Sendable {
        /// The convergence test fired within the iteration cap.
        case converged
        /// The optimizer used every available iteration without satisfying the
        /// convergence test. The returned coefficients are the best-so-far
        /// estimate, not corrupted — but a tighter learning rate or a larger
        /// iteration cap would reach the minimum. Compare the first and last
        /// entries of ``lossHistory`` to confirm meaningful descent.
        case maxIterationsReached
    }

    /// The fitted coefficient vector.
    ///
    /// When `hasIntercept` is true, the first element is the bias (intercept)
    /// term and the remaining elements are the feature weights — and the
    /// intercept is excluded from the penalty. When false, all elements are
    /// feature weights. Matches the layout used by ``LinearRegression/coefficients``
    /// and ``GradientDescent/coefficients`` so the regressors are interchangeable.
    public let coefficients: [Double]

    /// Number of features the model was trained on. Does not count the
    /// intercept term.
    public let featureCount: Int

    /// Whether this model includes an intercept (bias) term. The intercept,
    /// when present, is never penalized.
    public let hasIntercept: Bool

    /// The L2 penalty strength used during fitting. Larger values shrink the
    /// coefficients harder; `0.0` recovers ordinary least squares.
    public let lambda: Double

    /// The learning rate that was used during fitting.
    public let learningRate: Double

    /// Number of iterations the optimizer used before stopping. Equal to
    /// `lossHistory.count − 1` because the trajectory begins with the loss at
    /// the initial all-zeros parameters.
    public let iterations: Int

    /// The final penalized-loss value at the returned coefficients.
    public let finalLoss: Double

    /// The full loss trajectory of the penalized objective, beginning with the
    /// loss at θ = 0 and ending with ``finalLoss``. Length is `iterations + 1`.
    public let lossHistory: [Double]

    /// Whether the run converged or hit the iteration cap. See ``Outcome``.
    public let outcome: Outcome

    /// Fits a ridge regression model to the given training data.
    ///
    /// Runs batch gradient descent on the penalized objective
    /// L(θ) = (1/n)‖Xθ − y‖² + λ‖θ‖² until the signed relative loss-delta falls
    /// below `tolerance` or the iteration cap is reached. The intercept, when
    /// present, is excluded from the penalty term. The returned model carries the
    /// full loss trajectory and an ``Outcome`` distinguishing convergence from
    /// cap-reached.
    ///
    /// **Standardize the features.** A ridge penalty is only meaningful on
    /// features that share a scale; standardize with ``StandardScaler`` before
    /// fitting. The descent defaults are calibrated for unit-variance features.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a
    ///     feature. Assumed to be standardized.
    ///   - targets: 1D array of target values, one per sample.
    ///   - lambda: L2 penalty strength. Must be non-negative. `0.0` recovers
    ///     ordinary least squares.
    ///   - learningRate: Step size η. Defaults to `0.01`, the canonical value
    ///     for standardized features.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold for convergence. Defaults
    ///     to `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`. The
    ///     intercept is never penalized.
    /// - Complexity: O(*k*·*n*·*f*) where *k* is the iteration count, *n* the
    ///   number of samples, and *f* the feature count.
    /// - Returns: A trained ``Ridge`` model.
    /// - Throws: ``GradientDescentError/divergedNonFinite(iteration:loss:)`` when
    ///   the iterate produces a `NaN` or `±∞` loss;
    ///   ``GradientDescentError/divergedIncreasing(iteration:previousLoss:currentLoss:)``
    ///   when the penalized loss strictly increases beyond the tolerance.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        lambda: Double,
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> Ridge {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples")
        precondition(lambda >= 0,
            "Ridge penalty λ must be non-negative, got \(lambda)")
        precondition(learningRate > 0,
            "Learning rate must be positive, got \(learningRate)")
        precondition(maxIterations > 0,
            "Max iterations must be positive, got \(maxIterations)")
        precondition(tolerance > 0,
            "Tolerance must be positive, got \(tolerance)")

        let featureCount = features[0].count

        // Build the design matrix X, optionally prepending a ones column for the
        // bias term. Matches the convention used by `GradientDescent.fit` and
        // `_Regression.fitNormalEquation`.
        let X: [[Double]]
        let penalizeFromIndex: Int
        if intercept {
            X = features.map { row in
                var newRow = [1.0]
                newRow.append(contentsOf: row)
                return newRow
            }
            // Skip the leading ones column so the intercept is left unpenalized.
            penalizeFromIndex = 1
        } else {
            X = features
            penalizeFromIndex = 0
        }

        do {
            let result = try _GradientDescent.descend(
                features: X,
                targets: targets,
                learningRate: learningRate,
                maxIterations: maxIterations,
                tolerance: tolerance,
                lambda: lambda,
                penalizeFromIndex: penalizeFromIndex
            )

            let outcome: Outcome
            switch result.outcome {
            case .converged:
                outcome = .converged
            case .maxIterationsReached:
                outcome = .maxIterationsReached
            }

            return Ridge(
                coefficients: result.parameters,
                featureCount: featureCount,
                hasIntercept: intercept,
                lambda: lambda,
                learningRate: learningRate,
                iterations: result.iterations,
                finalLoss: result.lossHistory.last ?? .nan,
                lossHistory: result.lossHistory,
                outcome: outcome
            )
        } catch let cause as _GradientDescent.DivergenceCause {
            // Map the internal cause to the public error type, reusing the same
            // typed errors as GradientDescent — the optimizer is shared, so the
            // failure modes are identical.
            switch cause {
            case .nonFinite(let iteration, let loss):
                throw GradientDescentError.divergedNonFinite(iteration: iteration, loss: loss)
            case .increasing(let iteration, let previousLoss, let currentLoss):
                throw GradientDescentError.divergedIncreasing(
                    iteration: iteration,
                    previousLoss: previousLoss,
                    currentLoss: currentLoss
                )
            }
        }
    }

    /// Fits a ridge regression model from a single feature array.
    ///
    /// A convenience overload that accepts a flat `[Double]` instead of
    /// `[[Double]]` when training on a single feature. Each element is treated as
    /// one sample with one feature.
    ///
    /// - Parameters:
    ///   - features: 1D array of feature values, one per sample.
    ///   - targets: 1D array of target values, one per sample.
    ///   - lambda: L2 penalty strength. Must be non-negative.
    ///   - learningRate: Step size η. Defaults to `0.01`.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold. Defaults to `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`.
    /// - Returns: A trained ``Ridge`` model with `featureCount` of 1.
    /// - Throws: ``GradientDescentError`` on divergence.
    public static func fit(
        features: [Double],
        targets: [Double],
        lambda: Double,
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> Ridge {
        return try fit(
            features: features.map { [$0] },
            targets: targets,
            lambda: lambda,
            learningRate: learningRate,
            maxIterations: maxIterations,
            tolerance: tolerance,
            intercept: intercept
        )
    }

    /// Predicts target values for one or more samples.
    ///
    /// Computes ŷ = Xθ for each input sample using the fitted coefficients. When
    /// ``hasIntercept`` is true, the bias term is handled internally — callers
    /// pass raw feature rows, not augmented rows with a leading 1.0.
    ///
    /// - Parameter features: 2D array where each row is a sample to predict.
    /// - Returns: An array of predicted values, one per sample.
    public func predict(_ features: [[Double]]) -> [Double] {
        return features.map { sample in
            if hasIntercept {
                var sum = coefficients[0]
                for i in 0..<sample.count {
                    sum += coefficients[i + 1] * sample[i]
                }
                return sum
            } else {
                var sum = 0.0
                for i in 0..<sample.count {
                    sum += coefficients[i] * sample[i]
                }
                return sum
            }
        }
    }

    /// Predicts target values for single-feature regression.
    ///
    /// A convenience overload for single-feature models, mirroring the
    /// single-feature predict on the other regression models.
    ///
    /// - Parameter values: 1D array of feature values for single-feature
    ///   prediction.
    /// - Returns: An array of predicted values, one per input.
    public func predict(_ values: [Double]) -> [Double] {
        precondition(featureCount == 1,
            "Single-feature predict requires a model trained on 1 feature, got \(featureCount)")
        return predict(values.map { [$0] })
    }
}
