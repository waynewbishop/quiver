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

// MARK: - Gradient Descent

/// A regression model trained by batch gradient descent on mean squared error
/// loss.
///
/// Where ``LinearRegression`` solves the normal equation θ = (X'X)⁻¹X'y in one
/// shot, ``GradientDescent`` reaches the same minimum iteratively — starting
/// from θ = 0 and stepping opposite the gradient of the loss at each iteration.
/// For a linear hypothesis with squared-error loss, both routes converge to the
/// same coefficient vector. The iterative route exists because the models that
/// follow this one — logistic regression, support vector machines — have loss
/// functions with no closed-form minimum.
///
/// This is a value type — once created via ``fit(features:targets:learningRate:maxIterations:tolerance:intercept:)``,
/// the model is immutable. The fitted instance carries the full loss trajectory
/// so a reader can observe convergence rather than infer it from the final
/// number alone.
///
/// **Standardize features before fitting.** The defaults assume input features
/// already have unit variance — typically via ``StandardScaler``. On raw-scale
/// features the curvature of the loss surface scales with the squared feature
/// magnitude and the default learning rate diverges immediately. When in doubt,
/// scale.
///
/// Example:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [
///     [1.0], [2.0], [3.0], [4.0], [5.0]
/// ]
/// let targets = [2.1, 3.9, 6.1, 8.0, 9.8]
///
/// let model = try GradientDescent.fit(features: features, targets: targets)
/// let predictions = model.predict([[6.0], [7.0]])
/// ```
public struct GradientDescent: Regressor, Codable, CustomStringConvertible, Equatable, Sendable {

    public var description: String {
        let featureLabel = featureCount == 1 ? "feature" : "features"
        let lossFormatted = String(format: "%.4f", finalLoss)
        let convergence: String
        switch outcome {
        case .converged:
            convergence = "converged in \(iterations) iterations"
        case .maxIterationsReached:
            convergence = "reached max \(iterations) iterations"
        }
        return "GradientDescent: \(featureCount) \(featureLabel), \(convergence) (loss: \(lossFormatted))"
    }

    /// The outcome of the descent run that produced this model.
    public enum Outcome: String, Codable, Equatable, Sendable {
        /// The convergence test fired within the iteration cap.
        case converged
        /// The optimizer used every available iteration without satisfying the
        /// convergence test. The returned coefficients are the best-so-far
        /// estimate, not corrupted — but a tighter learning rate or a larger
        /// iteration cap would reach the minimum.
        ///
        /// This state is necessary but not sufficient for trustworthiness: a
        /// run that hit the cap *and made meaningful progress* is a useful
        /// best-so-far; a run that hit the cap *and barely moved* is
        /// essentially the initial θ = 0 with a rounding error. Confirm
        /// meaningful descent by comparing ``GradientDescent/lossHistory``'s
        /// first and last entries — if the loss has not fallen substantially,
        /// the learning rate was too small for the iteration cap, the
        /// coefficients are not yet informative, and either the rate must rise
        /// or the cap must grow.
        case maxIterationsReached
    }

    /// The fitted coefficient vector.
    ///
    /// When `hasIntercept` is true, the first element is the bias (intercept)
    /// term and the remaining elements are the feature weights. When false, all
    /// elements are feature weights. Matches the layout used by ``LinearRegression/coefficients``
    /// so the two regressors are interchangeable in downstream code.
    public let coefficients: [Double]

    /// Number of features the model was trained on. Does not count the
    /// intercept term.
    public let featureCount: Int

    /// Whether this model includes an intercept (bias) term.
    public let hasIntercept: Bool

    /// The learning rate that was used during fitting.
    public let learningRate: Double

    /// Number of iterations the optimizer used before stopping. Equal to
    /// `lossHistory.count − 1` because the trajectory begins with the loss at
    /// the initial all-zeros parameters.
    public let iterations: Int

    /// The final loss value at the returned coefficients.
    public let finalLoss: Double

    /// The full loss trajectory, beginning with the loss at θ = 0 and ending
    /// with ``finalLoss``. Length is `iterations + 1`. Inspecting this array
    /// is the canonical way to see whether descent was smooth, oscillated, or
    /// crawled — the diagnostic that turns ``GradientDescent`` from a black
    /// box into an observable process.
    public let lossHistory: [Double]

    /// Whether the run converged or hit the iteration cap. See ``Outcome``.
    public let outcome: Outcome

    /// Fits a gradient descent regression model to the given training data.
    ///
    /// Runs batch gradient descent on the mean squared error loss until the
    /// signed relative loss-delta falls below `tolerance` or the iteration cap
    /// is reached. The returned model carries the full loss trajectory and an
    /// ``Outcome`` distinguishing convergence from cap-reached, so the caller
    /// can diagnose a run without re-reading the math.
    ///
    /// **Standardize the features.** The defaults are calibrated for features
    /// with unit variance. On raw-scale inputs the loss surface curvature
    /// scales with the squared feature magnitude and the default learning rate
    /// diverges within a handful of iterations — surfacing as
    /// ``GradientDescentError/divergedNonFinite(iteration:loss:)``.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a
    ///     feature. Assumed to be standardized when defaults are used.
    ///   - targets: 1D array of target values, one per sample.
    ///   - learningRate: Step size η. Defaults to `0.01`, the canonical value
    ///     for standardized features.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold for convergence. Defaults
    ///     to `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`.
    /// - Complexity: O(*k*·*n*·*f*) where *k* is the iteration count, *n* the
    ///   number of samples, and *f* the feature count. Each iteration is a
    ///   matrix-vector product with the residual — no matrix inversion.
    /// - Returns: A trained ``GradientDescent`` model.
    /// - Throws: ``GradientDescentError/divergedNonFinite(iteration:loss:)``
    ///   when the iterate produces a `NaN` or `±∞` loss;
    ///   ``GradientDescentError/divergedIncreasing(iteration:previousLoss:currentLoss:)``
    ///   when the loss strictly increases between iterations beyond the
    ///   tolerance.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> GradientDescent {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == targets.count,
            "Features and targets must have the same number of samples")
        precondition(learningRate > 0,
            "Learning rate must be positive, got \(learningRate)")
        precondition(maxIterations > 0,
            "Max iterations must be positive, got \(maxIterations)")
        precondition(tolerance > 0,
            "Tolerance must be positive, got \(tolerance)")

        let featureCount = features[0].count

        // Build the design matrix X, optionally prepending a ones column for
        // the bias term. Matches the convention used by `_Regression.fitNormalEquation`.
        let X: [[Double]]
        if intercept {
            X = features.map { row in
                var newRow = [1.0]
                newRow.append(contentsOf: row)
                return newRow
            }
        } else {
            X = features
        }

        do {
            let result = try _GradientDescent.descend(
                features: X,
                targets: targets,
                learningRate: learningRate,
                maxIterations: maxIterations,
                tolerance: tolerance
            )

            let outcome: Outcome
            switch result.outcome {
            case .converged:
                outcome = .converged
            case .maxIterationsReached:
                outcome = .maxIterationsReached
            }

            return GradientDescent(
                coefficients: result.parameters,
                featureCount: featureCount,
                hasIntercept: intercept,
                learningRate: learningRate,
                iterations: result.iterations,
                finalLoss: result.lossHistory.last ?? .nan,
                lossHistory: result.lossHistory,
                outcome: outcome
            )
        } catch let cause as _GradientDescent.DivergenceCause {
            // Map the internal cause to the public error type. Keeping the two
            // separated lets the public surface evolve without touching the
            // descent loop's contract.
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

    /// Fits a gradient descent regression model from a single feature array.
    ///
    /// A convenience overload that accepts a flat `[Double]` instead of
    /// `[[Double]]` when training on a single feature. Each element is treated
    /// as one sample with one feature.
    ///
    /// - Parameters:
    ///   - features: 1D array of feature values, one per sample.
    ///   - targets: 1D array of target values, one per sample.
    ///   - learningRate: Step size η. Defaults to `0.01`.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold. Defaults to `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`.
    /// - Returns: A trained ``GradientDescent`` model with `featureCount` of 1.
    /// - Throws: ``GradientDescentError`` on divergence.
    public static func fit(
        features: [Double],
        targets: [Double],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> GradientDescent {
        return try fit(
            features: features.map { [$0] },
            targets: targets,
            learningRate: learningRate,
            maxIterations: maxIterations,
            tolerance: tolerance,
            intercept: intercept
        )
    }

    /// Predicts target values for one or more samples.
    ///
    /// Computes ŷ = Xθ for each input sample using the fitted coefficients.
    /// When ``hasIntercept`` is true, the implementation handles the bias term
    /// internally — callers pass raw feature rows, not augmented rows with a
    /// leading 1.0.
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
    /// A convenience overload mirroring ``LinearRegression/predict(_:)-1xvqe``.
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
