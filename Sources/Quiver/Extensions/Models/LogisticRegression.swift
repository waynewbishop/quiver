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

// MARK: - Logistic Regression

/// A binary classifier trained by gradient descent on cross-entropy loss.
///
/// Where ``LinearRegression`` predicts a continuous value and ``GradientDescent``
/// reaches that value iteratively, logistic regression predicts a *probability*
/// of class membership by pushing the same linear score Xθ through the sigmoid
/// function σ. The decision rule is a threshold: σ(Xθ) ≥ 0.5 predicts class 1,
/// otherwise class 0.
///
/// The optimizer is the *same* descent loop ``GradientDescent`` and ``Ridge`` use
/// — start from θ = 0 and step opposite the gradient — handed a different
/// objective. Logistic regression follows the cross-entropy gradient
/// ∇L = (1/n)Xᵀ(σ(Xθ) − y) rather than the squared-error gradient; the two share
/// the same shape, design-matrix transpose times the residual. Only the prediction
/// the residual is built from differs — σ(Xθ) here, Xθ for least squares — which is
/// why cross-entropy is the natural loss for a sigmoid hypothesis.
///
/// This is a value type — once created via `fit`, the model is immutable. The fitted instance carries the full loss trajectory so
/// a reader can observe convergence rather than infer it from the final number alone.
///
/// **Binary labels only.** Labels must be 0 or 1. Multinomial (softmax) logistic
/// regression is a separate model and out of scope here.
///
/// **Standardize features before fitting.** As with ``GradientDescent``, the default
/// `learningRate` of `0.01` assumes features with unit variance — typically via
/// ``StandardScaler``. On raw-scale features the loss-surface curvature scales with
/// the squared feature magnitude and the default rate diverges. The default rate is
/// deliberately conservative, so on small standardized datasets a faster rate (and a
/// higher iteration cap) reaches the minimum sooner — the example below passes
/// `learningRate: 0.5`.
///
/// Example:
/// ```swift
/// import Quiver
///
/// // One feature per sample (e.g. a spend score); classes overlap, so the
/// // maximum-likelihood fit is finite and descent converges.
/// let rawFeatures: [[Double]] = [
///     [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [3.5], [5.5]
/// ]
/// let labels = [0, 0, 1, 0, 1, 1, 1, 0]
///
/// // Standardize, then fit. A faster rate than the 0.01 default converges
/// // quickly on this well-scaled data.
/// let scaler = StandardScaler.fit(features: rawFeatures)
/// let model = try LogisticRegression.fit(
///     features: scaler.transform(rawFeatures), labels: labels, learningRate: 0.5)
///
/// // Transform query points through the same scaler before predicting.
/// let query = scaler.transform([[6.5], [2.5]])
/// let predicted = model.predict(query)              // [1, 0]
/// let probs = model.predictProbabilities(query)     // [0.758, 0.242]
/// ```
///
/// To bundle the scaler and model so query points are scaled automatically — and
/// the "forgot to scale" mistake becomes impossible — use ``Pipeline``. The
/// `labels:` argument selects this classifier unambiguously:
/// ```swift
/// let pipeline = try Pipeline.fit(features: rawFeatures, labels: labels, learningRate: 0.5)
///
/// // predict takes raw points — the pipeline scales them internally.
/// let predicted = pipeline.predict([[6.5], [2.5]])   // [1, 0]
/// ```
public struct LogisticRegression: Classifier, Codable, CustomStringConvertible, Equatable, Sendable {

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
        return "LogisticRegression: \(featureCount) \(featureLabel), \(convergence) (loss: \(lossFormatted))"
    }

    /// The outcome of the descent run that produced this model. Mirrors
    /// ``GradientDescent/Outcome`` — both objectives run the one shared loop, so
    /// they report convergence the same way.
    public enum Outcome: String, Codable, Equatable, Sendable {
        /// The convergence test fired within the iteration cap.
        case converged
        /// The optimizer used every available iteration without satisfying the
        /// convergence test. The returned coefficients are the best-so-far
        /// estimate. Confirm meaningful descent by comparing ``lossHistory``'s
        /// first and last entries before trusting the result.
        case maxIterationsReached
    }

    /// The fitted coefficient vector.
    ///
    /// When `hasIntercept` is true, the first element is the bias (intercept)
    /// term and the remaining elements are the feature weights. When false, all
    /// elements are feature weights. Matches the layout used by
    /// ``GradientDescent/coefficients``.
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

    /// The final cross-entropy loss at the returned coefficients.
    public let finalLoss: Double

    /// The full loss trajectory, beginning with the cross-entropy loss at θ = 0
    /// and ending with ``finalLoss``. Length is `iterations + 1`. Inspecting this
    /// array is the canonical way to see whether descent was smooth or stalled.
    public let lossHistory: [Double]

    /// Whether the run converged or hit the iteration cap. See ``Outcome``.
    public let outcome: Outcome

    /// Fits a logistic regression classifier to the given training data.
    ///
    /// Runs batch gradient descent on the cross-entropy loss until the signed
    /// relative loss-delta falls below `tolerance` or the iteration cap is
    /// reached. The returned model carries the full loss trajectory and an
    /// ``Outcome`` distinguishing convergence from cap-reached.
    ///
    /// **Standardize the features.** The defaults are calibrated for features
    /// with unit variance. On raw-scale inputs the default learning rate
    /// diverges — surfacing as ``GradientDescentError/divergedNonFinite(iteration:loss:)``.
    ///
    /// - Parameters:
    ///   - features: 2D array where each row is a sample and each column is a
    ///     feature. Assumed to be standardized when defaults are used.
    ///   - labels: 1D array of binary class labels, one per sample. Every value
    ///     must be 0 or 1.
    ///   - learningRate: Step size η. Defaults to `0.01`, the canonical value for
    ///     standardized features.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold for convergence. Defaults to
    ///     `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`.
    /// - Complexity: O(*k*·*n*·*f*) where *k* is the iteration count, *n* the
    ///   number of samples, and *f* the feature count. Each iteration is a
    ///   matrix-vector product with the residual — no matrix inversion.
    /// - Returns: A trained ``LogisticRegression`` model.
    /// - Throws: ``GradientDescentError/divergedNonFinite(iteration:loss:)`` when
    ///   the iterate produces a `NaN` or `±∞` loss;
    ///   ``GradientDescentError/divergedIncreasing(iteration:previousLoss:currentLoss:)``
    ///   when the loss strictly increases between iterations beyond the tolerance.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> LogisticRegression {
        precondition(!features.isEmpty, "Features array must not be empty")
        precondition(features.count == labels.count,
            "Features and labels must have the same number of samples")
        precondition(learningRate > 0,
            "Learning rate must be positive, got \(learningRate)")
        precondition(maxIterations > 0,
            "Max iterations must be positive, got \(maxIterations)")
        precondition(tolerance > 0,
            "Tolerance must be positive, got \(tolerance)")
        precondition(labels.allSatisfy { $0 == 0 || $0 == 1 },
            "LogisticRegression requires binary labels (0 or 1). Use a multinomial model for more than two classes.")

        let featureCount = features[0].count

        // Build the design matrix X, optionally prepending a ones column for the
        // bias term — same convention as `GradientDescent` and `_Regression`.
        let X: [[Double]]
        if intercept {
            X = features.map { [1.0] + $0 }
        } else {
            X = features
        }

        // Cross-entropy targets as Double, paired with the X just built. The
        // descent engine works in the regression family's numeric vocabulary
        // (`targets`); the public surface speaks the classifier's (`labels`).
        let y = labels.map { Double($0) }

        do {
            // The cross-entropy objective runs on the same shared descent loop as
            // the squared-error models — only the strategy differs.
            let result = try _GradientDescent.descend(
                features: X,
                targets: y,
                learningRate: learningRate,
                maxIterations: maxIterations,
                tolerance: tolerance,
                strategy: _CrossEntropyStrategy()
            )

            let outcome: Outcome
            switch result.outcome {
            case .converged:
                outcome = .converged
            case .maxIterationsReached:
                outcome = .maxIterationsReached
            }

            return LogisticRegression(
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
            // Map the internal cause to the public error type — the same shared
            // `GradientDescentError` the squared-error models throw, since both
            // optimizers can diverge the same two ways. Keeping the internal and
            // public types separate lets the public surface evolve without
            // touching either descent loop's contract.
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

    /// Fits a logistic regression classifier from a single feature array.
    ///
    /// A convenience overload that accepts a flat `[Double]` instead of
    /// `[[Double]]` when training on a single feature. Each element is treated as
    /// one sample with one feature.
    ///
    /// - Parameters:
    ///   - features: 1D array of feature values, one per sample.
    ///   - labels: 1D array of binary class labels (0 or 1), one per sample.
    ///   - learningRate: Step size η. Defaults to `0.01`.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold. Defaults to `1.0e-6`.
    ///   - intercept: Whether to fit a bias term. Defaults to `true`.
    /// - Returns: A trained ``LogisticRegression`` model with `featureCount` of 1.
    /// - Throws: ``GradientDescentError`` on divergence.
    public static func fit(
        features: [Double],
        labels: [Int],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> LogisticRegression {
        return try fit(
            features: features.map { [$0] },
            labels: labels,
            learningRate: learningRate,
            maxIterations: maxIterations,
            tolerance: tolerance,
            intercept: intercept
        )
    }

    /// Predicts class labels for one or more samples.
    ///
    /// For each sample, computes the probability σ(Xθ) and thresholds it at 0.5:
    /// a probability of 0.5 or greater predicts class 1, otherwise class 0. When
    /// ``hasIntercept`` is true the bias term is handled internally — callers pass
    /// raw feature rows, not augmented rows with a leading 1.0.
    ///
    /// - Parameter features: 2D array where each row is a sample to classify.
    /// - Returns: An array of predicted class labels (0 or 1), one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        // Thresholding σ(Xθ) at 0.5 is equivalent to thresholding the log-odds Xθ
        // at 0 — the same decision boundary, without the sigmoid round-trip.
        return scores(features).map { $0 >= 0.0 ? 1 : 0 }
    }

    /// Predicts the probability of class 1 for one or more samples.
    ///
    /// Returns σ(Xθ) for each input — the model's estimated probability that the
    /// sample belongs to class 1, in the open interval (0, 1). Unlike
    /// ``GaussianNaiveBayes/predictProbabilities(_:)``, which returns a per-class
    /// distribution that sums to 1.0, a binary logistic model has a single
    /// degree of freedom: the returned value is P(class = 1), and P(class = 0) is
    /// its complement. This single-probability shape is what threshold tuning and
    /// calibration recipes consume.
    ///
    /// - Parameter features: 2D array where each row is a sample.
    /// - Returns: An array of class-1 probabilities, one per sample.
    public func predictProbabilities(_ features: [[Double]]) -> [Double] {
        return scores(features).map { 1.0 / (1.0 + Foundation.exp(-$0)) }
    }

    /// Returns the raw linear score Xθ — the log-odds — for one or more samples.
    ///
    /// This is the value *before* the sigmoid: the model's estimate of
    /// log(P(class = 1) / P(class = 0)). Where ``predictProbabilities(_:)`` maps it
    /// into (0, 1) and ``predict(_:)`` thresholds it at 0.5, the log-odds is the
    /// quantity the decision boundary is linear in — zero is the boundary, positive
    /// favors class 1, negative favors class 0. It is the natural input for plotting
    /// a separating hyperplane, inspecting per-sample margins, or thresholding in
    /// log-odds space rather than probability space.
    ///
    /// - Parameter features: 2D array where each row is a sample.
    /// - Returns: An array of log-odds scores, one per sample, each in (−∞, +∞).
    public func decisionFunction(_ features: [[Double]]) -> [Double] {
        return scores(features)
    }

    /// Returns the log-odds score for a single-feature sample.
    ///
    /// A scalar convenience mirroring the protocol-provided scalar `predict`:
    /// pass one feature value, get one log-odds score back.
    ///
    /// - Parameter value: A single feature value for a model trained on one feature.
    /// - Returns: The log-odds score Xθ for that sample.
    public func decisionFunction(_ value: Double) -> Double {
        precondition(featureCount == 1,
            "Single-feature decisionFunction requires a model trained on 1 feature, got \(featureCount)")
        return decisionFunction([[value]])[0]
    }

    // MARK: - Private

    /// Computes the linear score Xθ for each sample, folding in the intercept when
    /// present. The single source of truth for the model's linear combination —
    /// ``predictProbabilities(_:)`` squashes it through the sigmoid,
    /// ``decisionFunction(_:)`` returns it directly, and ``predict(_:)``
    /// thresholds it at zero.
    private func scores(_ features: [[Double]]) -> [Double] {
        return features.map { sample in
            var z: Double
            if hasIntercept {
                z = coefficients[0]
                for i in 0..<sample.count {
                    z += coefficients[i + 1] * sample[i]
                }
            } else {
                z = 0.0
                for i in 0..<sample.count {
                    z += coefficients[i] * sample[i]
                }
            }
            return z
        }
    }
}
