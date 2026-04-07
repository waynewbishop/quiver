// Copyright 2026 Wayne W Bishop. All rights reserved.
// Licensed under the Apache License, Version 2.0.

import Foundation

// MARK: - Pipeline

/// A matched pair of scaler and model that travel together.
///
/// The most common mistake when persisting ML models is saving the model
/// without the scaler that was used to normalize its training data. When
/// the scaler is lost, predictions on new data produce wrong results
/// because the model expects scaled inputs. Pipeline eliminates this
/// by bundling the `FeatureScaler` and the model into a single value
/// that encodes and decodes as one unit.
///
/// Pipeline works with any model that conforms to ``Classifier`` or
/// ``Regressor``. The `transform` and `predict` steps are combined into
/// a single call — the caller passes raw features and Pipeline handles
/// scaling internally.
///
/// Example with a classifier:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
/// let labels = [0, 0, 1, 1]
///
/// let scaler = FeatureScaler.fit(features: features)
/// let model = KNearestNeighbors.fit(
///     features: scaler.transform(features),
///     labels: labels, k: 3
/// )
///
/// let pipeline = Pipeline(scaler: scaler, model: model)
///
/// // One call — scaling happens automatically
/// let predictions = pipeline.predict([[2, 3], [5, 7]])
/// // [0, 1]
///
/// // Encode the entire pipeline as one JSON blob
/// let data = try JSONEncoder().encode(pipeline)
/// let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)
/// ```
///
/// Example with a regressor:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [[1400], [1600], [1800], [2000]]
/// let targets = [245000.0, 312000.0, 378000.0, 440000.0]
///
/// let scaler = FeatureScaler.fit(features: features)
/// let model = try LinearRegression.fit(
///     features: scaler.transform(features),
///     targets: targets
/// )
///
/// let pipeline = Pipeline(scaler: scaler, model: model)
/// let prices = pipeline.predict([[1500], [1900]])
/// ```
public struct Pipeline<Model: Codable & Equatable & Sendable> {

    /// The feature scaler that normalizes raw inputs before prediction.
    public let scaler: FeatureScaler

    /// The trained model that produces predictions from scaled features.
    public let model: Model

    /// Creates a pipeline from a fitted scaler and a trained model.
    ///
    /// The scaler and model must have been fitted on the same data —
    /// the scaler learns column statistics, the model learns from
    /// the scaled output. Pipeline preserves this pairing permanently.
    ///
    /// - Parameters:
    ///   - scaler: A fitted `FeatureScaler`.
    ///   - model: A trained model (any `Codable & Equatable & Sendable` type).
    public init(scaler: FeatureScaler, model: Model) {
        self.scaler = scaler
        self.model = model
    }
}

// MARK: - Classifier Pipeline

extension Pipeline where Model: Classifier {

    /// Scales raw features and predicts class labels in one call.
    ///
    /// The caller passes unscaled features. Pipeline applies the scaler
    /// internally, then passes the scaled features to the classifier.
    ///
    /// - Parameter features: Raw (unscaled) feature matrix.
    /// - Returns: Predicted class labels, one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        let scaled = scaler.transform(features)
        return model.predict(scaled)
    }
}

// MARK: - Regressor Pipeline

extension Pipeline where Model: Regressor {

    /// Scales raw features and predicts continuous values in one call.
    ///
    /// The caller passes unscaled features. Pipeline applies the scaler
    /// internally, then passes the scaled features to the regressor.
    ///
    /// - Parameter features: Raw (unscaled) feature matrix.
    /// - Returns: Predicted values, one per sample.
    public func predict(_ features: [[Double]]) -> [Double] {
        let scaled = scaler.transform(features)
        return model.predict(scaled)
    }
}

// MARK: - Pipeline.fit() — Concrete Overloads

// Each overload scales the training data, trains the model, and
// bundles both into an immutable Pipeline in one call. This prevents
// the most common ML mistake: training on unscaled data or saving
// the model without the scaler.

extension Pipeline where Model == KNearestNeighbors {

    /// Scales features, trains a K-Nearest Neighbors classifier, and
    /// bundles the scaler and model into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(
    ///     features: trainingData, labels: labels,
    ///     scaler: scaler, k: 3
    /// )
    /// let predictions = pipeline.predict(newData)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Integer class labels, one per sample.
    ///   - scaler: A fitted `FeatureScaler`.
    ///   - k: Number of neighbors. Defaults to 3.
    ///   - metric: Distance metric. Defaults to `.euclidean`.
    ///   - weight: Vote weighting. Defaults to `.uniform`.
    /// - Returns: An immutable `Pipeline<KNearestNeighbors>`.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        scaler: FeatureScaler,
        k: Int = 3,
        metric: DistanceMetric = .euclidean,
        weight: VoteWeight = .uniform
    ) -> Pipeline<KNearestNeighbors> {
        let scaled = scaler.transform(features)
        let model = KNearestNeighbors.fit(
            features: scaled, labels: labels,
            k: k, metric: metric, weight: weight
        )
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == GaussianNaiveBayes {

    /// Scales features, trains a Gaussian Naive Bayes classifier, and
    /// bundles the scaler and model into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(
    ///     features: trainingData, labels: labels,
    ///     scaler: scaler
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Integer class labels, one per sample.
    ///   - scaler: A fitted `FeatureScaler`.
    /// - Returns: An immutable `Pipeline<GaussianNaiveBayes>`.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        scaler: FeatureScaler
    ) -> Pipeline<GaussianNaiveBayes> {
        let scaled = scaler.transform(features)
        let model = GaussianNaiveBayes.fit(features: scaled, labels: labels)
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == KMeans {

    /// Scales features, trains a K-Means clustering model, and
    /// bundles the scaler and model into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(
    ///     data: trainingData, scaler: scaler, k: 3
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - data: Raw (unscaled) feature matrix.
    ///   - scaler: A fitted `FeatureScaler`.
    ///   - k: Number of clusters.
    ///   - maxIterations: Maximum iterations. Defaults to 100.
    ///   - seed: Random seed for reproducibility.
    /// - Returns: An immutable `Pipeline<KMeans>`.
    public static func fit(
        data: [[Double]],
        scaler: FeatureScaler,
        k: Int,
        maxIterations: Int = 100,
        seed: UInt64? = nil
    ) -> Pipeline<KMeans> {
        let scaled = scaler.transform(data)
        let model = KMeans.fit(data: scaled, k: k, maxIterations: maxIterations, seed: seed)
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == LinearRegression {

    /// Scales features, trains a Linear Regression model, and
    /// bundles the scaler and model into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = try Pipeline.fit(
    ///     features: trainingData, targets: prices,
    ///     scaler: scaler
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - targets: Continuous target values, one per sample.
    ///   - scaler: A fitted `FeatureScaler`.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Returns: An immutable `Pipeline<LinearRegression>`.
    /// - Throws: `MatrixError.singular` if features are linearly dependent.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        scaler: FeatureScaler,
        intercept: Bool = true
    ) throws -> Pipeline<LinearRegression> {
        let scaled = scaler.transform(features)
        let model = try LinearRegression.fit(
            features: scaled, targets: targets, intercept: intercept
        )
        return Pipeline(scaler: scaler, model: model)
    }
}

// MARK: - Codable

extension Pipeline: Codable {
    enum CodingKeys: String, CodingKey {
        case scaler
        case model
    }
}

// MARK: - Equatable

extension Pipeline: Equatable {
    public static func == (lhs: Pipeline, rhs: Pipeline) -> Bool {
        lhs.scaler == rhs.scaler && lhs.model == rhs.model
    }
}

// MARK: - Sendable

extension Pipeline: Sendable {}

// MARK: - CustomStringConvertible

extension Pipeline: CustomStringConvertible {
    public var description: String {
        "Pipeline: \(scaler), model: \(model)"
    }
}
