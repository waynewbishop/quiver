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
/// by bundling a `StandardScaler` and the model into a single value
/// that encodes and decodes as one unit.
///
/// Pipeline works with any model that conforms to ``Classifier`` or
/// ``Regressor``. The `transform` and `predict` steps are combined into
/// a single call — the caller passes raw features and Pipeline handles
/// scaling internally.
///
/// A pipeline can also carry an optional dimensionality-reduction stage.
/// When a ``PCA`` reducer is present, predict runs scaler → reducer →
/// model, so raw query points are scaled and projected by the same fitted
/// values the model trained on:
/// ```swift
/// import Quiver
///
/// // Fit scaler, project three features onto two components, train KNN.
/// let pipeline = Pipeline.fit(features: workouts, labels: effortLabels,
///                             componentCount: 2, k: 3)
///
/// // Raw features in — scaling and projection happen internally.
/// let predictions = pipeline.predict(newWorkouts)
/// ```
///
/// Pipeline uses `StandardScaler` (z-score normalization) because it is
/// robust to outliers and is the default choice in most ML curricula.
/// Users who need bounded-range scaling (for example, image pixels in
/// 0...1) can use `FeatureScaler` directly without Pipeline.
///
/// Example with a classifier:
/// ```swift
/// import Quiver
///
/// let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
/// let labels = [0, 0, 1, 1]
///
/// // One call — fits the scaler, trains the model, bundles them together
/// let pipeline = Pipeline.fit(features: features, labels: labels, k: 3)
///
/// // Predict on raw features — scaling happens automatically
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
/// let pipeline = try Pipeline.fit(features: features, targets: targets)
/// let prices = pipeline.predict([[1500], [1900]])
/// ```
public struct Pipeline<Model: Codable & Equatable & Sendable>: Codable, Equatable, Sendable {

    /// The scaler that normalizes raw inputs before prediction.
    public let scaler: StandardScaler

    /// The optional dimensionality-reduction stage applied after scaling.
    ///
    /// When present, predict projects scaled features onto the reducer's
    /// components before the model sees them. When `nil`, the pipeline
    /// behaves exactly as a scaler-and-model pair.
    public let reducer: PCA?

    /// The trained model that produces predictions from transformed features.
    public let model: Model

    /// Creates a pipeline from a fitted scaler, an optional reducer, and a
    /// trained model.
    ///
    /// In most cases, prefer the `fit()` factory methods which handle
    /// scaling, projection, training, and bundling in a single call.
    ///
    /// - Parameters:
    ///   - scaler: A fitted `StandardScaler`.
    ///   - reducer: A fitted `PCA` applied after scaling. Defaults to `nil`.
    ///   - model: A trained model.
    public init(scaler: StandardScaler, reducer: PCA? = nil, model: Model) {
        self.scaler = scaler
        self.reducer = reducer
        self.model = model
    }

    private enum CodingKeys: String, CodingKey {
        case scaler
        case reducer
        case model
    }

    /// Decodes a pipeline, validating that its stages agree on shape.
    ///
    /// Archives written before the reducer stage existed carry no `reducer`
    /// key and decode with `reducer` set to `nil`, preserving their original
    /// behavior. When a reducer is present, its expected feature width must
    /// match the scaler's output width; a mismatch throws
    /// `DecodingError.dataCorrupted` rather than constructing a pipeline
    /// whose predict path would fail later.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let scaler = try container.decode(StandardScaler.self, forKey: .scaler)
        let reducer = try container.decodeIfPresent(PCA.self, forKey: .reducer)
        let model = try container.decode(Model.self, forKey: .model)

        if let reducer = reducer, reducer.featureCount != scaler.featureCount {
            throw DecodingError.dataCorruptedError(forKey: .reducer, in: container,
                debugDescription: "reducer expects \(reducer.featureCount) features but the scaler produces \(scaler.featureCount)")
        }
        self.init(scaler: scaler, reducer: reducer, model: model)
    }

    /// Encodes the pipeline, omitting the reducer key when no reducer is set
    /// so that reducer-free pipelines keep their original archive shape.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(scaler, forKey: .scaler)
        try container.encodeIfPresent(reducer, forKey: .reducer)
        try container.encode(model, forKey: .model)
    }
}

// MARK: - Classifier Pipeline

extension Pipeline where Model: Classifier {

    /// Scales raw features, applies the reducer when present, and predicts
    /// class labels in one call.
    ///
    /// The caller passes unscaled features. Pipeline applies the scaler
    /// internally, projects through the reducer when one is set, then
    /// passes the transformed features to the classifier.
    ///
    /// - Parameter features: Raw (unscaled) feature matrix.
    /// - Returns: Predicted class labels, one per sample.
    public func predict(_ features: [[Double]]) -> [Int] {
        let scaled = scaler.transform(features)
        let transformed = reducer?.transform(scaled) ?? scaled
        return model.predict(transformed)
    }
}

// MARK: - Regressor Pipeline

extension Pipeline where Model: Regressor {

    /// Scales raw features, applies the reducer when present, and predicts
    /// continuous values in one call.
    ///
    /// The caller passes unscaled features. Pipeline applies the scaler
    /// internally, projects through the reducer when one is set, then
    /// passes the transformed features to the regressor.
    ///
    /// - Parameter features: Raw (unscaled) feature matrix.
    /// - Returns: Predicted values, one per sample.
    public func predict(_ features: [[Double]]) -> [Double] {
        let scaled = scaler.transform(features)
        let transformed = reducer?.transform(scaled) ?? scaled
        return model.predict(transformed)
    }
}

// MARK: - Pipeline.fit() — Concrete Overloads

// Each overload fits a StandardScaler, trains the model, and bundles
// both into an immutable Pipeline in one call. This prevents the most
// common ML mistake: training on unscaled data or saving the model
// without the scaler.

extension Pipeline where Model == KNearestNeighbors {

    /// Fits a scaler, trains a K-Nearest Neighbors classifier, and
    /// bundles them into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(features: x, labels: y, k: 3)
    /// let predictions = pipeline.predict(newData)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Integer class labels, one per sample.
    ///   - k: Number of neighbors. Defaults to 3.
    ///   - metric: Distance metric. Defaults to `.euclidean`.
    ///   - weight: Vote weighting. Defaults to `.uniform`.
    /// - Returns: An immutable `Pipeline<KNearestNeighbors>`.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        k: Int = 3,
        metric: DistanceMetric = .euclidean,
        weight: VoteWeight = .uniform
    ) -> Pipeline<KNearestNeighbors> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let model = KNearestNeighbors.fit(
            features: scaled, labels: labels,
            k: k, metric: metric, weight: weight
        )
        return Pipeline(scaler: scaler, model: model)
    }

    /// Fits a scaler, projects the scaled features onto principal
    /// components, trains a K-Nearest Neighbors classifier on the
    /// projection, and bundles all three into a Pipeline.
    ///
    /// Correlated feature columns double-count the same information in
    /// every distance a nearest-neighbor model computes. Projecting onto
    /// `componentCount` principal components hands the classifier fewer,
    /// uncorrelated axes. The returned pipeline applies the same fitted
    /// scaler and reducer to every query, so raw features go in and the
    /// projection the model trained on is reproduced exactly.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(features: x, labels: y, componentCount: 2, k: 3)
    /// let predictions = pipeline.predict(newData)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Integer class labels, one per sample.
    ///   - componentCount: Number of principal components to retain, from 1
    ///     up to the number of feature columns.
    ///   - k: Number of neighbors. Defaults to 3.
    ///   - metric: Distance metric. Defaults to `.euclidean`.
    ///   - weight: Vote weighting. Defaults to `.uniform`.
    /// - Returns: An immutable `Pipeline<KNearestNeighbors>` carrying the
    ///   scaler, the fitted reducer, and the trained classifier.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        componentCount: Int,
        k: Int = 3,
        metric: DistanceMetric = .euclidean,
        weight: VoteWeight = .uniform
    ) -> Pipeline<KNearestNeighbors> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let reducer = PCA.fit(features: scaled, componentCount: componentCount)
        let projected = reducer.transform(scaled)
        let model = KNearestNeighbors.fit(
            features: projected, labels: labels,
            k: k, metric: metric, weight: weight
        )
        return Pipeline(scaler: scaler, reducer: reducer, model: model)
    }
}

extension Pipeline where Model == GaussianNaiveBayes {

    /// Fits a scaler, trains a Gaussian Naive Bayes classifier, and
    /// bundles them into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(features: x, labels: y)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Integer class labels, one per sample.
    /// - Returns: An immutable `Pipeline<GaussianNaiveBayes>`.
    public static func fit(
        features: [[Double]],
        labels: [Int]
    ) -> Pipeline<GaussianNaiveBayes> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let model = GaussianNaiveBayes.fit(features: scaled, labels: labels)
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == KMeans {

    /// Fits a scaler, trains a K-Means clustering model, and
    /// bundles them into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = Pipeline.fit(data: x, k: 3)
    /// ```
    ///
    /// - Parameters:
    ///   - data: Raw (unscaled) feature matrix.
    ///   - k: Number of clusters.
    ///   - maxIterations: Maximum iterations. Defaults to 100.
    ///   - seed: Random seed for reproducibility.
    /// - Returns: An immutable `Pipeline<KMeans>`.
    public static func fit(
        data: [[Double]],
        k: Int,
        maxIterations: Int = 100,
        seed: UInt64? = nil
    ) -> Pipeline<KMeans> {
        let scaler = StandardScaler.fit(features: data)
        let scaled = scaler.transform(data)
        let model = KMeans.fit(data: scaled, k: k, maxIterations: maxIterations, seed: seed)
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == LinearRegression {

    /// Fits a scaler, trains a Linear Regression model, and
    /// bundles them into a Pipeline.
    ///
    /// ```swift
    /// let pipeline = try Pipeline.fit(features: x, targets: y)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - targets: Continuous target values, one per sample.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Returns: An immutable `Pipeline<LinearRegression>`.
    /// - Throws: `MatrixError.singular` if features are linearly dependent.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        intercept: Bool = true
    ) throws -> Pipeline<LinearRegression> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let model = try LinearRegression.fit(
            features: scaled, targets: targets, intercept: intercept
        )
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == GradientDescent {

    /// Fits a scaler, trains a gradient descent regressor, and bundles them
    /// into a Pipeline.
    ///
    /// Gradient descent is sensitive to feature scale — the default learning
    /// rate assumes unit variance. Bundling the scaler with the model removes
    /// the two ways that sensitivity bites: training on unscaled data, and
    /// forgetting to scale query points at prediction time. The returned
    /// pipeline's ``predict(_:)->[Double]`` takes raw features and scales them internally.
    ///
    /// ```swift
    /// let pipeline = try Pipeline.fit(features: x, targets: y)
    /// let predictions = pipeline.predict(rawQueryPoints)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - targets: Continuous target values, one per sample.
    ///   - learningRate: Step size η. Defaults to `0.01`, the canonical value
    ///     for standardized features.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold. Defaults to `1.0e-6`.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Returns: An immutable `Pipeline<GradientDescent>`.
    /// - Throws: ``GradientDescentError`` on divergence.
    public static func fit(
        features: [[Double]],
        targets: [Double],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> Pipeline<GradientDescent> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let model = try GradientDescent.fit(
            features: scaled, targets: targets,
            learningRate: learningRate, maxIterations: maxIterations,
            tolerance: tolerance, intercept: intercept
        )
        return Pipeline(scaler: scaler, model: model)
    }
}

extension Pipeline where Model == LogisticRegression {

    /// Fits a scaler, trains a logistic regression classifier, and bundles them
    /// into a Pipeline.
    ///
    /// Like ``GradientDescent``, logistic regression is trained iteratively and
    /// is sensitive to feature scale. Bundling the scaler with the model means
    /// the pipeline's ``predict(_:)->[Int]`` takes raw features and scales them
    /// internally — the caller cannot forget to scale query points.
    ///
    /// ```swift
    /// let pipeline = try Pipeline.fit(features: x, labels: y, learningRate: 0.5)
    /// let predictions = pipeline.predict(rawQueryPoints)
    /// ```
    ///
    /// - Parameters:
    ///   - features: Raw (unscaled) training feature matrix.
    ///   - labels: Binary class labels (0 or 1), one per sample.
    ///   - learningRate: Step size η. Defaults to `0.01`, the canonical value
    ///     for standardized features.
    ///   - maxIterations: Hard cap on iterations. Defaults to `1000`.
    ///   - tolerance: Relative loss-delta threshold. Defaults to `1.0e-6`.
    ///   - intercept: Whether to include a bias term. Defaults to `true`.
    /// - Returns: An immutable `Pipeline<LogisticRegression>`.
    /// - Throws: ``GradientDescentError`` on divergence.
    public static func fit(
        features: [[Double]],
        labels: [Int],
        learningRate: Double = 0.01,
        maxIterations: Int = 1000,
        tolerance: Double = 1.0e-6,
        intercept: Bool = true
    ) throws -> Pipeline<LogisticRegression> {
        let scaler = StandardScaler.fit(features: features)
        let scaled = scaler.transform(features)
        let model = try LogisticRegression.fit(
            features: scaled, labels: labels,
            learningRate: learningRate, maxIterations: maxIterations,
            tolerance: tolerance, intercept: intercept
        )
        return Pipeline(scaler: scaler, model: model)
    }
}

// MARK: - CustomStringConvertible

extension Pipeline: CustomStringConvertible {
    public var description: String {
        if let reducer = reducer {
            return "Pipeline: \(scaler), reducer: \(reducer), model: \(model)"
        }
        return "Pipeline: \(scaler), model: \(model)"
    }
}
