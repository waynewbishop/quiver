// Copyright 2026 Wayne W Bishop. All rights reserved.
// Licensed under the Apache License, Version 2.0.

import XCTest
@testable import Quiver

final class PipelineTests: XCTestCase {

    // MARK: - Classifier Pipeline

    func testClassifierPipelinePredicts() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let pipeline = Pipeline(scaler: scaler, model: model)

        // Pipeline scales internally — pass raw features
        let predictions = pipeline.predict([[2, 3], [5, 7]])
        XCTAssertEqual(predictions, [0, 1])
    }

    func testClassifierPipelineMatchesManual() {
        let features: [[Double]] = [
            [1, 2], [3, 4], [5, 8], [6, 9]
        ]
        let labels = [0, 0, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let pipeline = Pipeline(scaler: scaler, model: model)
        let testData: [[Double]] = [[2, 3], [5, 7], [6, 8]]

        // Pipeline predict should match manual scale + predict
        let pipelineResult = pipeline.predict(testData)
        let manualResult = model.predict(scaler.transform(testData))
        XCTAssertEqual(pipelineResult, manualResult)
    }

    // MARK: - Regressor Pipeline

    func testRegressorPipelinePredicts() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        let scaler = FeatureScaler.fit(features: features)
        let model = try LinearRegression.fit(
            features: scaler.transform(features),
            targets: targets
        )

        let pipeline = Pipeline(scaler: scaler, model: model)
        let predictions = pipeline.predict([[3], [4]])

        // Should predict values close to 6.0 and 8.0
        XCTAssertEqual(predictions.count, 2)
        XCTAssertEqual(predictions[0], 6.0, accuracy: 0.5)
        XCTAssertEqual(predictions[1], 8.0, accuracy: 0.5)
    }

    func testRegressorPipelineMatchesManual() throws {
        let features: [[Double]] = [[1], [2], [3], [4]]
        let targets = [10.0, 20.0, 30.0, 40.0]

        let scaler = FeatureScaler.fit(features: features)
        let model = try LinearRegression.fit(
            features: scaler.transform(features),
            targets: targets
        )

        let pipeline = Pipeline(scaler: scaler, model: model)
        let testData: [[Double]] = [[2], [3]]

        let pipelineResult = pipeline.predict(testData)
        let manualResult = model.predict(scaler.transform(testData))
        XCTAssertEqual(pipelineResult, manualResult)
    }

    // MARK: - Codable Round-Trip

    func testClassifierPipelineCodable() throws {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [5, 8], [6, 9]
        ]
        let labels = [0, 0, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let original = Pipeline(scaler: scaler, model: model)

        // Encode and decode
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)

        // Predictions should be identical after round-trip
        let testData: [[Double]] = [[2, 3], [5, 7]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }

    func testRegressorPipelineCodable() throws {
        let features: [[Double]] = [[1], [2], [3], [4]]
        let targets = [10.0, 20.0, 30.0, 40.0]

        let scaler = FeatureScaler.fit(features: features)
        let model = try LinearRegression.fit(
            features: scaler.transform(features),
            targets: targets
        )

        let original = Pipeline(scaler: scaler, model: model)

        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<LinearRegression>.self, from: data)

        let testData: [[Double]] = [[2], [3]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }

    // MARK: - Equatable

    func testPipelineEquatable() {
        let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
        let labels = [0, 0, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let pipeline1 = Pipeline(scaler: scaler, model: model)
        let pipeline2 = Pipeline(scaler: scaler, model: model)

        XCTAssertEqual(pipeline1, pipeline2)
    }

    // MARK: - Sendable

    func testPipelineSendableAcrossTaskBoundary() async throws {
        let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
        let labels = [0, 0, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let pipeline = Pipeline(scaler: scaler, model: model)

        // Pipeline crosses Task boundary — requires Sendable
        let result = await Task {
            pipeline.predict([[2, 3], [5, 7]])
        }.value

        XCTAssertEqual(result, [0, 1])
    }

    // MARK: - CustomStringConvertible

    func testPipelineDescription() {
        let features: [[Double]] = [[1, 2], [3, 4]]
        let labels = [0, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 1
        )

        let pipeline = Pipeline(scaler: scaler, model: model)
        let desc = pipeline.description

        XCTAssertTrue(desc.contains("Pipeline"))
        XCTAssertTrue(desc.contains("FeatureScaler"))
        XCTAssertTrue(desc.contains("KNearestNeighbors"))
    }

    // MARK: - Naive Bayes Pipeline

    func testNaiveBayesPipeline() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let scaler = FeatureScaler.fit(features: features)
        let model = GaussianNaiveBayes.fit(
            features: scaler.transform(features),
            labels: labels
        )

        let pipeline = Pipeline(scaler: scaler, model: model)
        let predictions = pipeline.predict([[2, 3], [5, 7]])
        XCTAssertEqual(predictions, [0, 1])
    }

    // MARK: - Pipeline.fit() Convenience Overloads

    func testFitKNN() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaler = FeatureScaler.fit(features: features)

        // One call — scale, train, bundle
        let pipeline = Pipeline.fit(
            features: features, labels: labels,
            scaler: scaler, k: 3
        )

        // Should match manual approach
        let manual = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )
        let manualPipeline = Pipeline(scaler: scaler, model: manual)

        let testData: [[Double]] = [[2, 3], [5, 7]]
        XCTAssertEqual(pipeline.predict(testData), manualPipeline.predict(testData))
    }

    func testFitNaiveBayes() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaler = FeatureScaler.fit(features: features)

        let pipeline = Pipeline.fit(
            features: features, labels: labels,
            scaler: scaler
        )

        let predictions = pipeline.predict([[2, 3], [5, 7]])
        XCTAssertEqual(predictions, [0, 1])
    }

    func testFitKMeans() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let scaler = FeatureScaler.fit(features: features)

        let pipeline = Pipeline.fit(
            data: features, scaler: scaler, k: 2, seed: 42
        )

        // Should produce 2 clusters
        let scaled = scaler.transform(features)
        let labels = pipeline.model.predict(scaled)
        XCTAssertTrue(labels.contains(0))
        XCTAssertTrue(labels.contains(1))
    }

    func testFitLinearRegression() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]
        let scaler = FeatureScaler.fit(features: features)

        let pipeline = try Pipeline.fit(
            features: features, targets: targets,
            scaler: scaler
        )

        let predictions = pipeline.predict([[3], [4]])
        XCTAssertEqual(predictions[0], 6.0, accuracy: 0.5)
        XCTAssertEqual(predictions[1], 8.0, accuracy: 0.5)
    }

    func testFitMatchesManualForAllModels() throws {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]
        let scaler = FeatureScaler.fit(features: features)
        let testData: [[Double]] = [[2, 3], [5, 7]]

        // KNN: Pipeline.fit() == manual
        let knnPipeline = Pipeline.fit(features: features, labels: labels, scaler: scaler, k: 3)
        let knnManual = Pipeline(scaler: scaler, model: KNearestNeighbors.fit(
            features: scaler.transform(features), labels: labels, k: 3))
        XCTAssertEqual(knnPipeline.predict(testData), knnManual.predict(testData))

        // NB: Pipeline.fit() == manual
        let nbPipeline = Pipeline.fit(features: features, labels: labels, scaler: scaler) as Pipeline<GaussianNaiveBayes>
        let nbManual = Pipeline(scaler: scaler, model: GaussianNaiveBayes.fit(
            features: scaler.transform(features), labels: labels))
        XCTAssertEqual(nbPipeline.predict(testData), nbManual.predict(testData))
    }

    func testFitCodableRoundTrip() throws {
        let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
        let labels = [0, 0, 1, 1]
        let scaler = FeatureScaler.fit(features: features)

        let original = Pipeline.fit(features: features, labels: labels, scaler: scaler, k: 3)
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)

        let testData: [[Double]] = [[2, 3], [5, 7]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }
}
