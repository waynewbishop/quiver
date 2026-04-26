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

        let pipeline = Pipeline.fit(features: features, labels: labels, k: 3)

        // Pipeline scales internally — pass raw features
        let predictions = pipeline.predict([[2, 3], [5, 7]])
        XCTAssertEqual(predictions, [0, 1])
    }

    func testClassifierPipelineMatchesManual() {
        let features: [[Double]] = [
            [1, 2], [3, 4], [5, 8], [6, 9]
        ]
        let labels = [0, 0, 1, 1]

        let pipeline = Pipeline.fit(features: features, labels: labels, k: 3)

        // Manual equivalent
        let scaler = StandardScaler.fit(features: features)
        let model = KNearestNeighbors.fit(
            features: scaler.transform(features),
            labels: labels, k: 3
        )

        let testData: [[Double]] = [[2, 3], [5, 7], [6, 8]]
        let pipelineResult = pipeline.predict(testData)
        let manualResult = model.predict(scaler.transform(testData))
        XCTAssertEqual(pipelineResult, manualResult)
    }

    // MARK: - Regressor Pipeline

    func testRegressorPipelinePredicts() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        let pipeline = try Pipeline.fit(features: features, targets: targets)
        let predictions = pipeline.predict([[3], [4]])

        XCTAssertEqual(predictions.count, 2)
        XCTAssertEqual(predictions[0], 6.0, accuracy: 0.5)
        XCTAssertEqual(predictions[1], 8.0, accuracy: 0.5)
    }

    func testRegressorPipelineMatchesManual() throws {
        let features: [[Double]] = [[1], [2], [3], [4]]
        let targets = [10.0, 20.0, 30.0, 40.0]

        let pipeline = try Pipeline.fit(features: features, targets: targets)

        let scaler = StandardScaler.fit(features: features)
        let model = try LinearRegression.fit(
            features: scaler.transform(features),
            targets: targets
        )

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

        let original = Pipeline.fit(features: features, labels: labels, k: 3)

        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)

        let testData: [[Double]] = [[2, 3], [5, 7]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
        XCTAssertEqual(original, restored)
    }

    func testRegressorPipelineCodable() throws {
        let features: [[Double]] = [[1], [2], [3], [4]]
        let targets = [10.0, 20.0, 30.0, 40.0]

        let original = try Pipeline.fit(features: features, targets: targets)

        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<LinearRegression>.self, from: data)

        let testData: [[Double]] = [[2], [3]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }

    // MARK: - Equatable

    func testPipelineEquatable() {
        let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
        let labels = [0, 0, 1, 1]

        let pipeline1 = Pipeline.fit(features: features, labels: labels, k: 3)
        let pipeline2 = Pipeline.fit(features: features, labels: labels, k: 3)

        XCTAssertEqual(pipeline1, pipeline2)
    }

    // MARK: - Sendable

    func testPipelineSendableAcrossTaskBoundary() async throws {
        let features: [[Double]] = [[1, 2], [3, 4], [5, 8], [6, 9]]
        let labels = [0, 0, 1, 1]

        let pipeline = Pipeline.fit(features: features, labels: labels, k: 3)

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

        let pipeline = Pipeline.fit(features: features, labels: labels, k: 1)
        let desc = pipeline.description

        XCTAssertTrue(desc.contains("Pipeline"))
        XCTAssertTrue(desc.contains("StandardScaler"))
        XCTAssertTrue(desc.contains("KNearestNeighbors"))
    }

    // MARK: - Naive Bayes Pipeline

    func testNaiveBayesPipeline() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let pipeline: Pipeline<GaussianNaiveBayes> = Pipeline.fit(features: features, labels: labels)
        let predictions = pipeline.predict([[2, 3], [5, 7]])
        XCTAssertEqual(predictions, [0, 1])
    }

    // MARK: - Pipeline.fit() Convenience Overloads

    func testFitKMeans() {
        let features: [[Double]] = [
            [1, 2], [1.5, 1.8], [1.2, 2.1],
            [5, 8], [6, 9], [5.5, 7.5]
        ]

        let pipeline = Pipeline.fit(data: features, k: 2, seed: 42)

        // Should produce 2 clusters
        let scaled = pipeline.scaler.transform(features)
        let labels = pipeline.model.predict(scaled)
        XCTAssertTrue(labels.contains(0))
        XCTAssertTrue(labels.contains(1))
    }

    func testFitLinearRegression() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        let pipeline = try Pipeline.fit(features: features, targets: targets)
        let predictions = pipeline.predict([[3], [4]])
        XCTAssertEqual(predictions[0], 6.0, accuracy: 0.5)
        XCTAssertEqual(predictions[1], 8.0, accuracy: 0.5)
    }
}
