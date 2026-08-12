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

    // MARK: - GradientDescent Pipeline

    // The bundled pipeline scales query points internally, so raw inputs predict
    // correctly — this is the whole point on a scale-sensitive iterative model.
    func testFitGradientDescent() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.0, 4.0, 6.0, 8.0, 10.0]

        // Type annotation disambiguates from the LinearRegression (features:targets:)
        // overload, which Swift otherwise prefers (fewer parameters).
        let pipeline: Pipeline<GradientDescent> =
            try Pipeline.fit(features: features, targets: targets)

        // Raw query points — scaled inside predict.
        let predictions = pipeline.predict([[6], [7]])
        XCTAssertEqual(predictions[0], 12.0, accuracy: 0.5)
        XCTAssertEqual(predictions[1], 14.0, accuracy: 0.5)
    }

    // The pipeline result must equal the hand-rolled scaler + model + scaled-query
    // path exactly — proving the bundling is a faithful shortcut, not an approximation.
    func testFitGradientDescentMatchesManual() throws {
        let features: [[Double]] = [[1], [2], [3], [4], [5]]
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]

        let pipeline: Pipeline<GradientDescent> =
            try Pipeline.fit(features: features, targets: targets)

        let scaler = StandardScaler.fit(features: features)
        let model = try GradientDescent.fit(
            features: scaler.transform(features), targets: targets
        )

        let testData: [[Double]] = [[6], [7]]
        XCTAssertEqual(pipeline.predict(testData), model.predict(scaler.transform(testData)))
    }

    // MARK: - LogisticRegression Pipeline

    // `labels:` makes this overload unambiguous (no regressor overload takes it),
    // and the pipeline scales raw query points internally before classifying.
    func testFitLogisticRegression() throws {
        let features: [[Double]] = [[2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [3.5], [5.5]]
        let labels = [0, 0, 1, 0, 1, 1, 1, 0]

        let pipeline = try Pipeline.fit(features: features, labels: labels, learningRate: 0.5)
        XCTAssertTrue(type(of: pipeline) == Pipeline<LogisticRegression>.self)

        // Raw query points — scaled inside predict.
        XCTAssertEqual(pipeline.predict([[6.5], [2.5]]), [1, 0])
    }

    func testFitLogisticRegressionMatchesManual() throws {
        let features: [[Double]] = [[2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [3.5], [5.5]]
        let labels = [0, 0, 1, 0, 1, 1, 1, 0]

        let pipeline = try Pipeline.fit(features: features, labels: labels, learningRate: 0.5)

        let scaler = StandardScaler.fit(features: features)
        let model = try LogisticRegression.fit(
            features: scaler.transform(features), labels: labels, learningRate: 0.5
        )

        let testData: [[Double]] = [[6.5], [2.5], [4.0]]
        XCTAssertEqual(pipeline.predict(testData), model.predict(scaler.transform(testData)))
    }

    // MARK: - Reducer Pipeline

    // Three correlated features, two separable classes — the reducer suite's
    // shared fixture.
    private let reducerFeatures: [[Double]] = [
        [1.0, 2.0, 3.0], [1.2, 2.3, 3.1], [0.9, 1.8, 2.8],
        [6.0, 8.0, 9.0], [6.3, 8.4, 9.2], [5.8, 7.7, 8.9]
    ]
    private let reducerLabels = [0, 0, 0, 1, 1, 1]

    func testReducerPipelinePredicts() {
        let pipeline = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )

        // Raw features in — scaling and projection happen internally.
        let predictions = pipeline.predict([[1.1, 2.1, 3.0], [6.1, 8.1, 9.1]])
        XCTAssertEqual(predictions, [0, 1])
    }

    // The pipeline result must equal the hand-rolled scaler + PCA + model +
    // transformed-query path exactly — the bundling is a faithful shortcut.
    func testReducerPipelineMatchesManual() {
        let pipeline = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )

        let scaler = StandardScaler.fit(features: reducerFeatures)
        let scaled = scaler.transform(reducerFeatures)
        let reducer = PCA.fit(features: scaled, componentCount: 2)
        let model = KNearestNeighbors.fit(
            features: reducer.transform(scaled), labels: reducerLabels, k: 3
        )

        let testData: [[Double]] = [[1.1, 2.1, 3.0], [6.1, 8.1, 9.1], [3.5, 5.0, 6.0]]
        let manual = model.predict(reducer.transform(scaler.transform(testData)))
        XCTAssertEqual(pipeline.predict(testData), manual)

        // Expected labels cross-validated against an equivalent external
        // scale-project-classify pipeline on the same fixture.
        XCTAssertEqual(pipeline.predict(testData), [0, 1, 0])
    }

    func testReducerPipelineCodableRoundTrip() throws {
        let original = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )

        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)

        XCTAssertEqual(original, restored)
        let testData: [[Double]] = [[1.1, 2.1, 3.0], [6.1, 8.1, 9.1]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }

    // A reducer-free pipeline must encode without a reducer key — the exact
    // archive shape shipped before the reducer stage existed — and such an
    // archive must decode with reducer nil and predict identically.
    func testLegacyArchiveDecodesWithNilReducer() throws {
        let original = Pipeline.fit(features: reducerFeatures, labels: reducerLabels, k: 3)

        let data = try JSONEncoder().encode(original)
        if let json = String(data: data, encoding: .utf8) {
            XCTAssertFalse(json.contains("reducer"))
        } else {
            XCTFail("Encoded pipeline should be valid UTF-8")
        }

        let restored = try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)
        XCTAssertNil(restored.reducer)
        let testData: [[Double]] = [[1.1, 2.1, 3.0], [6.1, 8.1, 9.1]]
        XCTAssertEqual(original.predict(testData), restored.predict(testData))
    }

    // The validating decoder rejects an archive whose reducer disagrees with
    // the scaler on feature width, rather than constructing a pipeline whose
    // predict path would fail later.
    func testDecoderRejectsReducerWidthMismatch() throws {
        let scaler = StandardScaler.fit(features: reducerFeatures)  // 3 features

        let narrowFeatures: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let narrowReducer = PCA.fit(features: narrowFeatures, componentCount: 1)  // 2 features

        let model = KNearestNeighbors.fit(
            features: [[1.0], [2.0]], labels: [0, 1], k: 1
        )
        let mismatched = Pipeline(scaler: scaler, reducer: narrowReducer, model: model)

        let data = try JSONEncoder().encode(mismatched)
        XCTAssertThrowsError(try JSONDecoder().decode(Pipeline<KNearestNeighbors>.self, from: data)) { error in
            XCTAssertTrue(error is DecodingError)
        }
    }

    func testPipelineEquatableDiffersOnReducer() {
        let withReducer = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )
        let withoutReducer = Pipeline.fit(features: reducerFeatures, labels: reducerLabels, k: 3)

        XCTAssertNotEqual(withReducer, withoutReducer)
    }

    // The classifier trains on the projection, so its feature width is the
    // component count, not the raw column count.
    func testFittedModelWidthMatchesComponentCount() {
        let pipeline = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )

        XCTAssertEqual(pipeline.model.featureCount, 2)
        XCTAssertEqual(pipeline.reducer?.componentCount, 2)
        XCTAssertEqual(pipeline.reducer?.featureCount, 3)
    }

    func testReducerPipelineDescription() {
        let pipeline = Pipeline.fit(
            features: reducerFeatures, labels: reducerLabels,
            componentCount: 2, k: 3
        )

        XCTAssertTrue(pipeline.description.contains("reducer"))
        XCTAssertTrue(pipeline.description.contains("PCA"))

        // A reducer-free pipeline keeps its original description shape.
        let plain = Pipeline.fit(features: reducerFeatures, labels: reducerLabels, k: 3)
        XCTAssertFalse(plain.description.contains("reducer"))
    }

    // MARK: - Transformer Protocol

    // Both fitted stage types are usable through the Transformer existential.
    func testTransformerExistentialAppliesStages() {
        let scaler = StandardScaler.fit(features: reducerFeatures)
        let reducer = PCA.fit(features: scaler.transform(reducerFeatures), componentCount: 2)

        let stages: [any Transformer] = [scaler, reducer]

        var transformed = reducerFeatures
        for stage in stages {
            transformed = stage.transform(transformed)
        }

        XCTAssertEqual(transformed.count, reducerFeatures.count)
        XCTAssertEqual(transformed[0].count, 2)
    }
}
