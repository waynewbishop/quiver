// Copyright 2026 Wayne W Bishop. All rights reserved.
// Licensed under the Apache License, Version 2.0.

import XCTest
@testable import Quiver

/// Verifies all public types conform to Sendable and can cross
/// concurrency boundaries without warnings under Swift 6.
final class SendableTests: XCTestCase {

    func testModelsSendableAcrossTaskBoundary() async {
        let lr = try! LinearRegression.fit(features: [[1],[2],[3]], targets: [2.0, 4.0, 6.0])
        let knn = KNearestNeighbors.fit(features: [[1,2],[3,4]], labels: [0,1], k: 1)
        let km = KMeans.fit(data: [[1,2],[3,4],[5,6]], k: 2, seed: 42)
        let nb = GaussianNaiveBayes.fit(features: [[1,2],[3,4]], labels: [0,1])
        let scaler = FeatureScaler.fit(features: [[1,2],[3,4]])

        // Sending models into a Task requires Sendable conformance.
        // Without it, Swift 6 strict concurrency produces an error.
        let result = await Task {
            let lrPred = lr.predict([[4]])
            let knnPred = knn.predict([[2,3]])
            let kmPred = km.predict([[2,3]])
            let nbPred = nb.predict([[2,3]])
            let scaled = scaler.transform([[2,3]])
            return (lrPred.count, knnPred.count, kmPred.count, nbPred.count, scaled.count)
        }.value

        XCTAssertEqual(result.0, 1)
        XCTAssertEqual(result.1, 1)
        XCTAssertEqual(result.2, 1)
        XCTAssertEqual(result.3, 1)
        XCTAssertEqual(result.4, 1)
    }

    func testValueTypesSendableAcrossTaskBoundary() async {
        let cm = [0, 1, 1].confusionMatrix(actual: [0, 1, 0])
        let panel = Panel([("col", [1.0, 2.0, 3.0])])
        let frac = Fraction(numerator: 1, denominator: 3)

        let result = await Task {
            return (cm.accuracy, panel.rowCount, frac.value)
        }.value

        XCTAssertGreaterThan(result.0, 0)
        XCTAssertEqual(result.1, 3)
        XCTAssertGreaterThan(result.2, 0)
    }

    func testRetrievalTypesSendableAcrossTaskBoundary() async {
        var index = EmbeddingIndex<String>(embedder: SendableTestEmbedder())
        index.add("fast race", label: "racer")
        index.add("soft cushion", label: "trainer")
        let snapshot = index.snapshot
        let retrieval = index.retrieve("fast", k: 1)

        // The index, its snapshot and a retrieval result all cross into the task.
        let result = await Task {
            let restored = EmbeddingIndex(snapshot, embedder: SendableTestEmbedder())
            return (index.retrieve("soft", k: 1).hits.first?.label,
                    restored.count,
                    retrieval.hits.first?.label)
        }.value

        XCTAssertEqual(result.0, "trainer")
        XCTAssertEqual(result.1, 2)
        XCTAssertEqual(result.2, "racer")
    }

    func testGeneratorAndIteratorSendableAcrossTaskBoundary() async {
        let rng = SeededRandomNumberGenerator(seed: 42)
        var reference = SeededRandomNumberGenerator(seed: 42)
        let expected = reference.next()
        let iterator = Panel([("col", [1.0, 2.0, 3.0])]).makeIterator()

        // A seeded generator and a row iterator are values that cross into the task.
        let result = await Task {
            var generator = rng
            var rows = iterator
            return (generator.next(), rows.next()?["col"])
        }.value

        XCTAssertEqual(result.0, expected)
        XCTAssertEqual(result.1, 1.0)
    }
}

/// A two-dimensional embedder that maps the words "fast" and "soft" to separate axes.
private struct SendableTestEmbedder: Embedder {
    func embed(_ text: String) -> [Double]? {
        let words = text.split(separator: " ")
        let fast = words.contains("fast") ? 1.0 : 0.0
        let soft = words.contains("soft") ? 1.0 : 0.0
        return fast + soft == 0 ? nil : [fast, soft]
    }
}
