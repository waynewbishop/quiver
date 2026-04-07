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
}
