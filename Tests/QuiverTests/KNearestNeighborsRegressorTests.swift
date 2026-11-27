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

import XCTest
@testable import Quiver

/// Tests for `KNearestNeighborsRegressor`.
///
/// Reference predictions were computed independently with scikit-learn's `KNeighborsRegressor`
/// (uniform and distance weights; Euclidean, Manhattan and Chebyshev metrics) and agree to 1e-9.
final class KNearestNeighborsRegressorTests: XCTestCase {

    private let sqft = [[1000.0], [1500.0], [2000.0], [2500.0], [3000.0]]
    private let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

    private let points = [[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [5.0, 4.0], [3.0, 3.0]]
    private let pointTargets = [10.0, 12.0, 30.0, 32.0, 20.0]

    // MARK: - Predictions

    func testUniformAverageMatchesReference() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        let predictions = model.predict([[1800.0], [2750.0], [1500.0]])
        XCTAssertEqual(predictions[0], 230000.0, accuracy: 1e-6)
        XCTAssertEqual(predictions[1], 340000.0, accuracy: 1e-6)
        XCTAssertEqual(predictions[2], 175000.0, accuracy: 1e-6)
    }

    func testDistanceWeightedMatchesReference() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2, weight: .distance)
        let predictions = model.predict([[1800.0], [2750.0]])
        XCTAssertEqual(predictions[0], 236000.0, accuracy: 1e-6)
        XCTAssertEqual(predictions[1], 340000.0, accuracy: 1e-6)
    }

    func testExactMatchWithDistanceWeightingReturnsThatTarget() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2, weight: .distance)
        XCTAssertEqual(model.predict([[1500.0]])[0], 200000.0, accuracy: 1e-9)
    }

    func testExactMatchesAreAveragedWhenSeveralShareTheSamePoint() {
        let features = [[1.0], [1.0], [5.0]]
        let targets = [10.0, 20.0, 100.0]
        let model = KNearestNeighborsRegressor.fit(features: features, targets: targets, k: 3, weight: .distance)
        XCTAssertEqual(model.predict([[1.0]])[0], 15.0, accuracy: 1e-12)
    }

    func testMetricsMatchReference() {
        let queries = [[2.5, 2.5], [4.5, 4.5], [1.0, 2.0]]

        let euclidean = KNearestNeighborsRegressor.fit(
            features: points, targets: pointTargets, k: 3, metric: .euclidean, weight: .distance
        ).predict(queries)
        XCTAssertEqual(euclidean[0], 15.75077641, accuracy: 1e-7)
        XCTAssertEqual(euclidean[1], 29.42857143, accuracy: 1e-7)
        XCTAssertEqual(euclidean[2], 10.0, accuracy: 1e-9)

        let manhattan = KNearestNeighborsRegressor.fit(
            features: points, targets: pointTargets, k: 3, metric: .manhattan, weight: .distance
        ).predict(queries)
        XCTAssertEqual(manhattan[0], 15.5, accuracy: 1e-9)
        XCTAssertEqual(manhattan[1], 29.42857143, accuracy: 1e-7)

        let chebyshev = KNearestNeighborsRegressor.fit(
            features: points, targets: pointTargets, k: 3, metric: .chebyshev, weight: .distance
        ).predict(queries)
        XCTAssertEqual(chebyshev[0], 16.4, accuracy: 1e-9)
        XCTAssertEqual(chebyshev[1], 29.42857143, accuracy: 1e-7)
    }

    func testUniformPredictionsMatchReferenceForTwoFeatures() {
        let model = KNearestNeighborsRegressor.fit(features: points, targets: pointTargets, k: 3)
        let predictions = model.predict([[2.5, 2.5], [4.5, 4.5], [1.0, 2.0]])
        XCTAssertEqual(predictions[0], 14.0, accuracy: 1e-9)
        XCTAssertEqual(predictions[1], 27.333333333333332, accuracy: 1e-9)
        XCTAssertEqual(predictions[2], 14.0, accuracy: 1e-9)
    }

    func testKEqualToTrainingSizePredictsTheTargetMean() {
        let model = KNearestNeighborsRegressor.fit(features: points, targets: pointTargets, k: 5)
        XCTAssertEqual(model.predict([[0.0, 0.0]])[0], 20.8, accuracy: 1e-12)
        XCTAssertEqual(model.predict([[100.0, -50.0]])[0], 20.8, accuracy: 1e-12)
    }

    func testKOfOneReproducesTheNearestTarget() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 1)
        XCTAssertEqual(model.predict([[1900.0]])[0], 260000.0, accuracy: 1e-9)
        XCTAssertEqual(model.predict([[1600.0]])[0], 200000.0, accuracy: 1e-9)
    }

    func testPredictionsStayWithinTheRangeOfTheTargets() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 3)
        // An average of training targets can never extrapolate beyond them.
        let outside = model.predict([[500.0], [9000.0]])
        XCTAssertTrue(outside.allSatisfy { $0 >= 150000.0 && $0 <= 370000.0 })
    }

    func testTiesBreakByTrainingOrder() {
        // Two training points are equally far from the query; with k = 1 the earlier one wins.
        let model = KNearestNeighborsRegressor.fit(
            features: [[0.0], [2.0]], targets: [1.0, 99.0], k: 1
        )
        for _ in 0..<5 {
            XCTAssertEqual(model.predict([[1.0]])[0], 1.0)
        }
    }

    func testScalarPredictOverloadComesFromRegressor() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        XCTAssertEqual(model.predict(1800.0), 230000.0, accuracy: 1e-6)
    }

    func testPredictionIsDeterministic() {
        let model = KNearestNeighborsRegressor.fit(features: points, targets: pointTargets, k: 3, weight: .distance)
        XCTAssertEqual(model.predict([[2.5, 2.5]]), model.predict([[2.5, 2.5]]))
    }

    // MARK: - Model properties

    func testFitStoresConfiguration() {
        let model = KNearestNeighborsRegressor.fit(
            features: points, targets: pointTargets, k: 4, metric: .manhattan, weight: .distance
        )
        XCTAssertEqual(model.k, 4)
        XCTAssertEqual(model.metric, .manhattan)
        XCTAssertEqual(model.weight, .distance)
        XCTAssertEqual(model.featureCount, 2)
        XCTAssertEqual(model.trainingFeatures, points)
        XCTAssertEqual(model.trainingTargets, pointTargets)
    }

    func testDefaultsAreThreeNeighborsEuclideanUniform() {
        let model = KNearestNeighborsRegressor.fit(features: points, targets: pointTargets)
        XCTAssertEqual(model.k, 3)
        XCTAssertEqual(model.metric, .euclidean)
        XCTAssertEqual(model.weight, .uniform)
    }

    func testEquatable() {
        let first = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        let same = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        let differentK = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 3)
        let differentWeight = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2, weight: .distance)
        XCTAssertEqual(first, same)
        XCTAssertNotEqual(first, differentK)
        XCTAssertNotEqual(first, differentWeight)
    }

    func testDescription() {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        XCTAssertEqual(model.description,
            "KNearestNeighborsRegressor: k=2, euclidean, 5 training points, 1 features")
    }

    func testCodableRoundTrip() throws {
        let original = KNearestNeighborsRegressor.fit(
            features: points, targets: pointTargets, k: 3, metric: .minkowski(p: 3.0), weight: .distance
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(KNearestNeighborsRegressor.self, from: data)
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.predict([[2.5, 2.5]]), original.predict([[2.5, 2.5]]))
    }

    func testSendableAcrossTaskBoundary() async {
        let model = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)

        // Sending a model into a Task requires Sendable conformance.
        let prediction = await Task { model.predict([[1800.0]]) }.value
        XCTAssertEqual(prediction[0], 230000.0, accuracy: 1e-6)
    }

    // MARK: - Protocol

    func testWorksAsARegressor() {
        let model: any Regressor = KNearestNeighborsRegressor.fit(features: sqft, targets: prices, k: 2)
        XCTAssertEqual(model.predict([[1800.0]])[0], 230000.0, accuracy: 1e-6)
    }

    func testFullPipelineSplitFitPredictEvaluate() {
        // A smooth, non-linear target: nearest neighbors should track it closely.
        var features = [[Double]]()
        var targets = [Double]()
        for step in 0..<40 {
            let x = Double(step) * 0.25
            features.append([x])
            targets.append(x * x)
        }

        // The same seed shuffles both arrays identically, so the rows stay paired.
        let trainFeatures = features.trainTestSplit(testRatio: 0.25, seed: 7).train
        let testFeatures = features.trainTestSplit(testRatio: 0.25, seed: 7).test
        let trainTargets = targets.trainTestSplit(testRatio: 0.25, seed: 7).train
        let testTargets = targets.trainTestSplit(testRatio: 0.25, seed: 7).test
        XCTAssertEqual(trainFeatures.count, trainTargets.count)
        XCTAssertEqual(testFeatures.count, testTargets.count)

        let model = KNearestNeighborsRegressor.fit(
            features: trainFeatures, targets: trainTargets, k: 2, weight: .distance
        )
        let predicted = model.predict(testFeatures)
        let error = predicted.rootMeanSquaredError(actual: testTargets)

        // A model that always predicts the mean of the targets is the baseline to beat.
        let baselineValue = trainTargets.mean() ?? 0.0
        let baseline = [Double](repeating: baselineValue, count: testTargets.count)
        let baselineError = baseline.rootMeanSquaredError(actual: testTargets)
        XCTAssertLessThan(error, baselineError / 4.0)
    }

    // MARK: - Preconditions are exercised in the type's contract: empty, mismatched, and bad k
    // all trap by design, matching `KNearestNeighbors`, so they are documented rather than run.
}
