// Copyright 2025 Wayne W Bishop. All rights reserved.
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

final class LinearRegressionTests: XCTestCase {

    // Perfect linear data — model should recover exact coefficients
    func testPerfectLinearFit() throws {
        // y = 2x + 3
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [5.0, 7.0, 9.0, 11.0, 13.0]

        let model = try LinearRegression.fit(features: features, targets: targets)

        XCTAssertEqual(model.coefficients[0], 3.0, accuracy: 1e-9)  // intercept
        XCTAssertEqual(model.coefficients[1], 2.0, accuracy: 1e-9)  // slope
        XCTAssertEqual(model.featureCount, 1)
        XCTAssertTrue(model.hasIntercept)
    }

    // Multi-feature regression — y = 1 + 2x₁ + 3x₂
    func testMultiFeature() throws {
        let features: [[Double]] = [
            [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0], [2.0, 3.0]
        ]
        // y = 1 + 2*x1 + 3*x2
        let targets = features.map { 1.0 + 2.0 * $0[0] + 3.0 * $0[1] }

        let model = try LinearRegression.fit(features: features, targets: targets)

        XCTAssertEqual(model.coefficients[0], 1.0, accuracy: 1e-9)  // intercept
        XCTAssertEqual(model.coefficients[1], 2.0, accuracy: 1e-9)  // x1 weight
        XCTAssertEqual(model.coefficients[2], 3.0, accuracy: 1e-9)  // x2 weight
        XCTAssertEqual(model.featureCount, 2)
    }

    // Prediction produces expected values
    func testPredict() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [3.0, 5.0, 7.0, 9.0]  // y = 2x + 1

        let model = try LinearRegression.fit(features: features, targets: targets)
        let predictions = model.predict([[5.0], [10.0]])

        XCTAssertEqual(predictions[0], 11.0, accuracy: 1e-9)
        XCTAssertEqual(predictions[1], 21.0, accuracy: 1e-9)
    }

    // No intercept — y = 2x passes through origin
    func testNoIntercept() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]  // y = 2x

        let model = try LinearRegression.fit(
            features: features, targets: targets, intercept: false
        )

        XCTAssertEqual(model.coefficients.count, 1)
        XCTAssertEqual(model.coefficients[0], 2.0, accuracy: 1e-9)
        XCTAssertFalse(model.hasIntercept)
    }

    // R² should be close to 1.0 for near-perfect fit
    func testRSquared() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [2.1, 3.9, 6.1, 8.0, 9.8]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let predictions = model.predict(features)
        let r2 = predictions.rSquared(actual: targets)

        XCTAssertGreaterThan(r2, 0.99)
    }

    // Single-feature convenience predict accepts [Double] instead of [[Double]]
    func testSingleFeaturePredict() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [3.0, 5.0, 7.0, 9.0]  // y = 2x + 1

        let model = try LinearRegression.fit(features: features, targets: targets)

        // Convenience overload
        let convenience = model.predict([5.0, 10.0])
        // Standard overload
        let standard = model.predict([[5.0], [10.0]])

        XCTAssertEqual(convenience, standard)
        XCTAssertEqual(convenience[0], 11.0, accuracy: 1e-9)
        XCTAssertEqual(convenience[1], 21.0, accuracy: 1e-9)
    }

    // Single-feature convenience fit accepts [Double] instead of [[Double]]
    func testSingleFeatureFit() throws {
        let features = [1.0, 2.0, 3.0, 4.0]
        let targets  = [3.0, 5.0, 7.0, 9.0]  // y = 2x + 1

        // Convenience overload
        let convenience = try LinearRegression.fit(features: features, targets: targets)
        // Standard overload
        let standard = try LinearRegression.fit(features: features.map { [$0] }, targets: targets)

        // Both should produce the same coefficients
        XCTAssertEqual(convenience.coefficients, standard.coefficients)
        XCTAssertEqual(convenience.featureCount, 1)

        // Predictions should match
        let convPred = convenience.predict([5.0, 10.0])
        let stdPred  = standard.predict([5.0, 10.0])
        XCTAssertEqual(convPred, stdPred)
        XCTAssertEqual(convPred[0], 11.0, accuracy: 1e-9)
    }

    // Full pipeline: trainTestSplit → fit → predict → evaluate
    func testFullPipeline() throws {
        // Generate noisy linear data: y ≈ 3x + 2
        var features: [[Double]] = []
        var targets: [Double] = []

        for i in 0..<50 {
            let x = Double(i) * 0.2
            features.append([x])
            targets.append(3.0 * x + 2.0 + Double.random(in: -0.5...0.5))
        }

        // Split
        let (trainX, testX) = features.trainTestSplit(testRatio: 0.2, seed: 42)
        let (trainY, testY) = targets.trainTestSplit(testRatio: 0.2, seed: 42)

        // Fit and predict
        let model = try LinearRegression.fit(features: trainX, targets: trainY)
        let predictions = model.predict(testX)

        // Evaluate
        let r2 = predictions.rSquared(actual: testY)
        let rmse = predictions.rootMeanSquaredError(actual: testY)

        XCTAssertGreaterThan(r2, 0.95)
        XCTAssertLessThan(rmse, 1.0)
    }

    // Description shows intercept and slope for single-feature model
    func testDescriptionSingleFeature() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [3.0, 5.0, 7.0, 9.0]  // y = 2x + 1
        let model = try LinearRegression.fit(features: features, targets: targets)
        let desc = model.description
        XCTAssertTrue(desc.contains("1 feature"))
        XCTAssertTrue(desc.contains("intercept:"))
        XCTAssertTrue(desc.contains("slope:"))
    }

    // Description shows intercept and weights for multi-feature model
    func testDescriptionMultiFeature() throws {
        let features: [[Double]] = [
            [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0], [2.0, 3.0]
        ]
        let targets = features.map { 1.0 + 2.0 * $0[0] + 3.0 * $0[1] }
        let model = try LinearRegression.fit(features: features, targets: targets)
        let desc = model.description
        XCTAssertTrue(desc.contains("2 features"))
        XCTAssertTrue(desc.contains("intercept:"))
        XCTAssertTrue(desc.contains("weights: ["))
    }

    // Description shows slope without intercept when intercept is disabled
    func testDescriptionNoIntercept() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]  // y = 2x
        let model = try LinearRegression.fit(
            features: features, targets: targets, intercept: false
        )
        let desc = model.description
        XCTAssertFalse(desc.contains("intercept"))
        XCTAssertTrue(desc.contains("slope:"))
    }

    // Same training data produces equal models
    func testLinearRegressionEquatable() throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [3.0, 5.0, 7.0, 9.0]
        let model1 = try LinearRegression.fit(features: features, targets: targets)
        let model2 = try LinearRegression.fit(features: features, targets: targets)
        XCTAssertEqual(model1, model2)
    }
}
